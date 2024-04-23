import gin
import numpy as np
import torch
from einops import *
from scipy.spatial.transform import Rotation
from torch.linalg import *

from multi_slam.dpvo.lietorch import SE3, SO3
from .matrix_utils import *

@torch.no_grad()
def check_transl_mag(s, thresh=0.001):
    t_op, _ = torch.split(s.data, [3, 4], dim=-1)
    return torch.all(t_op.norm(dim=-1) > thresh)

def block_solve(A, B, ep=0.1, lm=0.0001):
    """ block matrix solve """
    *shapes, _, _ = A.shape
    A = A.view(np.prod(shapes, dtype=int), 6, 6)
    B = B.view(np.prod(shapes, dtype=int), 6, 1)
    I = torch.eye(6).to(A)

    A = A + (ep + lm * A) * I

    # Scaling
    scaling = A.abs().mean(dim=[1,2], keepdim=True)
    A = A / scaling
    B = B / scaling

    X = CholeskySolver.apply(A, B)
    return X.view(*shapes, 6)

I = torch.eye(3)

d_skew_sym = rearrange(torch.as_tensor(device='cpu', dtype=torch.float, data=[
    [[0, 0, 0],
    [0, 0, -1],
    [0, 1, 0]],

    [[0, 0, 1],
    [0, 0, 0],
    [-1, 0, 0]],

    [[0, -1, 0],
    [1, 0, 0],
    [0, 0, 0]],
    ]), 'i1 o1 o2 -> o1 o2 i1')

# (3 x 3) x 3
def calc_dE_dt(R_op):
    assert tuple(R_op.shape)[-2:] == (3, 3)
    dE_dtx = einsum(R_op, I.to(R_op), '... k i, j l -> ... i j k l')
    return einsum(dE_dtx, d_skew_sym.to(R_op), '... i j k l, k l m -> ... i j m')

# 3 x 3
def calc_dt_dxit(t_op):
    return -make_skew_sym(t_op)

# (3 x 3) x 3
def calc_dE_dxi(R_op, t_op): # (exp(xi) @ R_op).T @ P
    assert_shapes(R_op, (3, 3), t_op, (3,))
    t_op_x = make_skew_sym(t_op) # ... 3 x 3
    P = make_skew_sym(T(t_op_x)) # ... 3 x (3 x 3) [cols] x [row]^
    return einsum(R_op, P, '... j i, ... c j k -> ... i c k')

def calc_dF_dE(L, R):
    assert_shapes(L, (3, 3), R, (3, 3))
    return einsum(L, R, '... i k, ... l j -> ... i j k l')

def calc_dabc_dF(h1):
    assert h1.shape[-1] == 3
    return einsum(I.to(h1), h1, '... i j, ... k -> ... i j k')

# 2 x 3
def calc_dres_dabc(abc, h2):
    assert_shapes(abc, (3,), h2, (2,))
    a, b, c = abc.unbind(-1)
    px, py = h2.unbind(-1)
    d = (a*a + b*b)
    l = a*px + b*py + c
    aod = a / d
    bod = b / d
    row1 = [-2*aod*aod*l + aod*px + l/d,
            -2*aod*b*l/d + aod*py, aod]
    row2 = [-2*bod*a*l/d + bod*px,
            -2*bod*bod*l + bod*py + l/d, bod]
    return torch.stack(row1 + row2, -1).unflatten(-1, (2, 3))

def calc_res(abc, h2):
    assert_shapes(abc, (3,), h2, (2,))
    dp = lambda x1, x2: (x1 * x2).sum(dim=-1, keepdim=True)
    ab, c = abc.split([2, 1], dim=-1)
    s = dp(h2, ab) + c
    v = ab * (s / dp(ab, ab))
    return v.flatten(-2)

def calc_se3_grad(R_op, t_op, p1, p2, KL, KR, unit_transl):
    dE_dxi = calc_dE_dxi(R_op, t_op) # (3 x 3) x 3 # correct
    dE_dt = calc_dE_dt(R_op) # (3 x 3) x 3 # correct
    dE_dxit = einsum(dE_dt, calc_dt_dxit(t_op), '... k m n, ... n l -> ... k m l') # (3 x 3) x 3
    assert dE_dxit.shape == dE_dt.shape

    h1 = make_homog(p1)
    dF_dE = calc_dF_dE(KL, KR) # (3 x 3) x (3 x 3)
    dabc_dF = calc_dabc_dF(h1) # 3 x (3 x 3)
    dabc_dE = einsum(dabc_dF, dF_dE, '... P m i j, ... i j k n -> ... P m k n') # 3 x 3 correct
    E = einsum(R_op, make_skew_sym(t_op), '... j i, ... j k -> ... i k') # 3 x 3
    F = einsum(KL, E, KR, '... i j, ... j k, ... k l -> ... i l')
    abc = einsum(F, h1, '... i j, ... P j -> ... P i')
    dres_dabc = calc_dres_dabc(abc, p2)
    dres_dE = einsum(dres_dabc, dabc_dE, '... i j, ... j k m -> ... i k m')

    residual = calc_res(abc, p2)
    dres_dxi = einsum(dres_dE, dE_dxi, '... P r k m, ... k m n -> ... P r n')
    if unit_transl:
        dres_dxit = einsum(dres_dE, dE_dxit, '... P r k m, ... k m n -> ... P r n')
        J_se3 = rearrange(torch.cat((dres_dxit, dres_dxi), dim=-1), '... P r v -> ... (P r) v')
    else:
        dres_dt = einsum(dres_dE, dE_dt, '... P r k m, ... k m n -> ... P r n')
        J_se3 = rearrange(torch.cat((dres_dt, dres_dxi), dim=-1), '... P r v -> ... (P r) v')

    return J_se3, residual

@gin.configurable
def optimal_algorithm(se3_op_f2s, p1, p2, w, intrinsics1, intrinsics2, iters, gradient_on_matches=True, zero_weights=False, unit_transl=False):

    assert check_transl_mag(se3_op_f2s)
    se3_op_s2f = se3_op_f2s.inv()
    K1 = make_K(intrinsics1) / 100
    K2 = make_K(intrinsics2) / 100
    KL = inv(T(K2))
    KR = inv(K1)
    t_op, qrot_op = torch.split(se3_op_s2f.data, [3, 4], dim=-1)
    so3_op = SO3(qrot_op)
    w = repeat(w, '... P 1 -> ... (P 2) 1')

    for itr in range(iters):
        # Calc gradient
        R_op = so3_op.matrix()[...,:3,:3]
        J_se3, res = calc_se3_grad(R_op, t_op, p1, p2, KL, KR, unit_transl)
        # print(f"[{itr}]Residuals: {res.mean().item():.04f}")

        if not gradient_on_matches:
            J_se3 = J_se3.detach()
            res = res.detach()

        if zero_weights:
            w = torch.zeros_like(w)

        # Construct lhs and rhs
        Jw_se3 = J_se3 * w
        JtJ = einsum(Jw_se3, J_se3, '... n s1, ... n s2 -> ... s1 s2')
        Jtr = einsum(Jw_se3, res, '... n s1, ... n -> ... s1')

        # Solve for update
        delta = block_solve(JtJ, -Jtr)
        delta_t, delta_xi = rearrange(delta, '... (n m) -> n ... m', n=2, m=3)

        # Apply update
        if unit_transl:
            t_op = SO3.exp(delta_t) * t_op
        else:
            t_op = t_op + delta_t
        so3_op = so3_op.retr(delta_xi)

    se3_op_s2f = SE3(torch.cat((t_op, so3_op.data), dim=-1))
    return se3_op_s2f.inv()
