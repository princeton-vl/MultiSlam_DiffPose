from pathlib import Path

import gin
import numpy as np
import torch
from einops import *
from torch.linalg import det, inv, lstsq, matrix_rank, svd

from multi_slam.dpvo.lietorch import SE3, SO3
from multi_slam.metrics import batch_angle_error

from ..utils.external_functions import matrix_to_quaternion
from .iterative_pose_opt import optimal_algorithm
from .matrix_utils import *
from ..utils.misc import GradSafe


def map_coords_using_intrinsics(intrinsics, coords):
    assert {**parse_shape(intrinsics, 'B LR _'), "uv": 2} == parse_shape(coords, 'B LR _ _ _ uv')
    Ks = make_K(intrinsics) # B LR 3 3
    new_Ks = Ks[:, [1,0]] @ inv(Ks)
    coords_h = make_homog(coords) # B LR M 3 3 h
    return einsum(new_Ks[:,:,:2], coords_h, 'B LR i j, B LR M p1 p2 j -> B LR M p1 p2 i')

def epipolar_residuals(p1, p2, f_hat):
    abc = einsum(f_hat, make_homog(p1), '... i j, ... P j -> ... P i')
    ab, c = torch.split(abc, [2, 1], dim=-1)
    dp = lambda x1, x2: (x1 * x2).sum(dim=-1, keepdim=True)
    return -(dp(p2, ab) + c) * (ab/dp(ab, ab).clamp(min=1e-6))

def make_bidir_pts(points_2d, pred_matches):
    points_2d, pred_matches = torch.broadcast_tensors(points_2d, pred_matches)
    pts1 = torch.cat((points_2d[...,0,:,:], pred_matches[...,1,:,:]), dim=-2)
    pts2 = torch.cat((pred_matches[...,0,:,:], points_2d[...,1,:,:]), dim=-2)
    return torch.stack((pts1, pts2))

def prune_low_conf(w):
    thresh = torch.quantile(w, 0.2, dim=1, keepdim=True)
    return w * (thresh < w)

def project_pts(pts1, depth1, Ps, intrinsics):
    Ks = make_K(intrinsics)
    assert parse_shape(pts1, '... LR _ _') == parse_shape(Ks, '... LR _ _') == parse_shape(depth1, '... LR _ _')
    pts1 = make_homog(pts1) # B LR M uv1
    pts1_3d = einsum(pts1, inv(Ks), '... LR M j, ... LR i j -> ... LR M i') * depth1
    rel_pose = Ps[:,[1,0]] * Ps.inv()
    pts2_3d = rel_pose[:,:,None] * pts1_3d
    pts2 = einsum(pts2_3d, Ks[:,[1,0]], '... LR M j, ... LR i j -> ... LR M i')
    depth2 = pts2[...,2]
    pts2 = pts2[...,:2] / depth2[...,None]
    return pts2, depth2 > 0.001

@torch.no_grad()
def check_in_range(pts, bounds):
    assert parse_shape(pts, 'B LR _ uv') == parse_shape(bounds, 'B LR uv')
    bounds2 = bounds[:,[1,0],None]
    return ((pts >= 0) & (pts <= (bounds2-1))).all(dim=-1)

def id_pose_unit_t(*args, **kwargs):
    p = SE3.Identity(*args, **kwargs)
    p.data[...,2] = 1.0
    return p

def solve_for_pose(E):
    W = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).to(E)
    Z = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 0]]).to(E)

    U, SIG, VT = svd(E)
    t = U @ Z @ T(U)
    r1 = U @ W @ VT
    r2 = U @ T(W) @ VT
    pose_guesses = [
        (-t, r1),
        (-t, r2),
        ( t, r1),
        ( t, r2)
    ]

    se3_pose_guesses = []

    for i, (tx_guess, rot_guess) in enumerate(pose_guesses):
        rot_guess = torch.where(det(rot_guess)[...,None,None] > 0, 1, -1) * rot_guess
        assert (det(rot_guess) - 1).abs().max() < 1e-3
        assert check_skew_sym(tx_guess)
        assert check_orthonormal(rot_guess)
        t_guess = tx_guess[...,[2,0,1],[1,2,0]]
        q_guess = matrix_to_quaternion(rot_guess)[...,[1,2,3,0]]
        se3 = torch.cat((t_guess, q_guess), dim=-1)
        se3_pose_guesses.append(se3)

    return SE3(torch.stack(se3_pose_guesses, dim=-2))

def make_diag(t):
    dg = torch.as_tensor([1.,1.,0.]).to(t).diag()
    return t.unsqueeze(-1).expand(*([-1]*t.ndim), 3) * dg

def solve_rank2_svd(Y):
    *B, N, nine = Y.shape
    assert nine == 9
    _, _, V = svd(Y)
    f_vec = V[...,-1,:]
    f_hat_imp = f_vec.view(*B, 3, 3)

    # Enforce rank(F) = 2 
    s,v,d = svd(f_hat_imp)
    assert v.shape[-1] == 3
    f_hat = s @ make_diag(v) @ d
    residuals = torch.einsum('...ij,...j->...i', Y, f_vec)
    assert f_hat.requires_grad == residuals.requires_grad == Y.requires_grad
    return f_hat, residuals, f_hat_imp

def rescaleA2B(a, b):
    assert isinstance(a, SE3) and isinstance(b, SE3)
    sc = b.data[...,:3].norm(dim=-1) / a.data[...,:3].norm(dim=-1)
    sc = torch.where(torch.isinf(sc) | torch.isnan(sc), 1, sc)
    return a.scale(sc)

@gin.configurable
@torch.no_grad()
def chirality_test(pred_poses, w, K1, K2, p1, p2, chir_thresh=0.0):
    *_, N, _ = p1.shape
    h1 = make_homog(p1)
    h2 = make_homog(p2)
    h1 = einsum(inv(K1), h1, '... i j, ... N j -> ... N i')
    h2 = einsum(inv(K2), h2, '... i j, ... N j -> ... N i')
    m = pred_poses.inv().matrix()
    rot, transl = m[...,:3,:3], m[...,:3,3]
    rh2 = einsum(rot, h2, '... C i j, ... N j -> ... C N i')
    h1_expanded = repeat(h1, '... N i -> ... 4 N i', N=N, i=3)
    h1_rh2 = torch.stack((h1_expanded, -rh2), dim=-1)
    transl_expanded = repeat(transl, '... C i -> ... C N i 1', C=4, i=3, N=N)
    res = lstsq(h1_rh2, transl_expanded)
    z = res.solution.squeeze(-1)
    pos_z = (z > 0).all(dim=-1)
    th = w >= w.quantile(torch.as_tensor(chir_thresh).cuda(), dim=-2, keepdim=True)
    pos_z = pos_z & rearrange(th, '... N 1 -> ... 1 N', N=N)
    num_pos_z = pos_z.sum(dim=-1)
    return num_pos_z.argmax(dim=-1)

def closest_to_gt(pred_poses, gt_pose):
    pred_poses, gt_pose = map(lambda x: SE3(x.contiguous()), torch.broadcast_tensors(pred_poses.data, gt_pose.data.unsqueeze(-2)))
    pred_poses_scaled = rescaleA2B(pred_poses, gt_pose)
    assert pred_poses.shape == pred_poses_scaled.shape
    relative_poses = pred_poses_scaled * gt_pose.inv()
    magnitudes = relative_poses.log().norm(dim=-1)
    return magnitudes.argmin(dim=-1)


@gin.configurable
@torch.cuda.amp.autocast(True)
@torch.cuda.amp.custom_fwd(cast_inputs=torch.float)
def weighted_epa(p1, p2, w, intrinsics1, intrinsics2, gt_pose=None, image_size=None, calculate_pose=True, gradient_on_matches=True):
    *B, N, _ = p1.shape
    if not gradient_on_matches:
        p1 = p1.detach()
        p2 = p2.detach()
    if isinstance(gt_pose, torch.Tensor):
        gt_pose = SE3(gt_pose)
    assert tuple(p1.shape) == tuple(p2.shape) == (*B, N, 2), (p1.shape, p2.shape, (*B, N, 2))
    assert tuple(w.shape) == (*B, N, 1), (w.shape, (*B, N, 1))
    torch.broadcast_shapes(intrinsics1.shape, intrinsics2.shape, (*B, 4))
    (gt_pose is None) or torch.broadcast_shapes(gt_pose.shape, B)
    # if p1.dtype != torch.double:
    #     warnings.warn(f"Running the Weighted 8-Point-Algorithm with precision={p1.dtype}")
    if image_size is not None:
        H, W = image_size
        s = torch.as_tensor([2 / W, 2 / H]).cuda()
        p1 = p1 * s - 1
        p2 = p2 * s - 1
        intrinsics1 = intrinsics1 * s.repeat(2)
        intrinsics1[...,2:] = intrinsics1[...,2:] - 1
        intrinsics2 = intrinsics2 * s.repeat(2)
        intrinsics2[...,2:] = intrinsics2[...,2:] - 1

    x1, y1 = p1.unbind(-1)
    x2, y2 = p2.unbind(-1)
    i = torch.ones_like(x1)
    Y = torch.stack((
        x2 * x1,
        x2 * y1,
        x2,
        y2 * x1,
        y2 * y1,
        y2,
        x1,
        y1,
        i
    ), dim=-1)
    # Y = Y / Y.abs().mean(dim=[-1,-2], keepdim=True)
    Y = Y * w
    Y = GradSafe.apply(Y)

    f_hat, residuals, f_hat_imp = solve_rank2_svd(Y)
    assert torch.all(matrix_rank(f_hat) <= 2)

    if not calculate_pose:
        return f_hat

    K1 = make_K(intrinsics1)
    K2 = make_K(intrinsics2)
    E_hat = T(K2) @ f_hat @ K1
    E_hat = GradSafe.apply(E_hat)
    E_hat_imp = T(K2) @ f_hat_imp @ K1
    _, E_hat_imp_singv, _ = svd(E_hat_imp)
    pred_poses = solve_for_pose(E_hat)

    if gt_pose is None:
        chosen_idx = chirality_test(pred_poses, w, K1, K2, p1, p2)
    else:
        chosen_idx = closest_to_gt(pred_poses, gt_pose)
    chosen_idx = repeat(chosen_idx, '... -> ... 1 7')
    best_pred_pose = SE3(torch.gather(input=pred_poses.data, dim=-2, index=chosen_idx).squeeze(dim=-2))
    return best_pred_pose.float(None)

def se3_to_E(s): # s: first -> second
    rot, transl = s.inv().matrix()[...,:3,:].split([3,1], -1)
    tx = make_skew_sym(transl.squeeze(-1))
    return T(rot) @ tx

def se3_to_F(s, intrinsics):
    E = se3_to_E(s)
    K1, K2 = make_K(intrinsics).mul(1/100).unbind(dim=-3)
    F = inv(T(K2)) @ E @ inv(K1)
    return torch.stack((F, T(F)), dim=1)

def eval(name, pred, gt):
    ang_errs = batch_angle_error(asnumpy(pred.matrix().view(-1, 4, 4)), asnumpy(gt.matrix().view(-1, 4, 4)))
    rot_ang = ang_errs['R']
    transl_ang = ang_errs['T']
    print(f"{name.ljust(34)}: Rotation: {rot_ang.max():.04f} Translation: {transl_ang.max():.04f}")
