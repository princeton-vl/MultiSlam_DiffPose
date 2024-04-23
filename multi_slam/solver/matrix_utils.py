import torch
from torch.linalg import *

T = lambda x: x.transpose(-2, -1)

def make_skew_sym(v):
    *B, _ = v.shape
    out = torch.eye(3).to(v).tile(*B, 1, 1)
    out[...,[2,0,1],[1,2,0]] = v
    return out - T(out)

@torch.no_grad()
def check_skew_sym(x):
    return torch.round(T(x) + x).abs().max() < 1e-3

@torch.no_grad()
def check_orthonormal(x):
    return torch.round(inv(x) - T(x)).abs().max() < 1e-3

def make_K(intr):
    *B, _ = intr.shape
    K = torch.eye(3).to(intr).tile(*B, 1, 1)
    K[...,[0,1,0,1],[0,1,2,2]] = intr
    return K

class CholeskySolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b):
        # don't crash training if cholesky decomp fails
        U, info = torch.linalg.cholesky_ex(H)

        if torch.any(info):
            ctx.failed = True
            return torch.zeros_like(b)

        xs = torch.cholesky_solve(b, U)
        ctx.save_for_backward(U, xs)
        ctx.failed = False

        return xs

    @staticmethod
    def backward(ctx, grad_x):
        if ctx.failed:
            return None, None

        U, xs = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x, U)
        dH = -torch.matmul(xs, dz.transpose(-1,-2))

        return dH, dz

def assert_shape(x, *b):
    assert tuple(x.shape) == b, f"Expected {b} Got {x.shape}"

def assert_shapes(a, a_shape, b, b_shape):
    full_a_shape = tuple(a.shape)
    full_b_shape = tuple(b.shape)
    assert full_a_shape[:-len(a_shape)] == full_b_shape[:-len(b_shape)]
    assert full_a_shape[-len(a_shape):] == a_shape
    assert full_b_shape[-len(b_shape):] == b_shape

def make_homog(x, dim = -1):
    o = torch.ones_like(x).mean(dim=dim, keepdim=True)
    h = torch.cat((x, o), dim=dim)
    return h