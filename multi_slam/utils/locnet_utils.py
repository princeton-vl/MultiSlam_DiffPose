import torch
from einops import *
from torch.linalg import inv

from multi_slam.dpvo.lietorch import SE3, Sim3
from .misc import isbad
from multi_slam.solver.matrix_utils import make_homog, make_K


@torch.no_grad()
def compute_depth(src_2d, dst_2d, poses, intrinsics):
    """ poses are assumed to be "global" """
    assert parse_shape(src_2d, 'B LR M uv') == parse_shape(dst_2d, 'B LR M uv')
    assert parse_shape(intrinsics, 'B LR _') == parse_shape(dst_2d, 'B LR _ _')
    sh = parse_shape(src_2d, 'B LR M _')
    transf_se3 = SE3(torch.stack((poses.inv().data, poses.data), dim=1)) # src -> dst
    transf = transf_se3.matrix()
    transl = transf[...,:3,3]
    rot = transf[...,:3,:3]
    inv_Ks = inv(make_K(intrinsics))

    # Uncalibrate points
    h1 = make_homog(src_2d)
    h1 = einsum(inv_Ks, h1, 'B LR i j, B LR M j -> B LR M i')
    h2 = make_homog(dst_2d)
    h2 = einsum(inv_Ks[:,[1,0]], h2, 'B LR i j, B LR M j -> B LR M i')

    # Compute
    rh1 = einsum(rot, h1, 'B LR i j, B LR M j -> B LR M i')
    rh1_h2 = torch.stack((-rh1, h2), dim=-1)
    assert parse_shape(rh1_h2, 'B LR M _ _') == sh
    transl = repeat(transl, 'B LR i -> B LR M i 1', **sh)
    res = torch.linalg.lstsq(rh1_h2, transl)
    z = res.solution.squeeze(-1)

    # Sanity check
    # diff = (transf_se3[:,:,None] * (z[...,[0]] * h1)) - (z[...,[1]]*h2)
    # assert diff.abs().max() < 1e-4, diff.abs().median()

    return z

def inv_proj(p, intrinsics):
    sh = parse_shape(p, 'B _ _ _')
    u, v, d = rearrange(p, 'B LR M uvd -> uvd B LR M', uvd=3, LR=2, **sh)
    fx, fy, cx, cy = rearrange(intrinsics, 'B LR f -> f B LR 1', f=4, LR=2, **sh)
    z = 1/d
    x = (z/fx) * (u-cx)
    y = (z/fy) * (v-cy)
    return torch.stack((x, y, z), dim=-1)

def proj(p, intrinsics):
    sh = parse_shape(p, 'B _ _ _')
    x, y, z = rearrange(p, 'B LR M xyz -> xyz B LR M', xyz=3, LR=2, **sh)
    fx, fy, cx, cy = rearrange(intrinsics, 'B LR f -> f B LR 1', f=4, LR=2, **sh)
    u = (x * (fx/z)) + cx
    v = (y * (fy/z)) + cy
    return torch.stack((u, v), dim=-1), z

def compute_sim3(Gs, centroids, pre_update_matches, intrinsics, patch_disps, val_depth, dir_weights, depth_conf):

    # Triangulate depth
    z = compute_depth(centroids, pre_update_matches, Gs.inv(), intrinsics)[...,0]
    patch_depth = 1 / patch_disps.squeeze(3)
    assert z.shape ==  patch_depth.shape == val_depth.shape
    assert not isbad(patch_depth / z)

    # Mask to exclude obviously untrustworthy points
    valid_depth_mask = (dir_weights > 0.01) & val_depth & (z < 100) & (z > 1e-4) & (depth_conf > 0.7)

    # Compute ratio of SLAM depth to triangulated depth
    scale_ratios = (patch_depth / z)

    # Compare the scale differences between all pairs of points. We want to find a set of points which all agree
    scale_ratios_comparison = (scale_ratios[:,:,None,:] / scale_ratios[:,:,:,None]).log().abs().exp()
    scale_ratios_comparison[~(valid_depth_mask[:,:,None,:] & valid_depth_mask[:,:,:,None])] = torch.inf
    scale_ratios_num_inliers = (scale_ratios_comparison < 1.1).sum(dim=-1)

    # Identify an inlier set
    num_inliers, most_inliers = scale_ratios_num_inliers.max(dim=2) # B LR 1
    scale_ratio_both = torch.gather(scale_ratios, 2, most_inliers.unsqueeze(2)).squeeze(2) # B LR 1
    scale_ratio = scale_ratio_both[:,[0]] / scale_ratio_both[:,[1]]

    # Set the translation magnitude
    Gs = Gs.scale(scale_ratio_both[:,0])

    # Set the Sim3 scale
    r2l = Sim3(torch.cat((Gs.inv().data, scale_ratio), dim=1))
    inliers = num_inliers.min(dim=1).values
    print(f"#Inliers={asnumpy(inliers)}")

    # Return the Sim3 transformation, the number of inliers, an 3D patch centroids (for convenience)
    return r2l, inliers
