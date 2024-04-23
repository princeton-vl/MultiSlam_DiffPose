import os
from collections import OrderedDict
from itertools import chain

import gin
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import *
from einops.layers.torch import Rearrange

from .dpvo import altcorr, fastba
from .dpvo import projective_ops as pops
from .dpvo.ba import BA
from .dpvo.blocks import GatedResidual, GradientClip, SoftAgg
from .dpvo.lietorch import SE3
from .dpvo.utils import flatmeshgrid, pyramidify, set_depth
from .modules import Patchifier
from .solver.matrix_utils import make_homog
from .utils.keypoint_extractors import random_keypoints, superpoint_keypoints
from .utils.misc import isbad
from . import homog_ops as hops

autocast = torch.cuda.amp.autocast


import multi_slam.solver.epa_ops as epops
from multi_slam.utils.locnet_utils import compute_sim3

DIM = 384

@gin.configurable
class Update(nn.Module):

    def __init__(self, p, num_gru, corr_planes, grid_rad, num_heads, transf_dim):
        super(Update, self).__init__()
        
        self.norm = nn.LayerNorm(DIM, eps=1e-3)

        self.agg_kk = SoftAgg(DIM)

        self.c1 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))

        self.c2 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))

        self.transformer = nn.Sequential(nn.Linear(DIM, transf_dim), nn.TransformerEncoderLayer(d_model=transf_dim, nhead=num_heads, batch_first=True, dropout=0.0), nn.Linear(transf_dim, DIM))

        self.gru = nn.Sequential(*[nn.Sequential(nn.LayerNorm(DIM, eps=1e-3), GatedResidual(DIM)) for _ in range(num_gru)])

        self.corr = nn.Sequential(
            nn.Linear(corr_planes*((grid_rad*2+1)**2)*p*p, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
        )

        self.res = nn.Sequential(
            nn.Linear(2, DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
        )

        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2 * 1),
            GradientClip())

        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 1),
            GradientClip(),
            Rearrange('... 1 -> ...'),
            nn.Sigmoid())

        self.w_gn = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 1),
            GradientClip(),
            Rearrange('... 1 -> ...'),
            nn.Sigmoid())

    @gin.configurable("update_forward")
    def forward(self, net, inp, corr, residual, ii, jj, kk=None):
        """ update operator """

        *sh, D = net.shape
        net = net.flatten(0, -2)
        inp = inp.flatten(0, -2)
        corr = corr.flatten(0, -2)

        net = net + inp + self.corr(corr)
        if residual is not None:
            net = net + self.res(residual.flatten(0, -2))
        net = self.norm(net)

        if kk is not None:
            net = net.unsqueeze(0)
            ix, jx = fastba.neighbors(kk, jj)
            mask_ix = (ix >= 0).float().reshape(1, -1, 1)
            mask_jx = (jx >= 0).float().reshape(1, -1, 1)
            net = net + self.c1(mask_ix * net[:,ix])
            net = net + self.c2(mask_jx * net[:,jx])
            net = net + self.agg_kk(net, kk)
            net = net.squeeze(0)

        _, ix, counts = torch.unique_consecutive(ii + 12345*jj, return_counts=True, return_inverse=True)
        M = counts[0].item()
        transformer_out = self.transformer(net.view(-1, M, D)).view(-1, D)
        net = net + transformer_out

        net = self.gru(net)

        net = net.view(*sh, D)

        return net.float(), (self.d(net).float(), self.w(net).float(), self.w_gn(net).float())


@gin.configurable
class CorrBlock:
    def __init__(self, fmap, gmap, radius=3, dropout=0.2, levels=[1,4]):
        self.dropout = dropout
        self.radius = radius
        self.levels = levels

        self.gmap = rearrange(gmap, 'N M ... -> 1 (N M) ...')
        self.pyramid = pyramidify(fmap[None], lvls=levels)

    def __call__(self, ii, jj, coords):
        coords = rearrange(coords, 'N p1 p2 uv -> 1 N uv p1 p2', uv=2)
        corrs = []
        for i in range(len(self.levels)):
            corrs += [ altcorr.corr(self.gmap, self.pyramid[i], coords / self.levels[i], ii, jj, self.radius, self.dropout) ]
        return corrs

@gin.configurable
def calc_gt_matches(pts1, depth1, depths, intrinsics, Ps, cycle_thresh=np.inf):
    pts2, valid_repr = epops.project_pts(pts1, depth1, Ps, intrinsics)
    _, _, H, W = depths.shape
    sampling_pts2 = rearrange(pts2, 'B LR M uv -> (B LR) M 1 uv') * torch.tensor([2/(W-1), 2/(H-1)], device='cuda') - 1
    sampled_depth2 = F.grid_sample(rearrange(depths[:,[1,0]], 'B LR H W -> (B LR) 1 H W'), sampling_pts2, align_corners=True)
    sampled_depth2 = rearrange(sampled_depth2, '(B LR) 1 M 1 -> B LR M 1', LR=2)
    pts1_cycle, _ = epops.project_pts(pts2, sampled_depth2, Ps[:,[1,0]], intrinsics[:,[1,0]])
    cycle_consistency = (pts1 - pts1_cycle).norm(dim=-1)
    cycle_consistency = (cycle_consistency < (cycle_thresh * max(H, W))) & (sampled_depth2.squeeze(-1) > 0.001)
    return pts2, valid_repr, cycle_consistency

def sort_idxs(ix, kk, jj):
    _, x = torch.sort(ix[kk]*12345 + jj)
    kk = kk[x]
    jj = jj[x]
    return kk, jj

@gin.configurable
class LocNet(nn.Module):
    def __init__(self, P):
        super().__init__()
        self.P = P
        self.patchify = Patchifier(self.P)
        self.update = Update(self.P)
        self.DIM = DIM

    @property
    def device(self):
        return next(self.parameters()).device

    def load_weights(self, path: str):
        assert os.path.exists(path)
        state_dict = torch.load(path)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        state_dict = OrderedDict([(k.replace('module.',''),v) for k,v in state_dict.items()])
        self.load_state_dict(state_dict)

    @gin.configurable("locnet_forward")
    def forward(self, images, intrinsics, kp_or_detector, Ps=None, depths=None, bounds=None, M=None, STEPS=12, opt_iters=5, depth_conf=None, use_autocast=False, homography_pretrain=False):
        """ Estimates SE3 or Sim3 between pair of frames """

        B, *_ = images.shape
        disps = (1/depths).flatten(0,1) if (depths is not None) else None
        boundsf = (bounds/8.0).flatten(0,1) if (bounds is not None) else None

        if isinstance(kp_or_detector, torch.Tensor): # Provided keypoints as input, all w/ depth
            keypoints = kp_or_detector
        elif isinstance(kp_or_detector, tuple): # Provided some keypoints w/ depth, and some w/o
            keypoints1, keypoints2 = kp_or_detector
            keypoints = torch.cat((keypoints1, make_homog(keypoints2)), dim=2)
        else: # Provided only a keypoint detector, and the number of keypoints to use (i.e, 'M')
            keypoints, _ = kp_or_detector(images, bounds, M)

        with autocast(enabled=use_autocast, dtype=torch.bfloat16):
            choices = self.patchify(images.mul(2/255).add(-1).flatten(0,1), disps=disps, bounds=boundsf, coords=keypoints.flatten(0, 1))
        imap = rearrange(choices['imap'], '(B LR) M DIM 1 1 -> B LR M DIM', B=B, LR=2, DIM=DIM)
        patches = rearrange(choices['patches'], '(B LR) ... uvd p1 p2 -> B LR ... p1 p2 uvd', B=B, LR=2)
        _, _, hh ,ww = choices['fmap']['8'].shape

        corr_fns = [
            (4, CorrBlock(choices['fmap']['2'], choices['gmap']['2'], levels=[1])),
            (2, CorrBlock(choices['fmap']['4'], choices['gmap']['4'], levels=[1])),
            (1, CorrBlock(choices['fmap']['8'], choices['gmap']['8'])),
        ]

        p = self.P

        disp = patches[...,2]
        MIN_DISP = 0.001
        MAX_DISP = 1000
        val_depth = ~(
            reduce(torch.isnan(disp), '... p1 p2 -> ...', 'max') | \
            reduce(torch.isinf(disp), '... p1 p2 -> ...', 'max') | \
            (reduce(disp, '... p1 p2 -> ...', 'min') < MIN_DISP) | \
            (reduce(disp, '... p1 p2 -> ...', 'max') > MAX_DISP)
        )
        invalid_depth = (repeat(~val_depth, '... -> ... P1 P2 3', P1=p, P2=p)) & torch.tensor([False, False, True]).cuda()
        patches[invalid_depth] = 1.0

        _, _, M, _ = keypoints.shape
        ii = torch.arange(B*2).cuda().repeat_interleave(M)
        jj = torch.where((ii % 2) == 0, ii+1, ii-1)
        kk = torch.arange(B*2*M).cuda()

        centroids, patch_disps = patches[...,p//2,p//2,:].split([2, 1], dim=-1)
        if homography_pretrain:
            assert hops.is_H(Ps)
            rescale_H = torch.as_tensor(np.diag([8, 8, 1])).to(Ps)
            Ps = hops.inv(rescale_H) @ Ps @ rescale_H
            gt_matches = hops.apply_H(centroids, Ps)
            valid_repr_disp = torch.ones(B, 2, M, dtype=torch.bool, device='cuda')
            cycle_consistency = valid_repr_disp.clone()
        elif (Ps is not None) and (depths is not None):
            gt_matches, valid_repr_disp, cycle_consistency = calc_gt_matches(centroids * 8.0, 1/patch_disps, depths, intrinsics, Ps)
            gt_matches /= 8.0
        else:
            valid_repr_disp, cycle_consistency = torch.ones(2, B, 2, M, device='cuda', dtype=bool)
            gt_matches = torch.ones(B, 2, M, 2, device='cuda')
        has_accurate_gt = val_depth & valid_repr_disp

        patches = patches[...,:2]
        fixed = {"centroids": centroids * 8.0, "gt_matches": gt_matches * 8.0, "has_accurate_gt": has_accurate_gt, "cycle_consistent_gt": cycle_consistency}

        predictions = []
        net = torch.zeros(B, 2, M, DIM, device="cuda", dtype=torch.float)
        intrinsics = intrinsics / 8.0
        coords = epops.map_coords_using_intrinsics(intrinsics, patches)
        residuals = torch.zeros(B, 2, M, 2, device="cuda", dtype=torch.float)
        Gs = epops.id_pose_unit_t(B, device='cuda')
        if homography_pretrain:
            Gs = repeat(torch.eye(3).cuda(), 'i j -> B 2 i j', B=B)
        for it in range(STEPS):
            coords = coords.detach()
            Gs = Gs.detach()
            patches = patches.detach()

            if (it > 0) and (not homography_pretrain):
                f_hat = epops.se3_to_F(Gs, intrinsics)
                residuals = epops.epipolar_residuals(patches.flatten(-4,-2), coords.flatten(-4,-2), f_hat)
                residuals = rearrange(residuals, 'B LR (M p1 p2) uv -> B LR M p1 p2 uv', M=M, p1=p, LR=2, p2=p, B=B)
                coords = coords + residuals
                assert not residuals.requires_grad
                residuals = residuals[..., p//2, p//2, :]

            with autocast(enabled=use_autocast, dtype=torch.bfloat16):
                mf = coords.flatten(0, -4)
                corr = [corr_fn(kk, jj, mf * fac) for fac, corr_fn in corr_fns]
                corr = torch.cat(list(chain.from_iterable(corr)), dim=2).view(B, 2, M, -1).float()
                net, (delta, dir_weights, gn_weights) = self.update(net, imap, corr, residuals, ii, jj)

            # Rearrange branches
            delta = rearrange(delta, 'B LR M uv -> B LR M 1 1 uv', B=B, LR=2, M=M, uv=2)

            pre_update_matches = coords[..., p//2, p//2, :].clone()
            coords = coords + delta
            target = coords[..., p//2, p//2, :] # B LR M uv
            # target = gt_matches # [SANITY CHECK]
            # pre_update_matches = gt_matches

            # Confidence weights
            if bounds is None:
                tmp_bounds = repeat(torch.tensor([ww, hh]).cuda(), 'uv -> B 2 uv', B=B, uv=2)
                in_bounds_mask = epops.check_in_range(target, tmp_bounds)
            else:
                in_bounds_mask = epops.check_in_range(target, bounds / 8.0)
            dir_weights = in_bounds_mask * dir_weights
            gn_weights = in_bounds_mask * gn_weights

            # Solver step
            pts1, pts2 = epops.make_bidir_pts(centroids, target)
            dir_weightsX = rearrange(dir_weights, 'B LR M -> B (LR M) 1')
            gn_weightsX = rearrange(gn_weights, 'B LR M -> B (LR M) 1')

            if not homography_pretrain:
                try:
                    intr1, intr2 = intrinsics.unbind(-2)
                    Gs = epops.weighted_epa(pts1, pts2, dir_weightsX * 0.01, intr1, intr2, Ps[:,1] if self.training else None, (hh, ww))
                    Gs = epops.optimal_algorithm(Gs, pts1, pts2, gn_weightsX * 0.01, intr1, intr2, iters=opt_iters)
                    if isbad(Gs.data):
                        raise torch._C._LinAlgError()
                except torch._C._LinAlgError:
                    print("Warning: Optimization Layer Failed")
            else:
                predictions.append({"matches": target * 8.0})
                continue

            predictions.append({"matches": target * 8.0, "pre_update_matches": pre_update_matches * 8.0, "poses": Gs.inv(), "gn_weight": gn_weights, "weight": dir_weights}) # returns pose in world

            if (depth_conf is not None) and (it == (STEPS - 1)):
                if isinstance(kp_or_detector, tuple):
                    _, _, M, _ = kp_or_detector[0].shape
                    return compute_sim3(Gs, centroids[:,:,:M], pre_update_matches[:,:,:M], intrinsics, patch_disps[:,:,:M], val_depth[:,:,:M], dir_weights[:,:,:M], depth_conf)
                return compute_sim3(Gs, centroids, pre_update_matches, intrinsics, patch_disps, val_depth, dir_weights, depth_conf)

        return fixed, predictions

    def vo_forward(self, images, poses, disps, intrinsics, M=1024, STEPS=12, P=1, structure_only=False, rescale=False):
        """ Estimates SE3 or Sim3 between pair of frames """

        _, N, *_ = images.shape
        b = 1
        keypoints, _ = random_keypoints(images, None, M)
        # keypoints, _ = superpoint_keypoints(images, None, M)

        assert self.training
        images = 2 * (images / 255.0) - 1
        intrinsics = intrinsics / 8.0

        features = self.patchify(images.squeeze(0), disps=disps.squeeze(0), coords=keypoints.squeeze(0))
        patches = rearrange(features['patches'], 'N M uvd p1 p2 -> 1 (N M) uvd p1 p2')
        ix = repeat(torch.arange(N).cuda(), 'N -> (N M)', M=M)
        imap = features['imap'].unsqueeze(0)
        *_, h, w = features['fmap']['8'].shape

        corr_fns = [
            (4, CorrBlock(features['fmap']['2'], features['gmap']['2'], levels=[1])),
            (2, CorrBlock(features['fmap']['4'], features['gmap']['4'], levels=[1])),
            (1, CorrBlock(features['fmap']['8'], features['gmap']['8'], levels=[1, 2, 4, 8])),
        ]

        p = self.P

        patches_gt = patches.clone()
        Ps = poses

        d = patches[..., 2, p//2, p//2]
        patches = set_depth(patches, torch.rand_like(d))
        # patches = patches_gt.clone()

        kk, jj = flatmeshgrid(torch.where(ix < 8)[0], torch.arange(0,8, device="cuda"))
        kk, jj = sort_idxs(ix, kk, jj)
        ii = ix[kk]

        imap = imap.view(b, -1, DIM)
        net = torch.zeros(b, len(kk), DIM, device="cuda", dtype=torch.float)

        Gs = SE3.IdentityLike(poses)

        if structure_only:
            Gs.data[:] = poses.data[:]

        traj = []
        bounds = [-64//2, -64//2, w + 64//2, h + 64//2]

        while len(traj) < STEPS:
            Gs = Gs.detach()
            patches = patches.detach()

            n = ii.max() + 1
            if len(traj) >= 8 and n < images.shape[1]:
                if not structure_only: Gs.data[:,n] = Gs.data[:,n-1]
                kk1, jj1 = flatmeshgrid(torch.where(ix  < n)[0], torch.arange(n, n+1, device="cuda"))
                kk2, jj2 = flatmeshgrid(torch.where(ix == n)[0], torch.arange(0, n+1, device="cuda"))

                kk1, jj1 = sort_idxs(ix, kk1, jj1)
                kk2, jj2 = sort_idxs(ix, kk2, jj2)

                ii = torch.cat([ix[kk1], ix[kk2], ii])
                jj = torch.cat([jj1, jj2, jj])
                kk = torch.cat([kk1, kk2, kk])

                net1 = torch.zeros(b, len(kk1) + len(kk2), DIM, device="cuda")
                net = torch.cat([net1, net], dim=1)

                if np.random.rand() < 0.1:
                    k = (ii != (n - 4)) & (jj != (n - 4))
                    ii = ii[k]
                    jj = jj[k]
                    kk = kk[k]
                    net = net[:,k]

                patches[:,ix==n,2] = torch.median(patches[:,(ix == n-1) | (ix == n-2),2])
                n = ii.max() + 1

            coords = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
            # coords, valid, _ = pops.transform(Ps, patches_gt, intrinsics, ii, jj, kk, jacobian=True)

            mf = coords.flatten(0, -4)
            corr = torch.cat([corr_fn(kk, jj, mf * fac) for fac, corr_fn in corr_fns], dim=-1)
            net, (delta, _, weight) = self.update(net, imap[:,kk], corr, None, ii, jj, kk)
            weight = repeat(weight, '1 M -> 1 M 2')

            lmbda = 1e-4
            target = coords[...,p//2,p//2,:] + delta

            ep = 10
            for itr in range(2):
                Gs, patches = BA(Gs, patches, intrinsics, target, weight, lmbda, ii, jj, kk, 
                    bounds, ep=ep, fixedp=1, structure_only=structure_only)

            kl = torch.as_tensor(0)
            dij = (ii - jj).abs()
            k = (dij > 0) & (dij <= 2)

            coords = pops.transform(Gs, patches, intrinsics, ii[k], jj[k], kk[k])
            coords_gt, valid, _ = pops.transform(Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], jacobian=True)

            traj.append((valid, coords, coords_gt, Gs[:,:n], Ps[:,:n], kl))

        return traj
