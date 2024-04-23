import gin
import torch
from einops import *
from tensordict import TensorDict

from ..dpvo import projective_ops as pops
from ..dpvo.lietorch import SE3
from ..dpvo.utils import Timer
from ..utils.keypoint_extractors import random_keypoints, superpoint_keypoints
from .patchgraph import DIM, FKW, IKW, PatchGraph
from .slamend import FDEF, SubgraphBase


@gin.configurable
class Frontend(SubgraphBase):

    def __init__(self, DIM, device, dtype, update_op, patchify, graph: PatchGraph, buf_size, optimization_window):
        super().__init__(DIM, device, dtype, update_op, graph)
        assert graph.n == 0
        self.patchify = patchify
        self.optimization_window = optimization_window
        self.is_initialized = False
        self.buf_size = buf_size
        frame_size = self.graph.frame_size

        # Frames
        self.fmaps = TensorDict({
            '2': torch.full((buf_size, 24, *(frame_size // 2)), FDEF, device='cuda', dtype=torch.half),
            '4': torch.full((buf_size, 64, *(frame_size // 4)), FDEF, device='cuda', dtype=torch.half),
            '8': torch.full((buf_size, 128, *(frame_size // 8)), FDEF, device='cuda', dtype=torch.half),
            '16': torch.full((buf_size, 128, *(frame_size // 16)), FDEF, device='cuda', dtype=torch.half),
            '32': torch.full((buf_size, 128, *(frame_size // 32)), FDEF, device='cuda', dtype=torch.half),
            '64': torch.full((buf_size, 128, *(frame_size // 64)), FDEF, device='cuda', dtype=torch.half),
        }, batch_size=[buf_size])
        self.M = graph.M

    @property
    def n(self):
        return self.graph.n

    @n.setter
    def n(self, value):
        self.graph.n = value

    @property
    def ii(self):
        return self.graph.ii_active

    @property
    def jj(self):
        return self.graph.jj_active

    @property
    def kk(self):
        return self.graph.kk_active

    @ii.setter
    def ii(self, value):
        self.graph.ii_active = value

    @jj.setter
    def jj(self, value):
        self.graph.jj_active = value

    @kk.setter
    def kk(self, value):
        self.graph.kk_active = value

    @property
    def target(self):
        return self.graph.target_active

    @target.setter
    def target(self, value):
        self.graph.target_active = value

    @property
    def weight(self):
        return self.graph.weight_active

    @weight.setter
    def weight(self, value):
        self.graph.weight_active = value

    @property
    def fjj(self):
        return self.jj % self.buf_size

    def insert_frame(self, data):
        n = self.graph.n
        nb = n % self.buf_size
        self.fmaps[nb] = data.squeeze(0)

    def vid_length(self):
        return self.graph.n

    def get_opt_window(self):
        if not self.is_initialized:
            return 1, self.graph.length
        t0 = (self.graph.length) - self.optimization_window
        return max(t0, 1), self.graph.length

    @gin.configurable
    def edges_backw(self, patch_lifetime):
        st = max(0, self.graph.n - patch_lifetime)
        return [(self.graph.n-1, i) for i in range(st, self.graph.n)]

    @gin.configurable
    def edges_forw(self, patch_lifetime):
        st = max(0, self.graph.n - patch_lifetime)
        return [(i, self.graph.n-1) for i in range(st, self.graph.n-1)]

    def remove_factors(self, m, store=True):

        # store estimated factors
        if store:
            self.graph.ii_inac = torch.cat([self.graph.ii_inac, self.ii[m]], 0)
            self.graph.jj_inac = torch.cat([self.graph.jj_inac, self.jj[m]], 0)
            self.graph.kk_inac = torch.cat([self.graph.kk_inac, self.kk[m]], 0)
            self.graph.target_inac = torch.cat([self.graph.target_inac, self.target[m]], 0)
            self.graph.weight_inac = torch.cat([self.graph.weight_inac, self.weight[m]], 0)

        self.net = self.net[~m]
        self.residuals = self.residuals[~m]
        self.ii = self.ii[~m]
        self.jj = self.jj[~m]
        self.kk = self.kk[~m]
        self.target = self.target[~m]
        self.weight = self.weight[~m]

    @gin.configurable
    def prune_old_edges(self, removal_window):
        oldest_frame = (self.graph.n - removal_window)
        to_remove = (self.ii < oldest_frame) | (self.jj < oldest_frame)
        self.remove_factors(to_remove, store=True)

    def append_factors(self, edges):
        M = self.graph.M
        ii, jj = map(torch.as_tensor, zip(*edges))
        ptch = torch.arange(M * self.graph.N, **IKW).view(-1, M)
        kk = ptch[ii.cuda()].flatten()
        jj = repeat(jj.cuda(), 'n -> (n k)', k=ptch.shape[1])
        ii = self.graph.ix[kk]
        self.jj = torch.cat([self.jj, jj])
        self.kk = torch.cat([self.kk, kk])
        self.ii = torch.cat([self.ii, ii])

        net = torch.zeros(len(ii), self.DIM, device=self.device, dtype=self.dtype)
        self.net = torch.cat([self.net, net], dim=0)
        residuals = torch.zeros(len(ii), 2, **FKW)
        self.residuals = torch.cat([self.residuals, residuals], dim=0)

    def compute_flow_mag_at_frame(self, idx):
        M = self.graph.M
        N = self.graph.N
        ptch = torch.arange(M * self.graph.N, **IKW).view(-1, M)
        kk = ptch[[idx-1, idx+1]].flatten()
        ii = self.graph.ix[kk]
        jj = torch.tensor([idx+1, idx-1], device='cuda').repeat_interleave(M)
        patches = self.graph.patches[:N].view(1, N*self.M, 3, 3, 3)
        intrinsics = self.graph.intrinsics[:N].view(1, self.graph.N, 4)
        poses = self.graph.poses[:N].view(1, self.graph.N, 7)
        flow, _ = pops.flow_mag(SE3(poses), patches, intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def get_opt_edges(self):
        weight = repeat(self.weight, 'E -> 1 E 2')
        target = self.target.view(1, -1, 2)
        return self.ii, self.jj, self.kk, target, weight

    def delete_frame(self, k):
        to_remove = (self.ii == k) | (self.jj == k)
        self.remove_factors(to_remove, store=False)
        self.kk[self.ii > k] -= self.M
        self.ii[self.ii > k] -= 1
        self.jj[self.jj > k] -= 1
        self.graph.save_delta(k)

        for i in range(k, self.n):
            self.graph.images[i] = self.graph.images[i+1]
            self.graph.intrinsics[i] = self.graph.intrinsics[i+1]
            self.graph.frame_ids[i] = self.graph.frame_ids[i+1]
            self.graph.global_desc[i] = self.graph.global_desc[i+1]
            self.graph.poses[i] = self.graph.poses[i+1]

            self.fmaps[i%self.buf_size] = self.fmaps[(i+1)%self.buf_size]

            self.graph.patches[i] = self.graph.patches[i+1]
            self.graph.imap[i] = self.graph.imap[i+1]

            self.graph.gmaps[i] = self.graph.gmaps[i+1]
        
        self.n -= 1
        self.graph.N -= 1

    def motion_probe(self):
        kk = torch.arange((self.n-1)*self.M, self.n*self.M, device='cuda')
        ii = self.graph.ix[kk]
        jj = ii + 1
        coords = self.graph.reproject((ii, jj, kk))
        corr = self.corr(coords, (kk, jj)).float()
        ctx = self.graph.imap.flatten(0, 1)[kk]
        net = torch.zeros(self.M, DIM, device='cuda')
        _, (delta, _, _) = self.update_op(net, ctx, corr, None, ii, jj, kk)
        return delta.norm(dim=-1).mean().item()

    @gin.configurable('frontend_update')
    def __call__(self, image, tstamp, intrinsics, iters, keyframe_freq, keyframe_thresh):

        self.graph.set_frame_metadata(image.byte(), tstamp, intrinsics)
        image = image.unsqueeze(0) # 1 3 h w

        with torch.autocast(device_type="cuda"):
            keypoints, _ = superpoint_keypoints(image, None, self.M)
            # keypoints, _ = random_keypoints(image, None, self.M)

            image = 2 * (image / 255.0) - 1
        
            features = self.patchify(image, bounds=None, coords=keypoints, disps=None, padit=False)
        self.insert_frame(features['fmap'])

        imap, patches, gmaps = features['imap'], features['patches'], features['gmap']
        assert parse_shape(imap, 'b M DIM _ _') == dict(b=1, M=self.M, DIM=DIM)
        assert parse_shape(patches, 'b M uvd p1 p2') == dict(b=1, M=self.M, uvd=3, p1=self.P, p2=self.P)
        self.graph.imap[self.n] = imap.squeeze()
        self.graph.gmaps[self.n] = gmaps.squeeze(0)
        self.graph.global_desc[self.n] = self.graph.gc((image+1)/2)

        # TODO better depth initialization
        patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
        if self.is_initialized:
            s = torch.median(self.graph.patches[self.n-3:self.n,:,2])
            patches[:,:,2] = s
        self.graph.patches[self.n] = patches.squeeze(0)

        if self.graph.n > 1:
            P1 = SE3(self.graph.poses[self.graph.n-1])
            P2 = SE3(self.graph.poses[self.graph.n-2])
            xi = 0.5 * (P1 * P2.inv()).log()
            tvec_qvec = (SE3.exp(xi) * P1).data
            self.graph.poses[self.graph.n] = tvec_qvec


        if (self.n > 0) and (not self.is_initialized):
            if self.motion_probe() < 1.0:
               self.graph.save_delta(self.n)
               self.graph.N -= 1
               return

        self.graph.n += 1 # n is now the length

        if self.graph.n > 1:
            forw_edges = self.edges_forw()
            assert len(forw_edges) > 0
            self.append_factors(forw_edges)
            backw_edges = self.edges_backw()
            self.append_factors(backw_edges)

        if self.vid_length() >= 7:
            if not self.is_initialized:
                for itr in range(12):
                    self.run_update_op()
                self.is_initialized = True
            else:
                with Timer("Update:", enabled=False):
                    for itr in range(iters):
                        self.run_update_op()

                k = self.graph.n - 3
                flow_mag = self.compute_flow_mag_at_frame(k)
                if flow_mag < keyframe_thresh:
                    self.delete_frame(k)

                self.prune_old_edges()

        if self.is_initialized:
            kf_idx = self.n-5
            if (len(self.graph.keyframes) == 0) or ((kf_idx - self.graph.keyframes[-1]) >= keyframe_freq):
                self.graph.insert_keyframe(self.fmaps[kf_idx%self.buf_size], kf_idx)