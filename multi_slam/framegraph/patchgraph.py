import warnings

import gin
import numpy as np
import torch
from einops import *
from tensordict import TensorDict
from torch_scatter import scatter_max, scatter_sum

from multi_slam.dpvo.lietorch import SE3, SO3, Sim3

from ..dpvo import projective_ops as pops
from ..locnet import DIM
from .descriptors import GlobalDesc

IKW = dict(dtype=torch.long, device='cuda')
FKW = dict(dtype=torch.float, device='cuda')
WDEF = np.nan
TDEF = np.nan
FDEF = np.nan

warnings.filterwarnings("ignore", "The PyTorch API of nested tensors is in prototype stage")

BUFF_SIZE = 4000

@gin.configurable
class PatchGraph:

    gc = GlobalDesc()

    def __init__(self, N, M, name, frame_size, buf_size, device='cuda'):
        self.frame_history = []

        # store relative poses for removed frames
        self.delta = {}

        # Frames
        self.n = 0
        self.N = N
        self.frame_size = np.array(frame_size)
        self.buf_size = buf_size

        self.vid_name = name
        self.frame_ids = np.empty(BUFF_SIZE, dtype=object)

        self.images = torch.empty((BUFF_SIZE, 3, *frame_size), dtype=torch.uint8) # Not used?
        self.poses = SE3.Identity(BUFF_SIZE, device='cuda').data # cam -> world
        self.intrinsics = torch.full((BUFF_SIZE, 4), np.nan, device='cuda')
        self.fmaps = TensorDict({
            '2': torch.full((buf_size, 24, *(self.frame_size // 2)), FDEF, device='cuda', dtype=torch.half),
            '4': torch.full((buf_size, 64, *(self.frame_size // 4)), FDEF, device='cuda', dtype=torch.half),
            '8': torch.full((buf_size, 128, *(self.frame_size // 8)), FDEF, device='cuda', dtype=torch.half),
            '16': torch.full((buf_size, 128, *(self.frame_size // 16)), FDEF, device='cuda', dtype=torch.half),
            '32': torch.full((buf_size, 128, *(self.frame_size // 32)), FDEF, device='cuda', dtype=torch.half),
            '64': torch.full((buf_size, 128, *(self.frame_size // 64)), FDEF, device='cuda', dtype=torch.half),
        }, batch_size=[buf_size])
        self.global_desc = torch.full((BUFF_SIZE, 4096), FDEF, dtype=torch.float, device='cuda')
        self.ii2kf = torch.full((BUFF_SIZE,), -1, **IKW)
        self.keyframes = []

        # Edges
        self.ii_inac = torch.empty(0, **IKW)
        self.jj_inac = torch.empty(0, **IKW)
        self.kk_inac = torch.empty(0, **IKW)
        self.weight_inac = torch.empty(0, **FKW)
        self.target_inac = torch.empty(0, 2, **FKW)
        self.ii_active = torch.empty(0, **IKW)
        self.jj_active = torch.empty(0, **IKW)
        self.kk_active = torch.empty(0, **IKW)
        self.weight_active = torch.empty(0, **FKW)
        self.target_active = torch.empty(0, 2, **FKW)

        # Patches
        self.M = M
        self.P = 3
        self.patches = torch.full((BUFF_SIZE, self.M, 3, self.P, self.P), np.nan, device='cuda') # 1/8th resolution
        self.ix = torch.arange(BUFF_SIZE, **IKW).repeat_interleave(self.M)
        self.imap = torch.empty(self.N, self.M, DIM, device='cuda')
        self.gmaps = TensorDict({
            '2': torch.full((self.N, self.M, 24, 3, 3), FDEF, device='cuda', dtype=torch.half),
            '4': torch.full((self.N, self.M, 64, 3, 3), FDEF, device='cuda', dtype=torch.half),
            '8': torch.full((self.N, self.M, 128, 3, 3), FDEF, device='cuda', dtype=torch.half),
        }, batch_size=[self.N])

    @property
    def ii(self):
        return torch.cat((self.ii_inac, self.ii_active))

    @property
    def jj(self):
        return torch.cat((self.jj_inac, self.jj_active))

    @property
    def kk(self):
        return torch.cat((self.kk_inac, self.kk_active))

    @property
    def weight(self):
        return torch.cat((self.weight_inac, self.weight_active))

    @property
    def target(self):
        return torch.cat((self.target_inac, self.target_active))

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.delta[t]
        return dP * self.get_pose(t0)

    def predictions(self):
        self.traj = {}
        for idx in range(self.N):
            key = self.frame_ids[idx]
            self.traj[key] = self.poses[idx]

        output = []
        for key in self.frame_history:
            output.append((*key, self.get_pose(key)))
        del self.traj

        return [(v, t / 1e6, asnumpy(p.inv().data)) for v,t,p in output]

    def save_delta(self, k):
        dP = SE3(self.poses[k]) * SE3(self.poses[k-1]).inv()
        key0 = self.frame_ids[k-1]
        key1 = self.frame_ids[k]
        self.delta[key1] = (key0, dP)

    def set_frame_metadata(self, image, tstamp, intrinsics):
        self.frame_ids[self.n] = (self.vid_name, round(tstamp * 1e6))
        self.intrinsics[self.n] = intrinsics / 8
        self.images[self.n] = image
        self.frame_history.append(self.frame_ids[self.n])

    def insert_keyframe(self, data, idx):
        kf = len(self.keyframes)
        self.fmaps[kf] = data
        self.ii2kf[idx] = kf
        self.keyframes.append(idx)

    def topKpatches(self, k):
        n = self.length
        patch_activity = self.patch_confidence() # N M
        patch_activity = patch_activity.argsort(dim=1, descending=True)[:,:k]
        assert patch_activity.max() < self.M
        ptch = torch.arange(n*self.M, **IKW).view(n, self.M)
        ptch = ptch.gather(1, patch_activity) # N k
        assert parse_shape(ptch, 'n k') == dict(n=n, k=k)
        return ptch.flatten()

    def patch_confidence(self):
        n = self.length
        return scatter_max(self.weight, self.kk, dim_size=(n * self.M))[0].view(n, self.M)

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies
        patches = rearrange(self.patches[:self.N], 'N M ... -> 1 (N M) ...', M=self.M, N=self.N)
        coords = pops.transform(SE3(self.poses)[None], patches, self.intrinsics[None], ii, jj, kk)
        return rearrange(coords, '1 N p1 p2 uv -> N uv p1 p2', uv=2).contiguous()

    def compute_flow_mag(self, ii, jj, kk):
        assert ii.numel() > 1
        q = (self.P//2)
        patches = self.patches[:self.N].view(1, self.N*self.M, 3, 3, 3)[..., q:q+1, q:q+1]
        intrinsics = self.intrinsics[:self.N].view(1, self.N, 4)
        poses = self.poses[:self.N].view(1, self.N, 7)
        flow, d_val = map(torch.squeeze, pops.flow_mag(SE3(poses), patches, intrinsics, ii, jj, kk, beta=0.5))
        iijj = torch.stack((ii, jj), dim=0) # E 2
        (iiu, jju), inverse_indices, counts = torch.unique(iijj, return_counts=True, return_inverse=True, dim=1)

        num_val = scatter_sum(d_val.long(), inverse_indices)
        num_val_sq = torch.zeros(self.length, self.length, device='cuda', dtype=torch.long)
        num_val_sq[iiu, jju] = num_val
        flow_sum = scatter_sum(flow * d_val, inverse_indices, dim=0)
        flow_sum_sq = torch.full((self.length, self.length), np.inf, device='cuda')
        flow_sum_sq[iiu, jju] = flow_sum

        num_pts = torch.zeros(self.length, self.length, device='cuda', dtype=torch.long)
        num_pts[iiu, jju] = counts

        d = (flow_sum_sq / num_val_sq)
        d[(num_val_sq.float() / num_pts) < 0.75] = np.inf
        assert not torch.isnan(d).any()
        return d

    @property
    def length(self):
        assert self.n <= self.N
        return self.n # if self.n == self.N, the video is completed

    def is_complete(self):
        return (self.n == self.N)

    def normalize(self):
        """ normalize depth and poses """
        s = self.patches[:self.n,:,2].mean()
        rescaling = Sim3.Identity(1)[0]
        rescaling.data[-1] = s
        self.sim3_graph(rescaling)

    def sim3_graph(self, x: Sim3):
        assert isinstance(x, Sim3)
        assert parse_shape(x.data, 'D')['D'] == 8
        assert x.device == self.world_poses.device, (x.device, self.world_poses.device)

        self.world_poses = SE3((x[None] * Sim3(self.world_poses)).data[...,:7])
        s = x.data[7]
        self.patches[:self.n,:,2] /= s.item()
        s = s.cuda()
        for key1, (key0, dP) in self.delta.items():
            self.delta[key1] = (key0, dP.scale(s))

    @property
    def world_poses(self):
        return SE3(self.poses).inv()[:self.length].cpu()

    @world_poses.setter
    def world_poses(self, value):
        assert isinstance(value, SE3)
        assert parse_shape(value.data, 'n d') == dict(n=self.n, d=7), value.data.shape
        self.poses[:self.length] = value.inv().data.to(self.poses)

    @property
    def world_orientations(self):
        return SO3(self.world_poses[:self.length].data[:,3:])

    @property
    def world_positions(self):
        return self.world_poses[:self.length].data[:, :3].numpy()

class PatchGraphUnion(PatchGraph):

    def __init__(self, graph_a: PatchGraph, graph_b: PatchGraph):
        assert np.all(graph_a.frame_size == graph_b.frame_size)
        assert graph_a.M == graph_b.M
        assert graph_a.P == graph_b.P
        assert graph_a.is_complete()

        aN = graph_a.n
        bN = graph_b.n

        # Frames
        self.n = aN + bN # current frame index
        self.N = graph_a.N + graph_b.N # total number of frames
        self.frame_size = graph_a.frame_size
        self.vid_name = graph_b.vid_name

        self.frame_ids = np.concatenate((graph_a.frame_ids[:aN], graph_b.frame_ids))
        self.images = torch.cat((graph_a.images[:aN], graph_b.images))
        self.poses = torch.cat((graph_a.poses[:aN], graph_b.poses))
        self.intrinsics = torch.cat((graph_a.intrinsics[:aN], graph_b.intrinsics))
        self.buf_size = graph_a.buf_size + graph_b.buf_size
        self.fmaps = torch.cat((graph_a.fmaps, graph_b.fmaps))
        del graph_a.fmaps, graph_b.fmaps
        self.delta = {**graph_a.delta, **graph_b.delta}
        self.frame_history = graph_a.frame_history + graph_b.frame_history

        # Keyframes
        self.ii2kf = torch.cat((graph_a.ii2kf[:aN], graph_b.ii2kf + graph_a.buf_size))
        self.global_desc = torch.cat((graph_a.global_desc[:aN], graph_b.global_desc))
        self.keyframes = graph_a.keyframes + [(k + aN) for k in graph_b.keyframes]

        # Active Edges
        self.ii_active = graph_b.ii_active + aN
        self.jj_active = graph_b.jj_active + aN
        self.kk_active = graph_b.kk_active + (aN * graph_a.M)
        self.weight_active = graph_b.weight_active
        self.target_active = graph_b.target_active

        # Inactive Edges
        self.ii_inac = torch.cat((graph_a.ii_inac, graph_b.ii_inac + aN))
        self.jj_inac = torch.cat((graph_a.jj_inac, graph_b.jj_inac + aN))
        self.kk_inac = torch.cat((graph_a.kk_inac, graph_b.kk_inac + (aN*graph_a.M)))
        self.weight_inac = torch.cat((graph_a.weight_inac, graph_b.weight_inac))
        self.target_inac = torch.cat((graph_a.target_inac, graph_b.target_inac))

        # Patches
        self.M = graph_a.M
        self.P = graph_a.P
        self.patches = torch.cat((graph_a.patches[:aN], graph_b.patches))
        self.ix = torch.cat((graph_a.ix[:(aN*self.M)], graph_b.ix + aN))
        self.imap = torch.cat((graph_a.imap[:aN], graph_b.imap))
        self.gmaps = torch.cat((graph_a.gmaps[:aN], graph_b.gmaps))
        del graph_a.gmaps, graph_b.gmaps
 
