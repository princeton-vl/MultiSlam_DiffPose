
import gin
import torch
from einops import *
from torch.nn import functional as F

from .dpvo.lietorch import Sim3
from .locnet import DIM
from .framegraph import Backend, Frontend, PatchGraph, PatchGraphUnion


@gin.configurable
class FullSystem:

    def __init__(self, locnet, twoview_locnet, M):
        self.P = locnet.P
        self.twoview_network = twoview_locnet
        self.network = locnet
        self.network.eval()
        self.M = M
        self.device = locnet.device
        self.count = 0

        self.all_graphs = []
        self.frontend: Frontend = None
        self.graph: PatchGraph = None
        self.time_since_merge_attempt = 0

    def add_new_video(self, name, N, frame_size):
        assert (self.graph is None) or self.graph.is_complete()
        self.graph = PatchGraph(N, self.M, name, frame_size)
        self.all_graphs.append(self.graph)
        self.frontend = Frontend(DIM, self.device, torch.float, self.network.update, self.network.patchify, self.graph)

    def complete_video(self):
        assert len(self.all_graphs) == 1
        assert (self.graph is not None) and self.graph.is_complete()
        for itr in range(20):
            self.frontend.run_update_op()
        m = torch.ones_like(self.frontend.ii, dtype=torch.bool)
        self.frontend.remove_factors(m, store=True)
        assert self.frontend.ii.numel() == 0
        del self.frontend
        assert len(self.all_graphs) == 1

    def backend_update(self, iters=20):
        backend = Backend(384, 'cuda', torch.float, self.network.update, self.all_graphs[-1])
        backend.update(iters)
        del backend

    def terminate(self) -> PatchGraph:
        assert len(self.all_graphs) == 1
        graph = self.all_graphs.pop()
        return graph.predictions()

    @gin.configurable
    def rel_pose_batch(self, graph_I: PatchGraph, ii, graph_J: PatchGraph, jj, model_params):
        images = torch.stack((graph_I.images[ii], graph_J.images[jj]), dim=1).to(device=self.device)
        intrinsics = torch.stack((graph_I.intrinsics[ii], graph_J.intrinsics[jj]), dim=1) * 8
        centroids = torch.stack((graph_I.patches[ii, :, :, 1, 1], graph_J.patches[jj, :, :, 1, 1]), dim=1)
        centroids[:,:,:,:2] *= 8
        depth_confidence = torch.stack((graph_I.patch_confidence()[ii], graph_J.patch_confidence()[jj]), dim=1)

        Sim3_r2l, num_inliers = self.twoview_network(images, intrinsics, centroids, **model_params, depth_conf=depth_confidence)
        assert num_inliers.ndim == 1

        pi = graph_I.world_poses[ii].cpu()
        pj = graph_J.world_poses[jj].cpu()
        gt_C = Sim3(pi) * Sim3(Sim3_r2l.cpu()) * Sim3(pj.inv())
        num_inliers = num_inliers.double()
        num_inliers += torch.rand_like(num_inliers).mul(0.001)
        return list(zip(asnumpy(num_inliers).tolist(), gt_C))

    @staticmethod
    def _retrieve_image(i, W, desc_I, desc_J):
        ii = torch.arange(i, i+W, device='cuda')

        sims = einsum(desc_I[ii], desc_J, 'w d, n d -> w n')

        block_sims = F.unfold(sims[None,None], kernel_size=(W, 12))
        block_sims = rearrange(block_sims, '1 (si x) B -> B si x', si=W)
        block_sims_max, block_sims_argmax = block_sims.max(2)
        block_sims_red = reduce(block_sims_max, 'B si -> B', 'min')
        v, idx = block_sims_red.max(0)

        jj = block_sims_argmax[idx] + idx
        scores = block_sims_max[idx]
        return v, (ii, jj), scores

    def _merge_graphs(self, v):
        graph_I, graph_J = self.all_graphs
        assert graph_I.is_complete()

        # Apply Sim3
        graph_J.sim3_graph(v)

        # Union graphs
        self.frontend.graph = self.graph = PatchGraphUnion(graph_I, graph_J)

        # Update frontend
        idx_map = torch.arange(self.frontend.buf_size, device='cuda')
        idx_map = (idx_map + graph_I.N) % self.frontend.buf_size
        self.frontend.fmaps[idx_map] = self.frontend.fmaps.clone()

        # Perform Global BA
        # backend = Backend(384, 'cuda', torch.float, self.network.update, self.graph)
        # backend.update(10)
        # del backend

        self.all_graphs = [self.graph]
        self.graph.normalize()

    @gin.configurable('fs_insert_frame')
    def insert_frame(self, image, intrinsics, tstamp):
        assert not self.network.training
        assert parse_shape(image, 'rgb _ _') == dict(rgb=3)
        assert parse_shape(intrinsics, 'f') == dict(f=4)
        self.frontend(image, tstamp, intrinsics)
        self.time_since_merge_attempt += 1
        self.count += 1

        if (self.graph.ii_inac.numel() > 96*10) and (self.count % 50 == 0):
            # Perform Global BA
            backend = Backend(384, 'cuda', torch.float, self.network.update, self.graph)
            backend.update(2)
            del backend

        assert len(self.all_graphs) in [1,2]
        if (len(self.all_graphs) == 2) and (self.time_since_merge_attempt > 10):
            graph_J, graph_I = self.all_graphs
            RAD = 25 # Only add connections outside the optimization window.
            i = self.graph.n - RAD
            if i < 0:
                return
            descsI, descsJ = graph_I.global_desc, graph_J.global_desc[:graph_J.N]
            retr = self._retrieve_image(i, 6, descsI, descsJ)

            if retr[0] > 0.3:
                _, (ii, jj), scores = retr
            else:
                return

            self.time_since_merge_attempt = 0

            # graph_I.sim3_graph(Sim3.Random(1)[0])
            # graph_J.sim3_graph(Sim3.Random(1)[0])
            rel_poses = self.rel_pose_batch(graph_J, jj.cpu(), graph_I, ii.cpu())

            inl, v = max(rel_poses)
            if (inl >= 10):
                self._merge_graphs(v)
                print("Finished connecting disjoint trajectories")
                self.backend_update(iters=5)
