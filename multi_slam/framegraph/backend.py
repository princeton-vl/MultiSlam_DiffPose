import gin
import numba as nb
import numpy as np
import torch
from einops import asnumpy, repeat

from ..dpvo.utils import flatmeshgrid

from .patchgraph import FKW, IKW, PatchGraph
from .slamend import SubgraphBase
from ..dpvo.utils import flatmeshgrid

@nb.njit
def agg_edges(ii, jj, d, nms, thresh, max_edges):
    num_e = 0
    E = ii.shape[0]
    es = np.empty((max_edges, 2), dtype=np.int64)
    N = d.shape[1]

    for idx in range(E):
        i = ii[idx]
        j = jj[idx]

        if abs(i - j) < 15:
            continue

        if num_e >= max_edges:
            break

        if d[i, j] > thresh:
            continue

        es[num_e, 0] = i
        es[num_e, 1] = j

        for di in range(-nms, nms+1):
            i1 = i + di

            if (0 <= i1 < N):
                d[i1, j] = np.inf
                d[j, i1] = np.inf

        num_e += 1

    return es[:num_e]


class Backend(SubgraphBase):

    def __init__(self, DIM, device, dtype, update_op, graph: PatchGraph):
        super().__init__(DIM, device, dtype, update_op, graph)
        self.optimization_window = np.inf
        self.fmaps = graph.fmaps
        self.clear_edges()

    def get_opt_window(self):
        return 1, self.graph.length

    @property
    def fjj(self):
        return self.graph.ii2kf[self.jj]

    @gin.configurable
    def edges_global(self, nms, thresh, E, max_edges):
        assert self.graph.n
        assert len(self.graph.keyframes) > 0
        kfms = torch.as_tensor(self.graph.keyframes, device=self.device)
        ptch = self.graph.topKpatches(E)
        assert ptch.numel() == (self.graph.length) * E
        kk, jj = flatmeshgrid(ptch, kfms)

        ii = self.graph.ix[kk]
        d = self.graph.compute_flow_mag(ii, jj, kk)
        d[d > 200] = np.inf
        d.fill_diagonal_(np.inf)

        # Add edges for loop closure
        m = torch.argsort(d.view(-1)).cpu()
        ii, jj = flatmeshgrid(torch.arange(self.graph.length), torch.arange(self.graph.length))
        assert ii.shape == jj.shape == m.shape, (ii.shape, m.shape)
        ii, jj, d = ii[m].numpy(), jj[m].numpy(), asnumpy(d)
        assert torch.abs(self.graph.ii - self.graph.jj).all() < 15
        return agg_edges(ii, jj, d, nms, thresh, max_edges)

    def append_factors(self, edges, topk):
        ii, jj = map(torch.as_tensor, zip(*edges))
        ptch = self.graph.topKpatches(topk).view(self.graph.length, topk)
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

    def get_opt_edges(self):
        ii = torch.cat((self.ii, self.graph.ii))
        jj = torch.cat((self.jj, self.graph.jj))
        kk = torch.cat((self.kk, self.graph.kk))
        weight = repeat(torch.cat((self.weight, self.graph.weight)), 'E -> 1 E 2')
        target = torch.cat((self.target, self.graph.target)).view(1, -1, 2)
        return ii, jj, kk, target, weight

    def clear_edges(self):
        self.net = torch.empty(0, self.DIM, device=self.device, dtype=self.dtype)
        self.residuals = torch.empty(0, 2, device=self.device, dtype=self.dtype)
        self.ii = torch.empty(0, **IKW)
        self.jj = torch.empty(0, **IKW)
        self.kk = torch.empty(0, **IKW)
        self.weight = torch.empty(0, **FKW)
        self.target = torch.empty(0, 2, **FKW)

    @gin.configurable('backend_update')
    def update(self, iters, backend_k):
        for _ in range(iters):
            self.clear_edges()
            global_edges = self.edges_global()
            if global_edges.size == 0:
                return
            # print(f"Backend edges: {global_edges.shape}")
            self.append_factors(global_edges, topk=backend_k)
            self.run_update_op()
