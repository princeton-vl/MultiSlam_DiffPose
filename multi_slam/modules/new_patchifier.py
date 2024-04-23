import math
import sys
import time

import gin
import numpy as np
import torch
import torch.nn.functional as F
from einops import *
from fastcore.all import store_attr
from tensordict import TensorDict
from torch import nn

from ..dpvo import altcorr
from .extractor import MultiEncoder
autocast = torch.cuda.amp.autocast

DIM = 384

from .loftr_module.transformer import LoFTREncoderLayer

@gin.configurable
def pos_encode(x, bounds, linspace_limits):
    B, d_model, h, w = x.shape

    # Scaling
    scaling = torch.as_tensor(linspace_limits).float().view(1,-1)
    wh = torch.tensor([w,h]).float().view(1,2)
    bounds = bounds.float().cpu()
    torch.broadcast_tensors(wh, bounds, scaling)
    scaling = scaling * ((wh-1) / (bounds-1).cpu())

    # Encodings
    xy_position =  torch.stack(torch.meshgrid(torch.linspace(0, 1, w), torch.linspace(0, 1, h), indexing='xy'))
    xy_position = repeat(xy_position, 'xy H W -> xy B 1 H W', xy=2, H=h, W=w, B=B)
    assert parse_shape(scaling, 'B xy') == parse_shape(xy_position, 'xy B _ _ _') == dict(xy=2, B=B)
    x_position, y_position = xy_position * rearrange(scaling, 'B xy -> xy B 1 1 1')
    div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
    div_term = div_term.view(1,-1,1,1)  # [C//4, 1, 1]
    pe = torch.zeros(B, d_model, h, w, device=x.device)
    pe[:, 0::4, :, :] = torch.sin(x_position * div_term)
    pe[:, 1::4, :, :] = torch.cos(x_position * div_term)
    pe[:, 2::4, :, :] = torch.sin(y_position * div_term)
    pe[:, 3::4, :, :] = torch.cos(y_position * div_term)
    return x + pe

def random_coords_from_mask(mask, M):
    B, H, W = mask.shape
    xy = torch.stack(torch.meshgrid(torch.arange(W).cuda(), torch.arange(H).cuda(), indexing='xy'), dim=-1).view(H * W, 2)
    mask = mask.view(B, H * W).float()
    chosen = mask.multinomial(M, replacement=True) # B x nc
    return xy[chosen.flatten()].view(B, M, 2)

def random_coords(images, M, bounds):
    *B, _, H, W = images.shape
    bounds_x, bounds_y = bounds[0].split(dim=-1, split_size=1)
    assert bounds_x.dtype == torch.int64
    x = torch.randint(1, 1000000, size=[*B, M], device="cuda") % bounds_x
    y = torch.randint(1, 1000000, size=[*B, M], device="cuda") % bounds_y
    coords = torch.stack([x, y], dim=-1).float()
    return coords

def make_grid(d):
    B, _, H, W = d.shape
    uv = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')).float().cuda()
    uv = repeat(uv, 'uv H W -> B uv H W', B=B)
    return torch.cat((uv, d), dim=1)

@gin.configurable
class Patchifier(nn.Module):
    def __init__(self, patch_size, fnet, inet):
        super().__init__()
        store_attr()
        self.attention_layers = nn.ModuleList([
            LoFTREncoderLayer(DIM, 8, "linear"),
            LoFTREncoderLayer(DIM, 8, "linear"),
        ])

    def extract_patches(self, grid, fmap, imap, coords, fac):
        P = self.patch_size
        disp = None
        if parse_shape(coords, 'N M D')['D'] == 3:
            coords, disp = coords.split([2,1], dim=-1)
        coords = coords / fac
        if imap is not None:
            imap = altcorr.patchify(imap, coords, 0)
        gmap = altcorr.patchify(fmap, coords, P//2)
        patches = altcorr.patchify(grid, coords, P//2)
        if disp is not None:
            patches[:,:,2] = repeat(disp, 'N M 1 -> N M 3 3')
        return gmap, imap, patches

    def run_attention(self, inp, bounds):
        N, _, h, w = inp.shape

        mg = torch.stack(torch.meshgrid(torch.arange(w, device=inp.device), torch.arange(h, device=inp.device), indexing='xy'), dim=-1) # h w 2
        masks = (mg < bounds.view(N, 1, 1, 2)).all(dim=-1).view(N, -1)

        feats = rearrange(pos_encode(inp, bounds=bounds), 'N DIM h w -> N (h w) DIM', DIM=DIM, N=N, w=w)

        for att_layer in self.attention_layers:
            feats = att_layer(feats, feats, masks, masks)

        return rearrange(feats, 'N (h w) DIM -> N DIM h w', N=N, w=w, DIM=DIM)

    @gin.configurable('pfor')
    def forward(self, images, disps, coords, bounds=None, padit=True):
        B, _, H, W = images.shape

        _, _, imap = self.inet(images, padit=padit)
        fmaps = self.fnet(images, padit=padit)

        if bounds is None:
            bounds = torch.tensor([W, H], dtype=torch.float, device=images.device).tile(B, 1) / 8

        imap = imap + self.run_attention(imap, bounds)

        if disps is not None:
            disps = rearrange(disps, 'N H W -> N 1 H W')
        else:
            disps = torch.ones_like(images[:,:1])

        output = {"fmap": TensorDict({}, batch_size=[B]), "gmap": TensorDict({}, batch_size=[B])}
        for idx in [0, 1, 2]:
            fmap = fmaps[idx].float() / 4.0
            fac = 2 ** (idx + 1)
            grid = make_grid(F.avg_pool2d(disps, kernel_size=fac))
            gmap_p, imap_p, patches = self.extract_patches(grid, fmap, imap if (idx == 2) else None, coords, fac)
            output['imap'] = imap_p
            output['patches'] = patches
            output['fmap'][str(fac)] = fmap
            output['gmap'][str(fac)] = gmap_p

        for i in [2,4,8]:
            output['fmap'][str(i*8)] = F.avg_pool2d(fmap, kernel_size=i)

        return output
