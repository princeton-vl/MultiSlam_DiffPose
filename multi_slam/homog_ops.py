import gin
import numpy as np
import torch
from einops import *
from torch.linalg import inv

from .solver import epa_ops as epops
from .dpvo.lietorch import SE3
from .solver.matrix_utils import *
from .metrics import isbad

KWARGS = dict(device='cuda', dtype=torch.float32)

def is_H(inp):
    return (isinstance(inp, torch.Tensor) and (parse_shape(inp, '... i j') == dict(i=3, j=3)))

def apply_H(pts1, H):
    assert is_H(H)
    h1 = make_homog(pts1)
    h2 = einsum(H, h1, '... i j, ... M j -> ... M i')
    pts2 = h2[...,:2] / h2[...,[2]]
    assert pts1.shape == pts2.shape
    return pts2

@gin.configurable
def homography_forward_pass(data_dict, model, optimizer, scheduler):
    images, H_0_to_1 = (data_dict[key].to(**KWARGS) for key in ("images", "H_0_to_1"))
    B, _, _, H, W = images.shape

    H_gt = torch.stack((H_0_to_1, inv(H_0_to_1)), dim=1)
    intrinsics = repeat(torch.tensor([1000, 1000, W/2, H/2], **KWARGS), 'd -> B 2 d', B=B, d=4)
    depths = torch.ones(B, 2, H, W, **KWARGS)
    bounds = repeat(torch.tensor([H, W], device='cuda', dtype=torch.long), 'uv -> B 2 uv', B=B)
    fixed_output, model_output = model(images, intrinsics, Ps=H_gt, depths=depths, bounds=bounds, homography_pretrain=True)
    points_2d = fixed_output['centroids']
    gt_matches = fixed_output['gt_matches']
    epe_mask = in_bounds_gt = epops.check_in_range(gt_matches, bounds)#epops.check_in_mask(gt_matches.unsqueeze(0), masks.unsqueeze(0), M).view(B, 2, M) # gt_is_inside_frame_if_acc

    all_matching_loss = []
    #all_macd_loss = []
    loss_scaling = np.exp(np.linspace(0, 2, len(model_output)) * 0)
    loss_scaling = loss_scaling / loss_scaling.mean()
    for step, (mo, lm) in enumerate(zip(model_output, loss_scaling)):
        matches = mo['matches']
        #pred_homog = mo['pred_homographies']

        # Matching Error
        assert not isbad(matches)
        assert not isbad(gt_matches)
        epe = (matches - gt_matches).norm(dim=-1)
        assert not isbad(epe)
        epe_for_logging = epe[epe_mask]

        # Matching Loss
        epe_loss_reduced = epe[epe_mask].mean() / 8
        all_matching_loss.append(lm * epe_loss_reduced)

    all_matching_loss = torch.stack(all_matching_loss).mean()
    loss = all_matching_loss + all_matching_loss# + all_macd_loss
    metrics = {"Loss/Matching": all_matching_loss.item(), "Loss/Total": loss.item()}

    metrics['EPE'] = epe_for_logging.mean().item()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    scheduler.step()

    dummy_pred_summary = {
        "pred_pose_a2b": SE3.Random(1).matrix(),
        "gt_pose_a2b": SE3.Random(1).matrix(),
    }

    return metrics, dummy_pred_summary
