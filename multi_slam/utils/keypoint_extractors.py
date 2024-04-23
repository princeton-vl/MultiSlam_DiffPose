import contextlib
import io
import sys

import gin
import torch
from einops import *

autocast = torch.cuda.amp.autocast

superpoint_network = None

def bounds_from_img(images):
    *B, _, H, W = images.shape
    return torch.tensor([W, H], device=images.device).tile(*B, 1)

def initialize_superpoint():
    sys.path.append("thirdparty/Hierarchical-Localization")
    with contextlib.redirect_stderr(io.StringIO()):
        from hloc.extractors.superpoint import SuperPoint
    global superpoint_network
    with contextlib.redirect_stdout(io.StringIO()):
        superpoint_network = SuperPoint({'nms_radius': 3, 'max_keypoints': -1, 'remove_borders': 8, 'fix_sampling': True}).eval().to('cuda')

@gin.configurable
@torch.inference_mode()
def random_keypoints(images, bounds, M, remove_borders=8, **kwargs):
    if bounds is None:
        bounds = bounds_from_img(images)
    *B, _ = bounds.shape # B LR uv
    max_bounds = (bounds.unsqueeze(-2) - (2 * remove_borders))
    min_bounds = torch.full_like(max_bounds, remove_borders)
    keypoints = torch.randint(0, 1000000, size=(*B, M, 2), device=bounds.device)
    keypoints = (keypoints % max_bounds) + min_bounds
    return keypoints.float(), None

@gin.configurable
@torch.inference_mode()
def superpoint_helper(image, bounds, remove_borders, nms_radius, keypoint_threshold):
    assert not torch.is_grad_enabled()
    global superpoint_network
    if superpoint_network is None:
        initialize_superpoint()

    superpoint_network.net.config['nms_radius'] = nms_radius
    superpoint_network.net.config['remove_borders'] = remove_borders
    superpoint_network.net.config['keypoint_threshold'] = keypoint_threshold

    image = image.squeeze()
    assert image.ndim == 2 or ((image.ndim == 3) and image.shape[0] == 3)
    *_, H, W = image.shape
    image = image.view(1, 1, -1, H, W).mean(dim=2)

    bx, by = asnumpy(bounds.view(2))
    pred = superpoint_network({'image': image[...,:by,:bx]})
    sp_keypoints = pred['keypoints'][0]
    scores = pred['scores'][0]
    indices = torch.argsort(scores, descending=True)
    return sp_keypoints[indices]

@gin.configurable
@torch.inference_mode()
def superpoint_keypoints(images, bounds, M, remove_borders, nms_radius, keypoint_threshold, mixed_prec=False):
    assert not torch.is_grad_enabled()
    assert images.max() > 1.1

    images = images.float() / 255
    assert (-1e-4 < images.min()) and (images.max() < 1+1e-4)
    if bounds is None:
        bounds = bounds_from_img(images)
    *B, _ = bounds.shape
    images = reduce(images, '... 3 H W -> (...) 1 1 H W', 'mean')

    keypoints, _ = random_keypoints(None, bounds, M, remove_borders)
    keypoints = keypoints.flatten(0, -3)
    num_sp_kp = []

    for idx, (image, boundary) in enumerate(zip(images, bounds.view(-1, 2))):
        with autocast(enabled=mixed_prec, dtype=torch.bfloat16):
            sp_keypoints = superpoint_helper(image, boundary, remove_borders, nms_radius, keypoint_threshold)

        m = min(sp_keypoints.shape[0], M)
        keypoints[idx, :m] = sp_keypoints[:m]
        num_sp_kp.append(m)

    return keypoints.view(*B, M, 2), torch.as_tensor(num_sp_kp, device=keypoints.device).view(*B)
