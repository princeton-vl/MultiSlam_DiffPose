

import numpy as np
import torch
from einops import asnumpy

from .utils.external_functions import acos_linear_extrapolation
from multi_slam.dpvo.lietorch import SE3, SO3
from .utils.misc import isbad, GradClip

# --- METRIC AGGREGATION ---
# This function was taken from the src.utils.metrics file in the Loftr codebase
def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    thresholds = [5, 10, 20]
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}

# This function was taken from the src.utils.metrics file in the Loftr codebase
def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err

def batch_angle_error(pred_poses, gt_poses):
    B, _, _ = gt_poses.shape
    ang_errs = {"R": np.full(B, np.nan), "T": np.full(B, np.nan)}
    for i, (gt_pose, m) in enumerate(zip(gt_poses, pred_poses)):
        t = m[...,:3,3]
        r = m[...,:3,:3]
        if np.linalg.norm(t) < 1e-6:
            t = np.ones(3)
        t_err, R_err = relative_pose_error(gt_pose, r, t, ignore_gt_t_thr=0.0)
        ang_errs['R'][i] = R_err
        ang_errs['T'][i] = t_err
    assert not np.any(np.isnan(ang_errs['R']))
    return ang_errs

@torch.no_grad()
def compute_pose_error(pred_poses: SE3, gt_poses: SE3):
    ang_errs = batch_angle_error(asnumpy(pred_poses.matrix()), asnumpy(gt_poses.matrix()))
    all_metrics = {"Rotation Error (Deg)": ang_errs['R'],
                    "Translation Error (Deg)": ang_errs['T'],
                    "Rotation Error (Deg) Recall(%)": (ang_errs['R'] < 10).astype(np.float32)*100,
                    "Translation Error (Deg) Recall(%)": (ang_errs['T'] < 10).astype(np.float32)*100}
    return {f"PoseMetrics/{key}": np.mean(val) for key,val in all_metrics.items()}

# in radians
def compute_angle_loss(pred_poses: SE3, gt_poses: SE3):
    assert not isbad(pred_poses.data)
    assert not isbad(gt_poses.data)
    assert pred_poses.shape == gt_poses.shape
    pred_trans, pred_rot = torch.split(GradClip.apply(pred_poses.data), [3,4], dim=-1)
    gt_trans, gt_rot = torch.split(gt_poses.data, [3,4], dim=-1)
    rotation_ae = (SO3(pred_rot) * SO3(gt_rot).inv()).log().norm(dim=-1)
    assert rotation_ae.shape == pred_poses.shape
    denom = torch.clamp(pred_trans.norm(dim=-1) * gt_trans.norm(dim=-1), min=1e-4)
    numer = (pred_trans * gt_trans).sum(dim=-1)
    translation_ae = acos_linear_extrapolation(numer / denom)
    assert not isbad(translation_ae)
    assert not isbad(rotation_ae)
    return translation_ae, rotation_ae