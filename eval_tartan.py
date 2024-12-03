import argparse
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import cv2
import evo.main_ape as main_ape
import gin
import numpy as np
import torch
from einops import *
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from imageio.v3 import imread
from tqdm import tqdm

from multi_slam.dpvo.data_readers.tartan import test_split as val_split
from multi_slam.fullsystem import FullSystem
from multi_slam.locnet import LocNet

fx, fy, cx, cy = [320, 320, 320, 240]

STRIDE = 1

MAX_LEN = 1000

np.random.RandomState(0).shuffle(val_split)
val_split = val_split[:15]

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image[...,[2,1,0]] / 255.0)
    cv2.waitKey(t)

def video_iterator(imagedir):
    imfiles = sorted((Path(imagedir) / "image_left").glob("*.png"))[:MAX_LEN]
    assert len(imfiles) >= 20

    data_list = []
    for imfile in sorted(imfiles)[::STRIDE]:
        img = imread(imfile)
        image = torch.from_numpy(np.copy(img)).permute(2,0,1)
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        data_list.append((image, intrinsics))

    return data_list

@torch.no_grad()
def run(imagedir, network):

    model = FullSystem(network, network)
    scene_name = Path(imagedir).parts[2]
    data_list = video_iterator(imagedir)
    model.add_new_video(scene_name, len(data_list), (480, 640))

    for t, (image, intrinsics) in enumerate(tqdm(data_list, desc=f'Running on {imagedir}')):
        image = image.cuda()
        intrinsics = intrinsics.cuda()
        model.insert_frame(image, intrinsics, t)
    model.complete_video()

    preds = model.terminate()
    _, tstamps, poses = zip(*preds)
    poses = np.stack(poses)
    tstamps = np.array(tstamps)
    return tstamps, poses

def ate(traj_ref, traj_est, timestamps):

    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=timestamps)

    traj_ref = PoseTrajectory3D(
        positions_xyz=traj_ref[:,:3],
        orientations_quat_wxyz=traj_ref[:,3:],
        timestamps=timestamps)

    result = main_ape.ape(traj_ref, traj_est, est_name='traj',
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    return result.stats["rmse"]

def validate(network, plot=False):

    results = defaultdict(list)
    for scene in val_split:
        scene_dir = Path("datasets/TartanAir") / scene
        traj_ref = scene_dir / "pose_left.txt"
        assert traj_ref.exists()
        tstamps, traj_est = run(scene_dir, network)

        PERM = [1, 2, 0, 4, 5, 3, 6] # ned -> xyz
        traj_ref = np.loadtxt(traj_ref, delimiter=" ")[::STRIDE, PERM][:MAX_LEN]

        ate_score = ate(traj_ref, traj_est, tstamps)
        results[scene].append(ate_score)

        if plot:
            scene_name = '_'.join(scene.split('/')[1:]).title()
            Path("trajectory_plots").mkdir(exist_ok=True)
            j = 0
            plot_trajectory((traj_est, tstamps), (traj_ref, tstamps), f"TartanAir {scene_name.replace('_', ' ')} Trial #{j+1} (ATE: {ate_score:.03f})",
                            f"trajectory_plots/TartanAir_{scene_name}_Trial{j+1:02d}.pdf", align=True, correct_scale=True)

    results = dict(results)

    results_dict = dict([("Tartan/{}".format(k), np.median(v)) for (k, v) in results.items()])
    pprint(results_dict)
    return results_dict