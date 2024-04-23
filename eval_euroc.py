import argparse
from datetime import datetime
from pathlib import Path

import gin
import numpy as np
import torch
from einops import *
from tqdm import tqdm

from dataloaders.Euroc import Euroc
from multi_slam.fullsystem import FullSystem
from multi_slam.locnet import LocNet
from multi_slam.MultiTrajectory import MultiTrajectory

GROUPS = {
    'Machine Hall': ['MH_01_easy', 'MH_02_easy', 'MH_03_medium', 'MH_04_difficult', 'MH_05_difficult'],
    'Machine Hall0-3': ['MH_01_easy', 'MH_02_easy', 'MH_03_medium'],
    'Vicon 2': ['V2_01_easy', 'V2_02_medium', 'V2_03_difficult'],
    'Vicon 1': ['V1_01_easy', 'V1_02_medium', 'V1_03_difficult'],
}

@torch.no_grad()
def main(group_name):

    torch.manual_seed(1234)
    np.random.seed(1234)

    gt_mt = MultiTrajectory("Ground_Truth")
    pred_mt = MultiTrajectory("Estimated")
    scenes = [(s, Euroc(f"data/EuRoC/euroc_groundtruth/{s}.txt", stride=2)) for s in GROUPS[group_name]]

    for scene_name, scene_obj in scenes:
        for (gt_pose, _, tstamp, _) in scene_obj:
            gt_mt.insert(scene_name, tstamp, gt_pose)

    twoview_system = LocNet().cuda().eval()
    twoview_system.load_weights("twoview.pth")

    vo_system = LocNet().cuda().eval()
    vo_system.load_weights("vo.pth")

    model = FullSystem(vo_system, twoview_system)

    start_time = datetime.now()
    for scene_name, scene_obj in scenes:
        model.add_new_video(scene_name, len(scene_obj), (480, 736))
        for _, intrinsics, tstamp, rgb_path in tqdm(scene_obj):
            intrinsics = torch.as_tensor(intrinsics, dtype=torch.float, device='cuda')
            image = scene_obj.read_image(rgb_path)
            model.insert_frame(image, intrinsics, tstamp)
        model.complete_video()
        model.backend_update(iters=10)

    results = model.terminate()
    end_time = datetime.now()

    base_dir = Path("our_predictions") / group_name.replace(' ','_')
    base_dir.mkdir(exist_ok=True, parents=True)

    for scene_name, tstamp, pred_pose in results:
        pred_mt.insert(scene_name, tstamp, pred_pose)

    MultiTrajectory.plot_both(pred_mt, gt_mt, save_dir=base_dir)

    rmse_tr_err, rot_err, recalls = MultiTrajectory.error(pred_mt, gt_mt, ignore_rotation=True)
    text = f'{group_name} Err (t): {rmse_tr_err:.03f}m | Recall {recalls} | {end_time-start_time}'
    print(text)
    (base_dir / "results.txt").write_text(text)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('group', help='group_name', choices=list(GROUPS.keys()))
    args = parser.parse_args()

    gconfigs = [next(iter(Path('gconfigs').rglob(g)), None) for g in (["model/base.gin", "fullsystem.gin"])]
    assert all(gconfigs)
    gin.parse_config_files_and_bindings(gconfigs, [])

    main(args.group)
