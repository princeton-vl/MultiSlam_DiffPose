import argparse
from datetime import datetime
from pathlib import Path

import gin
import numpy as np
import torch
from einops import *
from tqdm import tqdm

from dataloaders.ETH3D import ETH3D
from multi_slam.fullsystem import FullSystem
from multi_slam.locnet import LocNet
from multi_slam.MultiTrajectory import MultiTrajectory

GROUPS = {
    'plant_scene': ['plant_scene_1', 'plant_scene_2', 'plant_scene_3'],
    'table': ['table_3', 'table_4'],
    'sofa': ['sofa_1', 'sofa_2', 'sofa_3', 'sofa_4'],
    'einstein': ['einstein_1', 'einstein_2'],
    'planar': ['planar_2', 'planar_3']
}

@torch.no_grad()
def main(group_name):

    torch.manual_seed(1234)
    np.random.seed(1234)

    gt_mt = MultiTrajectory("Ground_Truth")
    pred_mt = MultiTrajectory("Estimated")
    scenes = [(s, ETH3D(f"data/ETH3D/{s}", stride=2, rev=(i%2 == 1))) for i,s in enumerate(GROUPS[group_name])]

    for scene_name, scene_obj in scenes:
        for (gt_pose, _, tstamp, _) in scene_obj:
            if gt_pose is not None:
                gt_mt.insert(scene_name, tstamp, gt_pose)

    twoview_system = LocNet().cuda().eval()
    twoview_system.load_weights("twoview.pth")

    vo_system = LocNet().cuda().eval()
    vo_system.load_weights("vo.pth")

    model = FullSystem(vo_system, twoview_system)

    start_time = datetime.now()
    for scene_name, scene_obj in scenes:
        model.add_new_video(scene_name, len(scene_obj), (448,736))
        for _, intrinsics, tstamp, rgb_path in tqdm(scene_obj):
            intrinsics = torch.as_tensor(intrinsics, dtype=torch.float, device='cuda')
            image = scene_obj.read_image(rgb_path)
            model.insert_frame(image, intrinsics, tstamp)
        model.complete_video()
        model.backend_update(iters=10)

    results = model.terminate()
    end_time = datetime.now()

    base_dir = Path("our_predictions") / group_name
    base_dir.mkdir(exist_ok=True, parents=True)

    for scene_name, tstamp, pred_pose in results:
        pred_mt.insert(scene_name, tstamp, pred_pose)

    MultiTrajectory.plot_both(pred_mt, gt_mt, save_dir=base_dir)

    rmse_tr_err, rot_err, recalls = MultiTrajectory.error(pred_mt, gt_mt)
    text = f'Err (t): {rmse_tr_err:.03f}m | Err (R): {rot_err:.01f} deg | Recall {recalls} | {end_time-start_time}'
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
