from pathlib import Path

import gin
import numpy as np
import torch
from evo.core.sync import matching_time_indices
from evo.tools.file_interface import read_tum_trajectory_file
from imageio.v3 import imread

KWARGS = dict(dtype=torch.float, device='cuda')

@gin.configurable
class ETH3D:

    def __init__(self, base_path, stride, rev=False):
        self.base_path = Path(base_path)
        assert self.base_path.exists(), self.base_path
        traj = read_tum_trajectory_file(self.base_path / "groundtruth.txt")
        self.intrinsics = np.asarray(list(map(float, (self.base_path / "calibration.txt").read_text().split())))
        tstamps = []
        rgb_images = []

        all_lines = (self.base_path / "rgb.txt").read_text().splitlines()
        if rev:
            all_lines = all_lines[::-1]
            print(f"Reversing", self.base_path.name)
        for idx, line in enumerate(all_lines):
            if (idx % stride) != 0:
                continue
            tstamp, rgb = line.split()
            tstamps.append(float(tstamp))
            rgb_images.append(self.base_path / rgb)

        poses = np.array(traj.poses_se3)
        assignment = dict(zip(*matching_time_indices(tstamps, traj.timestamps)))
        self.files = []
        for idx, data in enumerate(zip(tstamps, rgb_images)):
            if j := assignment.get(idx, None):
                self.files.append((*data, poses[j]))
            else:
                self.files.append((*data, None))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        tstamp, rgb_path, pose = self.files[idx]
        return pose, np.copy(self.intrinsics), tstamp, rgb_path

    @staticmethod
    def read_image(rgb_path):
        img = imread(rgb_path)
        H, W, _ = img.shape
        img = img[:(H - H%32), :(W - W%32)]
        image = torch.as_tensor(np.copy(img), **KWARGS).permute(2,0,1)
        return image
