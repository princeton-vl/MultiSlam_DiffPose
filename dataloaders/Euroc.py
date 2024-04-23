from collections import OrderedDict
from pathlib import Path

import cv2
import gin
import numpy as np
import torch
from einops import *
from evo.core import sync
from evo.tools.file_interface import read_tum_trajectory_file
from imageio.v3 import imread

KWARGS = dict(dtype=torch.float, device='cuda')

INTRINSICS = np.array([458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])

@gin.configurable
class Euroc:

    def __init__(self, base_path, stride=2):
        self.base_path = Path(base_path)
        assert self.base_path.exists(), self.base_path # 
        traj = read_tum_trajectory_file(self.base_path)
        imagedir = self.base_path.parent.parent / self.base_path.stem / "mav0" / "cam0" / "data"
        assert imagedir.exists()
        image_list = sorted(imagedir.glob('*.png'))[::stride]
        self.files = {}
        image_tstamps = np.array([float(f.stem) for f in image_list])
        idx_img, idx_traj = sync.matching_time_indices(image_tstamps, traj.timestamps)
        for i,j in zip(idx_img, idx_traj):
            img = image_list[i]
            tstamp = int(img.stem)
            pose = traj.poses_se3[j]
            self.files[tstamp] = (img, pose)

        self.keys = list(self.files.keys())

    def __len__(self):
        return len(self.files)

    @staticmethod
    def read_image(rgb_path):
        img = imread(rgb_path)

        K = np.eye(3)
        K[[0,1,0,1],[0,1,2,2]] = INTRINSICS[:4]

        img = cv2.undistort(img, K, INTRINSICS[4:])

        H, W = img.shape
        img = img[:(H - H%32), :(W - W%32)]
        image = torch.as_tensor(np.copy(img), **KWARGS)
        return repeat(image, 'H W -> 3 H W')

    def __getitem__(self, idx):
        tstamp = self.keys[idx]
        rgb_path, pose = self.files[tstamp]

        return pose, np.copy(INTRINSICS[:4]), float(tstamp)/1e12, rgb_path
