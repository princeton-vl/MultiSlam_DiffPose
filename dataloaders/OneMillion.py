import time
from pathlib import Path

import albumentations as alb
import cv2
import gin
import numpy as np
import torch
from einops import *
from imageio.v3 import imread, imwrite
from scipy.spatial import ConvexHull, Delaunay
from shapely.geometry import Polygon
from torch.utils.data import DataLoader
from tqdm import tqdm

inv = np.linalg.inv

class DatagenException(Exception):
    pass

canonical_corners = np.array([
    [0, 0],
    [0, 999],
    [999, 999],
    [999, 0],
], dtype=np.float64)

def are_inside(bounds, query):
    assert parse_shape(bounds, '_ xy') == parse_shape(query, '_ xy') == {"xy": 2}
    return Delaunay(bounds).find_simplex(query) >= 0

def random_2d_rot(deg=45):
    angle = np.random.uniform(np.radians(-deg), np.radians(deg))
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, s), (-s, c)))

def apply_H(H, p):
    assert H.shape == (3, 3)
    n, d = p.shape
    assert d == 2
    p1 = np.concatenate((p, np.ones((n, 1))), axis=-1)
    p2 = einsum(H, p1, 'i j, n j -> n i')
    return p2[:,:2] / p2[:, 2:]

def calc_intersection(p1, p2):
    assert parse_shape(p1, '_ xy') == parse_shape(p2, '_ xy') == {"xy": 2}
    poly1 = Polygon(p1)
    poly2 = Polygon(p2)
    if poly1.intersects(poly2):
        return poly1.intersection(poly2).area
    return 0

@gin.configurable
def generate_random_corners(random_rotation=False):
    while True:
        # clockwise, starting from top left
        points = np.random.randint(0, 500, size=(4, 2)).astype(np.float64)
        assert points.shape == (4, 2)
        points[2:4, 0] += 500
        points[1:3, 1] += 500
        if ConvexHull(points).nsimplex == 4:
            break

        assert (points.min() >= 0) and (points.max() <= 999)

    center = np.mean(points, axis=0, keepdims=True)

    if random_rotation:
        for _ in range(100):
            rot = random_2d_rot()
            points_tmp = einsum(rot, points - center, 'i j, n j -> n i') + center
            if (points_tmp.min() >= 0) and (points_tmp.max() <= 999):
                points = points_tmp
                break

    assert np.all(are_inside(canonical_corners, points))
    assert are_inside(canonical_corners, np.array([[499.5, 499.5]]))

    while not np.all(are_inside(points, canonical_corners)):
        points -= center
        points *= 1.05
        points += center
        if np.abs(points).max() > 10000:
            raise DatagenException("Too much stretching")

    return points

"""
 blur, hue, saturation, sharpness, illumination, gamma and noise. Furthermore, we add random
additive shades into the image to simulate occlusions and
non-uniform illumination changes.
"""

@gin.configurable
class Distractors1M:

    def __init__(self, base_path, output_size, image_aug=False):
        self.base_path = Path(base_path)
        self.file_paths = (self.base_path / "revisitop1m.txt").read_text().splitlines()
        self.output_size = output_size
        if image_aug:
            self.augment = alb.Compose([
                # alb.Blur(blur_limit=(1, 3)),
                alb.HueSaturationValue(hue_shift_limit=7, sat_shift_limit=30, val_shift_limit=0),
                alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0)),
                alb.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True),
                alb.RandomGamma(gamma_limit=(40, 180), eps=None),
                alb.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8)),
                alb.ISONoise(intensity=(0.1, 0.5), color_shift=(0.01, 0.05)),
                alb.ToGray(always_apply=True)
            ])
        else:
            self.augment = dict

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.base_path / "jpg" / self.file_paths[idx]
        assert image_path.exists()

        try:
            rgb = imread(image_path)
            H, W, *_ = rgb.shape
        except Exception as e:
            print(str(e))
            return self[idx+1]

        if rgb.ndim == 2:
            rgb = repeat(rgb, 'H W -> H W 3')

        img_corners = np.array([
            [0, 0],
            [0, H-1],
            [W-1, H-1],
            [W-1, 0]
        ], dtype=np.float64)

        img_to_canon, _ = cv2.findHomography(img_corners, canonical_corners)

        while True:
            try:
                points_0 = generate_random_corners() # 4 x 2
                points_1 = generate_random_corners() # 4 x 2
                canon_to_frame0, _ = cv2.findHomography(canonical_corners, points_0)
                canon_to_frame1, _ = cv2.findHomography(canonical_corners, points_1)
                H_0_to_1 = canon_to_frame1 @ inv(canon_to_frame0)

                frame_0_in_1 = apply_H(H_0_to_1, canonical_corners)
                percent_overlap_in_1 = calc_intersection(canonical_corners, frame_0_in_1) / 1e6

                H_1_to_0 = canon_to_frame0 @ inv(canon_to_frame1)
                frame_1_in_0 = apply_H(H_1_to_0, canonical_corners)
                percent_overlap_in_0 = calc_intersection(canonical_corners, frame_1_in_0) / 1e6
                min_overlap = min(percent_overlap_in_1, percent_overlap_in_0)
                if min_overlap > 0.3:
                    break
            except DatagenException:
                pass


        H_rescale = np.diag([self.output_size / 1000, self.output_size / 1000, 1]).astype(np.float64)

        img_2_frame0 = H_rescale @ canon_to_frame0 @ img_to_canon
        img_2_frame1 = H_rescale @ canon_to_frame1 @ img_to_canon
        rgb0 = self.augment(image=rgb)["image"]
        rgb1 = self.augment(image=rgb)["image"]
        frame0 = cv2.warpPerspective(rgb0, img_2_frame0, (self.output_size, self.output_size))
        frame1 = cv2.warpPerspective(rgb1, img_2_frame1, (self.output_size, self.output_size))

        images = np.stack((frame0, frame1))
        images = rearrange(torch.as_tensor(images), 'LR H W rgb -> LR rgb H W', LR=2, rgb=3)
        return {
            "images": images,
            "H_0_to_1": torch.as_tensor(H_rescale @ H_0_to_1 @ inv(H_rescale)),
            "image_path": str(image_path),
            "id": Path(image_path).stem
        }

@gin.configurable
class DLDistractors(DataLoader):

    def __init__(self, dataset, *args, rank=0, world_size=1, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        dataset[0]
        self.prefix = "1M"

    def reinit(self):
        pass
