import colorsys
from copy import deepcopy
from pathlib import Path

import cv2
import evo.core.geometry as geometry
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from einops import parse_shape
from evo.core import lie_algebra as lie
from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import plot
from matplotlib import colors
from scipy.spatial.transform import Rotation as R

matplotlib.use('TKAgg')

COLORS = np.array([colors.to_rgba(f'C{i}') for i in range(10)])[:,:3]

def calc_umeyama(pred, gt, use_evo, **cv_kwargs):
    if use_evo:
        return geometry.umeyama_alignment(pred.T, gt.T, True)
    reval, out, inliers = cv2.estimateAffine3D(pred, gt, force_rotation=True, **cv_kwargs)
    print("inliers", inliers.astype(np.float64).mean())
    assert reval == 1
    s = np.linalg.norm(out[:,0])
    t_a = out[:3,3]
    r_a = out[:3,:3] / s
    return r_a, t_a, s

def desat(rgb):
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    return colorsys.hsv_to_rgb(h, s*0.7, v*0.7)

def make_4x4(d):
    out = np.eye(4)
    out[:3,3] = d[:3]
    out[:3,:3] = R.from_quat(d[3:]).as_matrix()
    return out

class MultiTrajectory:

    def __init__(self, name):
        self.name = name
        self.trajectories = {}
    
    def insert(self, key, timestamp: float, pose: np.ndarray):
        assert isinstance(timestamp, float), timestamp
        if key not in self.trajectories:
            self.trajectories[key] = dict(timestamps=[], poses_se3=[])
        if pose.shape != (4,4):
            pose = make_4x4(pose)
        self.trajectories[key]["timestamps"].append(timestamp)
        self.trajectories[key]["poses_se3"].append(pose)
    
    def get(self, key):
        return PoseTrajectory3D(**self.trajectories[key])

    def __set_traj(self, key, traj: PoseTrajectory3D):
        self.trajectories[key] = dict(timestamps=list(traj.timestamps), poses_se3=list(traj.poses_se3))

    @staticmethod
    def error(est: 'MultiTrajectory', ref: 'MultiTrajectory', ignore_rotation=False):
        est = deepcopy(est)
        ref = deepcopy(ref)
        est.align(ref, sync_traj=True, trim_gt=False)
        dists = np.linalg.norm(est.positions_xyz - ref.positions_xyz, axis=1)
        rotation_errors = np.rad2deg((est.orientations * ref.orientations.inv()).magnitude())
        assert dists.shape == rotation_errors.shape
        e = 1e4 * ignore_rotation
        recalls = [(t, a, np.around(np.mean((dists < t) & (rotation_errors < a))*100, 3)) for t,a in [(0.01, 2 + e), (0.025, 3 + e), (0.05, 5 + e), (0.1, 10 + e)]]
        return (dists**2).mean()**0.5, rotation_errors.mean(), recalls

    @staticmethod
    def __best_plotmode(*mts):
        positions_xyz = np.concatenate([mt.positions_xyz for mt in mts])
        _, i1, i2 = np.argsort(np.var(positions_xyz, axis=0))
        plot_axes = "xyz"[i2] + "xyz"[i1]
        return getattr(plot.PlotMode, plot_axes)

    @staticmethod
    def plot_both(est: 'MultiTrajectory', ref: 'MultiTrajectory', save_dir='/tmp/', **kwargs):
        est = deepcopy(est)
        ref = deepcopy(ref)

        rmse_err, _, recalls = MultiTrajectory.error(est, ref, **kwargs)

        est.align(ref, sync_traj=True, trim_gt=False)

        plot_collection = plot.PlotCollection("PlotCol")
        fig = plt.figure(figsize=(8, 8))
        plot_mode = MultiTrajectory.__best_plotmode(est, ref)
        ax = plot.prepare_axis(fig, plot_mode)
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        filename = f"{save_dir}/{est.name}_vs_{ref.name}.pdf"

        title = f'RMSE ATE: {rmse_err:.03f}m'

        ax.set_title(title)

        for color, key in zip(COLORS, ref.trajectories):
            plot.traj(ax, plot_mode, est.get(key), '-', color, f"{key} {est.name.title().replace('_',' ')}")
            plot.traj(ax, plot_mode, ref.get(key), '--', desat(color), f"{key} {ref.name.title().replace('_',' ')}")

        plot_collection.add_figure("traj (error)", fig)
        plot_collection.export(filename, confirm_overwrite=False)
        plt.close(fig=fig)
        print(f"Saved {filename}")

    def plot(self):
        plot_collection = plot.PlotCollection("PlotCol")
        fig = plt.figure(figsize=(8, 8))
        plot_mode = MultiTrajectory.__best_plotmode(self)
        ax = plot.prepare_axis(fig, plot_mode)
        filename = f"/tmp/{self.name}.pdf"

        for color, key in zip(COLORS, self.trajectories):
            plot.traj(ax, plot_mode, self.get(key), '-', color, key)

        plot_collection.add_figure("traj (error)", fig)
        plot_collection.export(filename, confirm_overwrite=False)
        plt.close(fig=fig)
        print(f"Saved {filename}")

    @property
    def positions_xyz(self):
        output = []
        for key in sorted(self.trajectories):
            output.append(self.get(key).positions_xyz)
        return np.concatenate(output)

    @property
    def orientations(self):
        output = []
        for key in sorted(self.trajectories):
            output.append(self.get(key).orientations_quat_wxyz)
        output = np.concatenate(output)
        return R.from_quat(output[:,[1,2,3,0]])

    def sync(self, other, trim_gt=False):
        for key in self.trajectories:
            traj_gt, traj_est = sync.associate_trajectories(other.get(key), self.get(key))
            self.__set_traj(key, traj_est)
            if trim_gt:
                other.__set_traj(key, traj_gt)

    @staticmethod
    def _calc_alignment(est: 'MultiTrajectory', other: 'MultiTrajectory'):
        est = deepcopy(est)
        other = deepcopy(other)
        est.sync(other, trim_gt=True)
        assert parse_shape(est.positions_xyz, 'N xyz') == parse_shape(other.positions_xyz, 'N xyz'), (est.positions_xyz.shape, other.positions_xyz.shape)
        assert est.positions_xyz.shape[1] == 3
        return calc_umeyama(est.positions_xyz, other.positions_xyz, use_evo=True, ransacThreshold=0.01)

    def align(self, other: 'MultiTrajectory', sync_traj, trim_gt=False):
        r_a, t_a, s = MultiTrajectory._calc_alignment(self, other)
        if sync_traj:
            self.sync(other, trim_gt=trim_gt)
        for key in self.trajectories:
            traj_est = self.get(key)
            traj_est.scale(s)
            traj_est.transform(lie.se3(r_a, t_a))
            self.__set_traj(key, traj_est)
