import time
from pathlib import Path
from subprocess import check_output

import albumentations as alb
import gin
import numpy as np
import torch
import torch.multiprocessing as mp
import wandb
from einops import parse_shape, rearrange, repeat
from scipy.spatial.transform import Rotation as R
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms.functional import InterpolationMode, resize
from tqdm import tqdm

from multi_slam.dpvo.lietorch import SE3

from .loftr_datasets import (MegaDepthDataset, RandomConcatSampler,
                             ScanNetDataset, get_local_split)


def assert_img(x):
    assert parse_shape(x, '_ _ rgb') == dict(rgb=3), (x.shape, x.dtype)
    assert any((x.dtype == d) for d in {np.uint8, torch.uint8})

def devshm_used():
    txt = check_output("df /dev/shm".split(), text=True)
    kb_used = int(txt.splitlines()[1].split()[2])
    return kb_used / (1024**2)

def process_data_dict(data, mode):
    assert_img(data['image0'])
    assert_img(data['image1'])
    H, W, _ = data['image0'].shape
    if 'bounds' not in data:
        data['bounds'] = torch.tensor(((W, H), (W, H)))
    if mode == "test":
        assert data['depth0'].numel() == 0
        m0 = data['image0'].sum(dim=2) > 0
        data['depth0'] = torch.rand(H, W).to(data['depth0']) * m0
        m1 = data['image1'].sum(dim=2) > 0
        data['depth1'] = torch.rand(H, W).to(data['depth1']) * m1
    return data

class GrayImage:

    def __init__(self, p):
        assert -0.01 < p < 1.01
        self.aug = alb.ToGray(always_apply=False, p=p)

    def __call__(self, img):
        img = img.numpy()
        img = self.aug(image=img)['image']
        return torch.as_tensor(img)


@gin.configurable
class ScanNetDatasetWrapper(ScanNetDataset):

    def __init__(self, root_dir, npz_path, intrinsics_path, mode, augmentor, gray_chance, pad_to_size=None, **kwargs):
        self.pad_to_size = pad_to_size
        assert (mode == "train") or (kwargs["min_overlap_score"] == 0)
        super().__init__(root_dir=root_dir, npz_path=npz_path, mode=mode, augment_fn=augmentor, intrinsic_path=intrinsics_path, **kwargs)
        self.img_augment = GrayImage(gray_chance)

    def __upscale(self, data, interp_mode):
        W, H = self.pad_to_size, (self.pad_to_size * 3) // 4
        data = rearrange(data, 'h w ... -> 1 (...) h w')
        _, c, h, w = data.shape
        output = torch.zeros(self.pad_to_size, self.pad_to_size, c, dtype=data.dtype)
        output[:H, :W] = rearrange(resize(data, (H, W), interpolation=interp_mode, antialias=True), '1 c h w -> h w c')
        return output.squeeze()

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        if self.pad_to_size is None:
            data['bounds'] = repeat(torch.tensor([640, 480]), 'uv -> 2 uv')
        else:
            w, h = self.pad_to_size, (self.pad_to_size * 3) // 4
            data['bounds'] = repeat(torch.tensor([w, h]), 'uv -> 2 uv')
            K_scale = torch.diag(torch.tensor([self.pad_to_size / 640, self.pad_to_size / 640, 1]))
            for i in range(2):
                data[f'K{i}'] = K_scale @ data[f'K{i}']
                data[f"image{i}"] = self.__upscale(data[f"image{i}"], InterpolationMode.BILINEAR)
                if data[f"depth{i}"].numel() > 0:
                    data[f"depth{i}"] = self.__upscale(data[f"depth{i}"], InterpolationMode.NEAREST)

        data = process_data_dict(data, self.mode)

        for i in range(2):
            data[f"image{i}"] = self.img_augment(data[f"image{i}"])

        return data

@gin.configurable
class MegaDepthDatasetWrapper(MegaDepthDataset):

    def __init__(self, root_dir, npz_path, intrinsics_path, mode, augmentor, gray_chance, **kwargs):
        assert (mode == "train") or (kwargs["min_overlap_score"] == 0)
        super().__init__(root_dir=root_dir, npz_path=npz_path, mode=mode, augment_fn=augmentor, intrinsic_path=intrinsics_path, **kwargs)
        self.img_augment = GrayImage(gray_chance)

    def __pad(self, data):
        W = H = self.img_resize
        data = rearrange(data, 'h w ... -> h w (...)')
        h, w, c = data.shape
        output = torch.zeros(H, W, c, dtype=data.dtype)
        output[:h, :w] = data[:H, :W]
        return output.squeeze()

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        assert torch.all(data['bounds'].max(dim=1).values == self.img_resize)
        for i in range(2):
            if data[f"depth{i}"].numel() > 0:
                data[f"depth{i}"] = self.__pad(data[f"depth{i}"])
            data[f"image{i}"] = self.__pad(data[f"image{i}"])
        data = process_data_dict(data, self.mode)

        for i in range(2):
            data[f"image{i}"] = self.img_augment(data[f"image{i}"])

        return data

@gin.configurable
class DLWrapper:

    def __init__(self, mode, shuffle, DatasetClass, data_root, intrinsics_path, npz_path_list, npz_root, seed, rank=0, world_size=1, **dl_kwargs):
        assert "__getitem__" in DatasetClass.__dict__
        assert mode in {"train", "val", "test", "train_holdval"}
        _, main_dataset_name, *_ = Path(data_root).parts
        npz_paths = sorted((Path(npz_root) / (n.rstrip('.npz')+'.npz')) for n in Path(npz_path_list).read_text().splitlines())
        if mode in {"val", "train_holdval"}: # not Test
            val_list = Path("data") / main_dataset_name / "val_scenes_lahav_seed82.txt"
            assert val_list.exists()
            val_scenes = set(val_list.read_text().splitlines())
            if mode == "val":
                npz_paths = sorted(filter(lambda p: p.stem.split('_')[0] in val_scenes, npz_paths))
            elif mode == "train_holdval":
                npz_paths = sorted(filter(lambda p: p.stem.split('_')[0] not in val_scenes, npz_paths))

        self.prefix = f"{main_dataset_name}/{mode}".title()
        self.prev_time = None
        self.mode = mode
        mode = mode.replace("train_holdval", "train")
        self.rank_world = f"(RANK:{rank} WORLD:{world_size})"

        assert len(npz_paths) > 0
        npz_paths = get_local_split(npz_paths, world_size, rank, seed).tolist()
        ds_args = [(data_root, str(npz_path), intrinsics_path, mode, None) for npz_path in npz_paths]
        DatasetClass(*(ds_args[0]))[0] # just a test
        itr = tqdm(ds_args, desc=f'Loading {len(ds_args)} scenes from ({self.prefix}) {self.rank_world}', disable=((mode != "train") or (len(ds_args) < 50)))
        if torch.distributed.is_initialized():
            scene_datasets = [DatasetClass(*dsa) for dsa in itr]
        else:
            with mp.Pool(processes=20) as pool:
                scene_datasets = list(pool.starmap(DatasetClass, itr))
        dataset = ConcatDataset(scene_datasets)
        self.dataset = dataset

        if mode != "test":
            dl_kwargs['sampler'] = RandomConcatSampler(dataset,
                                            (200 if ("train" == mode) else 50), # n_samples_per_subset,
                                            True, # subset_replacement,
                                            shuffle, # shuffle
                                            1,# repeat,
                                            seed) # seed)
            print(f"Built {self.prefix} dataset with {len(scene_datasets)} unique scenes, ~{round(np.mean([len(d) for d in scene_datasets])):,} pairs per scene, {len(self.dataset):,} overall. {self.rank_world}")
        elif mode == "test":
            dl_kwargs['shuffle'] = shuffle

        self.dl_kwargs = dl_kwargs
        self.reinit(True)

    def set_epoch(self, epoch):
        for _ in range(2, epoch):
            iter(self.dl_kwargs['sampler'])
        self.reinit(True)
        print(f"Skipped dataloader to epoch {epoch}")

    def reinit(self, force=False):
        if force or (self.dataloader_iterator is None):
            self.dataloader = DataLoader(self.dataset, **self.dl_kwargs)
            print(f"Created Dataloader ({self.prefix}): # Samples-per-epoch: {len(self.dataloader):,} {self.rank_world}")
            self.dataloader_iterator = iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def __next__(self):
        cur = time.time()
        if (self.prev_time is not None) and (wandb.run is not None):
            wandb.log({f"Timing/full_{self.mode}_step (seconds)": cur - self.prev_time}, commit=False)
        self.prev_time = cur
        data_dict = next(self.dataloader_iterator, None) # ['image0', 'depth0', 'image1', 'depth1', 'T_0to1', 'T_1to0', 'K0', 'K1', 'dataset_name', 'scene_id', 'pair_id', 'pair_names']
        if data_dict is None:
            self.dataloader_iterator = None
            raise StopIteration

        K0 = data_dict['K0'][:,[0,1,0,1],[0,1,2,2]]
        K1 = data_dict['K1'][:,[0,1,0,1],[0,1,2,2]]

        T = data_dict['T_1to0']
        T_np = T.numpy()
        q = torch.as_tensor(R.from_matrix(T_np[:,:3,:3]).as_quat())
        t = T[:,:3,3]
        pose_1 = SE3(torch.cat((t,q), dim=1))
        pose_0 = SE3.IdentityLike(pose_1)
        st = lambda a,b: torch.stack((a, b), dim=1)

        return {
            "images": rearrange(st(data_dict['image0'], data_dict['image1']).float().cuda(), 'B LR H W RGB -> B LR RGB H W', LR=2, RGB=3),
            "depths": st(data_dict['depth0'], data_dict['depth1']).cuda(),
            "poses": SE3(st(pose_0.data, pose_1.data).float().cuda()),
            "intrinsics": st(K0, K1).cuda(),
            "bounds": data_dict["bounds"].cuda(),
            "pair_names": list(zip(*data_dict['pair_names'])),
            **{k:data_dict[k] for k in ['dataset_name', 'scene_id', 'pair_id']},
         }
    
    def __iter__(self):
        return self
