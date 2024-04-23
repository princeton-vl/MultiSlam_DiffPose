import argparse
from pathlib import Path

import gin
import numpy as np
import torch

import wandb
from multi_slam.locnet import LocNet
from multi_slam.utils.misc import clean_state_dict
from train import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--overrides', nargs='+', type=str, default=[], help="Parameter overrides")
    parser.add_argument('--dataset', required=True, type=str, choices=["val_scannet", "test_scannet", "val_megadepth", "test_megadepth"], help="Eval dataset")
    parser.add_argument('--load_ckpt', type=Path, required=True)
    parser.add_argument('--batch_size', default=8, type=int)
    args = parser.parse_args()
    assert (args.load_ckpt is None) or args.load_ckpt.exists()

    torch.manual_seed(1234)
    np.random.seed(1234)

    mode, dataset = args.dataset.split('_')
    if dataset == "megadepth":
        args.overrides[:0] = [f"evaluate.DataLoader = @MD/{mode}/DLWrapper", f"MD/{mode}/MegaDepthDatasetWrapper.gray_chance=1.0"]
    if dataset == "scannet":
        args.overrides[:0] = [f"evaluate.DataLoader = @SN/{mode}/DLWrapper", f"SN/{mode}/ScanNetDatasetWrapper.gray_chance=1.0"]

    # Set gin configs
    gconfigs = [next(iter(Path('gconfigs').rglob(g)), None) for g in ["default.gin", "megadepth.gin", "scannet.gin"]]
    assert all(gconfigs) # ensure all .gin files were found
    gin.parse_config_files_and_bindings(gconfigs, args.overrides)
    wandb.init(anonymous="allow", mode='disabled')

    model = LocNet().cuda()
    model.load_state_dict(clean_state_dict(torch.load(args.load_ckpt)))
    evaluate(model, 1, batch_size=args.batch_size)

    print("Done!")
