import argparse
from collections import OrderedDict
from itertools import count
from pathlib import Path

import cv2
import gin
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from multi_slam.dpvo.data_readers.factory import dataset_factory
from multi_slam.dpvo.lietorch import SE3
from multi_slam.locnet import LocNet
from multi_slam.dpvo.logger import Logger
from eval_tartan import validate


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def image2gray(image):
    image = image.mean(dim=0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def kabsch_umeyama(A, B):
    n, m = A.shape
    EA = torch.mean(A, axis=0)
    EB = torch.mean(B, axis=0)
    VarA = torch.mean((A - EA).norm(dim=1)**2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = torch.svd(H)

    c = VarA / torch.trace(torch.diag(D))
    return c


def train(args):
    """ main training loop """

    # legacy ddp code
    rank = 0

    db = dataset_factory(['tartan'], datapath="datasets/TartanAir", n_frames=args.n_frames)
    train_loader = DataLoader(db, batch_size=1, shuffle=True, num_workers=4)

    # net = VONet()
    # net.train()
    # net.cuda()
    net = LocNet()
    net.cuda()
    assert net.training

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-6)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
        args.lr, args.steps, pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    if args.ckpt is not None:
        states = torch.load(args.ckpt)
        net.load_state_dict(states["model"])
        optimizer.load_state_dict(states["optimizer"])
        scheduler.load_state_dict(states["scheduler"])
        print("Loaded checkpoint")

    if rank == 0:
        logger = Logger(args.name, scheduler, ["Locnet"], args)
    print("Setup complete")

    total_steps = 0

    for epoch in count(1):
        for item_idx, data_blob in enumerate(train_loader):
            if item_idx == 0:
                print("Inner training loop began")
            images, poses, disps, intrinsics = [x.cuda().float() for x in data_blob]
            optimizer.zero_grad()

            # fix poses to gt for first 1k steps
            so = total_steps < 1000 and args.ckpt is None

            poses = SE3(poses).inv()
            traj = net.vo_forward(images, poses, disps, intrinsics, M=80, STEPS=18, structure_only=so)

            loss = 0.0
            for i, (v, x, y, P1, P2, kl) in enumerate(traj):
                e = (x - y).norm(dim=-1) * 2
                e = e.reshape(-1, net.P**2)[(v > 0.5).reshape(-1)].min(dim=-1).values

                N = P1.shape[1]
                ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
                ii = ii.reshape(-1).cuda()
                jj = jj.reshape(-1).cuda()

                k = ii != jj
                ii = ii[k]
                jj = jj[k]

                P1 = P1.inv()
                P2 = P2.inv()

                t1 = P1.matrix()[...,:3,3]
                t2 = P2.matrix()[...,:3,3]

                loss += args.flow_weight * e.mean()
                try:
                    s = kabsch_umeyama(t2[0], t1[0]).detach().clamp(max=10.0)
                    P1 = P1.scale(s.view(1, 1))

                    dP = P1[:,ii].inv() * P1[:,jj]
                    dG = P2[:,ii].inv() * P2[:,jj]

                    e1 = (dP * dG.inv()).log()
                    tr = e1[...,0:3].norm(dim=-1)
                    ro = e1[...,3:6].norm(dim=-1)

                    if not so and i >= 2:
                        loss += args.pose_weight * ( tr.mean() + ro.mean() )
                except:
                    print(f"kabsch_umeyama failed {i=}")
                    tr, ro = torch.tensor([1000, np.pi])

            # kl is 0 (not longer used)
            loss += kl
            loss.backward()

            try:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip, error_if_nonfinite=True)
            except Exception as e:
                print(f"{e}\nBAD GRADIENT. Reloading model and optimizer.")
                optimizer.zero_grad()
                net.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                continue

            optimizer.step()
            scheduler.step()

            total_steps += 1

            metrics = {
                "loss": loss.item(),
                "kl": kl.item(),
                "px1": (e < .25).float().mean().item(),
                "ro": ro.float().mean().item(),
                "tr": tr.float().mean().item(),
                "r1": (ro < .001).float().mean().item(),
                "r2": (ro < .01).float().mean().item(),
                "t1": (tr < .001).float().mean().item(),
                "t2": (tr < .01).float().mean().item(),
            }

            if rank == 0:
                logger.push(metrics)

            if total_steps % 10000 == 0:
                torch.cuda.empty_cache()

                if rank == 0:
                    PATH = 'checkpoints/%s_%06d.pth' % (args.name, total_steps)
                    state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "model": net.state_dict()}
                    torch.save(state, PATH)

                validation_results = validate(net)
                validation_results = {f"Validation/{key}":val for key,val in validation_results.items()}
                if rank == 0:
                    logger.write_dict(validation_results)

                torch.cuda.empty_cache()
                net.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='name your experiment')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--steps', type=int, default=240000)
    parser.add_argument('--lr', type=float, default=0.00008)
    parser.add_argument('--clip', type=float, default=10.0)
    parser.add_argument('--n_frames', type=int, default=15)
    parser.add_argument('--pose_weight', type=float, default=10.0)
    parser.add_argument('--flow_weight', type=float, default=0.1)
    args = parser.parse_args()

    gconfigs = [next(iter(Path('gconfigs').rglob(g)), None) for g in (["model.gin", "fullsystem.gin"] + [])]
    assert all(gconfigs)
    gin.parse_config_files_and_bindings(gconfigs, [])
    print(f"{args.name=}")

    train(args)
