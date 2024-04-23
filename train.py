import argparse
from itertools import count
from pathlib import Path

import gin
import gin.torch.external_configurables
import numpy as np
import torch
import wandb
from einops import asnumpy
from tqdm import tqdm

import multi_slam.solver.epa_ops as epops
from dataloaders import combined
from dataloaders.DLWrapper import devshm_used
from multi_slam.locnet import LocNet
from multi_slam.metrics import (batch_angle_error, compute_angle_loss,
                                compute_pose_error, error_auc)
from multi_slam.utils.ddp_utils import *
from multi_slam.utils.misc import Parachute, WBTimer, commit_sha, isbad
from multi_slam.homog_ops import homography_forward_pass


@gin.configurable
def train(job_name, DataLoader, optimizer_class, scheduler_class, rank, world_size, use_ddp, model_state, evaluations, batch_size, forward):

    torch.manual_seed(1234)
    np.random.seed(1234)
    device_id = f"cuda:{rank}"

    model = LocNet().to(device_id)
    if model_state is not None:
        model.load_weights(model_state)
    assert model.training
    if use_ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], static_graph=True)
    # for eval_func in evaluations:
    #     eval_func(net, 1)
    # exit(0)

    optimizer = optimizer_class(model.parameters())
    scheduler = scheduler_class(optimizer)

    train_dataloader = DataLoader(num_workers=calc_num_workers(), rank=rank, world_size=world_size, batch_size=batch_size)

    training_step = 0
    all_metrics = []
    all_preds = []
    all_gt = []
    for epoch in count(start=1):
        train_dataloader.reinit()
        wandb.log({f'{train_dataloader.prefix}/Misc/Epoch':epoch}, step=training_step)
        for batch_index, data_dict in enumerate(tqdm(train_dataloader, desc=f"Epoch [Rank {rank}]: {epoch}")):
            training_step += 1
            optimizer.zero_grad()
            with Parachute(data_dict):
                metrics, pred_summary = forward(data_dict, model, optimizer=optimizer, scheduler=scheduler)

            all_metrics.append(metrics)
            all_preds.append(asnumpy(pred_summary['pred_pose_a2b']))
            all_gt.append(asnumpy(pred_summary['gt_pose_a2b']))
            wandb.log({f"{train_dataloader.prefix}/"+key:val for key,val in metrics.items()}, step=training_step)
            if training_step % 10 == 0:
                wandb.log({"System/\dev\shm Used (GB)": devshm_used()}, step=training_step)

            with torch.no_grad():
                if (training_step % 200) == 0:
                    if rank == 0:
                        print('-'*30)
                        print(f"Training Step [Epoch: {epoch}] [Rank: {rank}]: {training_step}")
                        print('-'*20)
                    ang_errs = batch_angle_error(np.concatenate(all_preds), np.concatenate(all_gt))
                    all_preds.clear(); all_gt.clear()
                    auc = error_auc(np.maximum(ang_errs['R'], ang_errs['T']), [5, 10, 20])
                    if rank == 0:
                        print(f"AUC@10".ljust(35) + f" : {auc['auc@10']:.03f}")
                        wandb.log({f'{train_dataloader.prefix}/AUC/{m}':v for m, v in auc.items()}, step=training_step)
                        for key in ["Loss/Matching", "Loss/Pose", "Loss/Total", "FlowMetrics/Mean EPE"]:
                            if key in metrics:
                                print(f"{key.ljust(35)} : {np.mean([d[key] for d in all_metrics]):.03f}")
                    all_metrics.clear()
                    wandb.log({f"{train_dataloader.prefix}/Misc/Learning Rate": optimizer.param_groups[0]['lr']}, step=training_step)
                    if (training_step % 5000) == 0:
                        for eval_func in evaluations:
                            eval_func(model, training_step, rank=rank, batch_size=batch_size)
                        if rank == 0:
                            dir = Path("model_weights") / job_name
                            dir.mkdir(exist_ok=True, parents=True)
                            state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "model": model.state_dict(), "training_step": training_step, "epoch": epoch}
                            torch.save(state, dir / f"step_{training_step:06d}.pth")
                    if rank == 0:
                        print('-'*30)

            if training_step >= scheduler.total_steps:
                print(f"Done training (Step: {training_step})")
                return

@gin.configurable
@WBTimer("evaluation")
@torch.inference_mode()
def evaluate(model, step, DataLoader, batch_size, rank=0):
    assert not torch.is_grad_enabled()
    model.eval()
    val_dataloader = DataLoader(num_workers=calc_num_workers(), batch_size=batch_size)


    total_val_metrics = []
    all_preds = []
    all_gt = []

    for batch_index, data_dict in enumerate(tqdm(val_dataloader, position=0, bar_format=f'###### Validating on {val_dataloader.prefix}... {{n_fmt}}/{{total_fmt}} [{commit_sha()}] ######', disable=(rank > 0))):
        with Parachute(data_dict):
            metrics, pred_summary = forward_pass(data_dict, model, testing=True)
            all_preds.append(asnumpy(pred_summary['pred_pose_a2b']))
            all_gt.append(asnumpy(pred_summary['gt_pose_a2b']))
        total_val_metrics.append(metrics)

    if rank == 0:
        ang_errs = batch_angle_error(np.concatenate(all_preds), np.concatenate(all_gt))
        auc = error_auc(np.maximum(ang_errs['R'], ang_errs['T']), [5, 10, 20])
        for m, v in auc.items():
            print(f'{val_dataloader.prefix}/AUC/{m}',v)
            wandb.log({f'{val_dataloader.prefix}/AUC/{m}':v}, step=step)

        for key in total_val_metrics[0].keys():
            val = np.mean([e[key] for e in total_val_metrics])
            print(f"{val_dataloader.prefix}/{key} : {val}")
            wandb.log({f"{val_dataloader.prefix}/{key}":val}, step=step)
    model.train()

@gin.configurable
def forward_pass(data_dict, model, flow_loss_weight=0.0, cycle_consistent_epe=False, optimizer=None, scheduler=None, testing=False):

    images, depths, poses, intrinsics, bounds = (data_dict.get(key) for key in ("images", "depths", "poses", "intrinsics", "bounds"))
    B, _, _, _, _ = images.shape

    fixed_output, model_output = model(images, intrinsics, Ps=poses.inv(), depths=depths, bounds=bounds)
    gt_matches = fixed_output["gt_matches"]
    has_accurate_gt = fixed_output["has_accurate_gt"]
    cycle_consistent_gt = fixed_output["cycle_consistent_gt"]
    assert not isbad(gt_matches[has_accurate_gt])
    in_bounds_gt = epops.check_in_range(gt_matches, bounds) # gt_is_inside_frame_if_acc
    assert in_bounds_gt.shape == has_accurate_gt.shape
    assert in_bounds_gt.dtype == has_accurate_gt.dtype == torch.bool

    epe_mask = in_bounds_gt & has_accurate_gt
    if cycle_consistent_epe:
        epe_mask = epe_mask & cycle_consistent_gt
    if testing:
        epe_mask[:] = True # test data has no depth, so this prevents crashes

    nonempty_mask = epe_mask.view(B, -1).any(dim=1)
    if not nonempty_mask.all():
        print(f"WARNING: {np.arange(B)[asnumpy(~nonempty_mask)]} have no valid points to supervise on")
        if not nonempty_mask.any():
            return (None, (None, None), None)

    assert epe_mask.any()
    centroids = fixed_output["centroids"]
    gt_pose = poses[:,1]

    all_matching_loss = []
    all_rot_loss = []
    all_trans_loss = []
    for step, mo in enumerate(model_output):
        matches = mo['matches']

        # Matching Error
        epe = (matches - gt_matches).norm(dim=-1)
        epe_masked = epe[epe_mask]

        # Pose loss
        pred_poses = mo['poses']
        with Parachute((pred_poses, gt_pose)):
            tr, ro = compute_angle_loss(pred_poses, gt_pose)

        assert testing or (not isbad(epe_masked))
        all_matching_loss.append(epe_masked.mean() * flow_loss_weight)
        all_rot_loss.append(ro.mean())
        all_trans_loss.append(tr.mean())

    all_matching_loss = torch.stack(all_matching_loss).mean()
    all_rot_loss = torch.stack(all_rot_loss).mean()
    all_trans_loss = torch.stack(all_trans_loss).mean()
    all_pose_loss = all_rot_loss + all_trans_loss
    loss = all_matching_loss + all_pose_loss
    metrics = {"Loss/Matching": all_matching_loss.item(), "Loss/Rotation": all_rot_loss.item(), "Loss/Translation": all_trans_loss.item(), "Loss/Pose": all_pose_loss.item(), "Loss/Total": loss.item()}

    if optimizer is not None:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

    pred_poses = pred_poses.detach()
    pred_summary = {"pred_pose_a2b": pred_poses.matrix(), "gt_pose_a2b": gt_pose.matrix()}

    metrics.update(compute_pose_error(pred_poses, gt_pose))
    # print("\n",metrics['PoseMetrics/Rotation Error (Deg)'], metrics['PoseMetrics/Translation Error (Deg)'],"\n") # [SANITY CHECK]
    if epe_masked.numel() > 0:
        metrics["FlowMetrics/Mean EPE"] = epe_masked.mean().item()
        for px in [5, 10, 20, 40]:
            metrics[f"FlowMetrics/F{px}"] = (epe_masked < px).float().mean().item() * 100
    return metrics, pred_summary

def main(rank, world_size, args, use_ddp):
    gconfigs = [next(iter(Path('gconfigs').rglob(g)), None) for g in (["megadepth.gin", "scannet.gin", "training_hyperparams.gin"] + args.gconfigs)]
    assert all(gconfigs)
    gin.parse_config_files_and_bindings(gconfigs, args.overrides)

    if args.name is None:
        job_name = wandb.util.generate_id()
    else:
        job_name = args.name

    wandb.init(name=job_name, project='multi_slam_backbone', mode=('disabled' if (args.d or rank > 0) else 'online'), anonymous="allow")

    if use_ddp:
        print(f"Using DDP [{rank=} {world_size=}]")
        setup_ddp(rank, world_size)

    train(job_name, rank=rank, world_size=world_size, use_ddp=use_ddp, model_state=args.load_ckpt, batch_size=args.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gconfigs', nargs='+', type=str, default=[])
    parser.add_argument('-o', '--overrides', nargs='+', type=str, default=[])
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('-n', '--name', type=str, default=None)
    parser.add_argument('--load_ckpt', type=Path, default=None)
    parser.add_argument('-d', action='store_true')
    parser.add_argument('--use_ddp', action='store_true')
    args = parser.parse_args()
    assert (args.load_ckpt is None) or args.load_ckpt.exists()

    smp, world_size = init_ddp()
    if args.use_ddp or (world_size > 1):
        spwn_ctx = mp.spawn(main, nprocs=world_size, args=(world_size, args, True), join=False)
        spwn_ctx.join()
    else:
        main(0, 1, args, False)
    print("Done!")
