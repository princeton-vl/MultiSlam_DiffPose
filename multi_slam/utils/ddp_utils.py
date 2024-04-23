import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def calc_num_workers():
    try:
        world_size = dist.get_world_size()
    except:
        world_size = 1
    return len(os.sched_getaffinity(0)) // world_size

def setup_ddp(rank, world_size):
    dist.init_process_group(
        backend='nccl',
           init_method='env://',
        world_size=world_size,
        rank=rank)

    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.cuda.set_device(rank)

def init_ddp():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(12356 + np.random.randint(100))
    world_size = torch.cuda.device_count()
    assert world_size > 0, "You need a GPU!"
    smp = mp.get_context('spawn')
    return smp, world_size