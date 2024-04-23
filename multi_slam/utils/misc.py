import os
import sys
import time
import traceback
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import git
import torch
import wandb


def commit_sha():
    lambda: git.Repo().head.object.hexsha[:6]

def clean_state_dict(state_dict):
    return OrderedDict([(k.replace('module.',''),v) for k,v in state_dict.items()])

def isbad(t):
    return torch.any(torch.isnan(t) | torch.isinf(t))

class GradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        return grad_x.clamp(min=-0.01, max=0.01)


class GradSafe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        return torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)

@contextmanager
def Parachute(payload):
    try:
        yield
    except:
        _, _, tb = sys.exc_info()
        l1 = traceback.extract_tb(tb).pop()
        location = f"{os.path.basename(l1.filename)}:{l1.lineno} in {l1.name} | {l1.line}"
        filename = f"tmp/{uuid4().hex[:6]}.pt"
        Path("tmp").mkdir(exist_ok=True)
        torch.save(payload, filename)
        print(f"Saved {Path(filename).resolve()} [{type(payload)}] | {location}")
        raise

@contextmanager
def WBTimer(title, step=None):
    start = time.time()
    yield
    elapsed = time.time() - start
    if wandb.run is None:
        pass
    elif step is None: # commit=False is necessary
        wandb.log({f"Timing/{title} (seconds)": elapsed}, commit=False)
    else:
        wandb.log({f"Timing/{title} (seconds)": elapsed}, step=step)