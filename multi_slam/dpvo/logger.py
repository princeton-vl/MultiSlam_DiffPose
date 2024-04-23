import wandb
import os
import time
import sys

SUM_FREQ = 100

def log_time(name: str, since: float, step=None):
    time_delta = time.time() - since
    wandb.log({f"Timings/{name}": time_delta}, step=step)
    return time.time()

class Logger:

    def __init__(self, name, scheduler, tags, args):
        self.total_steps = 0
        self.running_loss = {}
        self.name = name
        self.scheduler = scheduler
        mode = "disabled" if (os.environ['SLURM_JOB_NAME'] == "interactive") else "online"
        config = {**vars(args), "job_id": int(os.environ['SLURM_JOB_ID'])}
        wandb.init(name=name, config=config, project='dpvo', mode=mode, entity='lahav', tags=tags)

    def _print_training_status(self):
        lr = self.scheduler.get_lr().pop()
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in self.running_loss.keys()]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, lr)
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        for key in self.running_loss:
            val = self.running_loss[key] / SUM_FREQ
            wandb.log({f"Stats/{key}": val}, step=self.total_steps)
            self.running_loss[key] = 0.0

    def push(self, metrics):

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

        self.total_steps += 1

    def write_dict(self, results):
        wandb.log(results, step=self.total_steps)

    def close(self):
        pass

