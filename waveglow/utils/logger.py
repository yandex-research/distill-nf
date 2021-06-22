import os
import torch
from .pprofiler import profiler
from torch.utils.tensorboard import SummaryWriter

import logging
log = logging.getLogger()


class Logger(SummaryWriter):
    def __init__(self, logdir, eval_interval=1000):
        self.logdir = logdir
        self.eval_interval = eval_interval
        super(Logger, self).__init__(logdir)

    def step_to_path(self, step):
        return os.path.join(self.logdir, "{:07d}_snapshot.pth".format(step))

    @staticmethod
    def load_last_checkpoint(logdir):
        steps = [
            int(filename.replace('_snapshot.pth', ''))
            for filename in os.listdir(logdir)
            if '_snapshot.pth' in filename
        ]

        if len(steps) == 0:
            return {
                'step': 0,
                'state_dict': None,
                'config': None,
                'optimizer_state_dict': None
            }

        return torch.load(os.path.join(logdir, "{:07d}_snapshot.pth".format(max(steps))), map_location="cpu")

    @profiler("save_checkpoint")
    def save_checkpoint(self, model, optimizer, step):
        torch.save({
            'step': step,
            'state_dict': model.state_dict(),
            'config': model.config,
            'optimizer_state_dict': optimizer.state_dict()
        }, self.step_to_path(step))

        # remove unnecessary snapshots
        if step > self.eval_interval and (step - self.eval_interval) % 20000 and os.path.exists(self.step_to_path(step - self.eval_interval)):
            os.remove(self.step_to_path(step - self.eval_interval))
