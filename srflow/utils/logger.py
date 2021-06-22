from torch.utils.tensorboard import SummaryWriter
import subprocess as sp
import torch
import os
import logging

from .pprofiler import profiler

log = logging.getLogger()


class Logger(SummaryWriter):
    def __init__(self, logdir, eval_interval=20):
        self.logdir = logdir
        self.eval_interval = eval_interval
        super(Logger, self).__init__(logdir)

    def step_to_path(self, step):
        return os.path.join(self.logdir, "{:07d}_snapshot.pth".format(step))

    def epoch_to_path(self, epoch):
        return os.path.join(self.logdir, "{:03d}_snapshot.pth".format(epoch))

    @staticmethod
    def load_last_checkpoint(logdir):
        epochs = [
            int(filename.replace('_snapshot.pth', ''))
            for filename in os.listdir(logdir)
            if '_snapshot.pth' in filename
        ]

        if len(epochs) == 0:
            return {
                'epoch': 0,
                'state_dict': None,
                'optimizer_state_dict': None,
                'scheduler_state_dict': None
            }

        return torch.load(os.path.join(logdir, "{:03d}_snapshot.pth".format(max(epochs))), map_location="cpu")

    @profiler("save_checkpoint")
    def save_checkpoint(self, model, optimizer, scheduler, epoch):
        torch.save({
            'epoch': epoch,
            'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, self.epoch_to_path(epoch))

        # remove unnecessary snapshots
        if epoch > self.eval_interval and (epoch - self.eval_interval) % 20 and os.path.exists(self.step_to_path(epoch - self.eval_interval)):
            os.remove(self.epoch_to_path(epoch - self.eval_interval))
