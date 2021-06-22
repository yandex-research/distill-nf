from torch.utils.tensorboard import SummaryWriter
import subprocess as sp
import torch
import os
import logging

from .pprofiler import profiler
from ya_tools import nirvana_tools

log = logging.getLogger()


class Logger(SummaryWriter):
    def __init__(self, logdir, eval_interval=1000, nirvana_state=None):
        self.logdir = logdir
        self.eval_interval = eval_interval
        self.nirvana_state = nirvana_state

        if nirvana_state and os.path.isfile(nirvana_state):
            log.info("restore logdir from {}".format(nirvana_state))
            sp.check_call("mkdir -p {}".format(logdir), shell=True)
            sp.check_call("tar xf {} --strip 1 -C {}".format(nirvana_state, logdir), shell=True)

        super(Logger, self).__init__(logdir)

    @profiler("log_training")
    def log_training(self, model, optimizer, loss, loss_parts, grad_norm, learning_rate, step, **kwargs):
        log.info("{:6d}: loss {:.4f}  grad_l2 {:.3f}".format(step, loss, grad_norm))

        self.add_scalar("loss/train", loss, step)

        z_like, coupling, invs = loss_parts
        self.add_scalar("loss/train/z_like", z_like, step)
        self.add_scalar("loss/train/coupling", coupling, step)
        self.add_scalar("loss/train/convinv", invs, step)

        # misc
        self.add_scalar("misc/grad_norm", grad_norm, step)
        self.add_scalar("misc/learning_rate", learning_rate, step)

        # "invertibility" by layer
        log_det_w_list = kwargs.get("log_det_w_list", [])
        for i, v in enumerate(log_det_w_list):
            self.add_scalar("convinv/logdet_W_{:02d}".format(i), v.item(), step)

    @profiler("log_validation")
    def log_validation(self, model, step, eval_data):
        (loss_gt, z_like_gt, coupling_gt, convinv_gt, per_speaker) = eval_data
        log.info("validation loss at {}: {:.4f}".format(step, loss_gt))

        self.add_scalar("loss/validation", loss_gt, step)
        self.add_scalar("loss/validation/z_like", z_like_gt, step)
        self.add_scalar("loss/validation/coupling", coupling_gt, step)
        self.add_scalar("loss/validation/convinv", convinv_gt, step)
        
        for speaker, loss_gt in per_speaker:
            self.add_scalar("loss/validation/{}".format(speaker), loss_gt, step)

        for writer in self.all_writers.values():
            writer.flush()

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

        nirvana_tools.copy_out_to_snapshot(out=self.logdir)

    def _save_state_dict(self, state, rank, step, suffix=""):
        if len(suffix) > 0 and not suffix.startswith("_"):
            suffix = "_" + suffix
        save_path = os.path.join(
            self.logdir,
            "{:02d}_{:07d}{}.pth".format(rank, step, suffix)
        )
        torch.save(state, save_path)
        # race safety
        if rank == 0:
            nirvana_tools.copy_out_to_snapshot(out=self.logdir)

    def save_spike_state(self, state, rank, step):
        self._save_state_dict(state, rank, step, "spike")

    def save_train_state(self, train_state, rank, step):
        self._save_state_dict(train_state, rank, step, "train_state")
