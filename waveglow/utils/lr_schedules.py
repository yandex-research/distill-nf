# coding: U8
# MultiCycleSchedule taken from https://github.com/bckenstler/CLR
# `exp_range` changed to match original paper idea: https://arxiv.org/abs/1506.01186
import logging
from collections import OrderedDict
import numpy as np

log = logging.getLogger()


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        if 'lr' in param_group:
            return param_group['lr']
    raise ValueError("Could not infer learning rate from optimizer {}".format(optimizer))


class OptimizerScheduleWrapper:
    def __init__(self, optimizer, **kwargs):
        self.opt = optimizer
        self.learning_rate_opts = kwargs

        self.step_count = 0
        self.current_lr = 0

    def update_learning_rate(self, t):
        lr = self.lr(t)
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr
        return lr

    def step(self, **kwargs):
        self.current_lr = self.update_learning_rate(self.step_count)
        res = self.opt.step(**kwargs)
        self.step_count += 1
        if self.step_count % 100 == 0:
            log.info("Step {} \t current learning rate: {}".format(self.step_count, self.current_lr))
        return res

    def state_dict(self, **kwargs):
        return OrderedDict([
            ('optimizer_state_dict', self.opt.state_dict(**kwargs)),
            ('learning_rate_opts', self.learning_rate_opts),
            ('step_count', self.step_count),
        ])

    def load_state_dict(self, state_dict, **kwargs):
        self.learning_rate_opts = state_dict['learning_rate_opts']
        for k, v in self.learning_rate_opts.items():
            setattr(self, k, v)
        self.step_count = state_dict['step_count']
        return self.opt.load_state_dict(state_dict['optimizer_state_dict'], **kwargs)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.opt, attr)

    def lr(self, t):
        raise NotImplementedError()


class OneCycleSchedule(OptimizerScheduleWrapper):
    def __init__(
        self, optimizer, 
        learning_rate_base=1e-3, 
        warmup_steps=10000, 
        decay_rate=0.2, 
        learning_rate_min=1e-5
        ):
        super(OneCycleSchedule, self).__init__(
            optimizer,
            learning_rate_base=learning_rate_base,
            warmup_steps=warmup_steps,
            decay_rate=decay_rate,
            learning_rate_min=learning_rate_min
        )
        self.learning_rate_base = learning_rate_base
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
        self.learning_rate_min = learning_rate_min
        
    def lr(self, t):
        lr = self.learning_rate_base * np.minimum(
            (t + 1.0) / self.warmup_steps,
            np.exp(self.decay_rate * ((self.warmup_steps - t - 1.0) / self.warmup_steps)),
        )
        lr = np.maximum(lr, self.learning_rate_min)
        return lr


class MultiCycleSchedule(OptimizerScheduleWrapper):
    def __init__(
        self, optimizer,
        mode="tri", step_size=20000, base_lr=1e-5, max_lr=1e-3, gamma=0.9
    ):
        super(MultiCycleSchedule, self).__init__(
            optimizer,
            mode=mode, 
            step_size=step_size, 
            base_lr=base_lr, 
            max_lr=max_lr,
            gamma=gamma
        )
        self.mode=mode
        self.step_size=step_size
        self.base_lr=base_lr
        self.max_lr=max_lr
        self.gamma=gamma

    def lr(self, t):
        cycle = np.floor(1 + t / (2 * self.step_size))
        x = np.abs(t / self.step_size - 2 * cycle + 1)
        if self.mode == "tri":
            scaling = 1.0
        elif self.mode == "tri2":
            scaling = 1.0 / float(2**(cycle - 1))
        elif self.mode == "exp_range":
            scaling = self.gamma ** (cycle - 1)
        else:
            raise ValueError("Unknown mode {}".format(mode))
        lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * scaling
        return lr