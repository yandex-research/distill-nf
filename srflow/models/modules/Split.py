import torch
from torch import nn as nn

from models.modules import thops
from models.modules.flow import Conv2dZeros, GaussianDiag


class Split2d(nn.Module):
    def __init__(self, num_channels, logs_eps=0, cond_channels=0, position=None, consume_ratio=0.5, opt=None):
        super().__init__()

        self.num_channels_consume = int(round(num_channels * consume_ratio))
        self.num_channels_pass = num_channels - self.num_channels_consume
        self.conv = Conv2dZeros(in_channels=self.num_channels_pass + cond_channels,
                                out_channels=self.num_channels_consume * 2)
        self.logs_eps = logs_eps
        self.position = position
        self.opt = opt

    def split2d_prior(self, z, ft):
        if ft is not None:
            z = torch.cat([z, ft], dim=1)
        h = self.conv(z)
        return thops.split_feature(h, "cross")

    def exp_eps(self, logs):
        return torch.exp(logs) + self.logs_eps

    def forward(self, input, eps_std=None, eps=None, ft=None):
        z1 = input
        mean, logs = self.split2d_prior(z1, ft)
        
        if eps is None:
            #print("WARNING: eps is None, generating eps untested functionality!")
            eps = GaussianDiag.sample_eps(mean.shape, eps_std).type(mean.dtype)
            # print("SAMPLE EPS")
        eps = eps.to(mean.device)
        z2 = mean + self.exp_eps(logs) * eps
        z = thops.cat_feature(z1, z2)
        return z

    def split_ratio(self, input):
        z1, z2 = input[:, :self.num_channels_pass, ...], input[:, self.num_channels_pass:, ...]
        return z1, z2