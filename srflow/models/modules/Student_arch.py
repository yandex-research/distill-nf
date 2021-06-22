import torch
from torch import nn as nn
import torch.nn.functional as F

from models.modules import flow
from utils.util import opt_get
from copy import deepcopy

from . import thops
from models.modules.RRDBNet_arch import RRDBNet, RRDB, SmallRRDB3C


class StudentX4(nn.Module):
    def __init__(self, K, rrdb_block=RRDB, flow_block=SmallRRDB3C,
                 pretrained_rrdb=None, freeze_rrdb=False, opt=None):
        super().__init__()
        self.L = opt_get(opt, ['network_G', 'flow', 'L'])
        self.K = [K] * (self.L + 1)

        # Assign pretrained RRDB module for LR preprocessing
        if pretrained_rrdb is not None:
            if freeze_rrdb:
                self.RRDB = pretrained_rrdb.eval()
                for param in self.RRDB.parameters():
                    param.requires_grad = False
            else:
                self.RRDB = deepcopy(pretrained_rrdb)
                self.RRDB.train()
        else:
            in_nc, out_nc = opt['network_G']['in_nc'], opt['network_G']['out_nc']
            nb, nf = opt['network_G']['nb'], opt['network_G']['nf']
            self.RRDB = RRDBNet(in_nc, out_nc, nf, nb, gc=32, scale=opt['scale'], rrdb_block=rrdb_block, opt=opt)

        self.opt = opt

        if opt['scale'] == 8:
            self.levelToName = {
                0: 'fea_up8',
                1: 'fea_up4',
                2: 'fea_up2',
                3: 'fea_up1',
                4: 'fea_up0'
            }
        elif opt['scale'] == 4:
            self.levelToName = {
                0: 'fea_up4',
                1: 'fea_up2',
                2: 'fea_up1',
                3: 'fea_up0',
                4: 'fea_up-1'
            }
        self._make_layers(flow_block)

    
    def _make_layers(self, flow_block):
        self.C = self.opt['flowUpsamplerNet']['C']
        H = int(self.opt['GT_size'] // self.opt['flowUpsamplerNet']['scaleH'])
        W = int(self.opt['GT_size'] // self.opt['flowUpsamplerNet']['scaleW'])
        n_rrdb = self.get_n_rrdb_channels(self.opt, opt_get)

        self.layers = nn.ModuleList()
        # Upsampler
        for level in range(self.L, 0, -1):
            # 1. Split
            split = Concat2d(n_rrdb + self.C, self.C)
            self.layers.append(split)
            if level < self.L - 1:
                self.C = 2 * self.C
        
            # 2. K FlowStep
            self.layers.extend([flow_block(self.C) for _ in range(self.K[level])])

            # 3. Transition Block
            self.layers.append(nn.Sequential(
                nn.Conv2d(self.C, self.C, 1, bias=False),
                StudentActNorm2d(self.C)
            ))

            # 4. Unsqueeze
            self.C, H, W = self.C // 4, H * 2, W * 2
            self.layers.append(flow.UnsqueezeLayer(factor=2))

        self.last_conv = nn.Conv2d(self.C, 3, 1)
        self.H = H
        self.W = W
        
    def forward(self, lr, rrdbResults=None, z=None, epses=None):
        if rrdbResults is None:
            rrdbResults = self.rrdbPreprocessing(lr)
        epses_copy = [eps for eps in epses] if isinstance(epses, list) else epses
        fl_fea = epses_copy.pop() if isinstance(epses_copy, list) else z

        level = self.L
        for layer in self.layers:
            if isinstance(layer, Concat2d):
                lr_enc = rrdbResults[self.levelToName[level]]
                eps = epses_copy.pop() if level < self.L - 1 else None
                fl_fea = layer(fl_fea, lr_enc, eps)
            elif isinstance(layer, flow.UnsqueezeLayer):
                fl_fea = layer(fl_fea)
                level -= 1
            else:
                fl_fea = layer(fl_fea)
        sr = self.last_conv(fl_fea)
        return sr

    def get_n_rrdb_channels(self, opt, opt_get):
        blocks = opt_get(opt, ['network_G', 'flow', 'stackRRDB', 'blocks'])
        n_rrdb = 64 if blocks is None else (len(blocks) + 1) * 64
        return n_rrdb

    def rrdbPreprocessing(self, lr):
        rrdbResults = self.RRDB(lr, get_steps=True)
        block_idxs = opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'blocks']) or []
        if len(block_idxs) > 0:
            concat = torch.cat([rrdbResults["block_{}".format(idx)] for idx in block_idxs], dim=1)

            if opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'concat']) or False:
                keys = ['last_lr_fea', 'fea_up1', 'fea_up2', 'fea_up4']
                if 'fea_up0' in rrdbResults.keys():
                    keys.append('fea_up0')
                if 'fea_up-1' in rrdbResults.keys():
                    keys.append('fea_up-1')
                if self.opt['scale'] >= 8:
                    keys.append('fea_up8')
                if self.opt['scale'] == 16:
                    keys.append('fea_up16')
                for k in keys:
                    h = rrdbResults[k].shape[2]
                    w = rrdbResults[k].shape[3]
                    rrdbResults[k] = torch.cat([rrdbResults[k], F.interpolate(concat, (h, w))], dim=1)
        return rrdbResults


class StudentX8(StudentX4):
    def _make_layers(self, flow_block):
        self.C = self.opt['flowUpsamplerNet']['C']
        H = int(self.opt['GT_size'] // self.opt['flowUpsamplerNet']['scaleH'])
        W = int(self.opt['GT_size'] // self.opt['flowUpsamplerNet']['scaleW'])
        n_rrdb = self.get_n_rrdb_channels(self.opt, opt_get)

        self.layers = nn.ModuleList()
        self.epsC = self.C
        # Upsampler
        for level in range(self.L, 0, -1):
            # 1. Split
            shift = 3 if self.L == 4 else 2  
            if level == 1:
                split = Concat2d(n_rrdb + self.C, self.C + shift)
                self.C = self.C + self.epsC + shift
                self.epsC = 2 * self.epsC
            elif level < self.L - 1:
                split = Concat2d(n_rrdb + self.C, self.C)
                self.C = self.C + self.epsC
                self.epsC = 2 * self.epsC
            else:
                split = Concat2d(n_rrdb + self.C, 2 * self.C)
                self.C = 2 * self.C

            self.epsC = self.epsC // 4
        
            self.layers.append(split)
        
            # 2. K FlowStep
            self.layers.extend([flow_block(self.C) for _ in range(self.K[level])])

            # 3. Transition Block
            self.layers.append(nn.Sequential(
                nn.Conv2d(self.C, self.C, 1, bias=False),
                StudentActNorm2d(self.C)
            ))

            # 4. Unsqueeze
            self.C, H, W = self.C // 4, H * 2, W * 2
            self.layers.append(flow.UnsqueezeLayer(factor=2))

        self.last_conv = nn.Conv2d(self.C, 3, 1)
        self.H = H
        self.W = W
    

class Concat2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=(1, 1), bias=True)

    def forward(self, input, lr_enc, eps=None):
        z = torch.cat([input, lr_enc], dim=1)
        z = self.conv(z)
        if eps is not None:
            z = torch.cat([z, eps], dim=1)
        return z


class StudentActNorm2d(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.

    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size)))
        self.num_features = num_features
        self.scale = float(scale)
        self.inited = False

    def _check_input_dim(self, input):
        assert len(input.size()) == 4
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()))

    def initialize_parameters(self, input):
        self._check_input_dim(input)
        if not self.training:
            return
        if (self.bias != 0).any():
            self.inited = True
            return
        assert input.device == self.bias.device, (input.device, self.bias.device)
        with torch.no_grad():
            bias = thops.mean(input.clone(), dim=[0, 2, 3], keepdim=True) * -1.0
            vars = thops.mean((input.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.inited = True

    def forward(self, x):
        if not self.inited:
            self.initialize_parameters(x)
        self._check_input_dim(x)
        # scale and center
        x = x * torch.exp(-self.logs) - self.bias
        return x
