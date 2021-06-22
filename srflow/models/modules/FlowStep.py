import torch
from torch import nn as nn

import models.modules
import models.modules.Permutations
from models.modules import flow, thops, FlowAffineCouplingsAblation
from utils.util import opt_get


def getConditional(rrdbResults, position):
    img_ft = rrdbResults if isinstance(rrdbResults, torch.Tensor) else rrdbResults[position]
    return img_ft


class FlowStep(nn.Module):
    FlowPermutation = {
        "reverse": lambda obj, z: obj.reverse(z, True),
        "shuffle": lambda obj, z: obj.shuffle(z, True),
        "invconv": lambda obj, z: obj.invconv(z),
        "squeeze_invconv": lambda obj, z: obj.invconv(z),
        "resqueeze_invconv_alternating_2_3": lambda obj, z: obj.invconv(z),
        "resqueeze_invconv_3": lambda obj, z: obj.invconv(z),
        "InvertibleConv1x1GridAlign": lambda obj, z: obj.invconv(z),
        "InvertibleConv1x1SubblocksShuf": lambda obj, z: obj.invconv(z),
        "InvertibleConv1x1GridAlignIndepBorder": lambda obj, z: obj.invconv(z),
        "InvertibleConv1x1GridAlignIndepBorder4": lambda obj, z: obj.invconv(z),
    }

    def __init__(self, in_channels, hidden_channels,
                 actnorm_scale=1.0, flow_permutation="invconv", flow_coupling="additive",
                 LU_decomposed=False, opt=None, image_injector=None, idx=None, acOpt=None, normOpt=None, in_shape=None,
                 position=None):
        # check configures
        assert flow_permutation in FlowStep.FlowPermutation, \
            "float_permutation should be in `{}`".format(
                FlowStep.FlowPermutation.keys())
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        self.image_injector = image_injector

        self.norm_type = normOpt['type'] if normOpt else 'ActNorm2d'
        self.position = normOpt['position'] if normOpt else None

        self.in_shape = in_shape
        self.position = position
        self.acOpt = acOpt

        # 1. actnorm
        self.actnorm = models.modules.FlowActNorms.ActNorm2d(in_channels, actnorm_scale)
        self.fused_actnorm = False
        
        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = models.modules.Permutations.InvertibleConv1x1(
                in_channels, LU_decomposed=LU_decomposed)

        # 3. coupling
        if flow_coupling == "CondAffineSeparatedAndCond":
            self.affine = models.modules.FlowAffineCouplingsAblation.CondAffineSeparatedAndCond(in_channels=in_channels,
                                                                                                opt=opt)
        elif flow_coupling == "noCoupling":
            pass
        else:
            raise RuntimeError("coupling not Found:", flow_coupling)

    def forward(self, z, rrdbResults=None):
        need_features = self.affine_need_features()
        # 1.coupling
        if need_features or self.flow_coupling in ["condAffine", "condFtAffine", "condNormAffine"]:
            img_ft = getConditional(rrdbResults, self.position)
            z = self.affine(input=z, ft=img_ft)

        # 2. permute
        z = FlowStep.FlowPermutation[self.flow_permutation](self, z)
        if not self.fused_actnorm:    
            z = self.actnorm(z, reverse=True)

        if not self.fused_actnorm and self.flow_permutation == 'invconv' and self.invconv.inversed_weights:
            self.fuse_actnorm()
        return z

    def fuse_actnorm(self):
        logs = self.actnorm.logs.squeeze()
        bias = self.actnorm.bias.squeeze()
        self.invconv.weight.data = self.invconv.weight.data * torch.exp(-logs)[:, None]
        self.invconv.bias = nn.Parameter(-bias, requires_grad=self.invconv.weight.requires_grad)
        self.fused_actnorm = True

    def affine_need_features(self):
        need_features = False
        try:
            need_features = self.affine.need_features
        except:
            pass
        return need_features
