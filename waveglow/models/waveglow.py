# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import copy
import numpy as np
import scipy.linalg

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from .waveglow_upsample import ConvInUpsampleNetwork


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a+input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def calc_loss(z, log_s_list, log_det_w_list, sigma=1.0):
    log_s_total = sum(map(torch.sum, log_s_list))
    log_det_w_total = sum(log_det_w_list)

    z_like = torch.sum(z * z) / (2 * sigma * sigma) / z.numel()
    coupling = -log_s_total / z.numel()
    invs = -log_det_w_total / z.numel()
    loss = z_like + coupling + invs
    return loss, z_like, coupling, invs


class LUInvertibleMM(torch.nn.Module):
    """ An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).

    Adopted from https://github.com/ikostrikov/pytorch-flows/blob/master/flows.py
    """

    def __init__(self, num_inputs):
        super(LUInvertibleMM, self).__init__()
        self.W = torch.Tensor(num_inputs, num_inputs)
        torch.nn.init.orthogonal_(self.W)

        # FIXME: refactor decomposition to a static function
        L_mask = torch.tril(torch.ones(self.W.size()), -1)
        self.L_mask = torch.nn.Parameter(L_mask, requires_grad=False)
        self.U_mask = torch.nn.Parameter(L_mask.t().clone(), requires_grad=False)

        P, L, U = scipy.linalg.lu(self.W.numpy())
        self.P = torch.nn.Parameter(torch.from_numpy(P), requires_grad=False)
        self.L = torch.nn.Parameter(torch.from_numpy(L))
        self.U = torch.nn.Parameter(torch.from_numpy(U))

        S = np.diag(U)
        sign_S = np.sign(S)
        log_S = np.log(abs(S))
        self.sign_S = torch.nn.Parameter(torch.from_numpy(sign_S), requires_grad=False)
        self.log_S = torch.nn.Parameter(torch.from_numpy(log_S))

        self.I = torch.nn.Parameter(torch.eye(self.L.size(0)), requires_grad=False)

    @staticmethod
    def w_from_plu(P, L, L_mask, I, U, U_mask, sign_S, log_S):
        LL = L * L_mask + I
        UU = U * U_mask + torch.diag(sign_S * torch.exp(log_S))
        W = P @ LL @ UU
        return W

    def forward(self, inputs, cond_inputs=None, reverse=False):
        batch_size, group_size, n_of_groups = inputs.size()

        if str(self.L_mask.device) != str(self.L.device):  # FIXME: why str?
            self.L_mask = self.L_mask.to(self.L.device)
            self.U_mask = self.U_mask.to(self.L.device)
            self.I = self.I.to(self.L.device)
            self.P = self.P.to(self.L.device)
            self.sign_S = self.sign_S.to(self.L.device)

        W = self.w_from_plu(self.P, self.L, self.L_mask, self.I, self.U, self.U_mask, self.sign_S, self.log_S)

        if not reverse:
            # forward
            out = inputs.permute(0, 2, 1) @ W
            return out.permute(0, 2, 1), batch_size * n_of_groups * self.log_S.sum()

        else:
            # inference
            out = inputs.permute(0, 2, 1) @ torch.inverse(W)
            return out.permute(0, 2, 1)


class Invertible1x1Conv(torch.nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """
    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
                                    bias=False)

        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]

        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:,0] = -1*W[:,0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z, reverse=False):
        # shape
        batch_size, group_size, n_of_groups = z.size()

        W = self.conv.weight.squeeze()

        if reverse:
            if not hasattr(self, 'W_inverse'):
                # Reverse computation
                W_inverse = W.inverse()
                W_inverse = Variable(W_inverse[..., None])
                if z.type() == 'torch.cuda.HalfTensor':
                    W_inverse = W_inverse.half()
                self.W_inverse = W_inverse
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            # Forward computation
            log_det_W = batch_size * n_of_groups * torch.logdet(W)
            z = self.conv(z)
            return z, log_det_W


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self._causal_padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self._causal_padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self._causal_padding != 0:
            return result[:, :, :-self._causal_padding]
        return result


class WN(torch.nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary difference
    from WaveNet is the convolutions need not be causal.  There is also no dilation
    size reset.  The dilation only doubles on each layer
    """
    def __init__(self, n_in_channels, n_mel_channels, n_layers, n_channels, 
                 kernel_size, ac_init=0, causal_layers=None):
        super(WN, self).__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(n_in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(n_channels, 2*n_in_channels, 1)
        if ac_init > 0:
            end.weight.data.normal_(std=ac_init)
        else:
            end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        cond_layer = torch.nn.Conv1d(n_mel_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
        
        self.causal_layers = []
        if causal_layers:
            self.causal_layers = causal_layers

        for i in range(n_layers):
            dilation = 2 ** i
            padding = int((kernel_size*dilation - dilation)/2)
            if i in self.causal_layers:
                in_layer = CausalConv1d(n_channels, 2 * n_channels, kernel_size, dilation=dilation)
            else:
                in_layer = torch.nn.Conv1d(n_channels, 2 * n_channels, kernel_size, dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, forward_input):
        audio, spect = forward_input
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])
        
        spect = self.cond_layer(spect)
        
        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels
            acts = fused_add_tanh_sigmoid_multiply(
                self.in_layers[i](audio),
                spect[:,spect_offset:spect_offset+2*self.n_channels,:],
                n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = res_skip_acts[:, :self.n_channels, :] + audio
                output = output + res_skip_acts[:, self.n_channels:, :]
            else:
                output = output + res_skip_acts
        return self.end(output)


class WaveGlow(torch.nn.Module):
    def __init__(
            self,
            n_mel_channels, n_flows, n_group, n_early_every, n_early_size,
            WN_config,
            upsample_multistage=False,
            decompose_convinv=False
    ):
        super(WaveGlow, self).__init__()

        self.config = {
            'n_mel_channels': n_mel_channels,
            'n_flows': n_flows,
            'n_group': n_group,
            'n_early_every': n_early_every,
            'n_early_size': n_early_size,
            'WN_config': WN_config,
            'upsample_multistage': upsample_multistage,
            "decompose_convinv": decompose_convinv
        }

        self.upsample_multistage = False
        if upsample_multistage:
            self.upsample = ConvInUpsampleNetwork(
                upsample_scales=[4, 4, 4, 4],
                cin_channels=n_mel_channels
            )
            self.upsample_multistage = True
        else:
            self.upsample = torch.nn.ConvTranspose1d(n_mel_channels,
                                                     n_mel_channels,
                                                     1024, stride=256)
        assert(n_group % 2 == 0)
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.WN = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()

        n_half = int(n_group/2)

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = n_group
        for k in range(n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size/2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
            if decompose_convinv:
                conv = LUInvertibleMM(n_remaining_channels)
            else:
                conv = Invertible1x1Conv(n_remaining_channels)

            wn = WN(n_half, n_mel_channels * n_group, **WN_config)

            self.convinv.append(conv)
            self.WN.append(wn)

        self.n_remaining_channels = n_remaining_channels  # Useful during inference

    def forward(self, forward_input):
        """
        forward_input[0] = mel_spectrogram:  batch x n_mel_channels x frames
        forward_input[1] = audio: batch x time
        """
        spect, audio = forward_input

        #  Upsample spectrogram to size of audio
        spect = self.upsample(spect)  # (B, M, F) -> (B, M, T)
        assert(spect.size(2) >= audio.size(1))
        if spect.size(2) > audio.size(1):
            spect = spect[:, :, :audio.size(1)]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)  # (B, M, T) -> (B, M, T//G, G) -> (B, T//G, M, G)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)  # (B, T//G, M, G) -> (B, T//G, M*G) -> (B, M*G, T//G)

        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)  # (B, T) -> (B, T//G, G) -> (B, G, T//G)
        output_audio = []
        log_s_list = []
        b_list = []
        log_det_W_list = []

        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                output_audio.append(audio[:, :self.n_early_size, :])
                audio = audio[:, self.n_early_size:, :]

            audio, log_det_W = self.convinv[k](audio)
            log_det_W_list.append(log_det_W)

            n_half = int(audio.size(1)/2)
            audio_0 = audio[:, :n_half, :] 
            audio_1 = audio[:, n_half:, :]

            wn_input = (audio_0, spect)
            output = self.WN[k](wn_input)
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = torch.exp(log_s)*audio_1 + b
            log_s_list.append(log_s)
            b_list.append(b)

            audio = torch.cat([audio_0, audio_1], 1)

        output_audio.append(audio)
        return torch.cat(output_audio, 1), log_s_list, log_det_W_list, b_list

    def infer(self, spect, sigma=1.0):
        spect = self.upsample(spect)

        # trim conv artifacts. maybe pad spec to kernel multiple
        if not self.upsample_multistage:
            time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
            spect = spect[:, :, :-time_cutoff]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)

        if spect.type() == 'torch.cuda.HalfTensor':
            audio = torch.cuda.HalfTensor(spect.size(0),
                                          self.n_remaining_channels,
                                          spect.size(2)).normal_()
        else:
            audio = torch.cuda.FloatTensor(spect.size(0),
                                           self.n_remaining_channels,
                                           spect.size(2)).normal_()

        audio = torch.autograd.Variable(sigma*audio)

        for k in reversed(range(self.n_flows)):
            n_half = int(audio.size(1)/2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            wn_input = (audio_0, spect)
            output = self.WN[k](wn_input)

            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b)/torch.exp(s)
            audio = torch.cat([audio_0, audio_1], 1)

            audio = self.convinv[k](audio, reverse=True)

            if k % self.n_early_every == 0 and k > 0:
                if spect.type() == 'torch.cuda.HalfTensor':
                    z = torch.cuda.HalfTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()
                else:
                    z = torch.cuda.FloatTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()
                audio = torch.cat((sigma*z, audio), 1)

        audio = audio.permute(0, 2, 1).contiguous().view(audio.size(0), -1).data
        return audio

    def remove_weightnorm(self):
        for WN in self.WN:
            WN.start = torch.nn.utils.remove_weight_norm(WN.start)
            WN.in_layers = remove(WN.in_layers)
            WN.cond_layers = torch.nn.utils.remove_weight_norm(WN.cond_layer)
            WN.res_skip_layers = remove(WN.res_skip_layers)
        
def remove(conv_list):
    new_conv_list = torch.nn.ModuleList()
    for old_conv in conv_list:
        old_conv = torch.nn.utils.remove_weight_norm(old_conv)
        new_conv_list.append(old_conv)
    return new_conv_list
