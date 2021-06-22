import torch
from torch import nn as nn

class _ActNorm(nn.Module):
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
        return NotImplemented

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

    def _center(self, input, reverse=False, offset=None):
        bias = self.bias

        if offset is not None:
            bias = bias + offset

        if not reverse:
            return input + bias
        else:
            return input - bias

    def _scale(self, input, reverse=False, offset=None):
        logs = self.logs

        if offset is not None:
            logs = logs + offset

        if not reverse:
            input = input * torch.exp(logs) # should have shape batchsize, n_channels, 1, 1
            # input = input * torch.exp(logs+logs_offset)
        else:
            input = input * torch.exp(-logs)
        return input

    def forward(self, input, offset_mask=None, reverse=False, logs_offset=None, bias_offset=None):
        if not self.inited:
            self.initialize_parameters(input)
        self._check_input_dim(input)

        if offset_mask is not None:
            logs_offset *= offset_mask
            bias_offset *= offset_mask
            
        # no need to permute dims as old version
        if not reverse:
            # center and scale
            input = self._center(input, reverse, bias_offset)
            input = self._scale(input, reverse, logs_offset)
        else:
            # scale and center
            input = self._scale(input, reverse, logs_offset)
            input = self._center(input, reverse, bias_offset)
        return input



class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1.):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 4
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()))


class MaskedActNorm2d(ActNorm2d):
    def __init__(self, num_features, scale=1.):
        super().__init__(num_features, scale)

    def forward(self, input, mask, logdet=None, reverse=False):

        assert mask.dtype == torch.bool
        output, logdet_out = super().forward(input, logdet, reverse)

        input[mask] = output[mask]
        logdet[mask] = logdet_out[mask]

        return input, logdet


class ForwardActNorm(nn.Module):
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
        return NotImplemented

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

    def _center(self, input, offset=None):
        bias = self.bias

        if offset is not None:
            bias = bias + offset
        return input + bias

    def _scale(self, input, offset=None):
        logs = self.logs

        if offset is not None:
            logs = logs + offset

        input = input * torch.exp(logs) # should have shape batchsize, n_channels, 1, 1    
        return input

    def forward(self, input, offset_mask=None, reverse=False, logs_offset=None, bias_offset=None):
        if not self.inited:
            self.initialize_parameters(input)
        self._check_input_dim(input)

        if offset_mask is not None:
            logs_offset *= offset_mask
            bias_offset *= offset_mask
            
        input = self._center(input, bias_offset)
        input = self._scale(input, logs_offset)
        return input