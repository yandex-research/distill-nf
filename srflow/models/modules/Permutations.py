import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.LU = LU_decomposed
        self.w_shape = w_shape
        self.inversed_weights = False
        self.bias = None
        
    def forward(self, input):
        """
        log-det = log|abs(|W|)| * pixels
        """
        if not self.inversed_weights:
            w_shape = self.w_shape
            dtype = self.weight.dtype
            self.weight[:] = torch.inverse(self.weight.double()).type(dtype).view(w_shape[0], w_shape[1])
            self.inversed_weights = True
        return F.conv2d(input, self.weight[:,:, None, None], bias=self.bias)


