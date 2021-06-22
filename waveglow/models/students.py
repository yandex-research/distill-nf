"""
   Student models for various design options
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.waveglow import WN


class FlowStudent(torch.nn.Module):
    def __init__(self, in_channels, mel_channels, hid_channels, *,
                 wavenet_channels=None, n_wavenets=6, wavenet_layers=8, **kwargs):
        super(FlowStudent, self).__init__()

        wavenet_channels = wavenet_channels or hid_channels
        self.config = dict(
            in_channels=in_channels, mel_channels=mel_channels, hid_channels=hid_channels,
            wavenet_channels=wavenet_channels, n_wavenets=n_wavenets, wavenet_layers=wavenet_layers, **kwargs)

        assert(in_channels % 2 == 0)
        n_half = int(in_channels / 2)
        self.WN = torch.nn.ModuleList([
            WN(n_half, mel_channels, wavenet_layers, wavenet_channels, **kwargs)
            for _ in range(n_wavenets)
        ])
        self.convinv = torch.nn.ModuleList([
            nn.Conv1d(in_channels, in_channels, kernel_size=1) 
            for _ in range(n_wavenets)
        ])

    def forward(self, audio, spect):
        for k in range(len(self.WN)):
            n_half = int(audio.size(1)/2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            wn_input = (audio_0, spect)
            output = self.WN[k](wn_input)

            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b)/torch.exp(s)
            audio = torch.cat([audio_0, audio_1], 1)

            audio = self.convinv[k](audio)
        return audio


class WideFlowStudent(FlowStudent):
    def __init__(self, in_channels, mel_channels, hid_channels, **kwargs):
        super(WideFlowStudent, self).__init__(hid_channels, mel_channels, hid_channels, **kwargs)
        self.in_proj = nn.Conv1d(in_channels, hid_channels, kernel_size=1)
        self.out_proj = nn.Conv1d(hid_channels, in_channels, kernel_size=1)

    def forward(self, audio, spect):
        audio = self.in_proj(audio)
        audio = super().forward(audio, spect)
        return self.out_proj(audio)


class AffineStudent(nn.Module):
    def __init__(self, in_channels, mel_channels, hid_channels, *,
                 wavenet_channels=None, n_wavenets=6, wavenet_layers=8, **kwargs):
        super(AffineStudent, self).__init__()

        wavenet_channels = wavenet_channels or hid_channels
        self.config = dict(
            in_channels=in_channels, mel_channels=mel_channels, hid_channels=hid_channels,
            wavenet_channels=wavenet_channels, n_wavenets=n_wavenets, wavenet_layers=wavenet_layers, **kwargs)

        self.WN = torch.nn.ModuleList([
            WN(in_channels, mel_channels, wavenet_layers, wavenet_channels, **kwargs)
            for _ in range(n_wavenets)
        ])
        self.convinv = torch.nn.ModuleList([
            nn.Conv1d(in_channels, in_channels, kernel_size=1) 
            for _ in range(n_wavenets)
        ])

    def forward(self, audio, spect):
        for k in range(len(self.WN)):
            wn_input = (audio, spect)
            output = self.WN[k](wn_input)

            s = output[:, audio.size(1):, :]
            b = output[:, :audio.size(1), :]
            audio = (audio - b) / torch.exp(s)
            audio = self.convinv[k](audio)
        return audio


class WaveNetStudent(nn.Module):
    def __init__(self, in_channels, mel_channels, hid_channels, *,
                 wavenet_channels=None, n_wavenets=6, wavenet_layers=8, **kwargs):
        """ Just a bunch of wavenets """
        super().__init__()
        wavenet_channels = wavenet_channels or hid_channels
        self.config = dict(
            in_channels=in_channels, mel_channels=mel_channels, hid_channels=hid_channels,
            wavenet_channels=wavenet_channels, n_wavenets=n_wavenets, wavenet_layers=wavenet_layers, **kwargs)

        self.in_proj = nn.Conv1d(in_channels, hid_channels, kernel_size=1)
        self.out_proj = nn.Conv1d(hid_channels, in_channels, kernel_size=1)

        self.wavenets = nn.ModuleList()
        for _ in range(n_wavenets):
            wn = WN(hid_channels, mel_channels, wavenet_layers, wavenet_channels, **kwargs)
            # Adjust the output channel dimensions
            wn.end = nn.Conv1d(wn.end.in_channels, hid_channels, kernel_size=1)
            self.wavenets.append(wn)

    def forward(self, audio, spect):
        audio = self.in_proj(audio)
        for wavenet in self.wavenets:
            wn_input = (audio, spect)
            audio = wavenet(wn_input)
        return self.out_proj(audio)
