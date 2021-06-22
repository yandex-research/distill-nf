"""
   Distillation loss for the student training
"""
from typing import Optional

import torch
import torch.nn as nn

import models.defaults as defaults
from utils.stft_losses import MultiResolutionSTFTLoss


class DistillationLoss(nn.Module):
    def __init__(self, student, teacher, *, infer_teacher: Optional[callable] = None,
                 teacher_dtype: torch.dtype, stft_loss_coeff: float,
                 # V-- below are spectrogram params
                 hop_length: int, win_length: int, num_mels: int, n_fft: Optional[int] = None,
                 sample_rate=defaults.SAMPLING_RATE, min_frequency=defaults.MEL_FMIN, max_frequency=defaults.MEL_FMAX,
                 eps=1e-5):
        """
        A pytorch module that computes a distillation loss.
        :type student: [Flow, WideFlow, Affine, WaveNet]Student
        :type teacher: WaveGlowTeacher
        :param infer_teacher: optionally provide a more efficient version of teacher.infer
        :param teacher_dtype: dtype of inputs and outputs to teacher,
        :param stft_loss_coeff: multiplicative coefficient for stft loss. loss = MAE + stft_loss_coeff * STFT_LOSS
        """
        super().__init__()
        self.student, self.teacher = student, teacher
        self.infer_teacher = infer_teacher or teacher.infer
        self.teacher_dtype, self.stft_loss_coeff = teacher_dtype, stft_loss_coeff
        
        self.n_fft = n_fft if n_fft is not None else win_length
        self.hop_length, self.win_length, self.num_mels = hop_length, win_length, num_mels
        self.sample_rate, self.min_frequency, self.max_frequency = sample_rate, min_frequency, max_frequency
        self.eps = eps

        self.multires_stft_loss = MultiResolutionSTFTLoss()

    def forward(self, mel: torch.Tensor, sigma=1.0):
        """
        :param mel: a batch of mel-spectrograms [batch_size, channels, length], already with matching device and dtype
        :returns: three scalar loss values: (total loss, MAE component, STFT loss component)
        """
        with torch.no_grad():
            upsampled_mels, *wg_noise = self.teacher.sample_inputs_for(mel.to(self.teacher_dtype), sigma=sigma)
            reference = self.infer_teacher(upsampled_mels, *wg_noise).to(mel.dtype)
            student_input = torch.cat(wg_noise, dim=1).to(mel.dtype)

        student_prediction = self.student(student_input, upsampled_mels.to(mel.dtype))
        student_prediction = student_prediction.permute(0, 2, 1).flatten(1)
        loss_mae = abs(student_prediction - reference).mean()
        loss_sc, loss_mag = self.multires_stft_loss(student_prediction, reference)
        loss_stft = loss_sc + loss_mag
        loss_full = loss_mae + loss_stft * self.stft_loss_coeff if self.stft_loss_coeff else loss_mae
        return loss_full, loss_mae, loss_stft
