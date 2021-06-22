import sys
sys.path.append('tacotron2')
import torch
from tacotron2.layers import STFT


class StudentDenoiser(torch.nn.Module):
    """ Removes model bias from audio produced with waveglow """

    def __init__(self, student, teacher, filter_length=1024, n_overlap=4,
                 win_length=1024, mode='zeros', mel_shape=(1, 80, 88), 
                 dtype=torch.float16, device='cuda'):
        super(StudentDenoiser, self).__init__()
        self.stft = STFT(filter_length=filter_length,
                         hop_length=int(filter_length/n_overlap),
                         win_length=win_length).to(device)
        if mode == 'zeros':
            mel_input = torch.zeros(mel_shape, dtype=dtype, device=device)
        elif mode == 'normal':
            mel_input = torch.randn(mel_shape, dtype=dtype, device=device)
        else:
            raise Exception("Mode {} if not supported".format(mode))

        inputs = teacher.sample_inputs_for(mel_input)
        student_mel, student_input = inputs[0], torch.cat(inputs[1:], dim=1)
        
        with torch.no_grad():
            bias_audio = student(student_input, student_mel).float().permute(0, 2, 1).flatten(1)
            bias_spec, _ = self.stft.transform(bias_audio)

        self.register_buffer('bias_spec', bias_spec[:, :, 0][:, :, None])

    def forward(self, audio, strength=0.1):
        audio_spec, audio_angles = self.stft.transform(audio.cuda().float())
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
        return audio_denoised


class TeacherDenoiser(torch.nn.Module):
    """ Removes model bias from audio produced with waveglow """

    def __init__(self, teacher, filter_length=1024, n_overlap=4,
                 win_length=1024, mode='zeros', mel_shape=(1, 80, 88),
                 dtype=torch.float16, device='cuda'):
        super(TeacherDenoiser, self).__init__()
        self.stft = STFT(filter_length=filter_length,
                         hop_length=int(filter_length/n_overlap),
                         win_length=win_length).to(device)
        if mode == 'zeros':
            mel_input = torch.zeros(mel_shape, dtype=dtype, device=device)
        elif mode == 'normal':
            mel_input = torch.randn(mel_shape, dtype=dtype, device=device)
        else:
            raise Exception("Mode {} if not supported".format(mode))

        inputs = teacher.sample_inputs_for(mel_input, sigma=0)
        
        with torch.no_grad():
            bias_audio = teacher(*inputs).float()
            bias_spec, _ = self.stft.transform(bias_audio)

        self.register_buffer('bias_spec', bias_spec[:, :, 0][:, :, None])

    def forward(self, audio, strength=0.1):
        audio_spec, audio_angles = self.stft.transform(audio.cuda().float())
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
        return audio_denoised
