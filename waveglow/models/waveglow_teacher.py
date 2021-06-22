"""
   WaveGlow teacher wrapper
"""

from functools import lru_cache
import numpy as np
import torch

from models.waveglow import WaveGlow


class WaveGlowTeacher(WaveGlow):
    """ A WaveGlow model optimized for use as a teacher in distillation """

    @classmethod
    def load(cls, path, fp16=True, train=False, device='cuda'):
        ckpt = torch.load(path)
        del ckpt['config']['n_speakers']
        del ckpt['config']['speaker_embedding_dim']
        
        waveglow = cls(**ckpt['config'])
        waveglow.remove_weightnorm()
        waveglow.load_state_dict(ckpt['state_dict'])

        if fp16:
            print("Cast WaveGlow to fp16")
            for convinv in waveglow.convinv:
                # precompute W_inverse
                # it should be used by WG automatically
                with torch.no_grad():
                    W = convinv.conv.weight.data.squeeze()
                    W_inverse = W.inverse()
                convinv.register_buffer("W_inverse", W_inverse[..., None])

            waveglow = waveglow.half()
        
        return waveglow.to(device).train(train)
    
    def sample_inputs_for(self, spect, sigma=1.0):
        """ upsample and generate noise: the non-distilled part of waveglow """
        assert self.n_early_every > 0 and self.n_flows > 0
        spect = self.upsample(spect)

        # trim conv artifacts. maybe pad spec to kernel multiple
        if not self.upsample_multistage:
            time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
            spect = spect[:, :, :-time_cutoff]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)
        
        num_noise_vectors = max(1, (self.n_flows - 1) // self.n_early_every)
        noise_audio = sigma * torch.randn(spect.shape[0], self.n_remaining_channels, spect.shape[2],
                                  device=spect.device, dtype=spect.dtype)
        noise_vectors = [sigma * torch.randn(spect.shape[0], self.n_early_size, spect.shape[2],
                                     device=spect.device, dtype=spect.dtype)
                         for _ in range(num_noise_vectors)]
        return (spect, noise_audio, *noise_vectors)

    def forward(self, spect, noise_audio, *noise_vectors):
        """ A deterministic version of waveglow.infer; use compute_inputs_for(mel_spect) for inputs """
        audio = noise_audio
        noise_index = 0

        for k in reversed(range(self.n_flows)):
            n_half = audio.size(1) // 2
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            wn_input = (audio_0, spect)
            output = self.WN[k](wn_input)

            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b) / torch.exp(s)
            audio = torch.cat([audio_0, audio_1], 1)

            audio = self.convinv[k](audio, reverse=True)

            if k % self.n_early_every == 0 and k > 0:
                z = noise_vectors[noise_index]
                noise_index += 1
                audio = torch.cat((z, audio), 1)

        assert noise_index == len(noise_vectors), f"Used {noise_index} noise vectors, but got {len(noise_vectors)}"
        audio = audio.permute(0, 2, 1).contiguous().view(audio.size(0), -1)
        
        return audio

    def flow_forward(self, spect, audio):
        #  Upsample spectrogram to size of audio
        spect = self.upsample(spect)  # (B, M, F) -> (B, M, T)
        assert(spect.size(2) >= audio.size(1))
        if spect.size(2) > audio.size(1):
            spect = spect[:, :, :audio.size(1)]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)  # (B, M, T) -> (B, M, T//G, G) -> (B, T//G, M, G)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)  # (B, T//G, M, G) -> (B, T//G, M*G) -> (B, M*G, T//G)

        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)  # (B, T) -> (B, T//G, G) -> (B, G, T//G)
        output_audio = []

        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                output_audio.append(audio[:, :self.n_early_size, :])
                audio = audio[:, self.n_early_size:, :]

            audio, _ = self.convinv[k](audio)

            n_half = int(audio.size(1)/2)
            audio_0 = audio[:, :n_half, :] 
            audio_1 = audio[:, n_half:, :]

            wn_input = (audio_0, spect)
            output = self.WN[k](wn_input)
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = torch.exp(log_s)*audio_1 + b
            audio = torch.cat([audio_0, audio_1], 1)

        output_audio.append(audio)
        return torch.cat(output_audio, 1), spect


class DeterministicWaveGlowTeacher(WaveGlowTeacher):
    """ WaveGlowTeacher that predicts deterministically based on seed """

    @lru_cache(maxsize=None)
    def generate_noise_common(self, *, seed=1337, device=torch.device('cpu'), 
                              dtype=torch.float32, channels=8, max_length=50_000):
        noise = np.random.RandomState(seed=seed).randn(1, channels, max_length)
        return torch.as_tensor(noise, device=device, dtype=dtype)

    def generate_noise_inputs(self, batch_size: int, channels: tuple, length: int, **kwargs):
        assert kwargs.get('max_length', 50_000) >= length
        common = self.generate_noise_common(channels=sum(channels), **kwargs)
        noise_inputs = []
        split_index = 0
        for num_channels in channels:
            noise_inputs.append(common[:, split_index: split_index + num_channels, :length].repeat(batch_size, 1, 1))
            split_index += num_channels
        return noise_inputs

    def sample_inputs_for(self, spect, **kwargs):
        """ upsample and generate noise: the non-distilled part of waveglow """
        assert self.n_early_every > 0 and self.n_flows > 0
        spect = self.upsample(spect)

        # trim conv artifacts. maybe pad spec to kernel multiple
        if not self.upsample_multistage:
            time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
            spect = spect[:, :, :-time_cutoff]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)
        num_noise_vectors = max(1, (self.n_flows - 1) // self.n_early_every)
        noise_num_channels = (self.n_remaining_channels,) + (self.n_early_size,) * num_noise_vectors
        noise_audio, *noise_vectors = self.generate_noise_inputs(
            spect.shape[0], noise_num_channels, spect.shape[2], device=spect.device, dtype=spect.dtype, **kwargs)
        return (spect, noise_audio, *noise_vectors)