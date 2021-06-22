# coding: U8

# data config
# TODO: borrow from audio_processing.py
SEGMENT_LENGTH = 16000
SAMPLING_RATE = 22050
STFT_FILTER_LENGTH = 1024
STFT_HOP_LENGTH = 256
STFT_WIN_LENGTH = 1024
MEL_FMIN = 0.0
MEL_FMAX = 8000.0
MEL_CHANNELS = 80

WG_UPSAMPLE_MULTISTAGE = False
WG_DECOMPOSE_CONVINV = False
WG_N_FLOWS = 12
WG_N_GROUP = 8
WG_N_EARLY_EVERY = 4
WG_N_EARLY_SIZE = 2
WAVENET_N_LAYERS = 8
WAVENET_N_CHANNELS = 256
WAVENET_KERNEL_SIZE = 3

def get_default_wg_config():
    return dict(
        n_mel_channels=MEL_CHANNELS, 
        n_flows=WG_N_FLOWS, 
        n_group=WG_N_GROUP, 
        n_early_every=WG_N_EARLY_EVERY, 
        n_early_size=WG_N_EARLY_SIZE,
        WN_config=dict(
            n_layers=WAVENET_N_LAYERS,
            n_channels=WAVENET_N_CHANNELS,
            kernel_size=WAVENET_KERNEL_SIZE,
            causal_layers=None
        ),
        upsample_multistage=WG_UPSAMPLE_MULTISTAGE,
        decompose_convinv=WG_DECOMPOSE_CONVINV,
    )