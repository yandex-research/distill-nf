# WaveGlow Distillation 

This code is based on the WaveGlow [repository](https://github.com/NVIDIA/waveglow). 

## Dataset

The training is performed on the [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) dataset. All models are evaluated on our [collected hold-out set](https://www.dropbox.com/sh/n14gejov1hocpso/AAC61FWhk0tKwFVOpFgTR9Qwa?dl=0). 

## Dependencies

* Python >= 3.6
* ```pip install -r requirements.txt```

## Reproduce

1. Download the data: `bash download_ljspeech.sh` (3 Gb);
2. Download the WaveGlow teacher and pretrained students:
```cd ./pretrained_models && bash download_pretrained_models.sh``` (718 Mb)
3. ***Training:*** `bash ./scripts/train_wg_student_v{1,2,3}.sh`

   ***Evaluation:*** `./notebooks/evaluation.ipynb`

## Audio Samples

The audio samples used for final evaluation are available [here](https://www.dropbox.com/sh/0wx2dswks9um90r/AAA5R78v_DPL_5I5RfugPKmWa?dl=0)