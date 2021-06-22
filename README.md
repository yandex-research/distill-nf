# Distill the Knowledge from Normalizing Flows

Official implementation of the paper [Distill the Knowledge from Normalizing Flows](TODO) (ICMLW'2021) by Dmitry Baranchuk, Vladimir Aliev, Artem Babenko.

## Overview

The approach transfers the knowledge from Normalizing Flows (NF) to efficient feed-forward models to speed up inference and reduce the model size. The effectiveness of this approach is demonstrated on two state-of-the-art conditional flow-based models for image super-resolution ([SRFlow](https://github.com/andreas128/SRFlow)) and speech synthesis ([WaveGlow](https://github.com/NVIDIA/waveglow)).

* [SRFlow distillation]() 
* [WaveGlow distillation]()

<img src="./resources/distill-nf-scheme.png" width="100%">

## Citation

@article{TODO,
  title={Distill the Knowledge from Normalizing Flows},
  author={Baranchuk, Dmitry and Aliev, Vladimir and Babenko, Artem}
  journal={arXiv preprint arXiv:TODO},
  year={2021}
}