# SRFlow Distillation 

This code is based on the SRFlow [repository](https://github.com/andreas128/SRFlow). 

## Datasets

The students are learned on the DF2K dataset (A merged training dataset of [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](https://github.com/LimBee/NTIRE2017)). All models are evaluated on 100 validation images from DIV2K. A few samples from [BSDS100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) and [Set14](https://drive.google.com/drive/folders/1gt5eT293esqY0yr1Anbm36EdnxWW_5oH?usp=sharing) datasets are used for qualitative evaluation.  


### Preparation Steps

* ***Training:*** Download the preprocessed train sets: ```cd ./data && bash download_train_data.sh``` (56 Gb). The data is preprocessed according to the [BasicSR](https://github.com/xinntao/BasicSR/blob/master/docs/DatasetPreparation.md#div2k) instructions.
* ***Evaluation:*** Download validation images: ```cd ./data && bash download_eval_data.sh``` (506 Mb).


## Dependencies

* Python >= 3.6
* ```pip install -r requirements.txt```


## Reproduce

1. Download the WaveGlow teacher and pretrained students:
```cd ./pretrained_models && bash download_pretrained_models.sh```
2. Prepare the datasets
3. ***Training:*** ` cd ./scripts && bash train_wg_student_v{1,2,3}.sh`

   ***Evaluation:*** `./notebooks/evaluate_x{4,8}.ipynb`

