## ReDet_mmclassification

This branch contains the codes for training ReResNet. 
We make minor modifications on the [mmclassification](https://github.com/open-mmlab/mmclassification).
The specific version is [4e6875d](https://github.com/open-mmlab/mmclassification/tree/4e6875d44e5e04d17c4afb146d97273b3a3f917a).

## Benchmark and model zoo

|         Model                                               |Group      | Top-1 (%) | Top-5 (%) | Download |
|:-----------------------------------------------------------:|:---------:|:---------:|:---------:|:--------:|
| [ReResNet-50](configs/re_resnet/re_resnet50_c8_batch256.py) |C<sub>8</sub>| 71.20     | 90.28     | [model](https://drive.google.com/file/d/1FshfREfLZaNl5FcaKrH0lxFyZt50Uyu2/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1VLW8YbU1kGpqd4hfvI9UItbCOprzo-v4/view?usp=sharing)|

*Alternative download link: [baiduyun](https://pan.baidu.com/s/1ENIkUVB_5-QRQhr0Vl-FMw) with extracting code `ABCD`.

## Installation

Please refer to [install.md](docs/install.md) for installation and dataset preparation.


## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMClassification. There are also tutorials for [finetuning models](docs/tutorials/finetune.md), [adding new dataset](docs/tutorials/new_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), and [adding new modules](docs/tutorials/new_modules.md).



## Citation

If you use this toolbox or benchmark in your research, please cite:

```
@misc{mmclassification,
  author =       {Yang, Lei and Li, Xiaojie and Lou, Zan and Yang, Mingmin and
                  Wang, Fei and Qian, Chen and Chen, Kai and Lin, Dahua},
  title =        {{MMClassification}},
  howpublished = {\url{https://github.com/open-mmlab/mmclassification}},
  year =         {2020}
}

@inproceedings{han2021ReDet,
  title={ReDet: A Rotation-equivariant Detector for Aerial Object Detection},
  author={Han, Jiaming and Ding, Jian and Xue, Nan and Xia, Gui-Song},
  booktitle =  {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```
