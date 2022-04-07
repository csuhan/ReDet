## ReDet_mmclassification

This branch contains the codes for training ReResNet. 
We make minor modifications on the [mmclassification](https://github.com/open-mmlab/mmclassification).
The specific version is [4e6875d](https://github.com/open-mmlab/mmclassification/tree/4e6875d44e5e04d17c4afb146d97273b3a3f917a).

## Benchmark and model zoo

|         Model                                               |Group      | Top-1 (%) | Top-5 (%) | Download |
|:-----------------------------------------------------------:|:---------:|:---------:|:---------:|:--------:|
| [ReResNet-50](configs/re_resnet/re_resnet50_c8_batch256.py) |C<sub>8</sub>| 71.20     | 90.28     |[raw](https://drive.google.com/file/d/1_d2igSp0wM8ypxTM1S14f5kCVjEyE6iI/view?usp=sharing) &#124; [publish](https://drive.google.com/file/d/1FshfREfLZaNl5FcaKrH0lxFyZt50Uyu2/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1VLW8YbU1kGpqd4hfvI9UItbCOprzo-v4/view?usp=sharing)|
| [ReResNet-101](configs/re_resnet/re_resnet101_c8_batch256.py) |C<sub>8</sub>| 74.92     | 92.22     |[raw](https://drive.google.com/file/d/1_SDzcwuv_0IkPPJ5VmM9Hfhyz47BtHA3/view?usp=sharing) &#124; [publish](https://drive.google.com/file/d/1w1KGCzYFPIJjjVOR2FOGgytYu4oCrjAM/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1w1KGCzYFPIJjjVOR2FOGgytYu4oCrjAM/view?usp=sharing)|


**Note**:

* Alternative download link: [baiduyun](https://pan.baidu.com/s/1ENIkUVB_5-QRQhr0Vl-FMw) with extracting code `ABCD`.
* The [raw](https://drive.google.com/file/d/1_d2igSp0wM8ypxTM1S14f5kCVjEyE6iI/view?usp=sharing) checkpoint is used to test the accuracy on ImageNet. The [publish](https://drive.google.com/file/d/1FshfREfLZaNl5FcaKrH0lxFyZt50Uyu2/view?usp=sharing) model is used for downstream tasks, e.g., object detection. We convert the raw model to publish model by [tools/publish_model.py](tools/publish_model.py).


## Installation

Please refer to [install.md](docs/install.md) for installation and dataset preparation.


## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMClassification. There are also tutorials for [finetuning models](docs/tutorials/finetune.md), [adding new dataset](docs/tutorials/new_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), and [adding new modules](docs/tutorials/new_modules.md).

### Convert ReResNet to standard Pytorch layers
We can export ReResNet to a standard ResNet by [tools/convert_re_resnet_to_torch.py](tools/convert_re_resnet_to_torch.py).

First, download the checkpoint from [here](https://drive.google.com/file/d/1_d2igSp0wM8ypxTM1S14f5kCVjEyE6iI/view?usp=sharing) and put it to `work_dirs/re_resnet50_c8_batch256/epoch_100.pth`.

Then, convert the raw checkpoint to a standard checkpoint for ResNet.
```
python tools/convert_re_resnet_to_torch.py configs/re_resnet/re_resnet50_c8_batch256.py \
        work_dirs/re_resnet50_c8_batch256/epoch_100.pth work_dirs/re_resnet50_c8_batch256/epoch_100_torch.pth
```

Now, we can test the accuracy with a standard ResNet.
```
bash tools/dist_test.sh configs/imagenet/resnet50_batch256.py work_dirs/re_resnet50_c8_batch256/epoch_100_torch.pth 8
```


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
