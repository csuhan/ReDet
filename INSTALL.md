## Installation

### Requirements

- Linux
- Python 3.5/3.6/3.7
- PyTorch 1.1/1.3.1
- CUDA 10.0/10.1
- NCCL 2+
- GCC 4.9+
- [mmcv<=0.2.14](https://github.com/open-mmlab/mmcv)


### Install ReDet

a. Create a conda virtual environment and activate it. Then install Cython.

```shell
conda create -n redet python=3.7 -y
source activate redet

conda install cython
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).

```
conda install pytorch=1.3.1 torchvision cudatoolkit=10.0 -c pytorch -y
```
Note:
1. If you want to use Pytorch>1.5, you have to made some modifications to the `cuda ops`. See [here](https://github.com/csuhan/ReDet/issues/1) for a reference.
2. There is a known [bug](https://github.com/csuhan/ReDet/issues/4) happened to some users but not all (As I have successfully run it on V100 and Titan Xp). If it occurs, please refer to [here](https://github.com/csuhan/ReDet/issues/4).
3. If you want to use Python<=3.6, you need to install [e2cnn@legacy_py3.6](https://github.com/QUVA-Lab/e2cnn/tree/legacy_py3.6) mamually, see [here](https://github.com/csuhan/ReDet/issues/14) for an instruction.

c. Clone the ReDet repository.

```shell
git clone https://github.com/csuhan/ReDet.git
cd ReDet
```

d. Compile cuda extensions.

```shell
bash compile.sh
```

e. Install ReDet (other dependencies will be installed automatically).

```shell
python setup.py develop
# or "pip install -e ."
```

Note:

1. It is recommended that you run the step e each time you pull some updates from github. If there are some updates of the C/CUDA codes, you also need to run step d.
The git commit id will be written to the version number with step e, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.

2. Following the above instructions, ReDet is installed on `dev` mode, any modifications to the code will take effect without installing it again.

### Install DOTA_devkit
```
    sudo apt-get install swig
    cd DOTA_devkit
    swig -c++ -python polyiou.i
    python setup.py build_ext --inplace
```
