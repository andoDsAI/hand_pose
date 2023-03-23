# HandOccNet

## Introduction

This repository is the offical [Pytorch](https://pytorch.org/) implementation of **[HandOccNet: Occlusion-Robust 3D Hand Mesh Estimation Network (CVPR 2022)](https://arxiv.org/abs/2203.14564)**. Below is the overall pipeline of HandOccNet.
![overall pipeline](./asset/model.png)

## Directory

### Root

The `${ROOT}` is described as below.

```bash
${ROOT}  
|-- data  
|-- demo
|-- common  
|-- main  
|-- output  
```

* `data` contains data loading codes and soft links to images and annotations directories.  
* `demo` contains demo codes.
* `common` contains kernel codes for HandOccNet.  
* `main` contains high-level codes for training or testing the network.  
* `output` contains log, trained models, visualized outputs, and test result.  

### Data

You need to follow directory structure of the `data` as below.

```bash
${ROOT}  
|-- data  
|   |-- DEX_YCB
|   |   |-- data
|   |   |   |-- 20200709-subject-01
|   |   |   |-- ......
|   |   |   |-- annotations
|   |   |   |   |--DEX_YCB_s0_train_data.json
|   |   |   |   |--DEX_YCB_s0_test_data.json
```

* Download DexYCB data and annotation files <[data](https://dex-ycb.github.io/)>[[annotation files](https://drive.google.com/drive/folders/1pmRpgv38PXvlLOODtoxpTYnIpYTkNV6b?usp=sharing)]

### Pytorch MANO layer

* For the MANO layer, I used [manopth](https://github.com/hassony2/manopth). The repo is already included in `common/utils/manopth`.
* Download `MANO_RIGHT.pkl` from [here](https://mano.is.tue.mpg.de/) and place at `common/utils/manopth/mano/models`.

### Output  

You need to follow the directory structure of the `output` folder as below.  

```bash
${ROOT}  
|-- output  
|   |-- log  
|   |-- model_dump  
|   |-- result  
|   |-- vis  
```  

* Creating `output` folder as soft link form is recommended instead of folder form because it would take large storage capacity.  
* `log` folder contains training log file.  
* `model_dump` folder contains saved checkpoints for each epoch.  
* `result` folder contains final estimation files generated in the testing stage.  
* `vis` folder contains visualized results.  

## Running

### Start  

* Install **[PyTorch](https://pytorch.org)** and Python >= 3.7.4 and run `pip3 install -r requirements.txt`.
* In the `main/config.py`, you can change settings of the model including dataset to use and input size and so on.  

### Train  

In the `main` folder, set train_set in `config.py` (as 'DEX_YCB') and run  

```bash  
python train.py --gpu 0-3
```  

to train on the GPU 0,1,2,3. `--gpu 0,1,2,3` can be used instead of `--gpu 0-3`.

### Test  

Place trained model at the `output/model_dump/`.
  
In the `main` folder, set test_set in `config.py` (as 'DEX_YCB') and run  

```bash  
python test.py --gpu 0-3 --test_epoch {test epoch}  
```  

`--gpu 0,1,2,3` can be used instead of `--gpu 0-3`.

## Acknowledgements

For this project, we relied on research codes from:

* [I2L-MeshNet_RELEASE](https://github.com/mks0601/I2L-MeshNet_RELEASE)
* [Semi-Hand-Object](https://github.com/stevenlsw/Semi-Hand-Object)
* [attention-module](https://github.com/Jongchan/attention-module)
