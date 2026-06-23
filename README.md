# Pyramid-Based Mesh Refinement for 3D Human Reconstruction in Body Language Assessment

<img style="max-width: 100%;" src="https://github.com/swerizwan/PMRR/blob/main/resources/wax.png" alt="Title Overview">

# Overview

This work introduces a Pyramid-based Mesh Refinement Reconstruction (PMRR) framework for accurate 3D human mesh reconstruction from a single image. The proposed method iteratively refines mesh parameters using multi-scale features to improve mesh-image alignment. It further incorporates emotional stability loss to maintain facial expression consistency and applies pixel-wise supervision to enhance feature quality. These improvements enable reliable analysis of body gestures, hand postures, and facial expressions for body language assessment.

# 👁️💬 Architecture

The comprehensive pipeline of the RealMock framework.

<img style="max-width: 100%;" src="https://github.com/swerizwan/PMRR/blob/main/resources/architecture.jpg" alt="PMRR Overview">

# PMRR Environment Setup

We evaluated PMRR using PVE, MPJPE, PA-MPJPE, and AP metrics on the COCO and 3DPW datasets. The method was compared against existing approaches, demonstrating superior performance in 3D pose and shape estimation.

The instructions for setting up a Conda environment named `pmrr` with the required dependencies:

## Requirements

- Python 3.8
```
conda create --no-default-packages -n pmrr python=3.8
conda activate pmrr
```

### Packages

- [PyTorch](https://www.pytorch.org) Tested with version 1.9.0. To install, use the following command:
```
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

- [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) Install the stable version with:
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

- All other necessary packages are listed in `requirements.txt`. Install them with:
```
pip install -r requirements.txt
```

### Required Files

> Mesh Downsampling and DensePose UV Data
- Execute the following script to download `mesh_downsampling.npz` and DensePose UV data from other repositories:

```
bash fetch_data.sh
```
> SMPL Model Files
- Obtain the SMPL model files from [SMPL](https://smpl.is.tue.mpg.de) and [UP](https://github.com/classner/up/blob/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl). Rename the model files as needed and place them in the `./files/smpl` directory.

> Preprocessed Data from SPIN
- Download the preprocessed data by following the instructions [here](https://github.com/nkolot/SPIN#fetch-data).

> Final Fits Data from SPIN
- Retrieve the final fits data as outlined [here](https://github.com/nkolot/SPIN#final-fits). Important Note: Using [EFT](https://github.com/facebookresearch/eft) fits for training is recommended. Compatible `.npz` files can be found [here](https://cloud.tsinghua.edu.cn/d/635c717375664cd6b3f5)

> Files Directory
- After gathering these necessary files, your `./data` directory structure should look like this:
```
./files
├── dataset_extras
│   └── .npz files
├── J_regressor
│   ├── J_regressor_extra.npy
│   └── J_regressor_h36m.npy
├── mesh_downsampling.npz
├── pretrained_model
│   └── emo-body-lang_checkpoint.pt
├── smpl
│   ├── SMPL_FEMALE.pkl
│   ├── SMPL_MALE.pkl
│   └── SMPL_NEUTRAL.pkl
├── smpl_mean_params.npz
├── UV_data
│   ├── UV_Processed.mat
│   └── UV_symmetry_transforms.mat
└── final_fits
    └── .npy files
```

## Datasets

### COCO

COCO (Common Objects in Context) is a large-scale dataset designed for object detection, segmentation, and image captioning. In this project, we use the val2014 subset to evaluate emotion-related body gestures from still images. You can download the dataset from the [COCO download page](https://cocodataset.org/#download). The preprocessed version we use is available [here (coco_2014_val.npz)](https://drive.google.com/file/d/1ew77AaaOT3SAF0fZpfPrg02P5c9bzTHe/view?usp=sharing).

### 3DPW

3DPW (3D Poses in the Wild) is a dataset containing outdoor video sequences with accurate 3D human poses captured using IMUs and cameras. It is widely used to evaluate 3D body posture and gesture recognition in natural environments. You can download the dataset from the [official 3DPW website](https://virtualhumans.mpi-inf.mpg.de/3DPW/).

### IEMOCAP

IEMOCAP (Interactive Emotional Dyadic Motion Capture) is a multimodal dataset containing acted conversational recordings with annotated emotional expressions. You can download the dataset from the [official 3DPW website](https://sail.usc.edu/iemocap/).

### AffectNet

AffectNet is a large-scale facial expression dataset collected from the internet, containing images annotated with discrete emotion categories and valence-arousal labels.You can download the dataset from the [official 3DPW website](https://mohammadmahoor.com/pages/databases/affectnet/).

## Preview of Demo Results:

### For Image Input:

```
python3 run_demo.py --checkpoint=files/pretrained_model/emo_body_lang_checkpoint.pt --img_file input/image.png
```

<p align="center">
    <img style="max-width: 100%;" src="https://github.com/swerizwan/PMRR/blob/main/resources/image.png" alt="PMRR Overview">
</p>

### For Video Input:

```
python3 run_demo.py --checkpoint=files/pretrained_model/emo_body_lang_checkpoint.pt --vid_file input/video.mp4
```

<p align="center">
    <img style="max-width: 100%;" src="https://github.com/swerizwan/PMRR/blob/main/resources/image.gif" alt="PMRR Overview">
</p>

## Training

We can monitor the training process by setting up a TensorBoard in the directory `./logs`.

```
CUDA_VISIBLE_DEVICES=0 python3 trainer.py --regressor emo_body_lang --single_dataset --misc TRAIN.BATCH_SIZE 64
```
