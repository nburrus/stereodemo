# Stereo Demo

Compare and visualize the output of various stereo reconstruction algorithms.

# Getting started

## Installation

Installing the module so you can run it from any folder:

```
pip3 install stereodemo
```

or without installing the module, directly from the repository

```
python3 -m stereodemo datasets/eth3d
```

## Running on some datasets

For convenience a subset of some popular datasets is included in this repository.

To browse all the images:

`stereodemo datasets`

Then click on `Next Image` to cycle through the images it could find in the `datasets` folder.

You can obviously just provide subfolders:

```
stereodemo datasets/drivingstereo
```

Or directly pairs of images:

```
stereodemo sceneflow/driving_{left,right}.png
```

# onnxruntime-gpu on Ubuntu 20.04

To get the CUDAProvider running on my system I had to follow these steps:

- Install CUDA 11.7 https://developer.nvidia.com/cuda-downloads

- Install nvidia-cudnn
```
pip install nvidia-pyindex
pip install nvidia-cudnn
```

- Make sure you have onnxruntime-gpu >= 1.10

# Dependencies

`pip` will install the dependencies automatically. Here is the list:

- [Open3D](https://open3d.org). For the point cloud visualization and the GUI.
- [OpenCV](https://opencv.org). For image loading and the traditional block matching baselines.
- [onnxruntime](https://onnxruntime.ai/). To run pretrained models in the ONNX format.
- [pytorch](https://pytorch.org/). To run pretrained models exported as torch script.
