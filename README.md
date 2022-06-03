[![Unit Tests](https://github.com/nburrus/stereodemo/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/nburrus/stereodemo/actions/workflows/unit_tests.yml)
<a href="https://twitter.com/nburrus">
<img src="https://img.shields.io/twitter/url?label=Twitter&style=social&url=https%3A%2F%2Ftwitter.com%2Fnburrus" alt="Twitter Badge"/>
</a>

# stereodemo

Small Python utility to **compare and visualize** the output of various **stereo depth estimation** algorithms:

- Make it easy to get a qualitative evaluation of several state-of-the-art models in the wild
- Feed it left/right images or capture live from an [OAK-D camera](https://store.opencv.ai/products/oak-d)
- Interactive colored point-cloud view since nice-looking disparity images can be misleading
- Try different parameters on the same image

**Included methods** (implementation/pre-trained models taken from their respective authors):

- [OpenCV](https://opencv.org) stereo block matching and Semi-global block matching baselines, with all their parameters
- [CREStereo](https://github.com/megvii-research/CREStereo): "Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation" (CVPR 2022)
- [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo): "Multilevel Recurrent Field Transforms for Stereo Matching" (3DV 2021)
- [Hitnet](https://github.com/google-research/google-research/tree/master/hitnet): "Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching" (CVPR 2021)
- [STereo TRansformers](https://github.com/mli0603/stereo-transformer): "Revisiting Stereo Depth Estimation From a Sequence-to-Sequence Perspective with Transformers" (ICCV 2021)
- [Chang et al. RealtimeStereo](https://github.com/JiaRenChang/RealtimeStereo): "Attention-Aware Feature Aggregation for Real-time Stereo Matching on Edge Devices" (ACCV 2020)

See below for more details / credits to get each of these working, and check this [blog post for more results, including performance numbers](https://nicolas.burrus.name/stereo-comparison/).

https://user-images.githubusercontent.com/541507/169557430-48e62510-60c2-4a2b-8747-f9606e405f74.mp4

# Getting started

## Installation

```
python3 -m pip install stereodemo
```

## Running it

### With an OAK-D camera

To capture data directly from an OAK-D camera, use:

```
stereodemo --oak
```

Then click on `Next Image` to capture a new one.

### With image files

If you installed stereodemo from pip, then just launch `stereodemo` and it will
show some embedded sample images captured with an OAK-D camera.

A tiny subset of some popular datasets is also included in this repository. Just
provide a folder to `stereodemo` and it'll look for left/right pairs (either
im0/im1 or left/right in the names):

```
# To evaluate on the oak-d images
stereodemo datasets/oak-d 

# To cycle through all images
stereodemo datasets
```

Then click on `Next Image` to cycle through the images.

Sample images included in this repository:
- [drivingstereo](datasets/drivingstereo/README.md): outdoor driving.
- [middlebury_2014](datasets/middlebury_2014/README.md): high-res objects.
- [eth3d](datasets/eth3d_lowres/README.md): outdoor and indoor scenes.
- [sceneflow](datasets/sceneflow/README.md): synthetic rendering of objects.
- [oak-d](datasets/oak-d/README.md): indoor images I captured with my OAK-D lite camera.
- [kitti2015](datasets/kitti2015/README.md): outdoor driving (only one image).

# Dependencies

`pip` will install the dependencies automatically. Here is the list:

- [Open3D](https://open3d.org). For the point cloud visualization and the GUI.
- [OpenCV](https://opencv.org). For image loading and the traditional block matching baselines.
- [onnxruntime](https://onnxruntime.ai/). To run pretrained models in the ONNX format.
- [pytorch](https://pytorch.org/). To run pretrained models exported as torch script.
- [depthai](https://docs.luxonis.com/en/latest/). Optional, to grab images from a Luxonis OAK camera.

# Credits for each method

I did not implement any of these myself, but just collected pre-trained models or converted them to torch script / ONNX.

- CREStereo
  - Official implementation and pre-trained models: https://github.com/megvii-research/CREStereo
  - Model Zoo for the ONNX models: https://github.com/PINTO0309/PINTO_model_zoo/tree/main/284_CREStereo
  - Port to ONNX + sample loading code: https://github.com/ibaiGorordo/ONNX-CREStereo-Depth-Estimation

- RAFT-Stereo
  - Official implementation and pre-trained models: https://github.com/princeton-vl/RAFT-Stereo
  - I exported the pytorch implementation to torch script via tracing, [with minor modifications of the source code](https://github.com/nburrus/RAFT-Stereo/commit/ebbb5a807227927ab4551274039e9bdd16a1b010).
  - Their fastest implementation was not imported.

- Hitnet
  - Official implementation and pre-trained models: https://github.com/google-research/google-research/tree/master/hitnet
  - Model Zoo for the ONNX models: https://github.com/PINTO0309/PINTO_model_zoo/tree/main/142_HITNET
  - Port to ONNX + sample loading code: https://github.com/ibaiGorordo/ONNX-HITNET-Stereo-Depth-estimation

- Stereo Transformers
  - Official implementation and pre-trained models: https://github.com/mli0603/stereo-transformer
  - Made [some small changes](https://github.com/nburrus/stereo-transformer/commit/0006a022c19f0c7c4d7683408531180a863603a5) to allow torch script export via tracing.
  - The exported model currently fails with GPU inference, so only CPU inference is enabled.

- Chang et al. RealtimeStereo
  - Official implementation and pre-trained models: https://github.com/JiaRenChang/RealtimeStereo
  - I exported the pytorch implementation to torch script via tracing with some minor changes to the code https://github.com/JiaRenChang/RealtimeStereo/pull/15 . See [chang_realtimestereo_to_torchscript_onnx.py](tools/chang_realtimestereo_to_torchscript_onnx.py).

# License

The code of stereodemo is MIT licensed, but the pre-trained models are subject to the license of their respective implementation.

The sample images have the license of their respective source, except for datasets/oak-d which is licenced under [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

