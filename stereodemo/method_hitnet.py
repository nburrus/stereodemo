from pathlib import Path
import shutil
import time
from dataclasses import dataclass
import urllib.request
import tempfile
import sys

import onnxruntime

import cv2
import numpy as np

from .methods import Config, EnumParameter, StereoMethod, InputPair, StereoOutput
from . import utils

urls = {
    "hitnet_eth3d_120x160.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_eth3d_120x160.onnx",
    "hitnet_eth3d_240x320.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_eth3d_240x320.onnx",
    "hitnet_eth3d_480x640.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_eth3d_480x640.onnx",
    "hitnet_eth3d_720x1280.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_eth3d_720x1280.onnx",
    "hitnet_middlebury_120x160.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_middlebury_120x160.onnx",
    "hitnet_middlebury_240x320.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_middlebury_240x320.onnx",
    "hitnet_middlebury_480x640.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_middlebury_480x640.onnx",
    "hitnet_middlebury_720x1280.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_middlebury_720x1280.onnx",
    "hitnet_sceneflow_120x160.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_sceneflow_120x160.onnx",
    "hitnet_sceneflow_240x320.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_sceneflow_240x320.onnx",
    "hitnet_sceneflow_480x640.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_sceneflow_480x640.onnx",
    "hitnet_sceneflow_720x1280.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_sceneflow_720x1280.onnx",
}

# Adapted from https://github.com/ibaiGorordo/ONNX-HITNET-Stereo-Depth-estimation
# Onnx models from https://github.com/PINTO0309/PINTO_model_zoo/tree/main/142_HITNET
# Official implementation https://github.com/google-research/google-research/tree/master/hitnet
class HitnetStereo(StereoMethod):
    def __init__(self, config: Config):
        super().__init__("Hitnet (CVPR 2021)",
                         "HITNet: Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching",
                         {},
                         config)
        self.reset_defaults()

        self._loaded_session = None
        self._loaded_model_path = None

    def reset_defaults(self):
        self.parameters.update ({
            "Shape": EnumParameter("Processed image size", 2, ["160x120", "320x240", "640x480", "1280x720"]),
            "Training Set": EnumParameter("Dataset used during training", 1, ["sceneflow", "middlebury", "eth3d"])
        })

    def compute_disparity(self, input: InputPair) -> StereoOutput:        
        cols, rows = self.parameters["Shape"].value.split('x')
        cols, rows = int(cols), int(rows)
        training_set = self.parameters["Training Set"].value

        model_path = self.config.models_path / f'hitnet_{training_set}_{rows}x{cols}.onnx'
        self._load_model (model_path)

        model_inputs = self._loaded_session.get_inputs()
        model_outputs = self._loaded_session.get_outputs()
        model_rows, model_cols = model_inputs[0].shape[2:] # B,C,H,W
        self.target_size = (model_cols, model_rows)

        grayscale = True if training_set == 'eth3d' else False
        combined_tensor = self._preprocess_input(input.left_image, input.right_image, grayscale)

        start = time.time()
        input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        outputs = self._loaded_session.run(['reference_output_disparity'], { 'input': combined_tensor })
        elapsed_time = time.time() - start

        disparity_map = self._process_output(outputs)
        if disparity_map.shape[:2] != input.left_image.shape[:2]:
            model_output_cols = disparity_map.shape[1]
            disparity_map = cv2.resize (disparity_map, (input.left_image.shape[1], input.left_image.shape[0]), cv2.INTER_NEAREST)
            x_scale = input.left_image.shape[1] / float(model_output_cols)
            disparity_map *= np.float32(x_scale)

        return StereoOutput(disparity_map, input.left_image, elapsed_time)

    def _preprocess_input (self, left: np.ndarray, right: np.ndarray, grayscale: bool):
        if grayscale:
            # H,W
            left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        else:
            # H,W,C=3
            left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
            right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

        left = cv2.resize(left, self.target_size, cv2.INTER_AREA)
        right = cv2.resize(right, self.target_size, cv2.INTER_AREA)

        # Grayscale needs expansion to reach H,W,C.
        # Need to do that now because resize would change the shape.
        if left.ndim == 2:
            left = left[..., np.newaxis]
            right = right[..., np.newaxis]

        # -> H,W,C=2 or 6 , normalized to [0,1]
        combined_img = np.concatenate((left, right), axis=-1) / 255.0
        # -> C,H,W
        combined_img = combined_img.transpose(2, 0, 1)
        # -> B=1,C,H,W
        combined_img = np.expand_dims(combined_img, 0).astype(np.float32)
        return combined_img

    def _process_output(self, outputs):
        disparity_map = outputs[0][0].squeeze(-1)
        return disparity_map

    def _load_model(self, model_path: Path):
        if (self._loaded_model_path == model_path):
            return
        
        if not model_path.exists():
            utils.download_model (urls[model_path.name], model_path)

        assert Path(model_path).exists()
        self._loaded_model_path = model_path
        self._loaded_session = onnxruntime.InferenceSession(str(model_path), providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
