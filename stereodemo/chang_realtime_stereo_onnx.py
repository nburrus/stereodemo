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

from .methods import EnumParameter, StereoMethod, InputPair, StereoOutput
from . import utils

urls = {
    "chang-realtime-stereo-cpu-1280x720.onnx": "",
    "chang-realtime-stereo-cpu-160x128.onnx": "",
    "chang-realtime-stereo-cpu-320x240.onnx": "",
    "chang-realtime-stereo-cpu-640x480.onnx": "",
}

# Adapted from https://github.com/ibaiGorordo/ONNX-CREStereo-Depth-Estimation
# https://github.com/PINTO0309/PINTO_model_zoo/tree/main/284_CREStereo
# IMPORTANT: these ONNX are not working, keeping in case things improve later on.
class ChangRealtimeStereoOnnx(StereoMethod):
    def __init__(self):
        super().__init__("Chang Real-time Onnx", "Attention-Aware Feature Aggregation for Real-time Stereo Matching on Edge Devices (ACCV 2020).", {})
        self.reset_defaults()

        self._loaded_session = None
        self._loaded_model_path = None

    def reset_defaults(self):
        self.parameters.update ({
            "Shape": EnumParameter("Processed image size", 1, ["160x128", "320x240", "640x480", "1280x720"])
        })

    def compute_disparity(self, input: InputPair) -> StereoOutput:
        if not models_path.exists():
            models_path.mkdir(parents=True, exist_ok=True)
        
        cols, rows = self.parameters["Shape"].value.split('x')
        cols, rows = int(cols), int(rows)
        self.target_size = (cols, rows)

        model_path = models_path / f'chang-realtime-stereo-cpu-{cols}x{rows}.onnx'
        self._load_model (model_path)

        left_tensor = self._preprocess_input(input.left_image)
        right_tensor = self._preprocess_input(input.right_image)

        start = time.time()
        model_inputs = self._loaded_session.get_inputs()
        model_outputs = self._loaded_session.get_outputs()
        input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        outputs = self._loaded_session.run(['disparity'], {'left': left_tensor,
                                                           'right': right_tensor})
        elapsed_time = time.time() - start

        disparity_map = self._process_output(outputs)
        if disparity_map.shape[:2] != input.left_image.shape[:2]:
            disparity_map = cv2.resize (disparity_map, (input.left_image.shape[1], input.left_image.shape[0]), cv2.INTER_NEAREST)
            x_scale = input.left_image.shape[1] / float(cols)
            disparity_map *= np.float32(x_scale)

        return StereoOutput(disparity_map, input.left_image, elapsed_time)

    def _preprocess_input (self, img: np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size, cv2.INTER_AREA)
        imagenet_stats = {'mean': np.array([0.485, 0.456, 0.406]), 'std': np.array([0.229, 0.224, 0.225])}
        img = (img.astype(np.float32) / 255.0) - imagenet_stats['mean'] / imagenet_stats['std']
        img = img.transpose(2, 0, 1) # C,H,W instead of H,W,C
        img = img[np.newaxis, :, :, :] # add batch dimension
        return img

    def _process_output(self, outputs):
        disparity_map = outputs[0].permute(1,2,0)
        return disparity_map

    def _load_model(self, model_path: Path):
        if (self._loaded_model_path == model_path):
            return
        
        if not model_path.exists():
            utils.download_model (urls[model_path.name], model_path)

        assert Path(model_path).exists()
        self._loaded_model_path = model_path
        self._loaded_session = onnxruntime.InferenceSession(str(model_path), providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
