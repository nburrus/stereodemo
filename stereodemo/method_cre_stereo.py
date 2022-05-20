from pathlib import Path
import time
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime

from .methods import Config, EnumParameter, StereoMethod, InputPair, StereoOutput
from . import utils

urls = {
    "crestereo_combined_iter10_240x320.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_combined_iter10_240x320.onnx",
    "crestereo_combined_iter10_480x640.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_combined_iter10_480x640.onnx",
    "crestereo_combined_iter10_720x1280.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_combined_iter10_720x1280.onnx",
    "crestereo_combined_iter20_240x320.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_combined_iter20_240x320.onnx",
    "crestereo_combined_iter20_480x640.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_combined_iter20_480x640.onnx",
    "crestereo_combined_iter20_720x1280.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_combined_iter20_720x1280.onnx",
    "crestereo_combined_iter2_240x320.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_combined_iter2_240x320.onnx",
    "crestereo_combined_iter2_480x640.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_combined_iter2_480x640.onnx",
    "crestereo_combined_iter2_720x1280.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_combined_iter2_720x1280.onnx",
    "crestereo_combined_iter5_240x320.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_combined_iter5_240x320.onnx",
    "crestereo_combined_iter5_380x480.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_combined_iter5_380x480.onnx",
    "crestereo_combined_iter5_480x640.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_combined_iter5_480x640.onnx",
    "crestereo_combined_iter5_720x1280.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_combined_iter5_720x1280.onnx",
    "crestereo_init_iter10_180x320.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_init_iter10_180x320.onnx",
    "crestereo_init_iter10_240x320.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_init_iter10_240x320.onnx",
    "crestereo_init_iter10_480x640.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_init_iter10_480x640.onnx",
    "crestereo_init_iter10_720x1280.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_init_iter10_720x1280.onnx",
    "crestereo_init_iter20_180x320.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_init_iter20_180x320.onnx",
    "crestereo_init_iter20_240x320.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_init_iter20_240x320.onnx",
    "crestereo_init_iter20_480x640.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_init_iter20_480x640.onnx",
    "crestereo_init_iter20_720x1280.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_init_iter20_720x1280.onnx",
    "crestereo_init_iter2_180x320.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_init_iter2_180x320.onnx",
    "crestereo_init_iter2_240x320.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_init_iter2_240x320.onnx",
    "crestereo_init_iter2_480x640.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_init_iter2_480x640.onnx",
    "crestereo_init_iter2_720x1280.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_init_iter2_720x1280.onnx",
    "crestereo_init_iter5_180x320.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_init_iter5_180x320.onnx",
    "crestereo_init_iter5_240x320.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_init_iter5_240x320.onnx",
    "crestereo_init_iter5_480x640.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_init_iter5_480x640.onnx",
    "crestereo_init_iter5_720x1280.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_init_iter5_720x1280.onnx"
}

# Adapted from https://github.com/ibaiGorordo/ONNX-CREStereo-Depth-Estimation
# https://github.com/PINTO0309/PINTO_model_zoo/tree/main/284_CREStereo
class CREStereo(StereoMethod):
    def __init__(self, config: Config):
        super().__init__("CRE Stereo (CVPR 2022)", 
                         "Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation. Pre-trained on a large range of datasets.", 
                         {},
                         config)
        self.reset_defaults()

        self._loaded_session = None
        self._loaded_model_path = None

    def reset_defaults(self):
        self.parameters.update ({
            "Iterations": EnumParameter("Number of iterations", 1, ["2", "5", "10", "20"]),
            "Mode": EnumParameter("Number of passes. The combined version does 2 passes, one to get an initial estimation and a second one to refine it.",
                                  1, ["init", "combined"]),
            "Shape": EnumParameter("Processed image size", 1, ["320x240", "640x480", "1280x720"])            
        })

    def compute_disparity(self, input: InputPair) -> StereoOutput:
        left_image, right_image = input.left_image, input.right_image
        cols, rows = self.parameters["Shape"].value.split('x')
        version = self.parameters["Mode"].value
        iters = self.parameters["Iterations"].value
        model_path = self.config.models_path / f'crestereo_{version}_iter{iters}_{rows}x{cols}.onnx'
        self._load_model (model_path)

        left_tensor = self._prepare_input(left_image)
        right_tensor = self._prepare_input(right_image)

        # Get the half resolution to calculate flow_init
        if self._has_flow:
            left_tensor_half = self._prepare_input(left_image, half=True)
            right_tensor_half = self._prepare_input(right_image, half=True)
            start = time.time()
            outputs = self._inference_with_flow(left_tensor_half,
                                               right_tensor_half,
                                               left_tensor,
                                               right_tensor)
        else:
            # Estimate the disparity map
            start = time.time()
            outputs = self._inference_without_flow(left_tensor, right_tensor)

        elapsed_time = time.time() - start
        
        disparity_map = self.process_output(outputs)

        if disparity_map.shape[:2] != left_image.shape[:2]:
            disparity_map = cv2.resize (disparity_map, (left_image.shape[1], left_image.shape[0]), cv2.INTER_NEAREST)
            x_scale = left_image.shape[1] / float(cols)
            disparity_map *= np.float32(x_scale)        
        return StereoOutput(disparity_map, input.left_image, elapsed_time)

    def _download_model (self, model_path: Path):
        utils.download_model (urls[model_path.name], model_path)

    def _load_model(self, model_path: Path):
        if (self._loaded_model_path == model_path):
            return
        
        if not model_path.exists():
            self._download_model (model_path)

        assert Path(model_path).exists()
        self._loaded_model_path = model_path
        self._loaded_session = onnxruntime.InferenceSession(str(model_path), providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
        
        # Get model info
        self.load_input_details()
        self.load_output_details()

        # Check if the model has init flow
        self._has_flow = len(self.input_names) > 2

    def _prepare_input(self, img, half=False):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if half:
            img_input = cv2.resize(
                img, (self.input_width//2, self.input_height//2), cv2.INTER_AREA)
        else:
            img_input = cv2.resize(
                img, (self.input_width, self.input_height), cv2.INTER_AREA)
        img_input = img_input.transpose(2, 0, 1)
        img_input = img_input[np.newaxis, :, :, :]
        return img_input.astype(np.float32)

    def _inference_without_flow(self, left_tensor, right_tensor):

        return self._loaded_session.run(self.output_names, {self.input_names[0]: left_tensor,
                                                            self.input_names[1]: right_tensor})[0]

    def _inference_with_flow(self, left_tensor_half, right_tensor_half, left_tensor, right_tensor):

        return self._loaded_session.run(self.output_names, {self.input_names[0]: left_tensor_half,
                                                            self.input_names[1]: right_tensor_half,
                                                            self.input_names[2]: left_tensor,
                                                            self.input_names[3]: right_tensor})[0]

    def process_output(self, output):
        return np.squeeze(output[:, 0, :, :])

    def load_input_details(self):
        model_inputs = self._loaded_session.get_inputs()
        self.input_names = [
            model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[-1].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def load_output_details(self):
        model_outputs = self._loaded_session.get_outputs()
        self.output_names = [
            model_outputs[i].name for i in range(len(model_outputs))]

        self.output_shape = model_outputs[0].shape
