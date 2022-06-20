from pathlib import Path
import shutil
import time
from dataclasses import dataclass
import urllib.request
import tempfile
import sys

import torch
from torchvision import transforms

import cv2
import numpy as np

from .methods import Calibration, Config, EnumParameter, StereoMethod, InputPair, StereoOutput
from . import utils

urls = {
    "dist-depth-256x256.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-distdepth/dist-depth-256x256.scripted.pt",
}

# https://github.com/facebookresearch/DistDepth
# Exported via torch tracing by tweaking the original demo.py.
# Changes here: https://github.com/nburrus/DistDepth/commit/fde3b427ef2ff31c34f08e99c51c8e6a2427b720
class DistDepth(StereoMethod):
    def __init__(self, config: Config):
        super().__init__("[Monocular] DistDepth (CVPR 2022)",
                         "Toward Practical Monocular Indoor Depth Estimation.",
                         {},
                         config)
        self.reset_defaults()

        self.net = None
        self._loaded_model_path = None

    def reset_defaults(self):
        self.parameters.update ({
            # "Device": EnumParameter("Device", 0, ["CPU", "CUDA"]),
            # For some reason it crashes with CUDA on my machine, disabling for now.
            "Device": EnumParameter("Device", 0, ["CPU"]),
        })

    def compute_disparity(self, input: InputPair) -> StereoOutput:
        # The pre-trained model is for 256x256. Their demo script resizes
        # all input images to that.
        self.target_size = (256, 256)
        device = torch.device('cuda') if self.parameters["Device"].value == 'CUDA' else 'cpu'
        
        model_path = self.config.models_path / f'dist-depth-256x256.scripted.pt'
        self._load_model (model_path)

        # raw_img can stay in BGR
        raw_img = np.transpose(input.left_image, (2, 0, 1))
        input_image = torch.from_numpy(raw_img).float().to(device)
        input_image = (input_image / 255.0).unsqueeze(0)
        input_image = torch.nn.functional.interpolate(
            input_image, (256, 256), mode="bilinear", align_corners=False
        )

        net = self.net.to(device)

        start = time.time()
        with torch.no_grad():
            outputs = net(input_image.to(device))
        elapsed_time = time.time() - start

        disparity_map = self._process_output(outputs, input.calibration)
        if disparity_map.shape[:2] != input.left_image.shape[:2]:
            disparity_map = cv2.resize (disparity_map, (input.left_image.shape[1], input.left_image.shape[0]), cv2.INTER_NEAREST)
            # not need to scale, the disparity values were already for the input full resolution calibration.

        return StereoOutput(disparity_map, input.left_image, elapsed_time)

    def _process_output(self, outputs, calib: Calibration):
        depth_meters = outputs[0].detach().squeeze(0).cpu().numpy()
        # The model directly gives a depth map in meters. Let's convert it
        # to disparity to fit in the stereo display.
        disparity_map = StereoMethod.disparity_from_depth_meters(depth_meters, calib)
        return disparity_map

    def _load_model(self, model_path: Path):
        if (self._loaded_model_path == model_path):
            return
        
        if not model_path.exists():
            utils.download_model (urls[model_path.name], model_path)

        assert Path(model_path).exists()
        self._loaded_model_path = model_path
        self.net = torch.jit.load(model_path)
        self.net.cpu ()
        self.net.eval ()
