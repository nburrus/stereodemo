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

from .methods import Config, EnumParameter, StereoMethod, InputPair, StereoOutput
from . import utils

urls = {
    "chang-realtime-stereo-cpu-1280x720.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-chang-realtimestereo/chang-realtime-stereo-cpu-1280x720.scripted.pt",
    "chang-realtime-stereo-cpu-160x128.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-chang-realtimestereo/chang-realtime-stereo-cpu-160x128.scripted.pt",
    "chang-realtime-stereo-cpu-320x240.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-chang-realtimestereo/chang-realtime-stereo-cpu-320x240.scripted.pt",
    "chang-realtime-stereo-cpu-640x480.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-chang-realtimestereo/chang-realtime-stereo-cpu-640x480.scripted.pt",
}

# https://github.com/JiaRenChang/RealtimeStereo
# I exported the pytorch implementation to torch script via tracing with
# some minor changes to the code https://github.com/JiaRenChang/RealtimeStereo/pull/15
# See chang_realtimestereo_to_torchscript_onnx.py
class ChangRealtimeStereo(StereoMethod):
    def __init__(self, config: Config):
        super().__init__("Chang Real-time (ACCV 2020)",
                         "Attention-Aware Feature Aggregation for Real-time Stereo Matching on Edge Devices. Pre-trained on SceneFlow + Kitti 2015.",
                         {},
                         config)
        self.reset_defaults()

        self.net = None
        self._loaded_model_path = None

        imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        self.img_to_tensor_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**imagenet_stats),
        ])

    def reset_defaults(self):
        self.parameters.update ({
            "Shape": EnumParameter("Processed image size", 2, ["160x128", "320x240", "640x480", "1280x720"])
        })

    def compute_disparity(self, input: InputPair) -> StereoOutput:
        cols, rows = self.parameters["Shape"].value.split('x')
        cols, rows = int(cols), int(rows)
        self.target_size = (cols, rows)

        model_path = self.config.models_path / f'chang-realtime-stereo-cpu-{cols}x{rows}.scripted.pt'
        self._load_model (model_path)

        left_tensor = self._preprocess_input(input.left_image)
        right_tensor = self._preprocess_input(input.right_image)

        start = time.time()
        with torch.no_grad():
            outputs = self.net(left_tensor, right_tensor)
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
        return self.img_to_tensor_transforms (img).unsqueeze(0)

    def _process_output(self, outputs):
        disparity_map = outputs[0].detach().cpu().permute(1,2,0).numpy()
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
