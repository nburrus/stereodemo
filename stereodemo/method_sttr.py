from pathlib import Path
import shutil
import time
from dataclasses import dataclass
import urllib.request
import gc
import tempfile
import re
import sys

import torch
from torchvision import transforms

import cv2
import numpy as np

from .methods import Config, EnumParameter, StereoMethod, InputPair, StereoOutput
from . import utils

urls = {
    "sttr-kitti-cpu-240x320-ds1.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-sttr/sttr-kitti-cpu-240x320-ds1.scripted.pt",
    "sttr-kitti-cpu-480x640-ds2.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-sttr/sttr-kitti-cpu-480x640-ds2.scripted.pt",
    "sttr-kitti-cpu-480x640-ds3.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-sttr/sttr-kitti-cpu-480x640-ds3.scripted.pt",
    "sttr-kitti-cpu-720x1280-ds3.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-sttr/sttr-kitti-cpu-720x1280-ds3.scripted.pt",
    "sttr-kitti-cuda-240x320-ds1.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-sttr/sttr-kitti-cuda-240x320-ds1.scripted.pt",
    "sttr-kitti-cuda-480x640-ds2.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-sttr/sttr-kitti-cuda-480x640-ds2.scripted.pt",
    "sttr-kitti-cuda-480x640-ds3.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-sttr/sttr-kitti-cuda-480x640-ds3.scripted.pt",
    "sttr-kitti-cuda-720x1280-ds3.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-sttr/sttr-kitti-cuda-720x1280-ds3.scripted.pt",

    "sttr-sceneflow-cpu-240x320-ds1.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-sttr/sttr-sceneflow-cpu-240x320-ds1.scripted.pt",
    "sttr-sceneflow-cpu-480x640-ds2.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-sttr/sttr-sceneflow-cpu-480x640-ds2.scripted.pt",
    "sttr-sceneflow-cpu-480x640-ds3.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-sttr/sttr-sceneflow-cpu-480x640-ds3.scripted.pt",
    "sttr-sceneflow-cpu-720x1280-ds3.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-sttr/sttr-sceneflow-cpu-720x1280-ds3.scripted.pt",
    "sttr-sceneflow-cuda-240x320-ds1.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-sttr/sttr-sceneflow-cuda-240x320-ds1.scripted.pt",
    "sttr-sceneflow-cuda-480x640-ds2.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-sttr/sttr-sceneflow-cuda-480x640-ds2.scripted.pt",
    "sttr-sceneflow-cuda-480x640-ds3.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-sttr/sttr-sceneflow-cuda-480x640-ds3.scripted.pt",
    "sttr-sceneflow-cuda-720x1280-ds3.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-sttr/sttr-sceneflow-cuda-720x1280-ds3.scripted.pt",
}

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()    

# https://github.com/mli0603/stereo-transformer
# Made some changes to allow torchscript tracing:
# https://github.com/nburrus/stereo-transformer/commit/0006a022c19f0c7c4d7683408531180a863603a5
class StereoTransformers(StereoMethod):
    def __init__(self, config: Config):
        super().__init__("STereo TRansformer (ICCV 2021)",
                         "Revisiting Stereo Depth Estimation From a Sequence-to-Sequence Perspective with Transformers.",
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
            "Shape": EnumParameter("Processed image size", 2, ["320x240 (ds1)", "640x480 (ds2)", "640x480 (ds3)", "1280x720 (ds3)"]),
            # "Model": EnumParameter("Pre-trained Model", 0, ["kitti-cpu", "sceneflow-cpu", "kitti-cuda", "sceneflow-cuda"])
            # The CUDA ones segfault with my Python 3.8 venv, but someone worked with Python 3.7.
            # Maybe related to the installed packages instead, need to investigate more.
            # Keeping only the CPU ones for now since it's enough to evaluate.
            "Model": EnumParameter("Pre-trained Model", 0, ["kitti-cpu", "sceneflow-cpu"])
        })

    def compute_disparity(self, input: InputPair) -> StereoOutput:
        stereo_output = self._compute_disparity (input)
        clear_gpu_memory ()
        return stereo_output

    def _compute_disparity(self, input: InputPair) -> StereoOutput:
        m = re.match("(\d+)x(\d+) \(ds(\d)\)", self.parameters["Shape"].value)
        cols, rows, downsample = map(lambda v: int(v), m.groups())
        self.target_size = (cols, rows)

        variant = self.parameters["Model"].value
        
        model_path = self.config.models_path / f'sttr-{variant}-{rows}x{cols}-ds{downsample}.scripted.pt'
        self._load_model (model_path)

        left_tensor = self._preprocess_input(input.left_image)
        right_tensor = self._preprocess_input(input.right_image)

        col_offset = int(downsample / 2)
        row_offset = int(downsample / 2)
        sampled_cols = torch.arange(col_offset, cols, downsample)[None,]
        sampled_rows = torch.arange(row_offset, rows, downsample)[None,]

        device = torch.device('cuda') if 'cuda' in variant else 'cpu'
        net = self.net.to(device)
        left_tensor = left_tensor.to(device)
        right_tensor = right_tensor.to(device)
        sampled_cols = sampled_cols.to(device)
        sampled_rows = sampled_rows.to(device)

        start = time.time()
        with torch.no_grad():
            outputs = net(left_tensor, right_tensor, sampled_cols, sampled_rows)
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
        disparity_map = outputs[0][0].detach().cpu().numpy()
        return disparity_map

    def _load_model(self, model_path: Path):
        # FIXME: always reload the model, for some reason
        # feeding multiple images to the same model freezes
        # with CUDA. Maybe due to multi-threading?
        # if (self._loaded_model_path == model_path):
        #     return
        
        if not model_path.exists():
            utils.download_model (urls[model_path.name], model_path)

        assert Path(model_path).exists()
        self._loaded_model_path = model_path
        self.net = torch.jit.load(model_path)
        self.net.eval ()
