from pathlib import Path
import shutil
import time
from dataclasses import dataclass
import urllib.request
import gc
import tempfile
import sys

import torch
from torchvision import transforms

import cv2
import numpy as np

from .methods import Config, EnumParameter, StereoMethod, InputPair, StereoOutput
from . import utils

urls = {
    "raft-stereo-eth3d-cpu-128x160.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-eth3d-cpu-128x160.scripted.pt",
    "raft-stereo-eth3d-cpu-256x320.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-eth3d-cpu-256x320.scripted.pt",
    "raft-stereo-eth3d-cpu-480x640.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-eth3d-cpu-480x640.scripted.pt",
    "raft-stereo-eth3d-cpu-736x1280.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-eth3d-cpu-736x1280.scripted.pt",
    "raft-stereo-eth3d-cuda-128x160.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-eth3d-cuda-128x160.scripted.pt",
    "raft-stereo-eth3d-cuda-256x320.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-eth3d-cuda-256x320.scripted.pt",
    "raft-stereo-eth3d-cuda-480x640.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-eth3d-cuda-480x640.scripted.pt",
    "raft-stereo-eth3d-cuda-736x1280.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-eth3d-cuda-736x1280.scripted.pt",
    "raft-stereo-fast-cpu-128x160.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-fast-cpu-128x160.scripted.pt",
    "raft-stereo-fast-cpu-256x320.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-fast-cpu-256x320.scripted.pt",
    "raft-stereo-fast-cpu-480x640.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-fast-cpu-480x640.scripted.pt",
    "raft-stereo-fast-cpu-736x1280.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-fast-cpu-736x1280.scripted.pt",
    "raft-stereo-fast-cuda-128x160.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-fast-cuda-128x160.scripted.pt",
    "raft-stereo-fast-cuda-256x320.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-fast-cuda-256x320.scripted.pt",
    "raft-stereo-fast-cuda-480x640.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-fast-cuda-480x640.scripted.pt",
    "raft-stereo-fast-cuda-736x1280.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-fast-cuda-736x1280.scripted.pt",
    "raft-stereo-middlebury-cpu-128x160.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-middlebury-cpu-128x160.scripted.pt",
    "raft-stereo-middlebury-cpu-256x320.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-middlebury-cpu-256x320.scripted.pt",
    "raft-stereo-middlebury-cpu-480x640.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-middlebury-cpu-480x640.scripted.pt",
    "raft-stereo-middlebury-cpu-736x1280.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-middlebury-cpu-736x1280.scripted.pt",
    "raft-stereo-middlebury-cuda-128x160.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-middlebury-cuda-128x160.scripted.pt",
    "raft-stereo-middlebury-cuda-256x320.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-middlebury-cuda-256x320.scripted.pt",
    "raft-stereo-middlebury-cuda-480x640.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-middlebury-cuda-480x640.scripted.pt",
    "raft-stereo-middlebury-cuda-736x1280.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-middlebury-cuda-736x1280.scripted.pt",
}

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()    

# https://github.com/princeton-vl/RAFT-Stereo
# I exported the pytorch implementation to torch script via tracing, with minor modifications of the source code.
# https://github.com/nburrus/RAFT-Stereo/commit/ebbb5a807227927ab4551274039e9bdd16a1b010
# Their fastest implementation was not imported.
class RaftStereo(StereoMethod):
    def __init__(self, config: Config):
        super().__init__("RAFT-Stereo (3DV 2021)",
                         "RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching.",
                         {},
                         config)
        self.reset_defaults()

        self.net = None
        self._loaded_model_path = None

    def reset_defaults(self):
        self.parameters.update ({
            "Shape": EnumParameter("Processed image size", 2, ["160x128", "320x256", "640x480", "1280x736"]),
            # "Model": EnumParameter("Pre-trained Model", 1, ["eth3d-cuda", "eth3d-cpu", "fast-cuda", "fast-cpu", "middlebury-cuda"])
            # The eth3d and fast cuda models required --corr_implementation alt to work once loaded via torchscript.
            # The supposedly faster "reg" is not working with a torch/cuda segfault, not sure why.
            "Model": EnumParameter("Pre-trained Model", 0, ["fast-cpu", "middlebury-cpu", "eth3d-cpu", "fast-cuda", "middlebury-cuda", "eth3d-cuda"])
        })

    def compute_disparity(self, input: InputPair) -> StereoOutput:
        stereo_output = self._compute_disparity (input)
        clear_gpu_memory ()
        return stereo_output

    def _compute_disparity(self, input: InputPair) -> StereoOutput:
        cols, rows = self.parameters["Shape"].value.split('x')
        cols, rows = int(cols), int(rows)
        self.target_size = (cols, rows)

        variant = self.parameters["Model"].value
        
        model_path = self.config.models_path / f'raft-stereo-{variant}-{rows}x{cols}.scripted.pt'
        self._load_model (model_path)

        left_tensor = self._preprocess_input(input.left_image)
        right_tensor = self._preprocess_input(input.right_image)

        device = torch.device('cuda') if 'cuda' in variant else 'cpu'
        net = self.net.to(device)
        left_tensor = left_tensor.to(device)
        right_tensor = right_tensor.to(device)

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
        # -> C,H,W
        # Normalization done in the model itself.
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

    def _process_output(self, outputs):
        disparity_map = outputs[1][0].detach().cpu().squeeze(0).squeeze(0).numpy() * -1.0
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
