from pathlib import Path

import numpy as np

import tempfile
import urllib.request
import shutil
import sys

def download_model (url: str, model_path: Path):
    filename = model_path.name
    with tempfile.TemporaryDirectory() as d:
        tmp_file_path = Path(d) / filename
        print (f"Downloading {filename} from {url} to {model_path}...")
        urllib.request.urlretrieve(url, tmp_file_path)
        shutil.move (tmp_file_path, model_path)

def pad_width (size: int, multiple: int):
    return 0 if size % multiple == 0 else multiple - (size%multiple)

class ImagePadder:
    def __init__(self, multiple, mode):
        self.multiple = multiple
        self.mode = mode
    
    def pad (self, im: np.ndarray):
        # H,W,C
        rows = im.shape[0]
        cols = im.shape[1]
        self.rows_to_pad = pad_width(rows, self.multiple)
        self.cols_to_pad = pad_width(cols, self.multiple)
        if self.rows_to_pad == 0 and self.cols_to_pad == 0:
            return im
        return np.pad (im, ((0, self.rows_to_pad), (0, self.cols_to_pad), (0, 0)), mode=self.mode)

    def unpad (self, im: np.ndarray):        
        w = im.shape[1] - self.cols_to_pad
        h = im.shape[0] - self.rows_to_pad
        return im[:h, :w, :]
