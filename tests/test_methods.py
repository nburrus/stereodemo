#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import math

from stereodemo import method_opencv_bm
from stereodemo import method_chang_realtime_stereo
from stereodemo import method_hitnet
from stereodemo import method_cre_stereo
from stereodemo import method_raft_stereo
from stereodemo import method_sttr
from stereodemo.methods import Config, InputPair, Calibration, StereoOutput, StereoMethod

data_folder = Path(__file__).parent.parent / 'datasets' / 'eth3d_lowres' / 'delivery_area_1l'
left_image = cv2.imread (str(data_folder / 'im0.png'), cv2.IMREAD_COLOR)
right_image = cv2.imread (str(data_folder / 'im1.png'), cv2.IMREAD_COLOR)
calib_file = data_folder / 'stereodemo_calibration.json'
calibration = Calibration.from_json (open(calib_file, 'r').read())
input = InputPair (left_image, right_image, calibration, "Test images loaded.", None)
models_path = Path(tempfile.gettempdir()) / 'models'
models_path.mkdir(parents=True, exist_ok=True)
config = Config(models_path)

class TestOpenCVBM(unittest.TestCase):

    def check_method(self, method: StereoMethod, expected_median: float, expected_coverage: float):
        output = method.compute_disparity (input)
        valid_pixels = output.disparity_pixels[output.disparity_pixels > 0.]
        coverage = valid_pixels.size / output.disparity_pixels.size
        median_value =  np.median (valid_pixels)
        self.assertAlmostEqual (median_value, expected_median, delta=0.01)
        self.assertAlmostEqual (coverage, expected_coverage, delta=0.01)

    def test_bm(self):
        self.check_method (method_opencv_bm.StereoBM(config), 4.8125, 0.4403)

    def test_sgbm(self):
        self.check_method (method_opencv_bm.StereoSGBM(config), 5.1875, 0.8515)

    def test_chang_realtime(self):
        m = method_chang_realtime_stereo.ChangRealtimeStereo(config)
        m.parameters["Shape"].set_value ("320x240")
        self.check_method (m, 12.7776, 1.0)

    def test_hitnet(self):
        m = method_hitnet.HitnetStereo(config)
        m.parameters["Shape"].set_value ("320x240")
        self.check_method (m, 4.9103, 1.0)

    def test_crestereo(self):
        m = method_cre_stereo.CREStereo(config)
        m.parameters["Shape"].set_value ("320x240")
        self.check_method (m, 4.6287, 1.0)

    def test_raft_stereo(self):
        m = method_raft_stereo.RaftStereo(config)
        m.parameters["Shape"].set_value ("320x256")
        self.check_method (m, 4.6408, 1.0)

    def test_sttr(self):
        m = method_sttr.StereoTransformers(config)
        m.parameters["Shape"].set_value ("640x480 (ds3)")
        self.check_method (m, 7.4198, 1.0)

if __name__ == '__main__':
    unittest.main()
