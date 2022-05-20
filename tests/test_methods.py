#!/usr/bin/env python3

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
from stereodemo.methods import InputPair, Calibration, StereoOutput, StereoMethod

data_folder = Path(__file__).parent.parent / 'datasets' / 'eth3d_lowres' / 'delivery_area_1l'
left_image = cv2.imread (str(data_folder / 'im0.png'), cv2.IMREAD_COLOR)
right_image = cv2.imread (str(data_folder / 'im1.png'), cv2.IMREAD_COLOR)
calib_file = data_folder / 'stereodemo_calibration.json'
calibration = Calibration.from_json (open(calib_file, 'r').read())
input = InputPair (left_image, right_image, calibration, "Test images loaded.", None)

class TestOpenCVBM(unittest.TestCase):

    def check_method(self, method: StereoMethod, expected_median: float, expected_coverage: float):
        output = method.compute_disparity (input)
        valid_pixels = output.disparity_pixels[output.disparity_pixels > 0.]
        coverage = valid_pixels.size / output.disparity_pixels.size
        median_value =  np.median (valid_pixels)
        self.assertAlmostEqual (median_value, expected_median, delta=0.01)
        self.assertAlmostEqual (coverage, expected_coverage, delta=0.01)

    def test_bm(self):
        self.check_method (method_opencv_bm.StereoBM(), 4.8125, 0.4403)

    def test_sgbm(self):
        self.check_method (method_opencv_bm.StereoSGBM(), 5.1875, 0.8515)

    def test_chang_realtime(self):
        self.check_method (method_chang_realtime_stereo.ChangRealtimeStereo(), 8.5121, 1.0)

    def test_hitnet(self):
        self.check_method (method_hitnet.HitnetStereo(), 5.1079, 1.0)

    def test_crestereo(self):
        self.check_method (method_cre_stereo.CREStereo(), 4.6947, 1.0)

    def test_raft_stereo(self):
        self.check_method (method_raft_stereo.RaftStereo(), 4.7230, 1.0)


if __name__ == '__main__':
    unittest.main()
