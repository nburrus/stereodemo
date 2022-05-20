import time

import numpy as np
import cv2

from .methods import Config, StereoMethod, IntParameter, EnumParameter, InputPair, StereoOutput

def odd_only(x):
    return x if x % 2 == 1 else x+1

def multiple_of_16(x):
    return max(16, x//16 * 16)

# Default parameters taken from 
# https://github.com/opencv/opencv/blob/4.x/samples/cpp/stereo_match.cpp
class StereoBM(StereoMethod):
    def __init__(self, config: Config):
        super().__init__("OpenCV BM", "OpenCV Simple Block Matching", {}, config)
        self.reset_defaults()

    def reset_defaults(self):
        # For more details:
        # https://learnopencv.com/depth-perception-using-stereo-camera-python-c/
        self.parameters.update ({
            "Num Disparities": IntParameter("Number of disparities (pixels)", 128, 16, 640, to_valid=multiple_of_16),
            "Block Size": IntParameter("Kernel size for block matching (odd)", 9, 3, 63, to_valid=odd_only),
            "TextureThreshold": IntParameter("Minimum SAD to consider the texture sufficient", 10, 0, 100),
            "Uniqueness Ratio": IntParameter("How unique the match each for each pixel", 15, 0, 100),
            "SpeckleWindowSize": IntParameter("Speckle window size in pixels (filter small objects). 0 to disable.", 100, 0, 1000),
            "SpeckleRange": IntParameter("Speckle range (max diff within a window)", 32, 0, 64),
            "Disp12MaxDiff": IntParameter("Maximum allowable difference in the right-left check", 1, 0, 64),
            "PreFilterCap": IntParameter("Max pre-filter output", 31, 1, 63),
            "PreFilterSize": IntParameter("Pre-filter size (odd)", 9, 5, 255, to_valid=odd_only),
        })

    def compute_disparity(self, input: InputPair) -> StereoOutput:
        left_image, right_image = input.left_image, input.right_image
        block_size = self.parameters['Block Size'].value
        if block_size % 2 == 0:
            block_size += 1
        stereoBM = cv2.StereoBM_create(numDisparities=self.parameters['Num Disparities'].value, 
                                       blockSize=block_size)
        stereoBM.setTextureThreshold(self.parameters['TextureThreshold'].value)
        stereoBM.setUniquenessRatio(self.parameters['Uniqueness Ratio'].value)
        stereoBM.setSpeckleWindowSize(self.parameters['SpeckleWindowSize'].value)
        stereoBM.setSpeckleRange(self.parameters['SpeckleRange'].value)
        stereoBM.setDisp12MaxDiff(self.parameters['Disp12MaxDiff'].value)
        stereoBM.setPreFilterCap(self.parameters['PreFilterCap'].value)
        stereoBM.setPreFilterSize(self.parameters['PreFilterSize'].value)

        gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY, left_image)
        gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY, right_image)
        # OpenCV returns 16x the disparity in pixels
        start = time.time()
        disparity = stereoBM.compute(gray_left, gray_right) / np.float32(16.0)
        return StereoOutput(disparity, input.left_image, time.time()-start)

class StereoSGBM(StereoMethod):
    def __init__(self, config: Config):
        super().__init__("OpenCV SGBM", "OpenCV Semi-Global Block Matching", {}, config)
        self.reset_defaults ()

    def reset_defaults(self):
        nchannels = 1
        # For more details:
        # https://learnopencv.com/depth-perception-using-stereo-camera-python-c/
        self.parameters.update ({
            "Num Disparities": IntParameter("Number of disparities (pixels)", 128, 2, 640),
            "Block Size": IntParameter("Kernel size for block matching (odd)", 3, 3, 63, to_valid=odd_only),
            
            "Mode": EnumParameter("Set it to StereoSGBM::MODE_HH to run the full-scale two-pass dynamic programming algorithm. It will consume O(W*H*numDisparities) bytes, which is large for 640x480 stereo and huge for HD-size pictures. By default, it is set to false .",
                                  0, ["MODE_SGBM", "MODE_HH", "MODE_SGBM_3WAY", "MODE_HH4"]),
            
            
            "P1": IntParameter("Penalty Cost (default=8*NChannels*BlockSize)", 8*nchannels*3*3, 0, 2000),
            "P2": IntParameter("Penalty Cost. Must be > P1 (default=32*NChannels*BlockSize).", 32*nchannels*3*3, 0, 2000),
            
            "Uniqueness Ratio": IntParameter("How unique the match each for each pixel", 10, 0, 100),
            "SpeckleWindowSize": IntParameter("Speckle window size in pixels (filter small objects). 0 to disable.", 100, 0, 1000),
            "SpeckleRange": IntParameter("Speckle range (max diff within a window)", 32, 0, 64),
            
            "Disp12MaxDiff": IntParameter("Maximum allowable difference in the right-left check", 1, 0, 64),
            "PreFilterCap": IntParameter("Max pre-filter output", 63, 1, 128),
        })

    def compute_disparity(self, input: InputPair) -> StereoOutput:
        left_image, right_image = input.left_image, input.right_image
        stereoSGBM = cv2.StereoSGBM_create(numDisparities=self.parameters['Num Disparities'].value, 
                                         blockSize=self.parameters['Block Size'].value)

        stereoSGBM.setMode(self.parameters['Mode'].index)
        stereoSGBM.setP1(self.parameters['P1'].value)
        stereoSGBM.setP2(self.parameters['P2'].value)
        stereoSGBM.setPreFilterCap(self.parameters['PreFilterCap'].value)
        stereoSGBM.setUniquenessRatio(self.parameters['Uniqueness Ratio'].value)
        stereoSGBM.setSpeckleWindowSize(self.parameters['SpeckleWindowSize'].value)
        stereoSGBM.setSpeckleRange(self.parameters['SpeckleRange'].value)
        stereoSGBM.setDisp12MaxDiff(self.parameters['Disp12MaxDiff'].value)

        gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY, left_image)
        gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY, right_image)
        # OpenCV returns 16x the disparity in pixels
        start = time.time()
        disparity = stereoSGBM.compute(gray_left, gray_right) / np.float32(16.0)
        return StereoOutput(disparity, input.left_image, time.time()-start)
