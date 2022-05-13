

from abc import abstractmethod
from dataclasses import dataclass

from typing import Dict, List

import numpy as np
import cv2

@dataclass
class IntParameter:
    description: str
    value: int
    min: int
    max: int

class StereoMethod:
    def __init__(self, name: str, description: str, parameters: Dict):
        self.name = name
        self.parameters = parameters
        self.description = description

    @abstractmethod    
    def compute_disparity(self, left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
        """Return the disparity map in pixels.
        
        Both input images are assumed to be rectified.
        """
        pass

class StereoBMMethod(StereoMethod):
    def __init__(self):
        params = {
            "Num Disparities": IntParameter("Number of disparities", 64, 2, 256),
            "Block Size": IntParameter("Kernel size for block matching", 15, 3, 63)
        }

        super().__init__("BM", "OpenCV Simple Block Matching", params)

    def compute_disparity(self, left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
        block_size = self.parameters['Block Size'].value
        if block_size % 2 == 0:
            block_size += 1
        stereoBM = cv2.StereoBM_create(numDisparities=self.parameters['Num Disparities'].value, 
                                       blockSize=block_size)
        gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY, left_image)
        gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY, right_image)
        disparity = stereoBM.compute(gray_left, gray_right)
        return disparity

class StereoSGBMMethod(StereoMethod):
    def __init__(self):
        params = {
            "Num Disparities": IntParameter("Number of disparities", 64, 2, 256),
            "Block Size": IntParameter("Kernel size for block matching", 3, 3, 21),
        }
        super().__init__("SGBM", "OpenCV Semi-Global Block Matching", params)

    def compute_disparity(self, left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
        stereoSGBM = cv2.StereoSGBM_create(numDisparities=self.parameters['Num Disparities'].value, 
                                         blockSize=self.parameters['Block Size'].value)
        gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY, left_image)
        gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY, right_image)
        disparity = stereoSGBM.compute(gray_left, gray_right)
        return disparity
