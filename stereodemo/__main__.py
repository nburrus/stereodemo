import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np

import cv2

from . import visualizer
from . import methods

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('left_image', type=Path, help='Left image (rectified)')
    parser.add_argument('right_image', type=Path, help='Right image (rectified)')
    parser.add_argument('calibration', type=Path, help='Calibration json')
    return parser.parse_args()

def main():
    method_list = [
        methods.StereoBMMethod(),
        methods.StereoSGBMMethod(),
    ]

    args = parse_args()

    left_img = cv2.imread(str(args.left_image), cv2.IMREAD_COLOR)
    right_img = cv2.imread(str(args.right_image), cv2.IMREAD_COLOR)

    with open(args.calibration, 'r') as f:
        d = json.loads(f.read())
        calib = SimpleNamespace(**d)

    method_dict = { method.name:method for method in method_list }    

    viz = visualizer.Visualizer(method_dict)
    viz.set_input (left_img, right_img, calib)
    cv2.imshow ("Input image", np.hstack([left_img, right_img]))
    while True:
        if not viz.update_once ():
            break
        time.sleep (1/30.0)
        cv2.waitKey (1)

main()
