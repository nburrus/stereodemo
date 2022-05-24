from abc import abstractmethod
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

import cv2

from . import visualizer
from . import methods

from .method_opencv_bm import StereoBM, StereoSGBM
from .method_raft_stereo import RaftStereo
from .method_cre_stereo import CREStereo
from .method_chang_realtime_stereo import ChangRealtimeStereo
from .method_hitnet import HitnetStereo
from .method_sttr import StereoTransformers

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--oak', action='store_true', help='Use an oak-D camera to grab images.')
    parser.add_argument('--oak-output-folder', type=Path, default=None, help='Output folder to save the images grabbed by the OAK camera')
    parser.add_argument('images',
                        help='rectified_left1 rectified_right1 ... [rectified_leftN rectified_rightN]. Load image pairs from disk. You can also specify folders.',
                        type=Path, 
                        default=None,
                        nargs='*')
    parser.add_argument('--calibration', type=Path, help='Calibration json. If unspecified, it will try to load a stereodemo_calibration.json file in the left image parent folder.', default=None)
    default_models_path = Path.home() / ".cache" / "stereodemo" / "models"
    parser.add_argument('--models-path', type=Path, help='Path to store the downloaded models.', default=default_models_path)
    return parser.parse_args()

def find_stereo_images_in_dir(dir: Path):
    left_files = []
    right_files = []

    def validated_lists():
        for f in right_files:
            assert f.exists()
        return left_files, right_files

    for ext in ['jpg', 'png']:
        left = sorted(list(dir.glob(f'**/*left*.{ext}')))
        if len(left) != 0:
            right = [f.parent / f.name.replace('left', 'right') for f in left]
            left_files += left
            right_files += right

    for ext in ['jpg', 'png']:
        left = sorted(list(dir.glob(f'**/im0.{ext}')))
        if len(left) != 0:
            right = [f.parent / f.name.replace('im0', 'im1') for f in left]
            left_files += left
            right_files += right            
    
    return validated_lists()

class FileListSource (visualizer.Source):
    def __init__(self, file_or_dir_list, calibration=None):        
        self.left_images_path = []
        self.right_images_path = []

        while file_or_dir_list:
            f = file_or_dir_list.pop(0)
            if f.is_dir():
                left, right = find_stereo_images_in_dir (f)
                self.left_images_path += left
                self.right_images_path += right
            else:
                if f.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                    print (f"Warning: ignoring {f}, not an image extension.")
                    continue
                try:
                    right_f = file_or_dir_list.pop(0)
                except:
                    print (f"Missing right image for {f}, skipping")
                    continue
                self.left_images_path.append(f)
                self.right_images_path.append(right_f)
        
        self.index = 0
        self.user_provided_calibration_path = calibration
        self.num_pairs = len(self.left_images_path)
        if self.num_pairs == 0:
            raise Exception("No image pairs.")

    def get_next_pair(self):

        if self.index >= self.num_pairs:
            self.index = 0

        def load_image(path):
            im =  cv2.imread(str(path), cv2.IMREAD_COLOR)
            assert im is not None
            return im

        left_image_path = self.left_images_path[self.index]
        left_image = load_image(left_image_path)
        if self.user_provided_calibration_path is None:
            calibration_path = left_image_path.parent / 'stereodemo_calibration.json'
            if not calibration_path.exists():
                print (f"Warning: no calibration file found {calibration_path}. Using default calibration, the point cloud won't be accurate.")
                calibration_path = None
        else:
            calibration_path = self.user_provided_calibration_path
        if calibration_path:
            calib = visualizer.Calibration.from_json (open(calibration_path, 'r').read())
        else:
            # Fake reasonable calibration.
            calib = visualizer.Calibration(left_image.shape[1],
                                           left_image.shape[0],
                                           left_image.shape[0]*0.8,
                                           left_image.shape[0]*0.8,
                                           left_image.shape[1]/2.0, # cx0
                                           left_image.shape[1]/2.0, # cx1
                                           left_image.shape[0]/2.0,
                                           0.075)
            
        right_image_path = self.right_images_path[self.index]
        self.index += 1
        status = f"{left_image_path} / {right_image_path}"
        return visualizer.InputPair (left_image, load_image(right_image_path), calib, status)

def main():
    args = parse_args()

    try:
        args.models_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        sys.stderr.write (f"Warning: cannot use the default models path {args.models_path}: {e}\n")
        sys.stderr.write ("A valid path is necessary to store the downloaded models.\n")
        args.models_path = Path(tempfile.gettempdir()) / 'stereodemo_models'
        sys.stderr.write (f"Going to use the temporary directory {args.models_path} instead, specify --model-paths to specify a custom persistent path instead.\n")
        try:
            args.models_path.mkdir(parents=True, exist_ok=True)
        except:
            sys.stderr.write ("Could not create a temporary directory to store the downloaded models.\n")
            sys.stderr.write ("Aborting, you need to specify --models-path with a valid writable path.\n")
            sys.exit (1)
    print (f"INFO: will store downloaded models in {args.models_path}")

    config = methods.Config(args.models_path)
    method_list = [
        StereoBM(config),
        StereoSGBM(config),
        CREStereo(config),
        RaftStereo(config),
        HitnetStereo(config),
        StereoTransformers(config),
        ChangRealtimeStereo(config),
    ]

    if args.images:
        source = FileListSource(args.images, args.calibration)
    elif args.oak:
        from .oakd_source import OakdSource, StereoFromOakInputSource
        source = OakdSource(args.oak_output_folder)
        method_list = [StereoFromOakInputSource(config)] + method_list
    else:
        print ("You need to specify --oak or provide images")
        sys.exit (1)

    method_dict = { method.name:method for method in method_list } 

    viz = visualizer.Visualizer(method_dict, source)

    while True:
        start_time = time.time()
        if not viz.update_once ():
            break
        cv2.waitKey (1)
        elapsed = time.time() - start_time
        time_to_sleep = 1/30.0 - elapsed
        if time_to_sleep > 0:
            time.sleep (time_to_sleep)


