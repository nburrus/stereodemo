from pathlib import Path
import sys
import time

from . import visualizer
from .methods import Config, InputPair, StereoMethod, StereoOutput

import cv2

try:
    import depthai as dai
except ImportError:
    print ("You need to 'pip install depthai' to use the OAK camera.")
    sys.exit (1)

def getFrame(queue):
  # Get frame from queue
  frame = queue.get()
  # Convert frame to OpenCV format and return
  return frame.getCvFrame()

def getMonoCamera(pipeline, isLeft):
  # Configure mono camera
  mono = pipeline.createMonoCamera()

  # Set Camera Resolution
  mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

  if isLeft:
      # Get left camera
      mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
  else :
      # Get right camera
      mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
  return mono

def getStereoPair(pipeline, monoLeft, monoRight):
    # Configure stereo pair for depth estimation
    stereo = pipeline.createStereoDepth()
    # Checks occluded pixels and marks them as invalid
    stereo.setLeftRightCheck(True)
    
    # Configure left and right cameras to work as a stereo pair
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    return stereo

class OakdSource (visualizer.Source):
    def __init__(self, output_folder: Path = None):
        self.connect ()
        self.output_folder = output_folder
        self.frameIndex = 0

    def connect (self):
        print ("Trying to connect to an OAK camera...")
        pipeline = dai.Pipeline()

        # Set up left and right cameras
        monoLeft = getMonoCamera(pipeline, isLeft = True)
        monoRight = getMonoCamera(pipeline, isLeft = False)

        # Combine left and right cameras to form a stereo pair
        stereo = getStereoPair(pipeline, monoLeft, monoRight)

        
        # Set XlinkOut for disparity, rectifiedLeft, and rectifiedRight
        xoutDisp = pipeline.createXLinkOut()
        xoutDisp.setStreamName("disparity")
        
        # xoutDepth = pipeline.create(dai.node.XLinkOut)
        # xoutDepth.setStreamName("depth")

        xoutRectifiedLeft = pipeline.createXLinkOut()
        xoutRectifiedLeft.setStreamName("rectifiedLeft")

        xoutRectifiedRight = pipeline.createXLinkOut()
        xoutRectifiedRight.setStreamName("rectifiedRight")

        stereo.disparity.link(xoutDisp.input)
        
        stereo.rectifiedLeft.link(xoutRectifiedLeft.input)
        stereo.rectifiedRight.link(xoutRectifiedRight.input)
        # stereo.depth.link(xoutDepth.input)

        self.device = dai.Device(pipeline).__enter__()

        oak_calib = self.device.readCalibration()
        w, h = monoLeft.getResolutionSize()

        # Intrinsics of disparity are taken from the right image.
        disparityIntrinsics = oak_calib.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, dai.Size2f(w, h))
        baselineMeters = 1e-2 * oak_calib.getBaselineDistance(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT)

        self.calibration = visualizer.Calibration(w, h,
                                                  fx=disparityIntrinsics[0][0],
                                                  fy=disparityIntrinsics[1][1],
                                                  cx0=disparityIntrinsics[0][2],
                                                  cx1=disparityIntrinsics[0][2],
                                                  cy=disparityIntrinsics[1][2],
                                                  baseline_meters=baselineMeters)

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        self.disparityQueue = self.device.getOutputQueue(name="disparity", maxSize=1, blocking=False)
        self.rectifiedLeftQueue = self.device.getOutputQueue(name="rectifiedLeft", maxSize=1, blocking=False)
        self.rectifiedRightQueue = self.device.getOutputQueue(name="rectifiedRight", maxSize=1, blocking=False)
        # depthQueue = self.device.getOutputQueue(name="depth", maxSize=1, blocking=False)

    def get_next_pair(self):
        leftFrame = getFrame(self.rectifiedLeftQueue)
        rightFrame = getFrame(self.rectifiedRightQueue)
        if self.output_folder is not None:
            self.output_folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.output_folder / f"img_{self.frameIndex:03d}_left.png"), leftFrame)
            cv2.imwrite(str(self.output_folder / f"img_{self.frameIndex:03d}_right.png"), rightFrame)
        disparityPixels = getFrame(self.disparityQueue)
        leftFrame = cv2.cvtColor (leftFrame, cv2.COLOR_GRAY2RGB)
        rightFrame = cv2.cvtColor (rightFrame, cv2.COLOR_GRAY2RGB)
        self.frameIndex += 1
        return visualizer.InputPair(leftFrame, rightFrame, self.calibration, "OAK-D Camera", disparityPixels)

class StereoFromOakInputSource(StereoMethod):
    def __init__(self, config: Config):
        super().__init__("Input Source", "Stereo computed by the input source", {}, config)

    def compute_disparity(self, input: InputPair) -> StereoOutput:
        # The disparity is aligned to the right image with OAK-D
        return StereoOutput(input.input_disparity, input.right_image, 0.0)
        