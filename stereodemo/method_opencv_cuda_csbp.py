import time

import numpy as np
import cv2

from .methods import Config, StereoMethod, IntParameter, EnumParameter, InputPair, StereoOutput


class StereoCudaCSBP(StereoMethod):
    def __init__(self, config: Config):
        super().__init__("OpenCV CUDA CSBP", "OpenCV CUDA Constant Space Belief Propagation", {}, config)
        self.reset_defaults()

    def reset_defaults(self):
        self.parameters.update({
            "Num Disparities": IntParameter("Number of disparities (pixels)", 128, 16, 256),
            "Num Iterations": IntParameter("Number of BP iterations on each pyramid level", 8, 1, 50),
            "Num Levels": IntParameter("Number of pyramid levels", 4, 1, 10),
            "Nr Plane": IntParameter("Number of active disparity levels on each pyramid level", 4, 1, 64),
            "Max Data Term x10": IntParameter("Truncation of data cost (divide by 10 for actual value)", 100, 1, 2000),
            "Data Weight x1000": IntParameter("Weight of data cost (divide by 1000 for actual value)", 70, 1, 1000),
            "Max Disc Term x10": IntParameter("Truncation of discontinuity cost (divide by 10 for actual value)", 17, 1, 200),
            "Disc Single Jump x10": IntParameter("Discontinuity single jump (divide by 10 for actual value)", 10, 1, 200),
            "Min Disp Threshold": IntParameter("Minimum disparity threshold for the level selection strategy", 0, 0, 100),
            "Msg Type": EnumParameter("Message type for internal computation buffers", 0, ["CV_32F", "CV_16S"]),
        })

    def compute_disparity(self, input: InputPair) -> StereoOutput:
        if cv2.cuda.getCudaEnabledDeviceCount() == 0:
            raise RuntimeError("No CUDA-enabled GPU detected. OpenCV CUDA CSBP requires a CUDA-capable device.")

        msg_type = cv2.CV_32F if self.parameters['Msg Type'].index == 0 else cv2.CV_16S
        stereo = cv2.cuda.createStereoConstantSpaceBP(
            ndisp=self.parameters['Num Disparities'].value,
            iters=self.parameters['Num Iterations'].value,
            levels=self.parameters['Num Levels'].value,
            nr_plane=self.parameters['Nr Plane'].value,
            msg_type=msg_type,
        )
        stereo.setMaxDataTerm(self.parameters['Max Data Term x10'].value / 10.0)
        stereo.setDataWeight(self.parameters['Data Weight x1000'].value / 1000.0)
        stereo.setMaxDiscTerm(self.parameters['Max Disc Term x10'].value / 10.0)
        stereo.setDiscSingleJump(self.parameters['Disc Single Jump x10'].value / 10.0)
        stereo.setMinDispTh(self.parameters['Min Disp Threshold'].value)

        gpu_left = cv2.cuda_GpuMat()
        gpu_right = cv2.cuda_GpuMat()
        gpu_left.upload(input.left_image)
        gpu_right.upload(input.right_image)

        start = time.time()
        gpu_disparity = stereo.compute(gpu_left, gpu_right)
        disparity = gpu_disparity.download().astype(np.float32)
        elapsed = time.time() - start

        return StereoOutput(disparity, input.left_image, elapsed)
