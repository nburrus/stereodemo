from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime

from .methods import Config, EnumParameter, StereoMethod, InputPair, StereoOutput
from . import utils


MODEL_NAME = "DA3METRIC-LARGE.onnx"
# Fixed ONNX export shape, in OpenCV (width, height) order. The model was
# exported for [1, 3, 280, 504], so inputs are resized to this shape even when
# that changes the source aspect ratio.
MODEL_INPUT_SIZE = (504, 280)
SKY_THRESHOLD = 0.3
SKY_DEPTH_CAP_METERS = 200.0
MIN_NON_SKY_FRACTION = 0.05
MAX_SKY_FRACTION_FOR_FILL = 0.6
REFERENCE_FOCAL_LENGTH_PIXELS = 300.0

urls = {
    MODEL_NAME: "https://huggingface.co/TillBeemelmanns/Depth-Anything-V3-ONNX/resolve/main/DA3METRIC-LARGE.onnx",
}


# ONNX model from https://huggingface.co/TillBeemelmanns/Depth-Anything-V3-ONNX
# Pre/post-processing follows https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt
class DepthAnythingV3(StereoMethod):
    def __init__(self, config: Config):
        super().__init__("[Monocular] Depth Anything V3 Metric",
                         "Depth Anything V3 metric monocular depth estimation.",
                         {},
                         config)
        self.reset_defaults()

        self._loaded_session = None
        self._loaded_model_path = None
        self._loaded_providers = None

    def reset_defaults(self):
        self.parameters.update({
            "Device": EnumParameter("Device", 0, ["Auto", "CPU"]),
            "Sky handling": EnumParameter("Sky handling", 0, ["Fill", "Ignore"]),
        })

    def compute_disparity(self, input: InputPair) -> StereoOutput:
        model_path = self.config.models_path / MODEL_NAME
        providers = self._selected_providers()
        self._load_model(model_path, providers)

        input_tensor = self._preprocess_input(input.left_image)
        model_inputs = self._loaded_session.get_inputs()
        model_outputs = self._loaded_session.get_outputs()
        output_names = [output.name for output in model_outputs]

        start = time.time()
        outputs = self._loaded_session.run(output_names, {model_inputs[0].name: input_tensor})
        elapsed_time = time.time() - start

        output_by_name = dict(zip(output_names, outputs))
        depth = output_by_name.get("depth", outputs[0])
        sky = output_by_name.get("sky", outputs[1] if len(outputs) > 1 else None)

        depth_meters = self._process_depth(depth, sky, input.calibration)
        invalid_mask = depth_meters <= 0.0
        if depth_meters.shape[:2] != input.left_image.shape[:2]:
            depth_meters = cv2.resize(
                depth_meters,
                (input.left_image.shape[1], input.left_image.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            invalid_mask = cv2.resize(
                invalid_mask.astype(np.uint8),
                (input.left_image.shape[1], input.left_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        depth_meters[invalid_mask] = 0.0
        disparity_map = StereoMethod.disparity_from_depth_meters(depth_meters, input.calibration)
        disparity_map[depth_meters <= 0.0] = 0.0
        disparity_map = disparity_map.astype(np.float32)
        return StereoOutput(disparity_map, input.left_image, elapsed_time)

    def _preprocess_input(self, image_bgr: np.ndarray):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, MODEL_INPUT_SIZE, interpolation=cv2.INTER_AREA)
        image = image_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = image.transpose(2, 0, 1)
        return image[np.newaxis, :, :, :].astype(np.float32)

    def _process_depth(self, depth_output, sky_output, calibration):
        depth = np.squeeze(depth_output).astype(np.float32)
        depth[depth < 0.0] = 0.0

        focal_pixels = np.float32((calibration.fx + calibration.fy) / 2.0)
        # Follow the ika-rwth-aachen ROS2/TensorRT adapter for the
        # DA3METRIC-LARGE ONNX export: the raw depth is scaled from the
        # model's 300 px reference focal length to the calibrated camera focal.
        # Source: https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt
        # documents metric_depth = depth * ((fx + fy) / 2) / 300.0.
        depth_meters = depth * (focal_pixels / np.float32(REFERENCE_FOCAL_LENGTH_PIXELS))

        if sky_output is None:
            return depth_meters

        sky = np.squeeze(sky_output).astype(np.float32)
        # The ONNX sky head uses low values for sky and high values for non-sky.
        sky_mask = sky < np.float32(SKY_THRESHOLD)
        if self.parameters["Sky handling"].value == "Ignore":
            depth_meters[sky_mask] = 0.0
            return depth_meters

        if np.mean(sky_mask) > MAX_SKY_FRACTION_FOR_FILL:
            # Treat large sky masks as unreliable. On some outdoor stereo
            # samples the sky head also marks road and background as sky,
            # and filling would collapse most of the disparity map.
            return depth_meters

        valid_non_sky = depth_meters[(~sky_mask) & (depth_meters > 0.0)]
        min_non_sky_pixels = int(depth_meters.size * MIN_NON_SKY_FRACTION)
        if valid_non_sky.size < min_non_sky_pixels:
            # The ONNX sky head can classify non-sky indoor scenes as mostly sky.
            # In that case a percentile fill would collapse the whole depth map.
            return depth_meters
        else:
            sky_depth = np.float32(min(np.percentile(valid_non_sky, 99), SKY_DEPTH_CAP_METERS))
        depth_meters[sky_mask] = sky_depth
        return depth_meters

    def _selected_providers(self):
        if self.parameters["Device"].value == "CPU":
            return ["CPUExecutionProvider"]

        available = onnxruntime.get_available_providers()
        preferred = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
        providers = [provider for provider in preferred if provider in available]
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
        return providers

    def _load_model(self, model_path: Path, providers):
        if self._loaded_model_path == model_path and self._loaded_providers == providers:
            return

        if not model_path.exists():
            utils.download_model(urls[model_path.name], model_path)

        if not model_path.exists():
            raise RuntimeError(f"Could not download Depth Anything V3 model to {model_path}")
        self._loaded_model_path = model_path
        self._loaded_providers = providers
        self._loaded_session = onnxruntime.InferenceSession(str(model_path), providers=providers)
