from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .methods import Config, EnumParameter, IntParameter, StereoMethod, InputPair, StereoOutput, Calibration
from . import utils


urls = {
    "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth": "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
    "DUSt3R_ViTLarge_BaseDecoder_224_linear.pth": "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth",
}


class Dust3rModel(nn.Module):
    def __init__(self, ckpt_dict, width: int, height: int, device: str):
        super().__init__()
        from .dust3r_lib.encoder import Dust3rEncoder
        from .dust3r_lib.decoder import Dust3rDecoder
        from .dust3r_lib.head import Dust3rHead

        torch_device = torch.device(device)
        self.encoder = Dust3rEncoder(ckpt_dict, batch=2, width=width, height=height, device=torch_device)
        self.decoder = Dust3rDecoder(ckpt_dict, batch=1, width=width, height=height, device=torch_device)
        self.head = Dust3rHead(ckpt_dict, width=width, height=height, device=torch_device)

    @torch.inference_mode()
    def forward(self, left, right):
        encoded = self.encoder(torch.cat([left, right], dim=0))
        f1, f2 = encoded[0:1], encoded[1:2]
        return self.head(*self.decoder(f1, f2))


class Dust3rStereo(StereoMethod):
    def __init__(self, config: Config):
        super().__init__(
            "DUSt3R (CVPR 2024)",
            "DUSt3R: Geometric 3D Vision Made Easy. Experimental metric stereo adapter using official PyTorch inference.",
            {},
            config)
        self.reset_defaults()
        self.net = None
        self._loaded_model = None
        self._loaded_device = None
        self._loaded_size = None

    def reset_defaults(self):
        self.parameters.update({
            "Device": EnumParameter("Inference device", 0, ["Auto", "CPU", "CUDA"]),
            "Model": EnumParameter("Pre-trained model", 0, ["512-dpt", "224-linear"]),
            "Min Confidence x100": IntParameter("Minimum DUSt3R confidence multiplied by 100", 25, 0, 1000),
        })

    def compute_disparity(self, input: InputPair) -> StereoOutput:
        device = self._selected_device()
        image_size = self._image_size()
        min_conf = self.parameters["Min Confidence x100"].value / 100.0

        left_view, _left_calib, resize_scale = self._prepare_view(
            input.left_image, input.calibration, image_size, idx=0)
        right_view, _right_calib, _ = self._prepare_view(
            input.right_image, input.calibration, image_size, idx=1)
        if left_view["img"].shape != right_view["img"].shape:
            raise RuntimeError("DUSt3R preprocessing produced mismatched left/right tensor shapes.")
        self._load_model(self._model_filename(), device, left_view["img"].shape[-1], left_view["img"].shape[-2])

        start = time.time()
        with torch.inference_mode():
            pts3d_left, conf_left, pts3d_right_in_left, conf_right = self.net(
                left_view["img"].to(device), right_view["img"].to(device))
        elapsed_time = time.time() - start

        pts3d_left = self._as_numpy(pts3d_left)[0]
        conf_left = np.squeeze(self._as_numpy(conf_left)[0])
        pts3d_right_in_left = self._as_numpy(pts3d_right_in_left)[0]
        conf_right = np.squeeze(self._as_numpy(conf_right)[0])

        # DUSt3R predicts a self-consistent (f_dust3r, pts3d) for the processed
        # crop with its principal point at the geometric image center, not at
        # the user's calibrated principal point. The actual pixel disparity in
        # the rectified pair is f_dust3r * b_dust3r / Z_dust3r, which we can
        # compute purely from DUSt3R's outputs — using the user's calibrated
        # focal length here would bias the recovered baseline.
        f_dust3r = self._estimate_dust3r_focal(pts3d_left, conf_left, min_conf)
        if f_dust3r is None:
            raise RuntimeError("DUSt3R: not enough confident predictions to estimate the implicit focal length.")
        b_dust3r = self._estimate_dust3r_baseline(
            pts3d_right_in_left, conf_right, f_dust3r, min_conf)
        if b_dust3r is None or b_dust3r <= 1e-6:
            raise RuntimeError("DUSt3R: failed to estimate the predicted baseline from the right view.")

        Z = pts3d_left[:, :, 2].astype(np.float32)
        actual_pixel_disparity_resized = np.float32(f_dust3r * b_dust3r)
        with np.errstate(divide="ignore", invalid="ignore"):
            disparity_map = np.where(Z > 0, actual_pixel_disparity_resized / Z, np.float32(-1.0)).astype(np.float32)
        invalid = (conf_left < min_conf) | ~np.isfinite(disparity_map) | (disparity_map <= 0)
        disparity_map[invalid] = -1.0

        if disparity_map.shape[:2] != input.left_image.shape[:2]:
            disparity_map = cv2.resize(
                disparity_map,
                (input.left_image.shape[1], input.left_image.shape[0]),
                interpolation=cv2.INTER_NEAREST)
            valid = disparity_map > 0
            # Disparity scales with the image-resize factor only; the centred
            # crop does not change disparity values.
            disparity_map[valid] *= np.float32(1.0 / resize_scale)

        return StereoOutput(disparity_map, input.left_image, elapsed_time)

    def _selected_device(self) -> str:
        requested = self.parameters["Device"].value
        if requested == "CPU":
            return "cpu"
        if requested == "CUDA":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was selected for DUSt3R, but torch.cuda.is_available() is false.")
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _model_filename(self):
        if self.parameters["Model"].value == "224-linear":
            return "DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"
        return "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

    def _image_size(self):
        return 224 if self.parameters["Model"].value == "224-linear" else 512

    def _load_model(self, model_name: str, device: str, width: int, height: int):
        size = (width, height)
        if self._loaded_model == model_name and self._loaded_device == device and self._loaded_size == size:
            return

        model_path = self.config.models_path / model_name
        if not model_path.exists():
            utils.download_model(urls[model_path.name], model_path)
        try:
            ckpt_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt_dict = torch.load(model_path, map_location="cpu")
        try:
            self.net = Dust3rModel(ckpt_dict, width, height, device).eval()
        except ImportError as e:
            raise RuntimeError("DUSt3R requires the small einops package for its local DPT head implementation.") from e
        self._loaded_model = model_name
        self._loaded_device = device
        self._loaded_size = size

    def _prepare_view(self, image_bgr: np.ndarray, calibration: Calibration, size: int, idx: int):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        processed, calib, scale = self._resize_and_crop(pil_image, calibration, size)
        img_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return {
            "img": img_norm(processed)[None],
            "true_shape": np.int32([processed.size[::-1]]),
            "idx": idx,
            "instance": str(idx),
        }, calib, scale

    def _resize_and_crop(self, image: Image.Image, calibration: Calibration, size: int):
        width, height = image.size
        if size == 224:
            long_edge_size = round(size * max(width / height, height / width))
        else:
            long_edge_size = size

        scale = long_edge_size / max(width, height)
        resized_width = int(round(width * scale))
        resized_height = int(round(height * scale))
        resampling = getattr(Image, "Resampling", Image)
        resample = resampling.LANCZOS if max(width, height) > long_edge_size else resampling.BICUBIC
        image = image.resize((resized_width, resized_height), resample)

        cx, cy = resized_width // 2, resized_height // 2
        if size == 224:
            half_width = min(cx, cy)
            half_height = half_width
        else:
            patch_size = 16
            half_width = int(((2 * cx) // patch_size) * patch_size / 2)
            half_height = int(((2 * cy) // patch_size) * patch_size / 2)
            if resized_width == resized_height:
                half_height = int(3 * half_width / 4)

        left = int(cx - half_width)
        top = int(cy - half_height)
        right = int(cx + half_width)
        bottom = int(cy + half_height)
        image = image.crop((left, top, right, bottom))

        calib = Calibration(
            width=right - left,
            height=bottom - top,
            fx=calibration.fx * scale,
            fy=calibration.fy * scale,
            cx0=calibration.cx0 * scale - left,
            cx1=calibration.cx1 * scale - left,
            cy=calibration.cy * scale - top,
            baseline_meters=calibration.baseline_meters,
            depth_range=calibration.depth_range,
            left_image_rect_normalized=np.array([0., 0., 1., 1.]),
            comment=calibration.comment,
        )
        return image, calib, scale

    def _estimate_dust3r_focal(self, pts3d: np.ndarray, confidence: np.ndarray, min_conf: float):
        """Estimate DUSt3R's implicit focal length from a per-pixel point map.

        DUSt3R places its principal point at the geometric image center of the
        processed crop. For pixel (u, v) and predicted point (X, Y, Z):
            (u - W/2) = f * X / Z, (v - H/2) = f * Y / Z
        so f = Z * sqrt((u - W/2)^2 + (v - H/2)^2) / sqrt(X^2 + Y^2). We take
        the per-pixel median for robustness.
        """
        height, width = pts3d.shape[:2]
        pp_x = width / 2.0
        pp_y = height / 2.0
        valid = (confidence >= min_conf) & np.isfinite(pts3d).all(axis=2) & (pts3d[:, :, 2] > 0)
        ys, xs = np.nonzero(valid)
        if xs.size < 64:
            return None
        points = pts3d[ys, xs]
        pixel_radius = np.hypot(xs - pp_x, ys - pp_y)
        point_radius = np.hypot(points[:, 0], points[:, 1])
        keep = point_radius > 1e-6
        if keep.sum() < 64:
            return None
        f_per_pixel = points[keep, 2] * pixel_radius[keep] / point_radius[keep]
        return float(np.median(f_per_pixel))

    def _estimate_dust3r_baseline(self, pts3d_right_in_left: np.ndarray, confidence: np.ndarray,
                                  f_dust3r: float, min_conf: float):
        """Estimate the predicted baseline assuming a rectified pair.

        For pixel (u, v) of the right image with predicted point (X, Y, Z) in
        the left camera frame, with the right camera at (b, 0, 0) and no
        rotation:
            (u - W/2) = f_dust3r * (X - b) / Z
        so b = X - (u - W/2) * Z / f_dust3r. Taking the per-pixel median is
        robust against the outlier predictions DUSt3R produces in low-texture
        or occluded regions.
        """
        height, width = pts3d_right_in_left.shape[:2]
        pp_x = width / 2.0
        valid = (confidence >= min_conf) & np.isfinite(pts3d_right_in_left).all(axis=2) & (pts3d_right_in_left[:, :, 2] > 0)
        ys, xs = np.nonzero(valid)
        if xs.size < 64:
            return None
        points = pts3d_right_in_left[ys, xs]
        b_per_pixel = points[:, 0] - (xs - pp_x) * points[:, 2] / f_dust3r
        return float(np.median(b_per_pixel))

    @staticmethod
    def _as_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)


DUSt3R = Dust3rStereo
