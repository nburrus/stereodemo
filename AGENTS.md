# AGENTS.md

Guidance for AI agents and contributors working in this repository.

## Project Overview

`stereodemo` is a Python package and GUI utility for comparing stereo depth estimation methods on rectified stereo pairs or live OAK-D camera input. The application loads image pairs, computes disparity maps with OpenCV, ONNX Runtime, or TorchScript-backed methods, and visualizes colored point clouds with Open3D.

Core code lives in `stereodemo/`. Sample datasets live in `datasets/`. Model conversion and capture helpers live in `tools/`. Tests live in `tests/`.

## Repository Layout

- `stereodemo/main.py`: CLI entry point, image discovery, method registration, and application startup.
- `stereodemo/visualizer.py`: Open3D GUI, point cloud rendering, method parameter controls, and source abstraction.
- `stereodemo/methods.py`: shared dataclasses and base `StereoMethod` interface.
- `stereodemo/method_*.py`: individual stereo or depth method adapters.
- `stereodemo/oakd_source.py`: live OAK-D capture support and OAK-provided disparity source.
- `stereodemo/utils.py`: model downloading and image padding helpers.
- `datasets/`: small checked-in stereo samples and calibration JSON files.
- `models/` and `dist/`: local/generated artifacts when present; do not treat them as canonical source unless explicitly asked.
- `tests/test_methods.py`: unittest-based inference regression tests.
- `pyproject.toml`: uv and setuptools packaging configuration.
- `setup.py`: minimal compatibility shim for older editable-install workflows.
- `build_release.sh`: release build script that temporarily symlinks `datasets/oak-d` into the package.

## Environment Setup

Use Python 3.8 or newer. CI currently uses Python 3.11.

Prefer `uv` for local development. Create or update the environment with:

```sh
uv sync
```

Then run project commands through `uv run` so they use the resolved environment:

```sh
uv run stereodemo datasets
uv run python tests/test_methods.py
```

CI installs from the uv lockfile with:

```sh
uv sync --frozen
```

Main runtime dependencies are declared in `pyproject.toml`: `numpy`, `opencv-python`, `open3d`, `torch`, `torchvision`, and ONNX Runtime. `depthai` is optional and only needed for OAK-D camera input; install it with `uv sync --extra oak` when needed. Keep `uv.lock` in sync when changing project dependencies, but do not churn it for unrelated source changes.

## Common Commands

Run the test suite:

```sh
uv run python tests/test_methods.py
```

Run the GUI on bundled datasets:

```sh
uv run stereodemo datasets
```

Run on one folder or explicit left/right files:

```sh
uv run stereodemo datasets/oak-d
uv run stereodemo left.png right.png --calibration stereodemo_calibration.json
```

Run with an OAK-D camera:

```sh
uv run stereodemo --oak
```

Build a release package:

```sh
uv build
```

For the project release flow, use `./build_release.sh` because it removes and recreates `stereodemo/datasets`, symlinks `datasets/oak-d` for package data, builds `dist/` with `uv build`, then removes the temporary package dataset folder.

## Testing Notes

The test suite instantiates multiple methods and checks median disparity and coverage against a fixed ETH3D sample. It uses `tempfile.gettempdir() / "models"` as the model cache.

Important implications:

- Tests may download model files from GitHub releases on first run.
- Network failures or missing model assets can fail otherwise valid code changes.
- CUDA OpenCV tests are skipped automatically when OpenCV CUDA is unavailable.
- Some methods prefer CUDA when available; preserve CPU fallback behavior unless intentionally changing device policy.
- Numerical changes should be justified. If an inference change intentionally alters output, update expected medians/coverage with a clear explanation.

For small edits that only touch docs, packaging metadata, or non-executed helper scripts, running the full inference suite may be unnecessary. For changes in `stereodemo/methods.py`, `stereodemo/main.py`, `stereodemo/visualizer.py`, or any `stereodemo/method_*.py`, run `uv run python tests/test_methods.py` when dependencies and model downloads are available.

## Coding Conventions

This codebase is small and mostly straightforward Python. Match the existing style in nearby files.

- Keep method adapters as subclasses of `StereoMethod`.
- Return `StereoOutput(disparity_pixels, color_image_bgr, computation_time, ...)` from `compute_disparity`.
- Treat images as OpenCV BGR arrays unless a model-specific preprocessing step explicitly converts to RGB or grayscale.
- Keep disparity values in pixels. Use `StereoMethod.depth_meters_from_disparity()` and `disparity_from_depth_meters()` for depth/disparity conversions.
- Add user-tunable options through `IntParameter` or `EnumParameter` so the Open3D UI can render controls.
- Cache loaded models/sessions inside method instances when practical; repeated UI parameter changes should not reload the same model unnecessarily.
- Keep downloads centralized through `utils.download_model()` or existing per-method URL maps.
- Do not introduce broad refactors while changing one method. These adapters carry method-specific preprocessing and scaling details.

## Data and Calibration

`FileListSource` discovers stereo pairs by looking for `*left*`/`*right*` or `im0`/`im1` image names under folders. Calibration defaults to `stereodemo_calibration.json` beside the left image unless `--calibration` is provided.

When adding dataset samples:

- Include left/right naming that `find_stereo_images_in_dir()` can discover.
- Include `stereodemo_calibration.json` when point cloud scale matters.
- Keep sample data small. Large datasets and generated model files should not be added casually.

## Model Assets

Most pretrained models are not package source. Methods download their required `.onnx` or `.scripted.pt` files into `Config.models_path`, which defaults to `~/.cache/stereodemo/models` for the CLI and a temporary models folder for tests.

When adding or changing a model-backed method:

- Add explicit URLs and expected filenames near the method implementation.
- Make the target input shape, scaling, and postprocessing obvious in code.
- Preserve deterministic output for the regression sample where possible.
- Avoid checking large generated model files into git unless the maintainer explicitly requests it.

## GUI and Runtime Caveats

The GUI uses Open3D and creates windows, so it may not run in headless environments. Prefer unit-level tests for automated validation and only run the GUI when a display is available.

OAK-D support requires `depthai` and hardware access. Do not make OAK-D required for normal installation, tests, or non-camera code paths.

## Packaging and Release Notes

Version numbers are duplicated in `pyproject.toml` and `stereodemo/__init__.py`; update both for a release.

`MANIFEST.in` excludes most sample datasets from source distributions. `pyproject.toml` package data expects `stereodemo/datasets/oak-d` to exist during release builds, which is why `build_release.sh` creates the temporary symlink.

Do not leave generated `stereodemo/datasets`, `build/`, `dist/`, or `*.egg-info/` changes in source edits unless the task is explicitly about release artifacts.
