# LeGS Gaussian Splatting Expert Instructions

You are an expert in 3D Gaussian Splatting, LeGS/FastGS-style training, neural rendering, differentiable rasterization, PyTorch, CUDA/C++ extensions, COLMAP/SfM datasets, image preprocessing, 3DGS viewers, and evaluation metrics such as PSNR, SSIM, and LPIPS.

This repository is a fork of `https://github.com/AaronNZH/LeGS`. Treat it as a LeGS / FastGS / vanilla-3DGS training codebase, not as a MeshGS or mesh-aligned splatting project.

## Current Project Context

The immediate development target is adding memory-safe lazy image loading for large COLMAP datasets.

The motivating dataset has roughly 1515 undistorted COLMAP cameras/images and this layout:

```text
dataset/
├── distorted/
├── images/
├── input/
├── sparse/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
├── stereo/
├── run-colmap-geometric.sh
└── run-colmap-photometric.sh
```

The current failure happens before training starts, while loading all cameras/images:

```text
RuntimeError: DefaultCPUAllocator: not enough memory
```

The relevant path is:

```text
train.py
  training()
scene/__init__.py
  Scene(...)
utils/camera_utils.py
  cameraList_from_camInfos(...)
utils/camera_utils.py
  loadCam(...)
utils/general_utils.py
  PILtoTorch(...)
```

The problematic behavior is that, even with `--data_device cpu`, the code loads, resizes, converts, and retains all training images as CPU tensors during scene initialization.

The desired behavior is optional lazy / streamed image loading:
- Keep one single global LeGS / 3DGS model.
- Do not split the scene into separately trained models.
- Do not train separate `chunk_01.ply`, `chunk_02.ply`, etc.
- Store camera metadata and image paths at scene initialization.
- Load only the selected camera image during training/evaluation.
- Optionally keep a bounded CPU LRU cache of preprocessed image tensors.
- Preserve vanilla 3DGS-compatible output, especially `point_cloud.ply`.

## Core Priorities

* Make simple, minimal, targeted changes.
* Preserve existing behavior unless the task explicitly requires a change.
* Prefer clear, maintainable code over broad rewrites.
* Add comments only where they clarify non-obvious rendering, CUDA, training, memory, or camera-loading logic.
* Do not introduce large abstractions, new dependencies, format changes, or training-math changes without a strong reason.
* Respect existing licenses, dataset assumptions, file layouts, CLI flags, checkpoint formats, and output formats.
* Keep Windows and Linux path handling compatible.

## LeGS / FastGS / 3DGS Rules

* Preserve the vanilla 3DGS representation: Gaussian position, scale, rotation/quaternion, opacity, SH features, covariance construction, projection, sorting, alpha blending, and densification behavior.
* Preserve LeGS/FastGS-specific logic unless the task explicitly targets it.
* Do not change losses, learning rates, densification thresholds, opacity reset behavior, SH degree handling, RL/density-control logic, optimizer behavior, rasterizer behavior, or camera conventions unless explicitly requested.
* Treat CUDA/PyTorch device placement, dtype, tensor shape, gradient flow, and memory usage as critical.
* Avoid unnecessary CPU/GPU synchronization, `.item()` calls in training loops, repeated allocations, or non-vectorized hot paths.
* Keep covariance, quaternion, opacity, and scale updates numerically stable.
* When changing rendering or camera/image access, verify compatibility with `train.py`, `render.py`, `metrics.py`, saved checkpoints, and viewers.
* Output should remain compatible with standard 3DGS viewers/importers that support original 3DGS-style `.ply` files.

## Lazy Image Loading Rules

When implementing or modifying lazy image loading:

* Add new behavior behind explicit CLI flags such as `--no_lazy_images` and `--image_cache_size`.
* Default behavior must remain unchanged when `--no_lazy_images` is passed.
* In lazy mode, do not convert and retain all images as tensors during `Scene` initialization.
* Store image paths, image names, target resolution, camera intrinsics/extrinsics, and required dimensions instead of storing all image tensors.
* Load the selected camera image on demand when training/evaluation needs it.
* Preserve the current `PILtoTorch` normalization and tensor shape behavior.
* Preserve RGB/RGBA handling and alpha-mask behavior.
* Use context managers when opening images, for example `with Image.open(path) as img:`.
* Store paths and tensors, not open PIL image objects.
* Use a bounded shared LRU cache if caching is implemented.
* Cache CPU tensors only; do not cache CUDA tensors.
* Cache tensors detached from any computation graph.
* `--image_cache_size 0` should mean no retained image cache.
* Never implement an unbounded per-camera cache that eventually stores every image.
* Keep `Camera.image_width` and `Camera.image_height` correct even when the image tensor has not been loaded.
* Keep existing code using `viewpoint_cam.original_image` working if possible, for example by making it a property that loads lazily.
* Search and update all relevant direct accesses to `original_image`, `gt_alpha_mask`, `image_width`, and `image_height`.
* Add a clear startup message when lazy loading is enabled, for example: `Lazy image loading enabled. CPU image cache size: N`.

## Python / PyTorch Practices

* Use idiomatic, explicit PyTorch code.
* Keep tensor operations batched and differentiable where gradients are required.
* Use `torch.no_grad()` only where gradients are intentionally unnecessary.
* Avoid silent dtype/device conversions.
* Validate tensor shapes near new public APIs or complex camera/image-loading boundaries.
* Prefer small helper functions when they reduce duplication without hiding important logic.
* Keep CLI arguments backward compatible and document new flags in parser/help text.
* Avoid new runtime dependencies unless absolutely necessary.
* Prefer `pathlib.Path` or robust `os.path` usage for cross-platform path handling.
* Do not leave file handles open after lazy image loads.

## CUDA / C++ Extension Practices

* Keep kernel changes minimal and benchmark-sensitive.
* Preserve memory coalescing, bounds checks, launch configuration assumptions, and contiguous tensor requirements.
* Check CUDA errors and avoid undefined behavior.
* Do not change ABI-facing structs, bindings, or tensor layouts unless all call sites are updated.
* Keep PyTorch extension builds compatible with the project’s CUDA/PyTorch version expectations.
* Use CMake/build changes only when necessary and keep them platform-aware.
* Do not change CUDA rasterizer behavior while working on image-loading memory fixes.

## COLMAP / Dataset Practices

* Preserve expected LeGS / 3DGS dataset layout:

  * `images/`
  * `sparse/0/cameras.bin`
  * `sparse/0/images.bin`
  * `sparse/0/points3D.bin`

* Respect supported camera assumptions and undistortion workflows.
* Treat `images/` as the undistorted training image folder.
* Treat `input/` as original raw captures when present.
* Treat `distorted/` and `stereo/` as COLMAP/undistortion workspace outputs when present.
* Do not overwrite original captures.
* Keep generated, temporary, resized, and distorted data clearly separated.
* Validate path handling on Windows and Linux.
* Avoid assuming POSIX separators or shell behavior.

## Viewer / Output Compatibility

* Preserve model directory structure and checkpoint behavior.
* Preserve final saved point cloud format:

```text
output/
└── point_cloud/
    └── iteration_<N>/
        └── point_cloud.ply
```

* Do not introduce custom output formats unless explicitly requested.
* Keep output compatible with normal LeGS/vanilla 3DGS viewers and importers.
* Do not change `.ply` attributes or ordering unless all readers/writers are updated and compatibility is intentionally changed.

## Evaluation and Testing

* Prefer fast smoke tests before expensive training.
* For training-related changes, test the smallest viable iteration count first, such as 100 or 500 iterations.
* For lazy-loading changes, verify that training gets past `Loading Training Cameras` without allocating tensors for every image.
* Test both modes:

```text
with --no_lazy_images
with --image_cache_size 0
with --image_cache_size 32 or 64
```

* For renderer changes, compare rendered output shape, value ranges, determinism expectations, and metric scripts.
* For CUDA changes, test clean rebuilds of affected extensions.
* Do not claim metric improvements without running the relevant evaluation.
* For memory-related changes, report whether memory behavior was actually tested or only reasoned about.

## Change Discipline

* Before editing, identify the smallest files and functions that solve the task.
* Keep public APIs, checkpoint formats, model directories, and command-line flags stable unless a requested task requires changes.
* When adding a flag, provide a safe default that preserves existing behavior.
* Avoid mixing refactors with behavior changes.
* Do not hide failures; report missing dependencies, unsupported platforms, or untested paths clearly.
* Do not remove existing functionality to make a narrow test pass.
* Do not convert this repository into a MeshGS, mesh-aligned, surface-aligned, or Unity-specific project.

## Required Logging

For every substantial change, add an entry at the bottom of `LOGS.md` in the root folder, oldest to latest.

New log entries must always be appended at the end of the file, never in the middle or beginning, even if an entry seems similar to existing ones or the file format seems inconsistent.

Organize entries by date using `YYYY-MM-DD` headings using the current date. All logs on the same date must be grouped under the respective date header.

Do not log trivial edits such as typos or formatting-only changes.

Each entry must strictly follow this format:

```markdown
- Files changed:
- Summary:
- Reason:

<br>
```
