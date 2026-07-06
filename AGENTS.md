# LazyLeGS Agent Instructions

You are an expert in 3D Gaussian Splatting, LeGS/FastGS-style training, neural rendering, differentiable rasterization, PyTorch, CUDA/C++ extensions, COLMAP/SfM datasets, image preprocessing, 3DGS viewers, and evaluation metrics such as PSNR, SSIM, and LPIPS.

This repository is a fork of `https://github.com/AaronNZH/LeGS`. Treat it as a LeGS / FastGS / vanilla-3DGS training codebase.

## Current Codebase State

LazyLeGS currently focuses on memory-safe LeGS training for large COLMAP datasets while preserving one global Gaussian model and vanilla 3DGS-compatible outputs.

Current implemented behavior:

* COLMAP lazy image loading is enabled by default.
* `--no_lazy_images` restores the original eager image loading behavior.
* `--image_cache_size` controls the bounded CPU image cache and defaults to `32`.
* `--image_cache_size 0` disables retained CPU image caching while still loading images on demand.
* Lazy COLMAP cameras store metadata, image paths, target resolution, and dimensions during `Scene` initialization.
* `Camera.original_image` remains the compatibility API. In lazy mode, it loads the image on demand.
* The lazy image cache is `Scene`-owned, shared across cameras, bounded, CPU-only, and stores detached tensors.
* Blender synthetic datasets still use the eager image preparation path. Track Blender lazy loading as future work in `TODO.md`.
* `convert.py` and `full_eval.py` use argument-list `subprocess.run(..., check=True)` calls so paths with spaces are handled correctly.

The expected COLMAP dataset layout is:

```text
dataset/
|-- distorted/
|-- images/
|-- input/
|-- sparse/
|   `-- 0/
|       |-- cameras.bin
|       |-- images.bin
|       `-- points3D.bin
|-- stereo/
|-- run-colmap-geometric.sh
`-- run-colmap-photometric.sh
```

The output must remain standard 3DGS-style:

```text
output/
`-- point_cloud/
    `-- iteration_<N>/
        `-- point_cloud.ply
```

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

When modifying camera/image loading:

* Preserve default-on lazy image loading for COLMAP datasets.
* Preserve `--no_lazy_images` as the compatibility opt-out for eager loading.
* Preserve `--image_cache_size`; default is `32`, `0` means no retained CPU cache, and negative values should fail clearly.
* Do not load, resize, tensorize, and retain every COLMAP image during `Scene` initialization in lazy mode.
* Keep camera metadata, image paths, image names, target resolution, intrinsics, extrinsics, `image_width`, and `image_height` available before image tensor loading.
* Preserve `PILtoTorch` normalization and tensor shape behavior.
* Preserve RGB/RGBA handling and alpha-mask behavior.
* Use context managers when opening images, for example `with Image.open(path) as img:`.
* Store paths and tensors, not open PIL image objects.
* Cache CPU tensors only. Never cache CUDA tensors.
* Cache tensors detached from any computation graph.
* Never implement an unbounded per-camera image cache.
* Keep `viewpoint_cam.original_image` working for training, validation, rendering, RL helpers, and metric-related code.
* Search for and update relevant direct accesses to `original_image`, `gt_alpha_mask`, `image_width`, and `image_height`.
* Keep the startup message when lazy loading is enabled: `Lazy image loading enabled. CPU image cache size: N`.
* Do not claim Blender lazy image loading exists until it is implemented.

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

## Script And Path Handling Practices

* Scripts must support directories and executable paths containing spaces on Windows and Linux.
* Do not use `os.system` or shell command strings for external commands that include paths.
* Use `subprocess.run([...], check=True)` or an equivalent argument-list API.
* Convert `Path` objects to `str` before passing them to subprocess argument lists.
* Use `pathlib.Path` or `os.path.join` for path construction.
* Do not concatenate filesystem paths with hard-coded `/` separators.
* If a script has dry-run output, prefer `shlex.join(cmd)` for readable command display while still executing the argument list.

## CUDA / C++ Extension Practices

* Keep kernel changes minimal and benchmark-sensitive.
* Preserve memory coalescing, bounds checks, launch configuration assumptions, and contiguous tensor requirements.
* Check CUDA errors and avoid undefined behavior.
* Do not change ABI-facing structs, bindings, or tensor layouts unless all call sites are updated.
* Keep PyTorch extension builds compatible with the project's CUDA/PyTorch version expectations.
* Use CMake/build changes only when necessary and keep them platform-aware.
* Do not change CUDA rasterizer behavior while working on image-loading memory fixes.

## COLMAP / Dataset Practices

* Preserve the standard LeGS / 3DGS COLMAP layout: `images/`, `sparse/0/cameras.bin`, `sparse/0/images.bin`, and `sparse/0/points3D.bin`.
* Treat `images/` as the undistorted training image folder.
* Treat `input/` as original raw captures when present.
* Treat `distorted/` and `stereo/` as COLMAP/undistortion workspace outputs when present.
* Respect supported camera assumptions and undistortion workflows.
* Do not overwrite original captures.
* Keep generated, temporary, resized, and distorted data clearly separated.
* Avoid assuming POSIX separators or shell behavior.

## Environment And Setup Practices

* Treat `README.md` as the source of truth for setup commands.
* The current documented Windows target stack is Python 3.8, PyTorch 2.0.0, CUDA 11.8, Visual Studio C++ build tools, and local CUDA extension installation after environment activation.
* Keep local extension build instructions compatible with `submodules/diff-gaussian-rasterization_fastgs`, `submodules/simple-knn`, and `submodules/fused-ssim`.
* COLMAP is optional for training already converted and undistorted datasets. It is only needed for conversion workflows such as `convert.py`.
* Report missing CUDA, compiler, PyTorch, or extension dependencies clearly.

## Viewer / Output Compatibility

* Preserve model directory structure and checkpoint behavior.
* Preserve final saved point cloud format under `output/point_cloud/iteration_<N>/point_cloud.ply`.
* Do not introduce custom output formats unless explicitly requested.
* Keep output compatible with normal LeGS, vanilla 3DGS, SIBR, and Supersplat-style viewers/importers.
* Do not change `.ply` attributes or ordering unless all readers/writers are updated and compatibility is intentionally changed.

## Evaluation And Testing

* Prefer fast smoke tests before expensive training.
* For training-related changes, test the smallest viable iteration count first, such as 100 or 500 iterations.
* For lazy-loading regressions, verify that training gets past `Loading Training Cameras` without allocating tensors for every image.
* Test these image-loading modes when camera loading is touched:

```text
with --no_lazy_images
with --image_cache_size 0
with --image_cache_size 32 or 64
```

* For path-handling changes, test source/output directories containing spaces. Use `full_eval.py --dry_run` where possible.
* For renderer changes, compare rendered output shape, value ranges, determinism expectations, and metric scripts.
* For CUDA changes, test clean rebuilds of affected extensions.
* Do not claim metric improvements without running the relevant evaluation.
* For memory-related changes, report whether memory behavior was actually tested or only reasoned about.
* If dependencies are unavailable in the current environment, report exactly which validation could not be run.

## Change Discipline

* Before editing, identify the smallest files and functions that solve the task.
* Keep public APIs, checkpoint formats, model directories, and command-line flags stable unless a requested task requires changes.
* When adding a flag, provide a safe and documented default.
* Avoid mixing refactors with behavior changes.
* Do not hide failures; report missing dependencies, unsupported platforms, or untested paths clearly.
* Do not remove existing functionality to make a narrow test pass.

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
