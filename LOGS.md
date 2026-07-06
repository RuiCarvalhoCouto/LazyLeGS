# Logs

## 2026-07-06

- Files changed:
  - README.md
- Summary:
  Replaced the upstream LeGS README with a LazyLeGS-specific README that identifies this repository as a fork, documents the current lazy image loading focus, and preserves upstream attribution and citation guidance without emoji.
- Reason:
  The repository README still presented the project as the original LeGS repository rather than this LazyLeGS fork.

<br>

- Files changed:
  - arguments/__init__.py
  - scene/dataset_readers.py
  - scene/cameras.py
  - scene/__init__.py
  - utils/camera_utils.py
  - README.md
  - TODO.md
- Summary:
  Added default-on lazy COLMAP image loading with a bounded Scene-owned CPU image cache, `--no_lazy_images` opt-out, and `--image_cache_size` cache sizing. Updated documentation and recorded Blender lazy loading as future work.
- Reason:
  Large COLMAP datasets were loading, resizing, tensorizing, and retaining all images during scene initialization, causing CPU memory exhaustion before training started.

<br>

- Files changed:
  - environment.yml
  - README.md
- Summary:
  Replaced the exported Linux/CUDA 12.8 environment with a concise LazyLeGS environment targeting Python 3.8, PyTorch 2.0.0, CUDA 11.8, and the runtime packages used by the repository. Updated setup instructions to create the Conda environment from `environment.yml` and build local CUDA extensions afterward.
- Reason:
  The previous environment file contained duplicate and unrelated packages, hard-pinned build strings, mirror channels, Python 3.9, Torch 2.8, and CUDA 12.8 packages, which did not match the README target stack and made environment creation unreliable.

<br>
