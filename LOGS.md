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
