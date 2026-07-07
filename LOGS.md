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

- Files changed:
  - convert.py
  - full_eval.py
- Summary:
  Replaced shell-string command execution with `subprocess.run([...], check=True)` argument lists for COLMAP, ImageMagick, training, rendering, and metrics helper commands.
- Reason:
  Paths containing spaces were being split by the shell when scripts built command strings with unquoted path values.

<br>

- Files changed:
  - AGENTS.md
- Summary:
  Updated repository agent instructions to describe the current default-on COLMAP lazy image loading behavior, `--no_lazy_images` compatibility mode, bounded CPU image cache rules, subprocess-safe path handling, and current Windows setup guidance.
- Reason:
  The previous instructions still described lazy image loading as a future implementation target and did not reflect the path-handling and setup changes already made in the repository.

<br>

- Files changed:
  - AGENTS.md
- Summary:
  Recreated repository agent instructions from the current codebase, including default-on COLMAP lazy loading, bounded CPU cache behavior, path-safe subprocess rules, setup expectations, output compatibility, testing guidance, and required logging policy.
- Reason:
  The repository instructions needed to be rebuilt around the implemented LazyLeGS behavior rather than the earlier future-work framing.

<br>

- Files changed:
  - README.md
  - scene/dataset_readers.py
  - metrics.py
  - convert.py
  - full_eval.py
  - TODO.md
  - LOGS.md
- Summary:
  Clarified that setup is currently documented through manual Conda and pip commands rather than `environment.yml`, closed PIL image handles in eager dataset and metrics paths, hardened `convert.py` subprocess and sparse-output handling, anchored `full_eval.py` helper script paths, and recorded inactive full-evaluation flags as future work.
- Reason:
  Follow-up review found setup documentation drift, remaining file-handle hygiene issues, fragile conversion rerun behavior, relative helper script paths, and parsed evaluation flags that should not be wired in this cleanup pass.

<br>

- Files changed:
  - convert.py
  - LOGS.md
- Summary:
  Added a `--max_num_matches` option to `convert.py` and passed it through to COLMAP as `--SiftMatching.max_num_matches`, with positive-value validation and a higher default of 32768.
- Reason:
  COLMAP matching can clamp large feature sets to its maximum match count, reducing usable features on high-detail datasets unless the matching limit is raised.

<br>

- Files changed:
  - convert.py
  - LOGS.md
- Summary:
  Added a `--max_num_features` option to `convert.py` and passed it to COLMAP feature extraction as `--SiftExtraction.max_num_features`, with a default of 65536 and positive-value validation.
- Reason:
  COLMAP can clamp detected SIFT features during feature extraction before matching starts, so raising only the matching limit does not prevent feature-count clamp warnings.

<br>

## 2026-07-07

- Files changed:
  - convert.py
  - LOGS.md
- Summary:
  Updated `convert.py` to inspect the installed COLMAP command help and select the supported option names for feature extraction GPU use, feature matching GPU use, and maximum match count. Added external command logging so COLMAP launches show the exact options being passed.
- Reason:
  Newer COLMAP versions use `FeatureExtraction.*` and `FeatureMatching.*` option namespaces, so passing only older `SiftMatching.max_num_matches` did not raise the active matching limit and feature-clamp warnings could remain at the old default.

<br>

- Files changed:
  - convert.py
  - LOGS.md
- Summary:
  Added `--no_gpu_matching` and `--force_gpu_matching` to `convert.py`, split feature-extraction GPU use from feature-matching GPU use, and automatically falls back to CPU matching when legacy `SiftMatching.use_gpu` would hit the OpenGL SiftGPU 16384-match limit.
- Reason:
  Older COLMAP builds can accept a larger `SiftMatching.max_num_matches` value but still clamp to 16384 when the OpenGL SiftGPU matcher is active, so the matcher must run without GPU to honor larger match limits on those builds.

<br>
