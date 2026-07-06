# LazyLeGS

LazyLeGS is a fork of [LeGS](https://github.com/AaronNZH/LeGS), which builds on FastGS and vanilla 3D Gaussian Splatting training. This fork is focused on making large COLMAP scene training more practical, with the immediate development target of memory-safe lazy image loading.

The goal is to keep one standard LeGS / 3DGS Gaussian model and standard viewer-compatible outputs while avoiding the current startup behavior where every training image is loaded, resized, converted to a tensor, and retained in CPU memory during scene initialization.

## Current Focus

- Preserve upstream LeGS / FastGS training behavior by default.
- Add lazy image loading for large COLMAP datasets.
- Load image tensors on demand during training and evaluation.
- Keep a bounded CPU cache of preprocessed images.
- Preserve vanilla 3DGS-compatible outputs, including `point_cloud.ply`.

## Dataset Layout

This fork targets the usual COLMAP / 3DGS dataset layout:

```text
dataset/
|-- images/
|-- sparse/
|   `-- 0/
|       |-- cameras.bin
|       |-- images.bin
|       `-- points3D.bin
`-- input/
```

Some COLMAP workflows also include `distorted/`, `stereo/`, and helper scripts. LazyLeGS treats `images/` as the undistorted training image folder.

## Setup

Clone this fork recursively, then create the Conda environment and install the bundled extensions:

```bash
git clone <this-fork-url> --recursive
cd LazyLeGS

# Windows only
SET DISTUTILS_USE_SDK=1

conda env create --file environment.yml
conda activate LeGS
pip install submodules/diff-gaussian-rasterization_fastgs submodules/simple-knn submodules/fused-ssim --no-build-isolation
```

## Training

Call `train.py` directly with your dataset path:

```bash
python train.py -s /path/to/dataset -m output/my_scene
```

For large datasets, use CPU image storage until lazy image loading is available:

```bash
python train.py -s /path/to/dataset -m output/my_scene --data_device cpu
```

`--no_lazy_images` disables lazy image loading and renables original LeGS behavior.
`--image_cache_size 0` is intended to disable retained CPU image caching while still loading images on demand. Default is 32.

## Evaluation and Rendering

The repository retains the standard LeGS / 3DGS scripts and output structure. Saved point clouds should remain compatible with standard 3DGS viewers:

```text
output/
`-- point_cloud/
    `-- iteration_<N>/
        `-- point_cloud.ply
```

## Upstream Project

This fork is based on:

- LeGS: [Beyond Heuristics: Learnable Density Control for 3D Gaussian Splatting](https://github.com/AaronNZH/LeGS)
- FastGS: [FastGS](https://github.com/fastgs/FastGS/tree/main)
- Vanilla 3DGS: [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)

Please follow the licenses and citation requirements of the upstream projects.

## Citation

If you use upstream LeGS functionality, cite the original LeGS paper:

```bibtex
@misc{ning2026heuristicslearnabledensitycontrol,
      title={Beyond Heuristics: Learnable Density Control for 3D Gaussian Splatting},
      author={Zhenhua Ning and Xin Li and Jun Yu and Guangming Lu and Yaowei Wang and Wenjie Pei},
      year={2026},
      eprint={2605.00408},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.00408},
}
```
