# LazyLeGS

LazyLeGS is a fork of [LeGS](https://github.com/AaronNZH/LeGS), which builds on FastGS and vanilla 3D Gaussian Splatting training. This fork is focused on making large COLMAP scene training more practical, with the immediate development target of memory-safe lazy image loading.

The goal is to keep one standard LeGS / 3DGS Gaussian model and standard viewer-compatible outputs while avoiding the current startup behavior where every training image is loaded, resized, converted to a tensor, and retained in CPU memory during scene initialization.

## Current Focus

- Preserve upstream LeGS / FastGS training behavior when `--no_lazy_images` is passed.
- Use lazy image loading by default for large COLMAP datasets.
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

Clone this fork recursively, then create the Conda environment:

```bash
git clone https://github.com/RuiCarvalhoCouto/LazyLeGS.git --recursive
cd LazyLeGS

# Windows only
SET DISTUTILS_USE_SDK=1

conda create -n lazylegs python=3.8 -y
# Optionally (not recommended), you can try to run the following instead: conda env create --file environment.yml

conda activate lazylegs
```

If environment was created manually, I recommend installing all dependencies individually:
```bash
conda install -y -c pytorch -c nvidia -c conda-forge pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 ffmpeg=4.2.2 pillow=10.2.0 pip=23.3.1 typing_extensions=4.9.0 colmap
python -m pip install numpy==1.24.4 scipy==1.10.1 tqdm==4.66.2 plyfile==0.8.1 opencv-python==4.8.1.78 imageio==2.34.0 scikit-image==0.21.0 matplotlib==3.7.5 tensorboard==2.14.0 lpips==0.1.4 websockets==12.0 && python -m pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html && python -m pip install -v submodules/diff-gaussian-rasterization_fastgs submodules/simple-knn submodules/fused-ssim --no-build-isolation
```

## Training

Call `train.py` directly with your dataset path:

```bash
python train.py -s /path/to/dataset -m output/my_scene
```

For large COLMAP datasets, lazy image loading is enabled by default. Use a bounded CPU cache for preprocessed images (default is on and is 32):

```bash
python train.py -s /path/to/dataset -m output/my_scene --data_device cpu --image_cache_size 64
```

Disable retained CPU image caching while keeping lazy loading enabled:

```bash
python train.py -s /path/to/dataset -m output/my_scene --image_cache_size 0
```

Pass `--no_lazy_images` to restore the original eager image loading behavior.

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
- FastGS: [FastGS](https://github.com/fastgs/FastGS)
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
