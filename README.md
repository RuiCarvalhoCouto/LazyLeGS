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

If done on Windows, I recommend running all the following commands under an Anaconda prompt, while Visual Studio C++ tools are activated.

To do that: 
1. Install the CUDA 11.8 toolkit (if you have any other version of CUDA, make sure 11.8 shows up first on PATH). 
2. Install Miniconda (check [Miniconda Dowloads](https://www.anaconda.com/download/success) or [Miniconda Install Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html) for more information).
3. Install Visual Studio Community 2022 (check [Visual Studio Older Downloads](https://visualstudio.microsoft.com/vs/older-downloads//) for more information). On the Visual Studio Installer, make sure to have "Desktop development with C++" installed, as well as "MSVC v143 - VS 2022 C++ x64/x86 build tools (v14.39-17.9)" in the Individual Components tab.
4. Open an Anaconda prompt.
5. Find "vcvars64.bat" by running `where /r "%ProgramFiles%\Microsoft Visual Studio\2022\Community" vcvars64.bat` and copy that path.
6. Run `call <insert your path here>` to activate "cl.exe".
7. Verify CUDA and "cl.exe" by running `where cl && nvcc --version`.

After that setup, clone this fork recursively, then create the Conda environment:

```bash
git clone https://github.com/RuiCarvalhoCouto/LazyLeGS.git --recursive
cd LazyLeGS

conda env create --file environment.yml
conda activate lazylegs
```

Build the local CUDA extensions after activating the environment. On Windows, run this from a Developer Command Prompt or a shell where the Visual Studio C++ tools and CUDA 11.8 toolkit are available:

```bash
# Windows only
SET DISTUTILS_USE_SDK=1

python -m pip install -v submodules/diff-gaussian-rasterization_fastgs submodules/simple-knn submodules/fused-ssim --no-build-isolation
```

COLMAP is optional for this repository when your dataset is already converted and undistorted. Install COLMAP separately only if you need to run `convert.py`.

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
