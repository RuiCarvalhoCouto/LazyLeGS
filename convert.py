#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import logging
import os
import shlex
import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action="store_true")
parser.add_argument("--skip_matching", action="store_true")
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument("--no_gpu_matching", action="store_true",
                    help="Disable GPU only for feature matching. Useful when OpenGL SiftGPU clamps matches to 16384.")
parser.add_argument("--force_gpu_matching", action="store_true",
                    help="Force GPU matching even when the script detects legacy OpenGL SiftGPU options.")
parser.add_argument("--max_num_features", default=65536, type=int,
                    help="COLMAP SiftExtraction.max_num_features value. Increase this if COLMAP clamps detected features during extraction.")
parser.add_argument("--max_num_matches", default=32768, type=int,
                    help="COLMAP maximum match count. The script maps this to the supported FeatureMatching or SiftMatching option.")
args = parser.parse_args()

source_path = Path(args.source_path)
colmap_command = args.colmap_executable if len(args.colmap_executable) > 0 else "colmap"
magick_command = args.magick_executable if len(args.magick_executable) > 0 else "magick"
feature_use_gpu = 1 if not args.no_gpu else 0
matching_use_gpu = 1 if not args.no_gpu and not args.no_gpu_matching else 0

if args.force_gpu_matching and (args.no_gpu or args.no_gpu_matching):
    parser.error("--force_gpu_matching cannot be combined with --no_gpu or --no_gpu_matching")

if args.max_num_features <= 0:
    parser.error("--max_num_features must be > 0")

if args.max_num_matches <= 0:
    parser.error("--max_num_matches must be > 0")


def supported_colmap_option(command_name, option_names):
    try:
        completed = subprocess.run(
            [colmap_command, command_name, "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
    except OSError as e:
        logging.error(f"Could not inspect COLMAP options: {e}. Exiting.")
        raise SystemExit(1)
    except subprocess.CalledProcessError as e:
        logging.error(f"Could not inspect COLMAP {command_name} options. Exiting.")
        raise SystemExit(e.returncode)

    help_text = completed.stdout + completed.stderr
    for option_name in option_names:
        if option_name in help_text:
            return option_name

    logging.error(f"COLMAP {command_name} does not support any of: {', '.join(option_names)}")
    raise SystemExit(1)


def run_checked(command, failure_message):
    try:
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=True)
    except OSError as e:
        logging.error(f"{failure_message} could not start: {e}. Exiting.")
        raise SystemExit(1)
    except subprocess.CalledProcessError as e:
        logging.error(f"{failure_message} failed with code {e.returncode}. Exiting.")
        raise SystemExit(e.returncode)


def normalize_sparse_output(sparse_path):
    sparse_zero_path = sparse_path / "0"
    os.makedirs(sparse_zero_path, exist_ok=True)

    expected_files = {
        "cameras.bin", "images.bin", "points3D.bin",
        "cameras.txt", "images.txt", "points3D.txt",
    }

    for filename in expected_files:
        source_file = sparse_path / filename
        if not source_file.is_file():
            continue
        destination_file = sparse_zero_path / filename
        if destination_file.exists():
            continue
        shutil.move(source_file, destination_file)


if not args.skip_matching:
    os.makedirs(source_path / "distorted" / "sparse", exist_ok=True)
    feature_use_gpu_option = supported_colmap_option(
        "feature_extractor",
        ["--FeatureExtraction.use_gpu", "--SiftExtraction.use_gpu"],
    )
    max_num_features_option = supported_colmap_option(
        "feature_extractor",
        ["--SiftExtraction.max_num_features"],
    )
    matching_use_gpu_option = supported_colmap_option(
        "exhaustive_matcher",
        ["--FeatureMatching.use_gpu", "--SiftMatching.use_gpu"],
    )
    max_num_matches_option = supported_colmap_option(
        "exhaustive_matcher",
        ["--FeatureMatching.max_num_matches", "--SiftMatching.max_num_matches"],
    )
    if (
        matching_use_gpu_option == "--SiftMatching.use_gpu"
        and args.max_num_matches > 16384
        and matching_use_gpu
        and not args.force_gpu_matching
    ):
        print(
            "[ INFO ] Legacy OpenGL SiftGPU matching is limited to 16384 matches. "
            "Using CPU feature matching so --max_num_matches can be honored. "
            "Pass --force_gpu_matching to keep GPU matching anyway.",
            flush=True,
        )
        matching_use_gpu = 0

    ## Feature extraction
    run_checked([
        colmap_command, "feature_extractor",
        "--database_path", str(source_path / "distorted" / "database.db"),
        "--image_path", str(source_path / "input"),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", args.camera,
        feature_use_gpu_option, str(feature_use_gpu),
        max_num_features_option, str(args.max_num_features),
    ], "Feature extraction")

    ## Feature matching
    run_checked([
        colmap_command, "exhaustive_matcher",
        "--database_path", str(source_path / "distorted" / "database.db"),
        matching_use_gpu_option, str(matching_use_gpu),
        max_num_matches_option, str(args.max_num_matches),
    ], "Feature matching")

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    run_checked([
        colmap_command, "mapper",
        "--database_path", str(source_path / "distorted" / "database.db"),
        "--image_path", str(source_path / "input"),
        "--output_path", str(source_path / "distorted" / "sparse"),
        "--Mapper.ba_global_function_tolerance=0.000001",
    ], "Mapper")

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
run_checked([
    colmap_command, "image_undistorter",
    "--image_path", str(source_path / "input"),
    "--input_path", str(source_path / "distorted" / "sparse" / "0"),
    "--output_path", str(source_path),
    "--output_type", "COLMAP",
], "Image undistortion")

normalize_sparse_output(source_path / "sparse")

if args.resize:
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(source_path / "images_2", exist_ok=True)
    os.makedirs(source_path / "images_4", exist_ok=True)
    os.makedirs(source_path / "images_8", exist_ok=True)
    files = os.listdir(source_path / "images")
    for file in files:
        source_file = source_path / "images" / file

        destination_file = source_path / "images_2" / file
        shutil.copy2(source_file, destination_file)
        run_checked([magick_command, "mogrify", "-resize", "50%", str(destination_file)], "50% resize")

        destination_file = source_path / "images_4" / file
        shutil.copy2(source_file, destination_file)
        run_checked([magick_command, "mogrify", "-resize", "25%", str(destination_file)], "25% resize")

        destination_file = source_path / "images_8" / file
        shutil.copy2(source_file, destination_file)
        run_checked([magick_command, "mogrify", "-resize", "12.5%", str(destination_file)], "12.5% resize")

print("Done.")
