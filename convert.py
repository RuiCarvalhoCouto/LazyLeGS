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
parser.add_argument("--max_num_features", default=65536, type=int,
                    help="COLMAP SiftExtraction.max_num_features value. Increase this if COLMAP clamps detected features during extraction.")
parser.add_argument("--max_num_matches", default=32768, type=int,
                    help="COLMAP SiftMatching.max_num_matches value. Increase this if COLMAP clamps features during matching.")
args = parser.parse_args()

source_path = Path(args.source_path)
colmap_command = args.colmap_executable if len(args.colmap_executable) > 0 else "colmap"
magick_command = args.magick_executable if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

if args.max_num_features <= 0:
    parser.error("--max_num_features must be > 0")

if args.max_num_matches <= 0:
    parser.error("--max_num_matches must be > 0")


def run_checked(command, failure_message):
    try:
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

    ## Feature extraction
    run_checked([
        colmap_command, "feature_extractor",
        "--database_path", str(source_path / "distorted" / "database.db"),
        "--image_path", str(source_path / "input"),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", args.camera,
        "--SiftExtraction.use_gpu", str(use_gpu),
        "--SiftExtraction.max_num_features", str(args.max_num_features),
    ], "Feature extraction")

    ## Feature matching
    run_checked([
        colmap_command, "exhaustive_matcher",
        "--database_path", str(source_path / "distorted" / "database.db"),
        "--SiftMatching.use_gpu", str(use_gpu),
        "--SiftMatching.max_num_matches", str(args.max_num_matches),
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
