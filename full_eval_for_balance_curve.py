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

import os, time
from argparse import ArgumentParser

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

grad_thresh = {
    "bicycle": [0.0002, 0.0008],
    "flowers": [0.0002, 0.001],
    "garden": [0.0002, 0.0003],
    "stump": [0.0002, 0.001],
    "treehill": [0.0002, 0.0018],
    "room": [0.0002, 0.0004],
    "counter": [0.0002, 0.0004],
    "kitchen": [0.0002, 0.0002],
    "bonsai": [0.0002, 0.0002],
    "truck": [0.0002, 0.0004],
    "train": [0.0002, 0.0004],
    "playroom": [0.0002, 0.0005],
    "drjohnson": [0.0002, 0.0005]
}

rl_grad_thresh = {
    "bicycle": [0.0001, 0.0004],
    "flowers": [0.0001, 0.0004],
    "garden": [0.0001, 0.0002],
    "stump": [0.0001, 0.0004],
    "treehill": [0.0001, 0.005],
    "room": [0.0001, 0.0002],
    "counter": [0.0001, 0.0002],
    "kitchen": [0.0001, 0.0001],
    "bonsai": [0.0001, 0.0001],
    "truck": [0.0001, 0.0002],
    "train": [0.0001, 0.0002],
    "playroom": [0.0001, 0.00025],
    "drjohnson": [0.0001, 0.00025]
}

special_args = {
    "bicycle": "",
    "flowers": " --dense 0.005 ",
    "garden": " --highfeature_lr 0.02 --loss_thresh 0.06 ",
    "stump": " --dense 0.004 ",
    "treehill": " --dense 0.01 ",
    "room": " --highfeature_lr 0.02 ",
    "counter": " --highfeature_lr 0.02 ",
    "kitchen": " --highfeature_lr 0.02 ",
    "bonsai": " --highfeature_lr 0.02 ",
    "truck": " --highfeature_lr 0.04 ",
    "train": " --highfeature_lr 0.042 --dense 0.015 ",
    "playroom": " --highfeature_lr 0.0025 --dense 0.005 ",
    "drjohnson": " --highfeature_lr 0.0015 --dense 0.003 "
}

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")
parser.add_argument("--optimizer_type", type=str, default="default")
parser.add_argument("--sh_lower", action="store_true")
parser.add_argument("--dry_run", action="store_true")
parser.add_argument("--addition_args", type=str, default="")
parser.add_argument("--densification_interval_list", nargs="+", type=int, default=[100,])
parser.add_argument("--rl", action="store_true")
args, _ = parser.parse_known_args()

# all_scenes = []
# all_scenes.extend(mipnerf360_outdoor_scenes)
# all_scenes.extend(mipnerf360_indoor_scenes)
# all_scenes.extend(tanks_and_temples_scenes)
# all_scenes.extend(deep_blending_scenes)

all_scenes = ["bicycle"]

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", default="/data2/ningzhh/data/mipnerf360", type=str)
    parser.add_argument("--tanksandtemples", "-tat", default="/data2/ningzhh/data/tanksandtemples", type=str)
    parser.add_argument("--deepblending", "-db", default="/data2/ningzhh/data/db", type=str)
    args = parser.parse_args()

def run_cmd(CMD, args):
    print(CMD)
    if not args.dry_run:
        os.system(CMD)


for scale in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]:
# for densification_interval in args.densification_interval_list:
    m360_timing = 0.
    tandt_timing = 0.
    db_timing = 0.

    current_output_path = args.output_path
    if args.rl:
        current_output_path += "_rl"
    else:
        current_output_path += "_fastgs"

    if not args.skip_training:
        common_args = " --quiet --eval "
        common_args += " --optimizer_type {}".format(args.optimizer_type)
        
        if args.sh_lower:
            common_args += " --sh_lower"

        if args.rl:
            common_args += " --use_rl_densification --rl_use_my_value --use_prune_estimator "
        
        current_output_path += f"/grad_thresh_scale_{scale}"

        for scene in all_scenes:
            start_time = time.time()

            grad_thresh_args = f" --grad_thresh {grad_thresh[scene][0] / scale} --grad_abs_thresh {grad_thresh[scene][1] / scale} "
            # fastgs_importance_score_thresh = 5 / scale
            # common_args += f" --fastgs_importance_score_thresh {fastgs_importance_score_thresh} "

            if scene in mipnerf360_outdoor_scenes:
                source = args.mipnerf360 + "/" + scene
                scene_args = common_args
                scene_args += grad_thresh_args
                CMD = "python train.py -s " + source + " -i images_4 -m " + current_output_path + "/" + f"{scene}" + scene_args + special_args[scene]

            if scene in mipnerf360_indoor_scenes:
                source = args.mipnerf360 + "/" + scene
                scene_args = common_args
                scene_args += grad_thresh_args
                CMD = "python train.py -s " + source + " -i images_2 -m " + current_output_path + "/" + f"{scene}" + scene_args + special_args[scene]

            if scene in tanks_and_temples_scenes:
                source = args.tanksandtemples + "/" + scene
                scene_args = common_args
                scene_args += grad_thresh_args
                CMD = "python train.py -s " + source + " -m " + current_output_path + "/" + f"{scene}" + scene_args + special_args[scene] + " --mult 0.7 "

            if scene in deep_blending_scenes:
                source = args.deepblending  + "/" + scene
                scene_args = common_args
                scene_args += grad_thresh_args
                CMD = "python train.py -s " + source + " -m " + current_output_path + "/" + f"{scene}" + scene_args + special_args[scene] + " --mult 0.7 "

            run_cmd(CMD, args)

            time_elapsed = (time.time() - start_time)/60.0
            if scene in mipnerf360_outdoor_scenes or scene in mipnerf360_indoor_scenes:
                m360_timing += time_elapsed
            elif scene in tanks_and_temples_scenes:
                tandt_timing += time_elapsed
            elif scene in deep_blending_scenes:
                db_timing += time_elapsed

        m360_timing /= 9
        tandt_timing /= 2
        db_timing /= 2

    if not args.dry_run:
        if not os.path.exists(current_output_path):
            os.makedirs(current_output_path, exist_ok=True)
        with open(os.path.join(current_output_path, "timing.txt"), 'w') as file:
            file.write(f"m360: {m360_timing} minutes \n tandt: {tandt_timing} minutes \n db: {db_timing} minutes\n")

    if not args.skip_rendering:
        for scene in all_scenes:
            output_path = current_output_path + "/" + scene
            CMD = f"python render.py -m {output_path} --skip_train"
            if scene in tanks_and_temples_scenes or scene in deep_blending_scenes:
                CMD += " --mult 0.7 "
            run_cmd(CMD, args)

    if not args.skip_metrics:
        for scene in all_scenes:
            output_path = current_output_path + "/" + scene
            CMD = f"python metrics.py -m {output_path}"
            run_cmd(CMD, args)