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

import os, random, time, shutil, pathlib, sys
import torch
import numpy as np
import uuid
from torch_scatter import scatter_mean, scatter_add
from random import randint
from lpipsPyTorch import lpips
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render_fastgs, network_gui_ws
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras
from utils.rl_utils import get_metric_score, get_gaussians_state_for_rl
from utils.general_utils import cosine_annealing


def saveRuntimeCode(dst: str) -> None:
    """
    备份运行时代码到输出目录，排除output文件夹
    """
    additionalIgnorePatterns = ['.git', '.gitignore', 'output']
    ignorePatterns = set()
    ROOT = '.'
    
    # 读取.gitignore文件中的忽略模式
    if os.path.exists(os.path.join(ROOT, '.gitignore')):
        with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
            for line in gitIgnoreFile:
                if not line.startswith('#') and line.strip():
                    if line.endswith('\n'):
                        line = line[:-1]
                    if line.endswith('/'):
                        line = line[:-1]
                    ignorePatterns.add(line)
    
    # 添加额外的忽略模式
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()
    
    # 执行备份
    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    print('Backup Finished! Code backed up to:', dst)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, websockets, use_rl_densification=False, rl_controller_path=None):
    if saving_iterations[-1] != opt.iterations:
        saving_iterations.append(opt.iterations)
    if len(testing_iterations) != 0 and testing_iterations[-1] != opt.iterations:
        testing_iterations.append(opt.iterations)

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, training_args=opt, use_rl_densification=use_rl_densification)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    if rl_controller_path:
        gaussians.rl_controller.restore(torch.load(rl_controller_path))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    if opt.use_validate_cam_list:
        validate_viewpoints = []
        for _ in range(10):
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            validate_viewpoints.append(viewpoint_stack.pop(rand_idx))
            _ = viewpoint_indices.pop(rand_idx)

    # record time
    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)
    total_time = 0.0

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    bg = torch.rand((3), device="cuda") if opt.random_background else background
    img_num = -1

    for iteration in range(first_iter, opt.iterations + 1):

        if websockets:
            if network_gui_ws.curr_id >= 0 and network_gui_ws.curr_id < len(scene.getTrainCameras()):
                cam = scene.getTrainCameras()[network_gui_ws.curr_id]
                net_image = render_fastgs(cam, gaussians, pipe, background, opt.mult, 1.0)["render"]
                network_gui_ws.latest_width = cam.image_width
                network_gui_ws.latest_height = cam.image_height
                network_gui_ws.latest_result = net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

        iter_start.record()
        
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

            # 随机 pop 10个视角作为验证集
            validate_viewpoints = []
            for _ in range(10):
                rand_idx = randint(0, len(viewpoint_indices) - 1)
                validate_viewpoints.append(viewpoint_stack.pop(rand_idx))
                _ = viewpoint_indices.pop(rand_idx)

            if img_num == -1:
                img_num = len(viewpoint_stack)

        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        _ = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            iter_time = iter_start.elapsed_time(iter_end)
            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time, testing_iterations, scene, render_fastgs, (pipe, background, opt.mult))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians to {}".format(iteration, dataset.model_path))
                scene.save(iteration)
            
            optim_start.record()

            # Optimization step
            if iteration < opt.iterations:
                if getattr(gaussians, "delay_prune_mask", None) is not None:
                    # 手动清除 delay prune 点的梯度，避免影响训练
                    gaussians._opacity.grad[gaussians.delay_prune_mask] = 0.0

                if opt.optimizer_type == "default":
                    gaussians.optimizer_step(iteration)
                elif opt.optimizer_type == "sparse_adam":
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
            
        # Densification
        if iteration < opt.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                
                if opt.use_validate_cam_list:
                    camlist = validate_viewpoints.copy()
                else:
                    my_viewpoint_stack = scene.getTrainCameras().copy()
                    camlist = sampling_cameras(my_viewpoint_stack)

                if use_rl_densification:
                    gaussians_state_for_rl, metric_score, visible_mask, global_metric, _, pre_metric_map_list, fastgs_score = get_gaussians_state_for_rl(scene, camlist, gaussians, pipe, bg, opt, iteration=iteration)
                    pre_global_metric = global_metric.clone()
                    pre_metric_score = metric_score.clone()
                    pre_visible_mask = visible_mask.clone()
                    # pre_uv_gradient_accum = uv_gradient_accum.clone()

                    if opt.verbose:
                        print(f"global_metric: {global_metric.item():.8f}")
                        print(f"metric_score min: {metric_score[visible_mask].min().item():.8f}, metric_score max: {metric_score[visible_mask].max().item():.8f}, metric_score mean: {metric_score[visible_mask].mean().item():.8f}")
                    prune_score = metric_score.clone()
                    # assert not metric_score.isnan().any()
                    # assert not metric_score.isinf().any()

                    pre_gradient_norm_accm = (gaussians.xyz_gradient_accum / gaussians.denom).squeeze(1)
                    pre_gradient_norm_accm[pre_gradient_norm_accm.isnan()] = 0.

                    with torch.no_grad():
                        if opt.use_fastgs_metric_score:
                            metric_score, prune_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt, DENSIFY=True)
                        gaussians.densify_and_prune_rl(size_threshold, 0.005, scene.cameras_extent, radii, opt,
                            gaussians_state_for_rl, iteration=iteration, tb_writer=tb_writer, importance_score=fastgs_score, metric_score=metric_score, prune_score=prune_score, visible_mask=visible_mask)
                else:
                    with torch.no_grad():
                        # The multiview consistent densification of fastgs
                        importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt, DENSIFY=True)                    
                        gaussians.densify_and_prune_fastgs(max_screen_size = size_threshold, 
                                                    min_opacity = 0.005, 
                                                    extent = scene.cameras_extent, 
                                                    radii=radii,
                                                    args=opt,
                                                    importance_score=importance_score,
                                                    pruning_score=pruning_score)

            if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                my_min_opacity = 0.005
                # if use_rl_densification:
                #     my_min_opacity = cosine_annealing(
                #         iteration - opt.densify_from_iter,
                #         opt.densify_until_iter - opt.densify_from_iter,
                #         opt.my_min_opacity_init,
                #         opt.my_min_opacity_final,
                #     )
                gaussians.reset_opacity(my_min_opacity + 0.005)
                # gaussians.reset_opacity(my_min_opacity * 2)

        # The multiview consistent pruning of fastgs. We do it every 3k iterations after 15k
        # In this stage, the model converge basically. So we can prune more aggressively without degrading rendering quality.
        # You can check the rendering results of 20K iterations in arxiv version (https://arxiv.org/abs/2511.04283), the rendering quality is already very good.
        if iteration % 3000 == 0 and iteration > opt.densify_until_iter and iteration < opt.iterations:
            if use_rl_densification:
                gaussians_state_for_rl, metric_score, visible_mask, global_metric, _, _, _ = get_gaussians_state_for_rl(scene, camlist, gaussians, pipe, bg, opt, iteration=iteration)
                prune_score = metric_score.clone()
                with torch.no_grad():
                    gaussians.final_prune_rl(min_opacity = opt.my_min_opacity_final, gaussians_state_for_rl=gaussians_state_for_rl, prune_score=prune_score, extent=scene.cameras_extent)
            else:
                with torch.no_grad():
                    _, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt)                  
                    gaussians.final_prune_fastgs(min_opacity = opt.my_min_opacity_final, pruning_score = pruning_score)
                    
            if len(gaussians.rl_controller.transition["state_list"]) == opt.rl_rollout_batch_size:
                lr = gaussians.rl_controller.update_learning_rate(delay_iteration - opt.densify_from_iter)
                if opt.verbose:
                    print("lr:", lr)
                # gaussians.rl_controller.learn(iteration=iteration, tb_writer=tb_writer)
                gaussians.parent_mapping = None
                gaussians.rl_controller.transition.clear()

        # record time
        optim_end.record()
        torch.cuda.synchronize()
        optim_time = optim_start.elapsed_time(optim_end)
        total_time += (iter_time + optim_time) / 1e3

    if use_rl_densification:
        torch.save(gaussians.rl_controller.capture(), os.path.join(dataset.model_path, "rl_controller.pth"))

    # scene.save(iteration)

    with open(os.path.join(dataset.model_path, "time_and_count.txt"), 'w') as file:
        file.write(f"Gaussian number: {gaussians._xyz.shape[0]}\n")
        file.write(f"Training time: {total_time}\n")

    print(f"Gaussian number: {gaussians._xyz.shape[0]}")
    print(f"Training time: {total_time}")

    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 10):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--websockets", action='store_true', default=False)
    parser.add_argument("--benchmark_dir", type=str, default=None)
    parser.add_argument("--use_rl_densification", action='store_true', default=False)
    parser.add_argument("--rl_controller_path", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # if args.use_rl_densification:
    #     # 备份代码到output dir
    #     try:
    #         saveRuntimeCode(os.path.join(args.model_path, 'backup'))
    #     except Exception as e:
    #         print(f"保存代码失败，原因: {e}")

    if(args.websockets):
        network_gui_ws.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint, 
        args.debug_from, 
        args.websockets,
        args.use_rl_densification,
        args.rl_controller_path
    )

    # All done
    print("\nTraining complete.")
