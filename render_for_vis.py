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

import torch
from scene import Scene
import os
import cv2
import numpy as np
import math
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_fastgs
import torchvision
from utils.general_utils import safe_state, build_scaling_rotation
from utils.sh_utils import eval_sh
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import time

from utils.loss_utils import gaussian



def compute_ellipses(gaussians, view, visibility_filter, mult, scale_threshold=None):
    visible_indices = visibility_filter.squeeze()
    if visible_indices.numel() == 0:
        return None, None, None

    

    # Get visible gaussian parameters
    
    scales = gaussians.get_scaling[visible_indices]

    if scale_threshold is not None:
        # print("scale_threshold:", scale_threshold)
        scale_valid_mask = scales.max(dim=1).values < scale_threshold
        visible_indices = visible_indices[scale_valid_mask]
        scales = scales[scale_valid_mask]

    xyz = gaussians.get_xyz[visible_indices]
    rots = gaussians.get_rotation[visible_indices]

    # Compute colors from SHs
    shs_view = gaussians.get_features[visible_indices].transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
    dir_pp = (xyz - view.camera_center.repeat(xyz.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
    colors = torch.clamp_min(sh2rgb + 0.5, 0.0)

    # Compute 3D Covariance
    L = build_scaling_rotation(scales * mult, rots)
    cov3D = L @ L.transpose(1, 2) # (N, 3, 3)

    # Project to View Space
    # view.world_view_transform is (4, 4). 
    # It assumes p_view = p_world @ world_view_transform (if p is row vector)
    # The top-left 3x3 is the rotation part W.
    W = view.world_view_transform[:3, :3] 
    t = view.world_view_transform[3, :3]

    # Transform means to view space
    means_view = xyz @ W + t
    
    # Transform covariance to view space
    # Sigma_view = W.T @ Sigma_world @ W
    cov_view = torch.matmul(W.T, torch.matmul(cov3D, W))

    # Project to 2D
    x, y, z = means_view[:, 0], means_view[:, 1], means_view[:, 2]
    
    tan_fovx = math.tan(view.FoVx * 0.5)
    tan_fovy = math.tan(view.FoVy * 0.5)
    
    # Clamping for Jacobian (as in CUDA implementation)
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    
    txtz = x / z
    tytz = y / z
    
    x_clamped = torch.min(torch.tensor(limx, device=x.device), torch.max(torch.tensor(-limx, device=x.device), txtz)) * z
    y_clamped = torch.min(torch.tensor(limy, device=y.device), torch.max(torch.tensor(-limy, device=y.device), tytz)) * z
    
    f_x = view.image_width / (2.0 * tan_fovx)
    f_y = view.image_height / (2.0 * tan_fovy)

    # Jacobian J using clamped coordinates
    N = x.shape[0]
    J = torch.zeros((N, 2, 3), device=xyz.device)
    J[:, 0, 0] = f_x / z
    J[:, 0, 2] = -(f_x * x_clamped) / (z * z)
    J[:, 1, 1] = f_y / z
    J[:, 1, 2] = -(f_y * y_clamped) / (z * z)

    cov2D = torch.matmul(J, torch.matmul(cov_view, J.transpose(1, 2)))
    
    # Check for NaNs or Infs
    valid_mask = ~torch.isnan(cov2D).any(dim=1).any(dim=1) & ~torch.isinf(cov2D).any(dim=1).any(dim=1)
    if not valid_mask.all():
        cov2D = cov2D[valid_mask]
        colors = colors[valid_mask]
        z = z[valid_mask]
        x = x[valid_mask]
        y = y[valid_mask]
    
    # Low-pass filter (as in CUDA implementation)
    cov2D[:, 0, 0] += 0.3
    cov2D[:, 1, 1] += 0.3
    
    # Means 2D (using original unclamped coordinates)
    means2D_x = (x / z) * f_x + view.image_width / 2.0
    means2D_y = (y / z) * f_y + view.image_height / 2.0
    means2D = torch.stack([means2D_x, means2D_y], dim=1)

    return means2D, cov2D, colors, z

def draw_ellipses(image, means2D, cov2D, colors, depths, thickness=1, alpha=0.9):
    # image: (H, W, 3) numpy array (BGR)
    # means2D: (N, 2) tensor (on CUDA)
    # cov2D: (N, 2, 2) tensor (on CUDA)
    # colors: (N, 3) tensor (RGB) (on CUDA)
    # depths: (N,) tensor (on CUDA)
    
    # Analytical eigenvalue computation for 2x2 matrices
    # M = [[a, b], [b, d]] (using d instead of c to avoid confusion)
    a = cov2D[:, 0, 0]
    b = cov2D[:, 0, 1]
    d = cov2D[:, 1, 1]
    
    # Trace and Determinant
    trace = a + d
    det = a * d - b * b
    
    # Eigenvalues
    # lambda = (trace +/- sqrt(trace^2 - 4*det)) / 2
    # lambda = (trace +/- sqrt((a-d)^2 + 4*b^2)) / 2
    # Gap = sqrt((a-d)^2 + 4*b^2) / 2
    gap = torch.sqrt(torch.clamp((a - d)**2 + 4 * b**2, min=0)) * 0.5
    lambda1 = trace * 0.5 + gap # Major
    lambda2 = trace * 0.5 - gap # Minor
    
    # Major and minor axes (3 sigma)
    major = 3 * torch.sqrt(torch.clamp(lambda1, min=0.1))
    minor = 3 * torch.sqrt(torch.clamp(lambda2, min=0.1))
    
    # Angle of the major axis
    # theta = 0.5 * atan2(2*b, a-d)
    theta = torch.rad2deg(0.5 * torch.atan2(2 * b, a - d))
    
    # Convert colors to BGR and 0-255
    bgr_colors = torch.stack([colors[:, 2], colors[:, 1], colors[:, 0]], dim=1) * 255
    
    # Move to CPU for drawing
    means_np = means2D.detach().cpu().numpy()
    major_np = major.detach().cpu().numpy()
    minor_np = minor.detach().cpu().numpy()
    theta_np = theta.detach().cpu().numpy()
    colors_np = bgr_colors.detach().cpu().numpy()
    depths_np = depths.detach().cpu().numpy()

    color_fill = (255, 229, 204)
    color_border = (102, 51, 0)
    thickness_border = 1

    H, W = image.shape[:2]
    TILE_SIZE = 16
    tiles_x = (W + TILE_SIZE - 1) // TILE_SIZE
    tiles_y = (H + TILE_SIZE - 1) // TILE_SIZE
    
    # Create buckets
    tile_buckets = [[[] for _ in range(tiles_x)] for _ in range(tiles_y)]
    
    N = means_np.shape[0]
    for i in range(N):
        cx, cy = means_np[i]
        r = major_np[i]

        # if r > 60:
        #     continue

        min_x = int(max(0, (cx - r)) // TILE_SIZE)
        max_x = int(min(W - 1, (cx + r)) // TILE_SIZE)
        min_y = int(max(0, (cy - r)) // TILE_SIZE)
        max_y = int(min(H - 1, (cy + r)) // TILE_SIZE)
        
        for ty in range(min_y, max_y + 1):
            for tx in range(min_x, max_x + 1):
                if 0 <= ty < tiles_y and 0 <= tx < tiles_x:
                    tile_buckets[ty][tx].append(i)
    
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            indices = tile_buckets[ty][tx]
            if not indices:
                continue
                
            # Sort by depth descending (far to close)
            indices.sort(key=lambda idx: depths_np[idx], reverse=True)
            
            # ROI
            x_start = tx * TILE_SIZE
            y_start = ty * TILE_SIZE
            x_end = min((tx + 1) * TILE_SIZE, W)
            y_end = min((ty + 1) * TILE_SIZE, H)
            
            roi = image[y_start:y_end, x_start:x_end]
            
            for i in indices:
                # Shift center to ROI coordinates
                center = (int(means_np[i, 0]) - x_start, int(means_np[i, 1]) - y_start)
                axes = (int(major_np[i]), int(minor_np[i]))
                angle = theta_np[i]
                
                # Draw on ROI
                overlay = roi.copy()
                cv2.ellipse(overlay, center, axes, angle, 0, 360, color_fill, -1)
                cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
                cv2.ellipse(roi, center, axes, angle, 0, 360, color_border, thickness_border)
        
    return image

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args, scale_threshold=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    total_time = 0.0

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx != args.view_id:
            continue
        start_time = time.time()
        rendering_dict = render_fastgs(view, gaussians, pipeline, background, args.mult)
        rendering = rendering_dict["render"]
        end_time = time.time()
        total_time += (end_time - start_time)
        gt = view.original_image[0:3, :, :]
        
        # Visualization
        visibility_filter = rendering_dict["visibility_filter"]
        
        # Converting rendering to numpy for drawing
        vis_img = rendering.detach().permute(1, 2, 0).cpu().numpy()
        vis_img = (np.clip(vis_img, 0, 1) * 255).astype(np.uint8)
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

        # vis_img = np.zeros_like(vis_img)

        # Compute and draw
        means2D, cov2D, colors, depths = compute_ellipses(gaussians, view, visibility_filter, args.mult, scale_threshold=scale_threshold)
        #  # 把colors变成随机的颜色
        #  colors = torch.rand_like(colors)
        if means2D is not None:
            vis_img = draw_ellipses(vis_img, means2D, cov2D, colors, depths, thickness=1, alpha=0.9)

        cv2.imwrite(os.path.join(render_path, '{0:05d}'.format(idx) + ".png"), vis_img)
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    
    num_frames = len(views)
    avg_time = total_time / num_frames if num_frames > 0 else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0
    print(f"[{name}] Rendered {num_frames} frames in {total_time:.2f} seconds. Average FPS: {fps:.2f}")


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args, action_mask=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, optimizer_type="default")
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # gaussians.load_ply("visualization/10000_iter_points.ply")

        # gaussians._scaling = gaussians.scaling_inverse_activation(gaussians.get_scaling * 0.5)

        print(f"Gaussian number: {gaussians.get_xyz.shape[0]}")

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # scale_threshold = 0.05 * scene.cameras_extent
        scale_threshold = None


        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args, scale_threshold=scale_threshold)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args, scale_threshold=scale_threshold)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mult", type=float, default=0.5)
    parser.add_argument("--view_id", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.9)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)