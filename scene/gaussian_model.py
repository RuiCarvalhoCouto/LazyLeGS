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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, identity_gate
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, cosine_annealing
from scene.rl_controller import GaussianDensificationController

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def modify_functions(self):
        old_opacities = self.get_opacity.clone()
        self.opacity_activation = torch.abs
        self.inverse_opacity_activation = identity_gate
        self._opacity = self.opacity_activation(old_opacities)

    def __init__(self, sh_degree, optimizer_type="default", training_args=None, use_rl_densification=False):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.shoptimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.training_args = training_args
        
        self.use_rl_densification = use_rl_densification
        self.rl_controller = None
        # 用于追踪稠密化后点和原点的映射关系
        self.parent_mapping = None
        if use_rl_densification:
            self.rl_controller = GaussianDensificationController(training_args=training_args, device="cuda")

        self.setup_functions()

    def capture(self, optimizer_type):
        if optimizer_type == "default":
            return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.optimizer.state_dict(),
            self.shoptimizer.state_dict(),
            self.spatial_lr_scale,
        )
        else:
            return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum,
        xyz_gradient_accum_abs, 
        denom,
        opt_dict, 
        shopt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.shoptimizer.load_state_dict(shopt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def clear_grad(self):
        """清空所有可训练参数的梯度"""
        # 清空所有 Gaussian 参数的梯度
        params = [
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
        ]
        for param in params:
            if param is not None and hasattr(param, 'grad') and param.grad is not None:
                param.grad = None

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.lowfeature_lr, "name": "f_dc"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        sh_l = [{'params': [self._features_rest], 'lr': training_args.highfeature_lr / 20.0, "name": "f_rest"}]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            self.shoptimizer = torch.optim.Adam(sh_l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            self.optimizer = SparseGaussianAdam(l + sh_l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        if self.use_rl_densification:
            self.rl_controller.training_setup(training_args)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        # if self.use_rl_densification:
        #     self.rl_controller.update_learning_rate(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def optimizer_step(self, iteration):
        ''' An optimization schdeuler. The goal is similar to the sparse Adam of taming 3dgs.'''
        if iteration <= 15000:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none = True)
            if iteration % 16 == 0:
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)
        elif iteration <= 20000:
            if iteration % 32 ==0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none = True)
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)
        else:
            if iteration % 64 ==0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none = True)
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self, min_opacity=0.01):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*min_opacity))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.tmp_radii is not None:
            self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # abs
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split_fastgs(self, metric_mask, filter, N=2):
        n_init_points = self.get_xyz.shape[0]

        selected_pts_mask = torch.zeros((n_init_points), dtype=bool, device="cuda")
        mask = torch.logical_and(metric_mask, filter)
        selected_pts_mask[:mask.shape[0]] = mask

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone_fastgs(self, metric_mask, filter):
        selected_pts_mask = torch.logical_and(metric_mask, filter)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune_fastgs(self, max_screen_size, min_opacity, extent, radii, args, importance_score = None, pruning_score = None):
        
        ''' 
            Densification and Pruning based on FastGS criteria:
            1.  The gaussians candidate for densification are selected based on the gradient of their position first.
            2.  Then, based on their average metric score (computed over multiple sampled views), they are either densified (cloned) or split.
                This is our main contribution compared to the vanilla 3DGS.
            3.  Finally, gaussians with low opacity or very large size are pruned.
        '''
        grad_vars = self.xyz_gradient_accum / self.denom
        grad_vars[grad_vars.isnan()] = 0.0
        self.tmp_radii = radii

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        grad_qualifiers = torch.where(torch.norm(grad_vars, dim=-1) >= args.grad_thresh, True, False)
        grad_qualifiers_abs = torch.where(torch.norm(grads_abs, dim=-1) >= args.grad_abs_thresh, True, False)
        clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= args.dense*extent
        split_qualifiers = torch.max(self.get_scaling, dim=1).values > args.dense*extent

        all_clones = torch.logical_and(clone_qualifiers, grad_qualifiers)
        all_splits = torch.logical_and(split_qualifiers, grad_qualifiers_abs)

        # This is our multi-view consisent metric for densification
        # We use this metric to further filter the candidates for densification, which is similar to taming 3dgs.
        metric_mask = importance_score > args.fastgs_importance_score_thresh

        self.densify_and_clone_fastgs(metric_mask, all_clones)
        self.densify_and_split_fastgs(metric_mask, all_splits)

        # n_all_qualifiers = torch.logical_and(torch.logical_or(clone_qualifiers, split_qualifiers), metric_mask).sum()
        # n_actual_clones = torch.logical_and(all_clones, metric_mask).sum()
        # n_actual_splits = torch.logical_and(all_splits, metric_mask).sum()
        # print(f"n_all_qualifiers: {n_all_qualifiers}, actual_clones: {n_actual_clones / n_all_qualifiers:.4f}, actual_splits: {n_actual_splits / n_all_qualifiers:.4f}")

        prune_mask = (self.get_opacity < min_opacity).squeeze(-1)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        scores = 1 - pruning_score 
        to_remove = torch.sum(prune_mask)
        remove_budget = int(0.5 * to_remove)

        # The budget is not necessary for our method.
        if remove_budget:
            n_init_points = self.get_xyz.shape[0]
            padded_importance = torch.zeros((n_init_points), dtype=torch.float32)
            padded_importance[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze(-1))
            selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool, device="cuda")
            sampled_indices = torch.multinomial(padded_importance, remove_budget, replacement=False)
            selected_pts_mask[sampled_indices] = True
            final_prune = torch.logical_and(prune_mask, selected_pts_mask)
            self.prune_points(final_prune)
        
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.8))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def densify_and_prune_rl(self, max_screen_size, min_opacity, extent, radii, args, gaussians_state_for_rl, iteration=None, tb_writer=None, importance_score=None, metric_score=None, prune_score=None, visible_mask=None, dataset=None):
        xyz = self.get_xyz
        curr_n_points = xyz.shape[0]
        opacity = self.get_opacity.squeeze(-1)

        opacity_median = opacity.median().item()
        opacity_mean = opacity.mean().item()
        if self.training_args.verbose:
            print("opacity_median:", opacity_median, "opacity_mean:", opacity_mean)

        grad_vars = self.xyz_gradient_accum / self.denom
        grad_vars[grad_vars.isnan()] = 0.0
        self.tmp_radii = radii

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        grad_qualifiers = torch.where(torch.norm(grad_vars, dim=-1) >= args.grad_thresh, True, False)
        grad_qualifiers_abs = torch.where(torch.norm(grads_abs, dim=-1) >= args.grad_abs_thresh, True, False)

        vanilla_grad_qualifiers = torch.where(torch.norm(grad_vars, dim=-1) >= 0.0002, True, False)
        clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= 0.01 * extent
        split_qualifiers = torch.max(self.get_scaling, dim=1).values > 0.01 * extent
        vanilla_all_clones = torch.logical_and(clone_qualifiers, vanilla_grad_qualifiers)
        vanilla_all_splits = torch.logical_and(split_qualifiers, vanilla_grad_qualifiers)

        #! 注意条件
        # valid_mask = grad_qualifiers
        valid_mask = torch.logical_or(grad_qualifiers, grad_qualifiers_abs)
        if self.training_args.verbose:
            print(f"n grad qualifiers points: {valid_mask.sum().item() / curr_n_points:.4f}")

        # if importance_score is not None and iteration > 3000:
        #     important_mask = importance_score > self.training_args.fastgs_metric_score_thresh
        #     valid_mask = torch.logical_or(valid_mask, important_mask)
        #     if self.training_args.verbose:
        #         print(f"n important points: {important_mask.sum().item() / curr_n_points:.4f}")

        valid_mask = torch.logical_and(valid_mask, visible_mask)

        # my_min_opacity = 0.005
        my_min_opacity = cosine_annealing(
            iteration - args.densify_from_iter,
            args.densify_until_iter - args.densify_from_iter,
            initial_temp=0.01, final_temp=0.1
        )
        prune_mask = opacity < (my_min_opacity if self.training_args.use_prune_estimator else min_opacity)

        # if metric_score is not None:
        #     metric_score = metric_score[visible_mask]
        #     metric_score = (metric_score - metric_score.min()) / (metric_score.max() - metric_score.min())
        #     metric_score_prune_mask = metric_score < 0.1
        #     if self.training_args.verbose:
        #         print(f"n metric score prune mask points: {metric_score_prune_mask.sum().item() / curr_n_points:.4f}")
        #     prune_mask[visible_mask] = metric_score_prune_mask

        if self.training_args.verbose:
            print(f"n final valid mask points: {valid_mask.sum().item() / curr_n_points:.4f}")
            print(f"n final prune mask points: {prune_mask.sum().item() / curr_n_points:.4f}")

        if args.use_prune_estimator and visible_mask is not None:
            prune_mask = torch.logical_and(prune_mask, visible_mask)
        if self.training_args.verbose:
            print(f"n opacity prune mask points: {prune_mask.sum().item() / curr_n_points:.4f}")

        if args.use_prune_estimator:
            # none_prune_mask = valid_mask.clone()

            valid_mask = torch.logical_or(valid_mask, prune_mask)

            # none_prune_mask = none_prune_mask[valid_mask]

            prune_mask = prune_mask[valid_mask]
            none_prune_mask = torch.logical_not(prune_mask)

            encoded_state = self.rl_controller.state_encoder(gaussians_state_for_rl[valid_mask])

            if prune_mask.any():
                prune_probs = self.rl_controller.prune_estimator(encoded_state[prune_mask])

            none_prune_probs = self.rl_controller.actor(encoded_state[none_prune_mask])

            if self.training_args.verbose:
                if args.use_delete_action:
                    print(f"Actor prob distribution: keep={none_prune_probs[:, 0].mean().item():.4f}, clone={none_prune_probs[:, 1].mean().item():.4f}, split={none_prune_probs[:, 2].mean().item():.4f}, delete={none_prune_probs[:, 3].mean().item():.4f}")
                else:
                    print(f"Actor prob distribution: keep={none_prune_probs[:, 0].mean().item():.4f}, clone={none_prune_probs[:, 1].mean().item():.4f}, split={none_prune_probs[:, 2].mean().item():.4f}")
                if prune_mask.any():
                    print(f"Prune estimator prob distribution: accept={prune_probs.mean().item():.4f}")

            dist = torch.distributions.Categorical(none_prune_probs)
            non_prune_action = dist.sample().long()

            if prune_mask.any():
                prune_action = torch.where(torch.bernoulli(prune_probs).bool(), 3, 0).squeeze(-1).long()
                if self.training_args.verbose:
                    print(f"prune action distribution: keep={(prune_action == 0).sum().item() / prune_action.shape[0]}, prune={(prune_action == 3).sum().item() / prune_action.shape[0]}")

            action = torch.zeros(encoded_state.shape[0], device="cuda", dtype=torch.long)
            action[none_prune_mask] = non_prune_action
            if prune_mask.any():
                action[prune_mask] = prune_action

            # none_prune_probs = none_prune_probs.gather(1, non_prune_action.unsqueeze(-1))
            # prune_probs = torch.where(prune_action.unsqueeze(-1) == 3, prune_probs, 1 - prune_probs)
            # _none_prune_probs = torch.zeros(encoded_state.shape[0], 1, device="cuda")
            # _prune_probs = torch.zeros(encoded_state.shape[0], 1, device="cuda")
            # _none_prune_probs[none_prune_mask] = none_prune_probs
            # _prune_probs[prune_mask] = prune_probs
            # prune_probs = _prune_probs
            # none_prune_probs = _none_prune_probs
            
            # # 重叠时由概率决定action归属none prune probs还是prune probs
            # overlap_mask = torch.logical_and(none_prune_mask, prune_mask)
            # if overlap_mask.any():
            #     overlap_prune_probs = prune_probs[overlap_mask]
            #     overlap_none_prune_probs = none_prune_probs[overlap_mask]
            #     # 归一化重叠的prune合不prune的概率
            #     overlap_probs = torch.stack([overlap_none_prune_probs, overlap_prune_probs], dim=-1)
            #     overlap_probs = overlap_probs / overlap_probs.sum(dim=-1, keepdim=True)
            #     overlap_action = torch.distributions.Categorical(overlap_probs).sample().long().squeeze(-1)
            #     action[overlap_mask] = overlap_action

            _action = torch.zeros(curr_n_points, device="cuda", dtype=torch.long)
            _action[valid_mask] = action
            action = _action

            _prune_mask = torch.zeros(curr_n_points, device="cuda", dtype=bool)
            _prune_mask[valid_mask] = prune_mask
            prune_mask = _prune_mask

        else:
            valid_mask[prune_mask] = False

            encoded_state = self.rl_controller.state_encoder(gaussians_state_for_rl[valid_mask])
            prob = self.rl_controller.actor(encoded_state)
            
            if self.training_args.verbose:
                if args.use_delete_action:
                    print(f"Actor prob distribution: keep={prob[:, 0].mean().item():.4f}, clone={prob[:, 1].mean().item():.4f}, split={prob[:, 2].mean().item():.4f}, delete={prob[:, 3].mean().item():.4f}")
                else:
                    print(f"Actor prob distribution: keep={prob[:, 0].mean().item():.4f}, clone={prob[:, 1].mean().item():.4f}, split={prob[:, 2].mean().item():.4f}")
            
            #* 决定gs点可执行的action
            dist = torch.distributions.Categorical(prob)
            action = dist.sample().long()# (n,)

            _action = torch.zeros(curr_n_points, device="cuda", dtype=torch.long)
            _action[valid_mask] = action
            action = _action
            action[prune_mask] = 3

        # 统计每个动作的比例
        actor_valid_mask = valid_mask & (~prune_mask)
        keep_ratio = torch.logical_and(action == 0, actor_valid_mask).sum().item() / actor_valid_mask.sum().item()
        clone_ratio = torch.logical_and(action == 1, actor_valid_mask).sum().item() / actor_valid_mask.sum().item()
        split_ratio = torch.logical_and(action == 2, actor_valid_mask).sum().item() / actor_valid_mask.sum().item()
        if prune_mask.any():
            delete_ratio = torch.logical_and(action == 3, prune_mask).sum().item() / prune_mask.sum().item()
        else:
            delete_ratio = 0.0
        tb_writer.add_scalar("rl/action_keep_ratio", keep_ratio, iteration)
        tb_writer.add_scalar("rl/action_clone_ratio", clone_ratio, iteration)
        tb_writer.add_scalar("rl/action_split_ratio", split_ratio, iteration)
        tb_writer.add_scalar("rl/action_delete_ratio", delete_ratio, iteration)

        # 保存action可视化
        if iteration == 5000 and getattr(args, "visualize_policy", False):
            vis_action = action.clone()
            vis_action[~valid_mask] = -1
            self.save_ply(os.path.join(dataset.model_path, "visualization/5000_iter_points.ply"))
            np.save(os.path.join(dataset.model_path, "visualization/5000_iter_action_from_rl.npy"), vis_action.detach().view(-1).cpu().numpy())

            action_from_fastgs = torch.zeros_like(action)
            action_from_fastgs[vanilla_all_clones] = 1
            action_from_fastgs[vanilla_all_splits] = 2
            np.save(os.path.join(dataset.model_path, "visualization/5000_iter_action_vanilla.npy"), action_from_fastgs.detach().view(-1).cpu().numpy())

        clone_action_mask = action == 1
        split_action_mask = action == 2
        delete_action_mask = action == 3

        actual_clone_indices = torch.where(clone_action_mask)[0]
        actual_split_indices = torch.where(split_action_mask)[0]
        actual_delete_indices = torch.where(delete_action_mask)[0]

        parent_mapping = torch.arange(curr_n_points, device="cuda")
        reward = torch.zeros(curr_n_points, device="cuda")

        # clone
        new_xyz = self._xyz[actual_clone_indices]
        new_features_dc = self._features_dc[actual_clone_indices]
        new_features_rest = self._features_rest[actual_clone_indices]
        new_opacity = self._opacity[actual_clone_indices]
        new_scaling = self._scaling[actual_clone_indices]
        new_rotation = self._rotation[actual_clone_indices]
        new_tmp_radii = self.tmp_radii[actual_clone_indices]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)
        parent_mapping = torch.cat([parent_mapping, actual_clone_indices])

        # split
        N=2
        new_xyz = self._xyz[actual_split_indices].repeat(N,1)
        stds = self.get_scaling[actual_split_indices].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[actual_split_indices]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[actual_split_indices].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[actual_split_indices].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[actual_split_indices].repeat(N,1)
        new_features_dc = self._features_dc[actual_split_indices].repeat(N,1,1)
        new_features_rest = self._features_rest[actual_split_indices].repeat(N,1,1)
        new_opacity = self._opacity[actual_split_indices].repeat(N,1)
        new_tmp_radii = self.tmp_radii[actual_split_indices].repeat(N)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)
        parent_mapping = torch.cat([parent_mapping, actual_split_indices.repeat(N)])

        # delete
        padded_prune_mask = torch.zeros(self.get_xyz.shape[0], device="cuda", dtype=bool)
        #* 注意先处理delete action，再删除split action的原点，以保证下标不错乱
        padded_prune_mask[actual_delete_indices] = True

        #* 注意别忘记split后要删除原点
        split_prune_mask = torch.zeros(self.get_xyz.shape[0], device="cuda", dtype=bool)
        split_prune_mask[actual_split_indices] = True
        self.prune_points(split_prune_mask)
        parent_mapping = parent_mapping[~split_prune_mask]
        padded_prune_mask = padded_prune_mask[~split_prune_mask]

        # #* 标记为delay prune的点，渲染时不透明度会置为负数并跳过，不影响渲染结果
        # self.delay_prune_mask = padded_prune_mask

        self.prune_points(padded_prune_mask)
        parent_mapping = parent_mapping[~padded_prune_mask]

        self.tmp_radii = None

        action = action.unsqueeze(-1)
        reward = reward.unsqueeze(-1)

        self.rl_controller.store_transition(
            gaussians_state_for_rl,
            action,
            reward,
            parent_mapping=self.parent_mapping,
            valid_mask=valid_mask,
            prune_mask=prune_mask,
        )

        self.parent_mapping = parent_mapping
        if self.training_args.verbose:
            print(f"n_points_after_densification: {self.get_xyz.shape[0]}, n_valid_points_after_densification: {(self.get_opacity >= min_opacity).sum().item()}")
    
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, 2:], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def final_prune_fastgs(self, min_opacity, pruning_score = None):
        """Final-stage pruning: remove Gaussians based on opacity and multi-view consistency.
        In the final stage we remove Gaussians that have low opacity or that are flagged by
        our multi-view reconstruction consistency metric (provided as `pruning_score`)."""
        prune_mask = (self.get_opacity < min_opacity).squeeze(-1) 
        if self.training_args.verbose:
            print("n opacity prune points:", prune_mask.sum().item())
        scores_mask = pruning_score > 0.9
        if self.training_args.verbose:
            print("n prune score prune points:", scores_mask.sum().item())
        final_prune = torch.logical_or(prune_mask, scores_mask)
        self.prune_points(final_prune)

    def final_prune_rl(self, min_opacity, gaussians_state_for_rl, prune_score=None, extent=None):
        prune_mask = (self.get_opacity < min_opacity).squeeze(-1)
        if self.training_args.verbose:
            print("n opacity prune points:", prune_mask.sum().item())
        final_prune = prune_mask

        if self.training_args.verbose:
            print(f"actual final prune: {final_prune.sum().item()}")

        self.prune_points(final_prune)
