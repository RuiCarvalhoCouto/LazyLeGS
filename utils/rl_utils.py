import torch
from fused_ssim import FusedSSIMMap
from fused_ssim import fused_ssim as fast_ssim
from utils.image_utils import psnr

from gaussian_renderer import render_fastgs
from utils.loss_utils import l1_loss


def fast_ssim_map(img1, img2, padding="same", train=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    img1 = img1.contiguous()
    map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)
    return map.mean(0)


def sign_log1p(x):
    return x.sign() * torch.log1p(x.abs())


def normalize_features(features):
    """对特征进行标准化归一化"""
    # features = sign_log1p(features)
    return (features - features.mean(0, keepdim=True)) / features.std(0, keepdim=True)
    

def get_metric_score(scene, camlist, gaussians, pipe, bg, opt, pre_metric_map_list=None):
    """
    计算每个Gaussian点对渲染误差的贡献度
    
    改进点:
    1. 减少重复渲染（复用render_pkg）
    2. 添加可选的归一化
    """
    num_points = gaussians.get_xyz.shape[0]
    metric_score = torch.zeros(num_points, device="cuda", dtype=torch.float32)
    # fastgs_score = torch.zeros(num_points, device="cuda", dtype=torch.float32)
    global_metric = torch.zeros(1, device="cuda", dtype=torch.float32)
    gs_weights = torch.zeros(num_points, device="cuda", dtype=torch.float32)
    uv_gradient_accum = torch.zeros(num_points, 1, device="cuda", dtype=torch.float32)

    for view in range(len(camlist)):
        my_viewpoint_cam = camlist[view]
        
        # 第一次渲染：获取渲染图像
        render_pkg = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, opt.mult)
        render_image = render_pkg['render']
        gt_image = my_viewpoint_cam.original_image.cuda()

        render_image.clamp_(min=0.0, max=1.0)
        gt_image.clamp_(min=0.0, max=1.0)

        # mse = torch.square(render_image - gt_image).mean(0)
        # metric_map = torch.log10(1.0 / (mse.sqrt() + 1e-10))
        # global_metric += metric_map.mean()

        # 计算梯度
        Ll1 = l1_loss(render_image, gt_image)
        ssim_value = fast_ssim(render_image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        visibility_filter = render_pkg["visibility_filter"].squeeze(-1)
        uv_gradient_accum[visibility_filter] += torch.norm(
            render_pkg["viewspace_points"].grad[visibility_filter, :2], dim=-1, keepdim=True
        )

        gaussians.optimizer.zero_grad(set_to_none=True)
        if getattr(gaussians, "shoptimizer", None) is not None:
            gaussians.shoptimizer.zero_grad(set_to_none=True)
        gaussians.clear_grad()
        
        # pre_metric_map = pre_metric_map_list[view]
        # # metric_map = (metric_map - pre_metric_map) / pre_metric_map
        # metric_map = pre_metric_map - metric_map

        render_pkg = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, opt.mult, get_flag=True, metric_map=None, gt_image=gt_image)
        # accum_metric_counts = render_pkg["accum_metric_counts"]
        accum_metric_per_gs = render_pkg["accum_metric_per_gs"]
        accum_gs_weight = render_pkg["accum_gs_weight"]

        metric_score += accum_metric_per_gs
        gs_weights += accum_gs_weight

    # 平均多视角的metric
    visible_mask = gs_weights > 0
    global_metric = global_metric / len(camlist)
    if opt.verbose:
        print("avg gs pixel count:", gs_weights[visible_mask].float().mean().item())
        print("original score max:", metric_score.max().item(), "min:", metric_score.min().item(), "mean:", metric_score[visible_mask].mean().item())

    metric_score = sign_log1p(metric_score).clamp(min=-6.0, max=6.0)

    uv_gradient_accum = uv_gradient_accum.squeeze(1)
    # uv_gradient_accum[visible_mask] = uv_gradient_accum[visible_mask] / gs_weights[visible_mask]
    # metric_score[visible_mask] = metric_score[visible_mask] / gs_weights[visible_mask]
    # assert (metric_score[~visible_mask] == 0).all()

    return metric_score, global_metric, visible_mask, uv_gradient_accum


def get_gaussians_state_for_rl(scene, camlist, gaussians, pipe, bg, opt, iteration=None, return_metric_map=False):
    """
    构建RL的状态特征
    
    改进点:
    1. 添加更多信息特征（局部密度、可见性统计、训练进度）
    2. 分组归一化（梯度特征、属性特征、统计特征分别归一化）
    3. 使用鲁棒归一化减少异常值影响
    4. 添加训练进度信息
    
    状态特征维度（当前）:
    - 梯度特征 (7维): xyz_grads(3), scale_grads(3), uv_gradient(1)
    - 属性特征 (3维): opacity(1), max_scale(1), metric_score(1)
    总计: 10维
    """
    num_points = len(gaussians.get_xyz)
    
    # 梯度累积
    xyz_grads = torch.zeros(num_points, 3, device="cuda", dtype=torch.float32)
    scale_grads = torch.zeros(num_points, 3, device="cuda", dtype=torch.float32)
    opacity_grads = torch.zeros(num_points, 1, device="cuda", dtype=torch.float32)
    feature_dc_grads = torch.zeros(num_points, 3, device="cuda", dtype=torch.float32)
    uv_gradient_accum = torch.zeros(num_points, 1, device="cuda", dtype=torch.float32)
    metric_score = torch.zeros(num_points, device="cuda", dtype=torch.float32)
    gs_weights = torch.zeros(num_points, device="cuda", dtype=torch.float32)
    fastgs_score = torch.zeros(num_points, device="cuda", dtype=torch.float32)

    metric_map_list = []

    global_metric = torch.zeros(1, device="cuda")
    n_views = len(camlist)
    
    for view in range(n_views):
        my_viewpoint_cam = camlist[view]
        render_pkg = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, opt.mult)

        visibility_filter = render_pkg["visibility_filter"].squeeze(-1)
        render_image = render_pkg["render"]
        gt_image = my_viewpoint_cam.original_image.cuda()

        render_image.clamp_(min=0.0, max=1.0)
        gt_image.clamp_(min=0.0, max=1.0)

        # # 计算metric map
        # mse = torch.square(render_image - gt_image).mean(0)
        # metric_map = torch.log10(1.0 / (mse.sqrt() + 1e-10))
        # # metric_map = mse
        # global_metric += metric_map.mean()
        # metric_map_list.append(metric_map)

        # with torch.no_grad():
        #     metric_map = torch.mean(torch.abs(render_image - gt_image), 0)
        #     metric_map = (metric_map - torch.min(metric_map)) / (torch.max(metric_map) - torch.min(metric_map))
        #     metric_map = (metric_map > opt.loss_thresh).float()

        # 计算梯度
        Ll1 = l1_loss(render_image, gt_image)
        ssim_value = fast_ssim(render_image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        # 累积梯度（使用绝对值或原值，这里保持原值以保留方向信息）
        if gaussians._xyz.grad is not None:
            xyz_grads += gaussians._xyz.grad
        if gaussians._scaling.grad is not None:
            scale_grads += gaussians._scaling.grad
        if gaussians._opacity.grad is not None:
            opacity_grads += gaussians._opacity.grad
        if gaussians._features_dc.grad is not None:
            # _features_dc: (N, 1, 3) -> grad: (N, 1, 3); squeeze(1) 保留 (N, 3)
            feature_dc_grads += gaussians._features_dc.grad.squeeze(1)
        if render_pkg["viewspace_points"].grad is not None:
            uv_gradient_accum += torch.norm(
                render_pkg["viewspace_points"].grad[:, :2], dim=-1, keepdim=True
            )

        gaussians.optimizer.zero_grad(set_to_none=True)
        if getattr(gaussians, "shoptimizer", None) is not None:
            gaussians.shoptimizer.zero_grad(set_to_none=True)
        gaussians.clear_grad()

        # 累积metric score
        with torch.no_grad():
            render_pkg = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, opt.mult, get_flag=True, metric_map=None, gt_image=gt_image)
            visibility_filter = render_pkg["visibility_filter"].squeeze(-1)
            accum_metric_counts = render_pkg["accum_metric_counts"]
            accum_metric_per_gs = render_pkg["accum_metric_per_gs"]
            accum_gs_weight = render_pkg["accum_gs_weight"]

            metric_score += accum_metric_per_gs
            gs_weights += accum_gs_weight
            fastgs_score += accum_metric_counts


    visible_mask = gs_weights > 0

    metric_score = sign_log1p(metric_score)
    metric_score_feature = metric_score.clone().detach().unsqueeze(-1)
    
    metric_score_feature = normalize_features(metric_score_feature)
    global_metric = global_metric / n_views
    fastgs_score /= n_views

    # fastgs_score = sign_log1p(fastgs_score)
    # fast_score = fastgs_score - fastgs_score.mean(0, keepdim=True) / fastgs_score.std(0, keepdim=True)

    # ============ 构建状态特征 ============
    
    # # 1. 梯度特征 (8维) - 使用鲁棒归一化
    grad_features = torch.cat([
        xyz_grads.clone().detach(),
        scale_grads.clone().detach(),
        opacity_grads.clone().detach(),    # 1维
        feature_dc_grads.clone().detach(), # 3维
        # uv_gradient_accum.clone().detach() # 1维
    ], dim=-1)
    grad_features = normalize_features(grad_features)

    # # 2. 属性特征 (5维)
    # opacity = gaussians.get_opacity.clone().detach()  # 1维
    # scale = gaussians.get_scaling.clone().detach() / scene.cameras_extent
    
    attr_features = torch.cat([
        metric_score_feature,
        # fast_score.unsqueeze(-1),
    ], dim=-1)

    # 5. 组合所有特征
    states = torch.cat([
        grad_features,
        attr_features,
    ], dim=-1)

    if opt.use_sparse_conv_state_encoder:
        states = torch.cat([states, gaussians.get_xyz.clone().detach()], dim=-1)

    if opt.verbose:
        print(f"states mean: {states.mean(0)}")

    uv_gradient_accum = uv_gradient_accum.squeeze(1)

    # assert (metric_score[~visible_mask] == 0).all()

    return states, metric_score, visible_mask, global_metric, uv_gradient_accum, metric_map_list, fastgs_score