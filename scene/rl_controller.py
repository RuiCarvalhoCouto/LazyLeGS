from turtle import forward
import torch
import random
import math
import spconv.pytorch as spconv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_scatter import scatter_mean
from torch.utils.checkpoint import checkpoint
from utils.general_utils import get_expon_lr_func, cosine_annealing
from collections import defaultdict

class BasicSPConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, indice_key=None) -> None:
        super().__init__()
        self.conv = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            indice_key=indice_key,
        )
        self.ln = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = x.replace_feature(self.ln(x.features))
        x = x.replace_feature(self.relu(x.features))
        return x

class SparseConvStateEncoder(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, voxel_size=0.02) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.voxel_size = voxel_size
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.conv_net = []
        for _ in range(2):
            self.conv_net.append(
                BasicSPConvBlock(hidden_dim, hidden_dim)
            )
        self.conv_net = nn.Sequential(*self.conv_net)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, state_features):
        xyz = state_features[:, -3:]
        x = self.input_proj(state_features)

        points_coords = (xyz / self.voxel_size).long()
        voxel_coords, inverse_ids = torch.unique(points_coords, return_inverse=True, dim=0)
        voxel_features = torch.zeros(voxel_coords.shape[0], x.shape[-1], device=x.device)
        voxel_features.index_reduce_(0, inverse_ids, x, reduce="mean")

        batch = torch.zeros(voxel_features.shape[0], dtype=torch.int32, device=x.device)
        sparse_shape = torch.add(torch.max(voxel_coords, dim=0).values, 1).tolist()
        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=torch.cat(
                [batch.unsqueeze(-1).int(), voxel_coords.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=batch[-1].tolist() + 1,
        )
        
        x = self.conv_net(x)
        x = self.out_proj(x.features)
        x = x[inverse_ids]

        return x



class MLPStateEncoder(nn.Module):
    """
    浅层宽网络 + SwiGLU激活
    对于小输入维度，浅而宽通常比深而窄效果更好
    """
    def __init__(self, input_dim=12, hidden_dim=64):
        super(MLPStateEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        expand_dim = hidden_dim * 4  # 更大的扩展因子
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 只用2层，但更宽 + SwiGLU
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(hidden_dim),
                'gate': nn.Linear(hidden_dim, expand_dim),
                'up': nn.Linear(hidden_dim, expand_dim),
                'down': nn.Linear(expand_dim, hidden_dim),
            }) for _ in range(3)
        ])
        
        self.output_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, state_features, xyz_coords=None):
        x = self.input_proj(state_features)
        
        for layer in self.layers:
            residual = x
            x = layer['norm'](x)
            # SwiGLU: 比GELU更有效
            gate = F.silu(layer['gate'](x))
            up = layer['up'](x)
            x = layer['down'](gate * up)
            x = x + residual
        
        return self.output_norm(x)


class PPOActor(nn.Module):
    """
    PPO策略网络 - 输出动作概率分布
    
    改进点:
    1. 简化为1层隐藏层（StateEncoder已做足够特征提取）
    2. 输出层零初始化，使初始策略接近均匀分布
    """
    def __init__(self, hidden_dim=64, action_dim=4):
        super(PPOActor, self).__init__()
        self.action_dim = action_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # 输出层零初始化，使初始策略接近均匀分布
        self._init_output_layer()
    
    def _init_output_layer(self):
        # 使用小标准差初始化，使初始logits接近0 -> softmax接近均匀分布
        # 但保留足够的梯度信号以便学习
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)
        if self.mlp[-1].bias.shape[0] == 4:
            self.mlp[-1].bias[-1].data -= 1

    def forward(self, x, temperature=1.0):
        logits = self.mlp(x) / temperature
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def get_action(self, state):
        """
        获取每个Gaussian点的动作
        
        Args:
            state: (n, hidden_dim) 每个Gaussian点的编码特征
            
        Returns:
            action: (n, 1) 每个点的动作索引
        """
        probs = self.forward(state)  # (n, action_dim)
        action = torch.argmax(probs, dim=-1, keepdim=True)  # (n, 1)
        return action


class PPOPruneEstimator(nn.Module):
    """
    Prune估计器 - 专门负责判断是否删除点
    
    改进点:
    1. 简化为1层隐藏层
    2. 输出层零初始化，使初始输出接近0.5
    """
    def __init__(self, hidden_dim=64):
        super(PPOPruneEstimator, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # 输出层零初始化，使初始输出经过Sigmoid后接近0.5
        self._init_output_layer()
    
    def _init_output_layer(self):
        # 使用小标准差初始化，保留梯度信号
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)
    
    def forward(self, x):
        return torch.sigmoid(self.mlp(x))

    def get_action(self, state):
        return (self.forward(state) > 0.5).int()


class PPOCritic(nn.Module):
    """
    PPO价值网络 - 评估状态价值
    
    改进点:
    1. 简化为1层隐藏层（与Actor保持一致）
    2. 添加LayerNorm稳定训练
    
    输入: (n, hidden_dim) 其中n是gaussian点的数量
    输出: (n, 1) 每个点的状态价值
    """
    def __init__(self, hidden_dim=64):
        super(PPOCritic, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.mlp(x)


class GaussianDensificationController:
    """使用PPO算法控制Gaussian点的densification过程"""
    def __init__(self,
                 training_args,
                 device="cuda"):
        self.device = device
        self.gamma = training_args.rl_gamma
        self.policy_clip = training_args.rl_policy_clip
        self.gae_lambda = training_args.rl_gae_lambda
        self.rl_rollout_batch_size = training_args.rl_rollout_batch_size  # time steps of one trajectory
        self.lr_scheduler = None
        self.training_args = training_args
        # 显存调试开关：优先使用 rl_debug_memory，其次复用 pipeline 的 debug
        self.mem_debug = getattr(training_args, 'rl_debug_memory', getattr(training_args, 'debug', False))
        
        # PPO Loss权重系数
        self.vf_coef = getattr(training_args, 'rl_vf_coef', 0.5)  # Value loss权重
        self.entropy_coef_init = getattr(training_args, 'entropy_loss_init', 0.1)
        self.entropy_coef_final = getattr(training_args, 'entropy_loss_final', 0.01)  # 保持一定探索
        self.value_clip = getattr(training_args, 'rl_value_clip', 0.2)  # Value clipping范围
        self.use_my_value = getattr(training_args, 'rl_use_my_value', False)

        state_dim = training_args.rl_state_dim
        hidden_dim = training_args.rl_net_hidden_dim

        self.actor = PPOActor(hidden_dim, action_dim=4 if training_args.use_delete_action else 3).to(device)

        if training_args.use_prune_estimator:
            self.prune_estimator = PPOPruneEstimator(hidden_dim).to(device)
            if not self.use_my_value:
                self.prune_critic = PPOCritic(hidden_dim).to(device)

        if not self.use_my_value:
            self.critic = PPOCritic(hidden_dim).to(device)
        if training_args.use_sparse_conv_state_encoder:
            self.state_encoder = SparseConvStateEncoder(state_dim + 3, hidden_dim).to(device)
        else:
            self.state_encoder = MLPStateEncoder(state_dim, hidden_dim).to(device)

        # 打印网络参数大小，换算成MB单位
        if self.training_args.verbose:
            print(f"Actor parameters size: {sum(p.numel() for p in self.actor.parameters()) / 1e6:.4f}MB")
            if not self.use_my_value:
                print(f"Critic parameters size: {sum(p.numel() for p in self.critic.parameters()) / 1e6:.4f}MB")
            print(f"State encoder parameters size: {sum(p.numel() for p in self.state_encoder.parameters()) / 1e6:.4f}MB")
            if training_args.use_prune_estimator:
                print(f"prune_estimator parameters size: {sum(p.numel() for p in self.prune_estimator.parameters()) / 1e6:.4f}MB")
                if not self.use_my_value:
                    print(f"prune_critic parameters size: {sum(p.numel() for p in self.prune_critic.parameters()) / 1e6:.4f}MB")
        
        # 优化器
        self.actor_optimizer = optim.AdamW(list(self.actor.parameters()), weight_decay=1e-4)
        if not self.use_my_value:
            self.critic_optimizer = optim.AdamW(list(self.critic.parameters()), weight_decay=1e-4)
        self.state_encoder_optimizer = optim.AdamW(list(self.state_encoder.parameters()), weight_decay=1e-4)

        if training_args.use_prune_estimator:
            self.prune_estimator_optimizer = optim.AdamW(list(self.prune_estimator.parameters()), weight_decay=1e-4)
            if not self.use_my_value:
                self.prune_critic_optimizer = optim.AdamW(list(self.prune_critic.parameters()), weight_decay=1e-4)

        self.transition = defaultdict(list)
        
    def _log_cuda_mem(self, tag):
        """调试用：打印当前和峰值显存（MB）"""
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                alloc = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                peak_alloc = torch.cuda.max_memory_allocated()
                peak_reserved = torch.cuda.max_memory_reserved()
                print(f"[MEM][{tag}] alloc={alloc/1e6:.1f}MB, reserved={reserved/1e6:.1f}MB, peak_alloc={peak_alloc/1e6:.1f}MB, peak_reserved={peak_reserved/1e6:.1f}MB")
        except Exception as _:
            pass
    
    def store_transition(self, state, action, reward, parent_mapping, **kwargs):
        self.transition["state_list"].append(state)
        self.transition["action_list"].append(action)
        self.transition["reward_list"].append(reward)
        self.transition["parent_mapping_list"].append(parent_mapping)
        for key, value in kwargs.items():
            self.transition[key + "_list"].append(value)
        
    def learn(self, iteration=None, tb_writer=None):
        """PPO学习过程"""
        if self.mem_debug and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self._log_cuda_mem("learn:start")
        state_list, action_list, reward_list, parent_mapping_list = self.transition["state_list"], self.transition["action_list"], self.transition["reward_list"], self.transition["parent_mapping_list"]
        valid_mask_list = self.transition.get("valid_mask_list", [torch.ones(x.shape[0], device="cuda", dtype=bool) for x in state_list])
        prune_mask_list = self.transition.get("prune_mask_list", [torch.zeros(x.shape[0], device="cuda", dtype=bool) for x in state_list])
        

        if self.mem_debug:
            self._log_cuda_mem("after_encode")

        if not self.use_my_value:
            value_list = []
        old_log_prob_list = []

        with torch.no_grad():
            for state, action, valid_mask, prune_mask in zip(state_list, action_list, valid_mask_list, prune_mask_list):
                n_points = state.shape[0]

                state = state[valid_mask]
                action = action[valid_mask]
                prune_mask = prune_mask[valid_mask]

                encoded = self.state_encoder(state)

                if not self.use_my_value:
                    # value保持原来的shape方便计算优势函数
                    value = torch.zeros(n_points, 1, device="cuda", dtype=torch.float)
                    # 使用分离的critic：prune点用prune_critic，其他点用主critic
                    if self.training_args.use_prune_estimator:
                        # 非prune点使用主critic
                        if (~prune_mask).any():
                            value_non_prune = self.critic(encoded[~prune_mask])
                        # prune点使用prune_critic
                        if prune_mask.any():
                            value_prune = self.prune_critic(encoded[prune_mask])
                        
                        # 组装回完整的value tensor
                        valid_indices = torch.where(valid_mask)[0]
                        if (~prune_mask).any():
                            value[valid_indices[~prune_mask]] = value_non_prune
                        if prune_mask.any():
                            value[valid_indices[prune_mask]] = value_prune
                    else:
                        value[valid_mask] = self.critic(encoded)
                    value_list.append(value)

                if self.training_args.use_prune_estimator:
                    probs = torch.zeros(encoded.shape[0], 1, device="cuda", dtype=torch.float)
                    
                    # 处理 prune 点
                    if prune_mask.any():
                        prune_encoded_state = encoded[prune_mask]
                        prune_probs = self.prune_estimator(prune_encoded_state)
                        prune_probs = torch.where(action[prune_mask] == 3, prune_probs, 1 - prune_probs)
                        probs[prune_mask] = prune_probs
                    
                    # 处理 non-prune 点
                    if (~prune_mask).any():
                        non_prune_encoded_state = encoded[~prune_mask]
                        non_prune_probs = self.actor(non_prune_encoded_state)
                        non_prune_probs = non_prune_probs.gather(-1, action[~prune_mask])
                        probs[~prune_mask] = non_prune_probs
                else:
                    probs = self.actor(encoded)
                    probs = probs.gather(-1, action)

                log_prob = torch.log(probs)
                # assert not log_prob.isnan().any() and not log_prob.isinf().any()
                old_log_prob_list.append(log_prob.detach())

            if self.mem_debug:
                self._log_cuda_mem("after_values_probs")


            if self.use_my_value:
                value_list = self.transition["value_list"]

            # ============ 改进的GAE计算 ============
            # lastgaelam 会在循环中变成 (n_points,1) 张量；初始化为 0 不影响首轮计算
            lastgaelam = 0
            advantage_list_reversed = []
            t_length = len(reward_list)
            
            for t in reversed(range(t_length)):
                reward_t = reward_list[t]
                # 注意parent mapping是当前点对应的父节点
                value_t = value_list[t]
                if t < t_length - 1:
                    value_t_next = value_list[t + 1]
                    # 这里的映射应来自“时间步 t 的动作”，也就是把 s_{t+1} 的点聚合回 s_t 的父点
                    parent_mapping_t_next = parent_mapping_list[t + 1]

                    # 聚合子点的value到父点（处理clone/split后点数变化）
                    next_value_aggregated = scatter_mean(
                        value_t_next,
                        parent_mapping_t_next.unsqueeze(-1),
                        dim=0,
                        dim_size=value_t.shape[0],
                    )
                    
                    # TD误差: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
                    delta = reward_t + self.gamma * next_value_aggregated - value_t

                    # GAE: A_t = δ_t + (γλ) * A_{t+1}
                    lastgaelam_aggregated = scatter_mean(
                        lastgaelam,
                        parent_mapping_t_next.unsqueeze(-1),
                        dim=0,
                        dim_size=value_t.shape[0],
                    )
                    lastgaelam = delta + self.gamma * self.gae_lambda * lastgaelam_aggregated
                else:
                    # 最后一个时间步，没有下一个状态
                    lastgaelam = reward_t - value_t
                advantage_list_reversed.append(lastgaelam)
            advantage_list = advantage_list_reversed[::-1]  # 反转得到正序

        # # Advantage标准化 - 分别对prune和non-prune点归一化
        # # 因为两个任务使用不同的critic，value分布和目标不同，需要分开归一化以避免梯度失衡
        # if self.training_args.rl_adv_norm:
        #     if self.training_args.use_prune_estimator:
        #         # 收集所有prune和non-prune点的advantage
        #         all_prune_adv = []
        #         all_non_prune_adv = []
        #         for adv, valid_mask, prune_mask in zip(advantage_list, valid_mask_list, prune_mask_list):
        #             valid_adv = adv[valid_mask]
        #             valid_prune_mask = prune_mask[valid_mask]
        #             if valid_prune_mask.any():
        #                 all_prune_adv.append(valid_adv[valid_prune_mask].flatten())
        #             if (~valid_prune_mask).any():
        #                 all_non_prune_adv.append(valid_adv[~valid_prune_mask].flatten())
                
        #         # 分别计算统计量
        #         if all_prune_adv:
        #             all_prune_adv = torch.cat(all_prune_adv)
        #             prune_adv_mean = all_prune_adv.mean()
        #             prune_adv_std = all_prune_adv.std()
        #             print(f"prune advantage stats: mean={prune_adv_mean.item()}, std={prune_adv_std.item()}")
        #         else:
        #             prune_adv_mean, prune_adv_std = 0., 1.
                    
        #         if all_non_prune_adv:
        #             all_non_prune_adv = torch.cat(all_non_prune_adv)
        #             non_prune_adv_mean = all_non_prune_adv.mean()
        #             non_prune_adv_std = all_non_prune_adv.std()
        #             print(f"non-prune advantage stats: mean={non_prune_adv_mean.item()}, std={non_prune_adv_std.item()}")
        #         else:
        #             non_prune_adv_mean, non_prune_adv_std = 0., 1.
                
        #         # 分别归一化
        #         new_advantage_list = []
        #         for adv, valid_mask, prune_mask in zip(advantage_list, valid_mask_list, prune_mask_list):
        #             new_adv = adv.clone()
        #             valid_prune_mask = prune_mask[valid_mask]
        #             # 对prune点归一化
        #             if valid_prune_mask.any():
        #                 prune_indices = torch.where(valid_mask)[0][valid_prune_mask]
        #                 new_adv[prune_indices] = (adv[prune_indices] - prune_adv_mean) / prune_adv_std
        #             # 对non-prune点归一化
        #             if (~valid_prune_mask).any():
        #                 non_prune_indices = torch.where(valid_mask)[0][~valid_prune_mask]
        #                 new_adv[non_prune_indices] = (adv[non_prune_indices] - non_prune_adv_mean) / non_prune_adv_std
        #             new_advantage_list.append(new_adv)
        #         advantage_list = new_advantage_list
        #         tb_writer.add_scalar("rl/prune_advantage_mean", prune_adv_mean, iteration)
        #         tb_writer.add_scalar("rl/prune_advantage_std", prune_adv_std, iteration)
        #         tb_writer.add_scalar("rl/non_prune_advantage_mean", non_prune_adv_mean, iteration)
        #         tb_writer.add_scalar("rl/non_prune_advantage_std", non_prune_adv_std, iteration)
        #     else:
        #         # 不使用prune estimator时，统一归一化
        #         # 先使用valid_mask过滤，只计算有效点的advantage统计量
        #         all_advantages = []
        #         for adv, valid_mask in zip(advantage_list, valid_mask_list):
        #             valid_adv = adv[valid_mask]
        #             if valid_adv.numel() > 0:  # 确保不是空张量
        #                 all_advantages.append(valid_adv.flatten())
                
        #         if all_advantages:
        #             all_advantages = torch.cat(all_advantages)
        #             adv_mean = all_advantages.mean()
        #             adv_std = all_advantages.std()
        #         else:
        #             adv_mean, adv_std = 0., 1.
            
        #         advantage_list = [(x - adv_mean) / adv_std for x in advantage_list]
        #         print(f"advantage stats: mean={adv_mean.item()}, std={adv_std.item()}")
        #         tb_writer.add_scalar("rl/advantage_mean", adv_mean, iteration)
        #         tb_writer.add_scalar("rl/advantage_std", adv_std, iteration)

        # all_advantages = []
        # for adv, valid_mask in zip(advantage_list, valid_mask_list):
        #     valid_adv = adv[valid_mask]
        #     if valid_adv.numel() > 0:  # 确保不是空张量
        #         all_advantages.append(valid_adv.flatten())
        
        # if all_advantages:
        #     all_advantages = torch.cat(all_advantages)
        #     adv_mean = all_advantages.mean()
        #     adv_std = all_advantages.std()
        # else:
        #     adv_mean, adv_std = 0., 1.
    
        # advantage_list = [(x - adv_mean) / adv_std for x in advantage_list]
        # print(f"advantage stats: mean={adv_mean.item()}, std={adv_std.item()}")
        # tb_writer.add_scalar("rl/advantage_mean", adv_mean, iteration)
        # tb_writer.add_scalar("rl/advantage_std", adv_std, iteration)

        state_list = [x[valid_mask] for x, valid_mask in zip(state_list, valid_mask_list)]
        action_list = [x[valid_mask] for x, valid_mask in zip(action_list, valid_mask_list)]
        advantage_list = [x[valid_mask].detach() for x, valid_mask in zip(advantage_list, valid_mask_list)]
        value_list = [x[valid_mask] for x, valid_mask in zip(value_list, valid_mask_list)]
        prune_mask_list = [x[valid_mask] for x, valid_mask in zip(prune_mask_list, valid_mask_list)]

        all_action = torch.cat(action_list).squeeze(-1)
        all_advantage = torch.cat(advantage_list).squeeze(-1)

        if self.training_args.verbose:
            all_prune_mask = torch.cat(prune_mask_list)
            print("Adavantage distribution: keep={}, clone={}, split={}, delete={}".format(
                all_advantage[(all_action == 0) & (~all_prune_mask)].mean().item(),
                all_advantage[(all_action == 1)].mean().item(),
                all_advantage[(all_action == 2)].mean().item(),
                all_advantage[(all_action == 3)].mean().item()
            ))
            if tb_writer:
                tb_writer.add_scalar("rl/clone_advantage_mean", all_advantage[(all_action == 1)].mean().item(), iteration)
                tb_writer.add_scalar("rl/split_advantage_mean", all_advantage[(all_action == 2)].mean().item(), iteration)
                tb_writer.add_scalar("rl/delete_advantage_mean", all_advantage[(all_action == 3)].mean().item(), iteration)
        
        n_epochs = self.training_args.ppo_n_epochs
        n_rollout = len(state_list)
        pg_loss_avg = 0.
        vf_loss_avg = 0.
        entropy_loss_avg = 0.
        ratio_avg = 0.

        for _i in range(n_epochs):
            self.actor_optimizer.zero_grad(set_to_none=True)
            if not self.use_my_value:
                self.critic_optimizer.zero_grad(set_to_none=True)
            self.state_encoder_optimizer.zero_grad(set_to_none=True)
            if self.training_args.use_prune_estimator:
                self.prune_estimator_optimizer.zero_grad(set_to_none=True)
                if not self.use_my_value:
                    self.prune_critic_optimizer.zero_grad(set_to_none=True)

            loss = 0.
            for state, action, advantage, old_value, old_log_prob, prune_mask in zip(state_list, action_list, advantage_list, value_list, old_log_prob_list, prune_mask_list):
                if self.mem_debug and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    self._log_cuda_mem("before_state_encode")
                
                state = self.state_encoder(state)
                    
                if self.mem_debug:
                    self._log_cuda_mem("after_state_encode")

                actual_value = (advantage + old_value)

                if self.training_args.use_prune_estimator:
                    probs = torch.zeros(state.shape[0], 1, device="cuda", dtype=torch.float)
                    if not self.use_my_value:
                        value = torch.zeros(state.shape[0], 1, device="cuda", dtype=torch.float)
                    
                    # 处理 prune 点
                    if prune_mask.any():
                        prune_encoded_state = state[prune_mask]
                        prune_probs = self.prune_estimator(prune_encoded_state)
                        prune_probs = torch.where(action[prune_mask] == 3, prune_probs, 1 - prune_probs)
                        probs[prune_mask] = prune_probs

                        if not self.use_my_value:
                            prune_value = self.prune_critic(prune_encoded_state)
                            value[prune_mask] = prune_value
                    
                    # 处理 non-prune 点
                    if (~prune_mask).any():
                        non_prune_encoded_state = state[~prune_mask]
                        non_prune_probs = self.actor(non_prune_encoded_state)
                        non_prune_probs = non_prune_probs.gather(-1, action[~prune_mask])
                        probs[~prune_mask] = non_prune_probs

                        if not self.use_my_value:
                            non_prune_value = self.critic(non_prune_encoded_state)
                            value[~prune_mask] = non_prune_value
                else:
                    if not self.use_my_value:
                        value = self.critic(state)
                    probs = self.actor(state)
                    probs = probs.gather(-1, action)

                log_prob = torch.log(probs)
                # assert not log_prob.isnan().any() and not log_prob.isinf().any()
                ratio = torch.exp(log_prob - old_log_prob)

                # ============ 改进的Loss计算 ============
                
                # 1. 策略梯度损失 (PPO-Clip) - 标准实现
                pg_loss1 = -advantage * ratio
                pg_loss2 = -advantage * torch.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip)
                pg_loss = torch.mean(torch.max(pg_loss1, pg_loss2))

                if not self.use_my_value:
                    vf_loss = 0.5 * torch.mean((value - actual_value) ** 2)
                else:
                    vf_loss = 0.

                # # 2. 价值函数损失 - 添加clipping防止value更新过大
                # # 使用PPO-Clip风格的value loss
                # value_clipped = old_value + torch.clamp(
                #     value - old_value, 
                #     -self.value_clip, 
                #     self.value_clip
                # )
                # vf_loss1 = (value - actual_value) ** 2
                # vf_loss2 = (value_clipped - actual_value) ** 2
                # vf_loss = 0.5 * torch.mean(torch.max(vf_loss1, vf_loss2))
                
                # 检查loss是否有效，如果无效则跳过这个batch
                # if torch.isnan(pg_loss) or torch.isinf(pg_loss) or torch.isnan(vf_loss) or torch.isinf(vf_loss):
                if torch.isnan(pg_loss) or torch.isinf(pg_loss):
                    print(f"警告: 检测到无效的loss值 (pg_loss={pg_loss.item()}, vf_loss={vf_loss.item()})，跳过此batch")
                    print("log_prob:", log_prob.mean().item(), "old_log_prob:", old_log_prob.mean().item())
                    raise ValueError("Invalid loss values")

                # 3. 熵正则化 - 鼓励探索，保持策略多样性
                entropy = 0.
                if self.training_args.use_prune_estimator:
                    # 处理 prune 点的熵
                    if prune_mask.any():
                        prune_probs = self.prune_estimator(state[prune_mask])
                        prune_probs = torch.cat([prune_probs, 1 - prune_probs], dim=-1)
                        entropy += -torch.sum(prune_probs * torch.log(prune_probs), dim=-1).mean()
                    
                    # 处理 non-prune 点的熵
                    if (~prune_mask).any():
                        non_prune_probs = self.actor(state[~prune_mask])
                        entropy += -torch.sum(non_prune_probs * torch.log(non_prune_probs), dim=-1).mean()
                else:
                    probs = self.actor(state)
                    entropy += -torch.sum(probs * torch.log(probs), dim=-1).mean()
                
                # 动态调整熵系数：使用cosine annealing但保持最小探索
                entropy_coef = cosine_annealing(
                    iteration - self.training_args.densify_from_iter,
                    self.training_args.densify_until_iter - self.training_args.densify_from_iter,
                    initial_temp=self.entropy_coef_init, 
                    final_temp=self.entropy_coef_final
                )
                entropy_loss = -entropy_coef * entropy  # 负号因为我们想最大化熵

                pg_loss_avg += pg_loss.item()
                if not self.use_my_value:
                    vf_loss_avg += vf_loss.item()
                entropy_loss_avg += entropy_loss.item()
                ratio_avg += ratio.mean().item()

                # 4. 总损失 - 使用权重系数平衡各项
                loss += pg_loss + vf_loss + entropy_loss

            loss /= n_rollout
            loss.backward()

            if self.mem_debug:
                self._log_cuda_mem("after_backward")

            # # 检查网络梯度，只打印第一个参数
            # for name, param in self.actor.named_parameters():
            #     if param.grad is not None:
            #         print(f"Actor {name} grad norm: {param.grad.norm().item()}")
            #     break

            # # 梯度裁剪防止爆炸
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            # torch.nn.utils.clip_grad_norm_(self.state_encoder.parameters(), max_norm=1.0)
            # if self.training_args.use_prune_estimator:
            #     torch.nn.utils.clip_grad_norm_(self.prune_estimator.parameters(), max_norm=1.0)

            self.actor_optimizer.step()
            if not self.use_my_value:
                self.critic_optimizer.step()
            self.state_encoder_optimizer.step()
            if self.training_args.use_prune_estimator:
                self.prune_estimator_optimizer.step()
                if not self.use_my_value:
                    self.prune_critic_optimizer.step()

        pg_loss_avg /= n_epochs * n_rollout
        vf_loss_avg /= n_epochs * n_rollout
        entropy_loss_avg /= n_epochs * n_rollout
        ratio_avg /= n_epochs * n_rollout

        if self.mem_debug:
            self._log_cuda_mem("after_opt_step")

        if self.training_args.verbose:
            print(f"pg_loss_avg: {pg_loss_avg}, vf_loss_avg: {vf_loss_avg}, entropy_loss_avg: {entropy_loss_avg}, ratio_avg: {ratio_avg}")
        
        # 计算value和advantage的统计信息用于诊断
        all_values = torch.cat(value_list)

        if self.training_args.verbose:
            print(f"value stats: mean={all_values.mean().item()}, std={all_values.std().item()}")
        
        if tb_writer:
            tb_writer.add_scalar("rl/pg_loss_avg", pg_loss_avg, iteration)
            tb_writer.add_scalar("rl/vf_loss_avg", vf_loss_avg, iteration)
            # tb_writer.add_scalar("rl/entropy_loss_avg", entropy_loss_avg, iteration)
            tb_writer.add_scalar("rl/ratio_avg", ratio_avg, iteration)
            # 诊断信息
            tb_writer.add_scalar("rl/value_mean", all_values.mean().item(), iteration)
            tb_writer.add_scalar("rl/value_std", all_values.std().item(), iteration)

        # self.transition.clear()
        # torch.cuda.empty_cache()

    def save_models(self, path):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            # 'actor_optimizer': self.actor_optimizer.state_dict(),
            # 'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)
        
    def load_models(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        # self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        # self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

    def training_setup(self, training_args):
        self.actor_lr_scheduler = get_expon_lr_func(
            lr_init=training_args.rl_actor_lr_init,
            lr_final=training_args.rl_actor_lr_final,
            lr_delay_steps=training_args.rl_lr_delay_steps,
            lr_delay_mult=training_args.rl_lr_delay_mult,
            max_steps=training_args.densify_until_iter - training_args.densify_from_iter,
        )
        self.critic_lr_scheduler = get_expon_lr_func(
            lr_init=training_args.rl_critic_lr_init,
            lr_final=training_args.rl_critic_lr_final,
            lr_delay_steps=training_args.rl_lr_delay_steps,
            lr_delay_mult=training_args.rl_lr_delay_mult,
            max_steps=training_args.densify_until_iter - training_args.densify_from_iter,
        )

        self.state_encoder_lr_scheduler = get_expon_lr_func(
            lr_init=training_args.rl_state_encoder_lr_init,
            lr_final=training_args.rl_state_encoder_lr_final,
            lr_delay_steps=training_args.rl_lr_delay_steps,
            lr_delay_mult=training_args.rl_lr_delay_mult,
            max_steps=training_args.densify_until_iter - training_args.densify_from_iter,
        )
        
        self.prune_estimator_lr_scheduler = get_expon_lr_func(
            lr_init=training_args.rl_prune_estimator_lr_init,
            lr_final=training_args.rl_prune_estimator_lr_final,
            lr_delay_steps=training_args.rl_lr_delay_steps,
            lr_delay_mult=training_args.rl_lr_delay_mult,
            max_steps=training_args.densify_until_iter - training_args.densify_from_iter,
        )
        
        # prune_critic使用与主critic相同的学习率调度
        self.prune_critic_lr_scheduler = get_expon_lr_func(
            lr_init=training_args.rl_critic_lr_init,
            lr_final=training_args.rl_critic_lr_final,
            lr_delay_steps=training_args.rl_lr_delay_steps,
            lr_delay_mult=training_args.rl_lr_delay_mult,
            max_steps=training_args.densify_until_iter - training_args.densify_from_iter,
        )

    def update_learning_rate(self, iteration):
        lr = self.actor_lr_scheduler(iteration)
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = lr

        if not self.use_my_value:
            lr = self.critic_lr_scheduler(iteration)
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = lr

        lr = self.state_encoder_lr_scheduler(iteration)
        for param_group in self.state_encoder_optimizer.param_groups:
            param_group['lr'] = lr

        if self.training_args.use_prune_estimator:
            lr = self.prune_estimator_lr_scheduler(iteration)
            for param_group in self.prune_estimator_optimizer.param_groups:
                param_group['lr'] = lr
            if not self.use_my_value:
                lr = self.prune_critic_lr_scheduler(iteration)
                for param_group in self.prune_critic_optimizer.param_groups:
                    param_group['lr'] = lr

        return lr


    def capture(self):
        return (
            self.actor.state_dict(),
            None if self.use_my_value else self.critic.state_dict(),
            self.state_encoder.state_dict(),
            # self.actor_optimizer.state_dict(),
            # self.critic_optimizer.state_dict(),
            # self.state_encoder_optimizer.state_dict()
        )

    def restore(self, rl_controller_state):
        self.actor.load_state_dict(rl_controller_state[0])
        
        if not self.use_my_value:
            self.critic.load_state_dict(rl_controller_state[1])

        self.state_encoder.load_state_dict(rl_controller_state[2])
        # self.actor_optimizer.load_state_dict(rl_controller_state[3])
        # self.critic_optimizer.load_state_dict(rl_controller_state[4])
        # self.state_encoder_optimizer.load_state_dict(rl_controller_state[5])


