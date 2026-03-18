from tqdm import tqdm
import math
import copy
import numpy as np
from PIL import Image
from collections import defaultdict, ChainMap
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.distributions import Normal
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
import clip
from omegaconf import DictConfig
import wandb
import utils.math3d as math3d
from utils.clip import clip_encode_text
from utils.optim import Lamb, GradualWarmupScheduler
from utils.structure import ActResult
from point_renderer.rvt_renderer import RVTBoxRenderer
from tvve_stage1 import PolicyNetwork
from preprocess import (
    preprocess_images_in_batch,
    flatten_img_pc_to_points,
    clamp_pc_in_bound,
    place_pc_in_cube,
    generate_heatmap_from_screen_pts,
    apply_se3_augmentation,
    transform_pc,
    grid_sample_from_heatmap,
    add_uniform_noise
)
from tvve_stage1 import TaskMoEMVT
from point_renderer.cameras import OrthographicCameras, PerspectiveCameras
from pytorch3d.transforms import euler_angles_to_matrix
from torch_cluster import fps
from torch.distributions import Normal, Independent
from time import time
from torch.utils.data import DataLoader
from runstats import Statistics

# Switch to disable decorators
ENABLE_TIMER = False

def timer_decorator(func):
    if not ENABLE_TIMER:
        return func  # When not disabled, return directly to the original function
    
    def wrapper(*args, **kwargs):
        import time
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__} Execution time: {end_time - start_time:.6f} Second")
        return result
    return wrapper


class CameraPolicyNetwork(nn.Module):
    """6-DoFcamera policy network"""
    def __init__(self, feat_dim=256, hidden_dim=128):
        super().__init__()
        
        # feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Strategy Head (meanμandlogstandard deviationσ）
        self.actor = nn.Linear(hidden_dim, 6*2)  # every camera6parameters, each parameterμandσ
        
        # value function header
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        hidden = self.encoder(x)
        
        # Decomposition parameters
        params = self.actor(hidden).view(-1, 6, 2)
        mu = params[..., 0]  # [B, 6]
        log_std = params[..., 1]  # [B, 6]
        
        # Apply constraints
        mu = self._apply_constraints(mu)
        std = torch.exp(log_std.clamp(min=-20, max=2))
        
        return mu, std, self.critic(hidden)
    
    def _apply_constraints(self, mu):
        """Parametric physical constraints"""
        # Positional parameters (theta, phi, r)
        mu[:, 0] = torch.sigmoid(mu[:, 0]) * np.pi        # theta ∈ [0, π]
        mu[:, 1] = torch.sigmoid(mu[:, 1]) * 2 * np.pi    # phi ∈ [0, 2π)
        mu[:, 2] = F.softplus(mu[:, 2]) + 0.1             # r > 0.1
        
        # Rotation parameters (alpha, beta, gamma)
        mu[:, 3:] = torch.tanh(mu[:, 3:]) * torch.tensor(
            [np.pi, np.pi/2, np.pi/2], device=mu.device)  # Constraint range
        
        return mu


class MultiViewCameraPolicy(nn.Module):
    """Supports multi-view prediction6-DoFcamera policy network"""
    def __init__(self, feat_dim=256, hidden_dim=128, num_views=3):
        super().__init__()
        self.num_views = num_views
        
        # shared feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        # Multi-view prediction head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, num_views * 6 * 2)  # [μ, log_std] * 6parameter * 3perspective
        )
        # value function header
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.encoder(x)
        
        # Decompose multi-view parameters
        params = self.actor(hidden).view(batch_size, self.num_views, 6, 2)
        mu = params[..., 0]  # [B, 3, 6]
        log_std = params[..., 1]  # [B, 3, 6]
        
        # Apply physical constraints
        mu = self._apply_constraints(mu)
        std = torch.exp(log_std.clamp(min=-20, max=2))
        
        return mu, std, self.critic(hidden)
    
    def _apply_constraints(self, mu):
        """Apply parametric constraints on a per-view basis"""
        # Positional parameters (theta, phi, r)
        mu[..., 0] = torch.sigmoid(mu[..., 0]) * np.pi        # theta ∈ [0, π]
        mu[..., 1] = torch.sigmoid(mu[..., 1]) * 2 * np.pi    # phi ∈ [0, 2π)
        mu[..., 2] = F.softplus(mu[..., 2]) + 0.1             # r > 0.1
        
        # Rotation parameters (alpha, beta, gamma)
        rot_params = mu[..., 3:]
        rot_params = torch.tanh(rot_params) * torch.tensor(
            [np.pi, np.pi/2, np.pi/2], device=mu.device
        )
        mu[..., 3:] = rot_params
        return mu

    def sample(self, x, deterministic=False):
        """Sampling multi-view parameters"""
        mu, std, value = self(x)
        dist = Normal(mu, std)
        
        if deterministic:
            actions = mu
        else:
            actions = dist.rsample()
            
        log_probs = dist.log_prob(actions).sum(dim=[1,2])  # Joint probability of each view
        return actions, log_probs, value


class MultiViewLookAtCameraPolicy(nn.Module):
    """based onlook-atmode camera policy network (predicted position andupvector)"""
    def __init__(self, feat_dim=256, hidden_dim=128, num_views=3):
        super().__init__()
        self.num_views = num_views
        
        # shared feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Predictive Header: Position(theta,phi,r) + upvector(theta_up,phi_up)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, num_views * 5 * 2)  # 5parameter/Perspective:3Location + 2updirection
        )
        
        # value function header
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.encoder(x)
        
        # Decomposition parameters [B, 3, 5, 2]
        params = self.actor(hidden).view(batch_size, self.num_views, 5, 2)
        mu = params[..., 0]  # [B, 3, 5]
        log_std = params[..., 1]  # [B, 3, 5]
        
        # Apply physical constraints
        mu = self._apply_constraints(mu)
        std = torch.exp(log_std.clamp(min=-20, max=2))
        
        return mu, std, self.critic(hidden)
    
    def _apply_constraints(self, mu):
        """Parameter constraints"""
        mu =  mu.clone()
        # Positional parameters (theta, phi, r)
        mu[..., 0] = torch.sigmoid(mu[..., 0]) * np.pi        # theta ∈ [0, π]
        mu[..., 1] = torch.sigmoid(mu[..., 1]) * 2 * np.pi    # phi ∈ [0, 2π)
        mu[..., 2] = F.softplus(mu[..., 2]) + 0.1             # r > 0.1
        
        # Upvector direction parameter (theta_up, phi_up)
        mu[..., 3] = torch.sigmoid(mu[..., 3]) * np.pi        # theta_up ∈ [0, π]
        mu[..., 4] = torch.sigmoid(mu[..., 4]) * 2 * np.pi    # phi_up ∈ [0, 2π)
        
        return mu

    def sample(self, x, deterministic=False):
        """Sample camera parameters"""
        mu, std, value = self(x)
        dist = Normal(mu, std)
        
        actions = mu if deterministic else dist.rsample()
        log_probs = dist.log_prob(actions).sum(dim=[1,2])
        
        return actions, log_probs, value


class ChannelLastGroupNorm(nn.Module):
    """support [B, N, C] Formatted GroupNorm"""
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps)
    
    def forward(self, x):
        # x shape: [B, N, C]
        # Convert to [B, C, N]
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        # Convert back [B, N, C]
        return x.permute(0, 2, 1)

class PointCloudEncoder(nn.Module):
    """point cloud+Image feature encoder"""
    def __init__(self, point_dim=3, img_dim=3, hidden_dim=128):
        super().__init__()
        
        # Point-level feature extraction (coordinates+RGB）
        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim + img_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            ChannelLastGroupNorm(8, 128)
        )
        
        self.attention = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # global feature encoding
        self.global_mlp = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, pc, img_feat):
        # Stitching point cloud coordinates and image features [B, N, 6]
        x = torch.cat([pc, img_feat], dim=-1)
        
        # Point-by-point feature extraction [B, N, 128]
        point_feat = self.point_mlp(x)
        
        # attention weight [B, N, 1]
        attn = F.softmax(self.attention(point_feat), dim=1)
        
        # weighted aggregation [B, 128]
        global_feat = (point_feat * attn).sum(dim=1)
        
        # global encoding [B, hidden_dim]
        return self.global_mlp(global_feat)

class MultiViewLookAtCameraPolicyPlus(nn.Module):
    """Optimized multi-view camera policy network"""
    def __init__(self, 
                 point_dim=3, 
                 img_dim=3,
                 hidden_dim=512,
                 num_views=3, cfg=None):
        super().__init__()
        self.rmin = cfg.rmin
        self.rmax = cfg.rmax
        self.num_views = num_views
        self.encoder = PointCloudEncoder(point_dim, img_dim, hidden_dim)
        
        # Multi-view prediction head (independent branch)
        self.view_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 5*2)  # 5parameter(3Location+2updirection) × (μ,σ)
            ) for _ in range(num_views)
        ])
        
        # Shared value function header

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )


    def forward(self, pc, img_feat):
        # Encoding global features [B, hidden_dim]
        with autocast(enabled=False):
            global_feat = self.encoder(pc, img_feat)
            
            # Parallel prediction of parameters for each viewing angle
            params = []
            for head in self.view_heads:
                param = head(global_feat)  # [B, 10]
                params.append(param.view(-1, 5, 2))
            
            # Combination parameters [B, 3, 5, 2]
            params = torch.stack(params, dim=1)
            mu = params[..., 0]  # [B, 3, 5]
            log_std = params[..., 1]  # [B, 3, 5]
            
            # Apply physical constraints
            mu = self._apply_constraints(mu)
            std = torch.exp(log_std.clamp(min=-20, max=2))
            
            # value function estimation
            value = self.critic(global_feat)
        return mu, std, value
        
    def _apply_constraints(self, mu):
        """New implementation to avoid in-place operations"""
        # Create a copy instead of modifying it in place
        theta = torch.sigmoid(mu[..., 0]) * np.pi
        phi = torch.sigmoid(mu[..., 1]) * 2 * np.pi
        # r_min, r_max = 0.75, 1.3
        # r_min, r_max = 0.6, 1.56
        # r_min, r_max = 0.9, 1.04
        r_min, r_max = self.rmin, self.rmax
        r = r_min + torch.sigmoid(mu[..., 2]) * (r_max - r_min)
        theta_up = torch.sigmoid(mu[..., 3]) * np.pi
        phi_up = torch.sigmoid(mu[..., 4]) * 2 * np.pi
        
        # Construct a new tensor
        constrained_mu = torch.stack([
            theta, phi, r, theta_up, phi_up
        ], dim=-1)
        
        return constrained_mu

    @timer_decorator
    def sample(self, pc, img_feat, deterministic=False):
        """Sample camera parameters"""
        mu, std, value = self(pc, img_feat)
        if not deterministic:
            dist = Normal(mu, std)
            actions = dist.rsample()
            log_probs = dist.log_prob(actions).sum(dim=[1,2])  # joint probability
            entropy_bonus = dist.entropy().mean()
            return actions, log_probs, value, entropy_bonus
        else:
            actions = mu
            return actions

class MultiViewLookAtCameraPolicyPlus_multi_version(nn.Module):
    """Optimized multi-view camera policy network"""
    def __init__(self, 
                 point_dim=3, 
                 img_dim=3,
                 hidden_dim=256,
                 num_views=3):
        super().__init__()
        
        self.num_views = num_views
        self.encoder = PointCloudEncoder(point_dim, img_dim, hidden_dim)
        self.version = 1
        # Multi-view prediction head (independent branch)
        self.view_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 5*2)  # 5parameter(3Location+2updirection) × (μ,σ)
            ) for _ in range(num_views)
        ])
        
        # Shared value function header
        if self.version == 1:
            self.critic = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        elif self.version == 2:
            self.critic = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                ) for _ in range(num_views)
            ])


    def forward(self, pc, img_feat):
        # Encoding global features [B, hidden_dim]
        with autocast(enabled=False):
            global_feat = self.encoder(pc, img_feat)
            
            # Parallel prediction of parameters for each viewing angle
            params = []
            for head in self.view_heads:
                param = head(global_feat)  # [B, 10]
                params.append(param.view(-1, 5, 2))
            
            # Combination parameters [B, 3, 5, 2]
            params = torch.stack(params, dim=1)
            mu = params[..., 0]  # [B, 3, 5]
            log_std = params[..., 1]  # [B, 3, 5]
            
            # Apply physical constraints
            mu = self._apply_constraints(mu)
            std = torch.exp(log_std.clamp(min=-20, max=2))
            
            # value function estimation
            value = None
            if self.version == 1:
                value = self.critic(global_feat)
            elif self.version == 2:
                values_per_view = torch.stack([
                    head(global_feat)  # [B, 1]
                    for i, head in enumerate(self.critic)
                ], dim=1)  # [B, num_views, 1]

                value = values_per_view.mean(dim=1)  # [B, 1]
        return mu, std, value
        
    def _apply_constraints(self, mu):
        """New implementation to avoid in-place operations"""
        # Create a copy instead of modifying it in place
        theta = torch.sigmoid(mu[..., 0]) * np.pi
        phi = torch.sigmoid(mu[..., 1]) * 2 * np.pi
        r = F.softplus(mu[..., 2]) + 0.1
        theta_up = torch.sigmoid(mu[..., 3]) * np.pi
        phi_up = torch.sigmoid(mu[..., 4]) * 2 * np.pi
        
        # Construct a new tensor
        constrained_mu = torch.stack([
            theta, phi, r, theta_up, phi_up
        ], dim=-1)
        
        return constrained_mu

    @timer_decorator
    def sample(self, pc, img_feat, deterministic=False):
        """Sample camera parameters"""
        mu, std, value = self(pc, img_feat)
        if not deterministic:
            dist = Normal(mu, std)
            actions = dist.rsample()
            log_probs = dist.log_prob(actions).sum(dim=[1,2])  # joint probability
            entropy_bonus = dist.entropy().mean()
            return actions, log_probs, value, entropy_bonus
        else:
            actions = mu
            return actions

def spherical_to_cartesian(theta, phi, r):
    """Convert spherical coordinates to Cartesian coordinates"""
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack([x, y, z])

class DynamicCameraRenderer(RVTBoxRenderer):
    """
    Renderer extension to support dynamic camera strategies
    Inherited from RVTBoxRenderer and add the following functionality:
    1. 根据策略网络parametergenerate相机位姿
    2. Dynamically update camera configuration
    3. Compatible with original fixed camera mode
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dynamic_cameras = None
        self.last_camera_params = None
        self.now_eyes = None
        self.now_camera_look_at_params = None
        self.camera_params_pattern = "look_at" #  "6d"

    def _spherical_to_cartesian(self, theta, phi, r):
        """Convert spherical coordinates to Cartesian coordinates"""
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)
        return [x.item(), y.item(), z.item()]

    def _get_look_at(self, view):
        theta, phi, r, theta_up, phi_up = view
        # calculateeyePosition (spherical coordinates converted to Cartesian)
        eye = self._spherical_to_cartesian(theta, phi, r)
        # calculateupvector (based on spherical coordinates)
        up = self._spherical_to_cartesian(theta_up, phi_up, 1.0)
        return eye, up


    def _params_to_cameras(self, params):
        """
        Convert policy network output toOrthographicCamerasobject
        enterparametershape: [B, 6] (theta, phi, r, alpha, beta, gamma) B=3
        or [B, 5] (eye,up)
        """
        if self.camera_params_pattern == "look_at":
            eyes, ats, ups = [], [], []
            for p_i in params:
                eye, up = self._get_look_at(p_i)
                eyes.append(eye)
                ats.append([0, 0, 0])
                ups.append(up)
            self.now_eyes = eyes
            self.now_camera_look_at_params = [eyes, ups]
            cameras = OrthographicCameras.from_lookat(eyes, ats, ups, img_sizes_w=[2, 2], img_size_px=self.img_size)
            return cameras
        else:
            # Decomposition parameters (remain unchanged)
            theta = params[:, 0]  # polar angle ∈ [0, π]
            phi = params[:, 1]    # Azimuth ∈ [0, 2π)
            r = params[:, 2]      # distance
            alpha = params[:, 3]  # Zaxis rotation
            beta = params[:, 4]   # Yaxis rotation
            gamma = params[:, 5]  # Xaxis rotation

            # Convert spherical coordinates to Cartesian coordinates (remain unchanged)
            x = r * torch.sin(theta) * torch.cos(phi)
            y = r * torch.sin(theta) * torch.sin(phi)
            z = r * torch.cos(theta)
            T = torch.stack([x, y, z], dim=-1)  # [B, 3]

            # Calculate the rotation matrix (remain unchanged)
            R = euler_angles_to_matrix(
                torch.stack([alpha, beta, gamma], dim=-1),
                convention="ZYX"
            )
            
            return OrthographicCameras.from_rotation_and_translation(
                R=R,
                T=T,
                img_sizes_w=[2, 2],
                img_size_px=self.img_size
            )

    def update_dynamic_cameras(self, camera_params):
        """Update dynamic camera configuration"""
        self.dynamic_cameras = self._params_to_cameras(camera_params)
        self.dynamic_cameras = self.dynamic_cameras.to(self.device)
        self.last_camera_params = camera_params

    def __call__(self, pc, feat, camera_params=None, **kwargs):
        """
        Extended rendering calls:
        - camera_params: Camera parameters output by the policy network [B, 6]
        """
        # Decide whether to use a dynamic or static camera
        if camera_params is not None:
            if self.last_camera_params is not None:
                if not torch.allclose(camera_params.float(), self.last_camera_params.float()):
                    self.update_dynamic_cameras(camera_params)
            else:
                self.update_dynamic_cameras(camera_params)
            cameras = self.dynamic_cameras
        else:
            cameras = self.cameras

        # Perform rendering
        self._check_device(pc, "pc")
        self._check_device(pc, "feat")

        pc_images, pc_depths = self.renderer.render_batch(
            pc, feat,
            cameras=cameras,
            img_size=self.img_size,
            splat_radius=self.splat_radius,
            default_color=self.default_color,
            default_depth=self.default_depth,
            aa_factor=self.aa_factor
        )

        # Post-processing remains the same
        if self.normalize_output:
            _, h, w = pc_depths.shape
            depth_0 = pc_depths == -1
            depth_sum = torch.sum(pc_depths, (1, 2)) + torch.sum(depth_0, (1, 2))
            depth_mean = depth_sum / ((h * w) - torch.sum(depth_0, (1, 2)))
            pc_depths -= depth_mean.unsqueeze(-1).unsqueeze(-1)
            pc_depths[depth_0] = -1

        if self.with_depth:
            img_out = torch.cat([pc_images, pc_depths[:, :, :, None]], dim=-1)
        else:
            img_out = pc_images
        if camera_params is not None:
            return img_out, self.now_camera_look_at_params
        else:
            return img_out

class RewardCalculator:
    """Multi-objective reward calculator with normalization"""
    def __init__(self, weights=None, normalize=True):
        self.weights = weights or {
            # 'task': 1.0,
            'task_stage2_hm1': 1.0,
            'task_stage2_hm2': 1.0,
            'task_stage2_hm3': 1.0,
            'rot_grip_log_probs_x': 1.0,
            'rot_grip_log_probs_y': 1.0,
            'rot_grip_log_probs_z': 1.0,
            'rot_grip_log_probs_grip': 1.0,
            'rot_grip_log_probs_coll': 1.0,
            'confidence': 0.5,
            'diversity': 0.3,
            # 'stability': -0.1
        }
        self.normalize = normalize

    def sum_loss(self, loss_dict):
        
        loss_copy = {k: v for k, v in loss_dict.items() if k != 'stat_dict'}
        return sum(loss_copy.values())

    def compute(self, data_batch):
        rewards = {}
        # Task reward
        rewards['task_stage2_hm1'] = data_batch['default_loss']['stage2_log_probs_list'][0][:,0]-data_batch['policy_loss']['stage2_log_probs_list'][0][:,0]
        rewards['task_stage2_hm2'] = data_batch['default_loss']['stage2_log_probs_list'][1][:,0]-data_batch['policy_loss']['stage2_log_probs_list'][1][:,0]
        rewards['task_stage2_hm3'] = data_batch['default_loss']['stage2_log_probs_list'][2][:,0]-data_batch['policy_loss']['stage2_log_probs_list'][2][:,0]
       
        rewards['rot_grip_log_probs_x'] = data_batch['default_loss']['rot_grip_log_probs'][:, 6]-data_batch['policy_loss']['rot_grip_log_probs'][:, 6]
        rewards['rot_grip_log_probs_y'] = data_batch['default_loss']['rot_grip_log_probs'][:, 7]-data_batch['policy_loss']['rot_grip_log_probs'][:, 7]
        rewards['rot_grip_log_probs_z'] = data_batch['default_loss']['rot_grip_log_probs'][:, 8]-data_batch['policy_loss']['rot_grip_log_probs'][:, 8]
        rewards['rot_grip_log_probs_grip'] = data_batch['default_loss']['rot_grip_log_probs'][:, 9]-data_batch['policy_loss']['rot_grip_log_probs'][:, 9]
        rewards['rot_grip_log_probs_coll'] = data_batch['default_loss']['rot_grip_log_probs'][:, 10]-data_batch['policy_loss']['rot_grip_log_probs'][:, 10]

        # rewards['task'] = self.sum_loss(data_batch['default_loss']) - self.sum_loss(data_batch['policy_loss'])

        # Confidence Reward: Normalized by Channel -> entropy
        rewards['confidence'] = self._confidence_reward(data_batch['heatmaps'])

        # Perspective Diversity Bonus
        rewards['diversity'] = self._view_diversity(data_batch['camera_params'])

        # stability penalty
        # rewards['stability'] = -self._camera_movement(data_batch['camera_params'])

        unnorm_reward = {k: v for k, v in rewards.items()}
        
        # 2. Normalize each component individually
        for k in rewards.keys():
            rewards[k] = self._normalize_component(rewards[k], k)

        # 3. Weighted sum (remains unchanged)
        total_reward = sum(self.weights[k] * rewards[k] for k in self.weights)
        unnorm_reward['total'] = total_reward.detach().clone()
        
        # 4. Total tailoring reward
        total_reward = torch.clamp(total_reward, -10.0, 10.0)
        return total_reward, rewards, unnorm_reward

    def _normalize_component(self, component, component_name):
        """Run normalization on individual reward components"""
        # Initialize statistics
        if not hasattr(self, f"{component_name}_mean"):
            setattr(self, f"{component_name}_mean", 0.0)
            setattr(self, f"{component_name}_var", 1.0)  # The initial variance is set to1avoid division by zero
            setattr(self, f"{component_name}_count", 1e-4)
        
        # Get current statistics
        mean = getattr(self, f"{component_name}_mean")
        var = getattr(self, f"{component_name}_var")
        count = getattr(self, f"{component_name}_count")
        
        # Calculate batch statistics
        batch_mean = component.mean().item()
        batch_var = component.var().item()
        batch_count = component.shape[0]
        
        # Update global statistics (Welfordalgorithm)
        delta = batch_mean - mean
        new_count = count + batch_count
        new_mean = mean + delta * batch_count / new_count
        
        # Variance pooling formula
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / new_count
        new_var = M2 / new_count
        
        # Update properties
        setattr(self, f"{component_name}_mean", new_mean)
        setattr(self, f"{component_name}_var", new_var)
        setattr(self, f"{component_name}_count", new_count)
        
        # Normalize current components using historical statistics
        return (component - mean) / (torch.sqrt(torch.tensor(var)) + 1e-8)
    
    def _confidence_reward(self, heatmaps):
        """
        enter: [B, 3, H, W] -> 3heat map from each perspective
        Separately for each perspective softmax，Then calculate the average entropy
        """
        B, V, H, W = heatmaps.shape
        hm = heatmaps.view(B * V, -1)  # [B*3, H*W]
        hm_prob = torch.nn.functional.softmax(hm, dim=-1)
        entropy = -(hm_prob * torch.log(hm_prob + 1e-8)).sum(dim=-1)  # [B*3]
        entropy = entropy.view(B, V)  # [B, 3]
        return -entropy.mean(dim=1)   # [B]，The more concentrated the better, negentropy is the reward

    def _view_diversity(self, params):
        """Calculate cosine similarity between three views"""
        pos = params[:, :, :3]  # [B, 3, 3]
        pos_vec = pos / (pos.norm(dim=-1, keepdim=True) + 1e-8)
        sim_matrix = torch.einsum('bic,bjc->bij', pos_vec, pos_vec)  # [B, 3, 3]
        return (1 - sim_matrix).mean(dim=(1, 2))  # [B]

    def _camera_movement(self, params):
        """Calculate offset from default position"""
        default_pos = torch.tensor(
            [[0, 0, 1], [1, 0, 0], [0, 0.5, 0]],
            device=params.device
        ).expand_as(params[:, :, :3])
        return torch.norm(params[:, :, :3] - default_pos, dim=-1).mean(dim=1)  # [B]


class RewardCalculator_old:
    """Multi-goal reward calculator"""
    def __init__(self, weights=None):
        self.weights = weights or {
            'task': 1.0,
            'confidence': 0.5,
            'diversity': 0.3,
            'stability': -0.1
        }

    def sum_loss(self, loss_dict):
        loss_dict.pop('stat_dict')
        return sum(loss_dict.values())

    @timer_decorator
    def compute(self, data_batch):
        """
        data_batchInclude:
        - default_loss: Loss of fixed viewing angle
        - policy_loss: Loss of Strategic Perspective
        - heatmaps: Heat map generated by strategy [B, 3, H, W]
        - camera_params: Camera parameters [B, 3, 6]
        """
        rewards = {}
        
        # Task reward
        
        rewards['task'] = self.sum_loss(data_batch['default_loss']) - self.sum_loss(data_batch['policy_loss'])
        
        # Confidence reward
        hm = data_batch['heatmaps'].flatten(2)
        entropy = -(hm * torch.log(hm + 1e-8)).sum(dim=-1)
        rewards['confidence'] = -entropy.mean(dim=1)
        
        # Perspective Diversity Bonus
        rewards['diversity'] = self._view_diversity(
            data_batch['camera_params']
        )
        
        # stability penalty
        rewards['stability'] = -self._camera_movement(
            data_batch['camera_params']
        )
        
        # weighted combination
        total_reward = sum(
            self.weights[k] * rewards[k] for k in self.weights
        )
        return total_reward
    
    def _view_diversity(self, params):
        """Calculate cosine similarity between three views"""
        # Extract position vector
        pos = params[:, :, :3]  # [B, 3, 3]
        pos_vec = pos / pos.norm(dim=-1, keepdim=True)
        
        # Calculate pairwise similarity
        sim_matrix = torch.einsum('bic,bjc->bij', pos_vec, pos_vec)
        return (1 - sim_matrix).mean(dim=(1,2))
    
    def _camera_movement(self, params):
        """Calculate offset from default position"""
        data = [
            [0, 0, 1],
            [1, 0, 0],
            [0, 0.5, 0]
        ]
        default_pos = torch.tensor(
            data, device=params.device
        ).expand_as(params[:, :, :3])
        return torch.norm(params[:, :, :3] - default_pos, dim=-1).mean(dim=1)

class PointSampler:
    @timer_decorator
    def process_batch(self, points, img, radius=5, mask=None, target_num=2048):
        """
        usetorch_clusterofFPSPerform high-speed point sampling
        enter:
            points: Point cloud data, list[B]tensors, each of shape[N_i, 3]
            img: image data, list[B]tensors, each of shape[N_i, 3]
            target_num: Target number of sampling points for each sample
        return:
            sampled_points: Sample point cloud with shape[B, target_num, 3]
            sampled_img: Sample image with shape[B, target_num, 3]
        """
        device = points[0].device
        B = len(points)
        
        # 1. Merge all point clouds and create batch index
        all_points = torch.cat(points, dim=0)  # [total_points, 3]
        all_img = torch.cat(img, dim=0)  # [total_points, 3]
        
        # Create batch index [0,0,...,0, 1,1,...,1, ...]
        batch_idx = torch.cat([
            torch.full((len(points[i]),), i, dtype=torch.long, device=device)
            for i in range(B)
        ])
        
        # 2. calculate每个sampleof采样比例（目标point数/original points)
        ratios = torch.tensor([
            min(1.0, target_num / len(p))  # Make sure the ratio does not exceed1.0
            for p in points
        ], device=device)
        
        # 3. usetorch_clusteroffpsPerform efficient sampling
        fps_idx = fps(
            src=all_points, 
            batch=batch_idx, 
            ratio=ratios,  # Use the corresponding proportion for each sample
            random_start=True
        )
        
        # 4. Collect sampling results
        sampled_points = all_points[fps_idx]
        sampled_img = all_img[fps_idx]
        
        # 5. Reorganize into batch form
        # First determine the actual number of sampling points for each sample
        unique_batches, counts = torch.unique_consecutive(
            batch_idx[fps_idx], return_counts=True
        )
        
        # Create the final result tensor
        final_points = torch.zeros(B, target_num, 3, device=device)
        final_img = torch.zeros(B, target_num, 3, device=device)
        
        # 6. 填充结果，处理采样point数不足of情况
        start_idx = 0
        for i in range(B):
            end_idx = start_idx + counts[i]
            sample_points = sampled_points[start_idx:end_idx]
            sample_img = sampled_img[start_idx:end_idx]
            
            n_sampled = len(sample_points)
            
            if n_sampled >= target_num:
                # if采样point数足够，直接取前target_numpoint
                final_points[i] = sample_points[:target_num]
                final_img[i] = sample_img[:target_num]
            else:
                print("Not enough points! ! !")
                # If the number of sampling points is insufficient, random repeated sampling will make up for it.
                final_points[i, :n_sampled] = sample_points
                final_img[i, :n_sampled] = sample_img
                
                # Randomly select additional points to top up the amount
                num_extra = target_num - n_sampled
                extra_indices = torch.randint(0, n_sampled, (num_extra,), device=device)
                final_points[i, n_sampled:] = sample_points[extra_indices]
                final_img[i, n_sampled:] = sample_img[extra_indices]
            
            start_idx = end_idx
        
        return final_points, final_img


class CameraPolicy(PolicyNetwork):
    """End-to-end training system"""
    def __init__(self, model_cfg, env_cfg, render_device):
        super().__init__(model_cfg, env_cfg, render_device)
        self.camera_params_pattern = "look_at" #  "6d"
        if self.camera_params_pattern == "look_at":
            self.multiview_camera_policy = MultiViewLookAtCameraPolicyPlus(cfg=model_cfg)
        else:
            self.multiview_camera_policy = MultiViewCameraPolicy()
        self.cpp_renderer = DynamicCameraRenderer(device=render_device,
                                               img_size=(model_cfg.img_size, model_cfg.img_size),
                                               three_views=True,
                                               with_depth=model_cfg.add_depth)
        self.reward_calc = RewardCalculator()
        self.point_sampler = PointSampler()
        self.config = {
            # PPOhyperparameters
            'clip_epsilon': 0.2,  # PPO clipparameter
            'value_clip_epsilon': 0.2,  # value functionclip
            'value_loss_coef': 0.5,  # value loss weight
            'entropy_coef': 0.01,  # Entropy reward coefficient
            
            # Stabilization parameters
            'max_grad_norm': 0.5,  # gradient clipping threshold
            'reward_clip_min': -10.0,
            'reward_clip_max': 10.0,
            'task_loss_clip': 100.0,  # Prevent extreme loss values
        }
        self.buffer = None

    def save_tensor_as_image(self, tensor, output_path, normalize=True):
        """
        WillPyTorchSave tensor as image file
        
        parameter:
            tensor: The shape is[C,H,W]ofPyTorchTensor
            output_path: Path to output image
            normalize: Whether to normalize tensor values ​​to[0,1]scope
        """
        # Make sure the tensor is inCPUsuperior
        tensor = tensor.cpu()
        
        # If necessary, normalize the tensor values ​​to[0,1]scope
        if normalize:
            tensor_min = tensor.min()
            tensor_max = tensor.max()
            if tensor_max > tensor_min:
                tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        
        # Convert the tensor toPILimage
        if tensor.dim() == 3:  # [C,H,W]
            # Handle channel dimensions
            if tensor.size(0) == 3:  # RGBimage
                # Convert tonumpyArray, channel order starts from[C,H,W]Convert to[H,W,C]
                img_array = tensor.permute(1, 2, 0).numpy()
                # Convert toPILImage (make sure the data type isuint8）
                img = Image.fromarray((img_array * 255).astype(np.uint8))
            elif tensor.size(0) == 1:  # grayscale image
                # Remove channel dimensions
                img_array = tensor.squeeze(0).numpy()
                # Convert toPILimage
                img = Image.fromarray((img_array * 255).astype(np.uint8), mode='L')
            else:
                raise ValueError(f"Number of channels not supported: {tensor.size(0)}")
        else:
            raise ValueError(f"Unsupported tensor dimensions: {tensor.dim()}")
        
        # save image
        img.save(output_path)
        print(f"Image saved to: {output_path}")

    def farthest_point_sample(self, xyz, npoint):
        """
        Farthest point sampling (FPS）
        enter:
            xyz: Point cloud data, the shape is [B, N, 3]
            npoint: Target sampling points
        return:
            centroids: Sampling index, of shape [B, npoint]
        """
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids

    @timer_decorator
    def process_batch(self, points, img, radius, mask=None, target_num=2048):
        """
        Batch processing of point cloud and image data
        enter:
            points: Point cloud data, the shape is [B, N, 3]
            img: Image data with the shape of [B, N, 3]
            mask: The initial valid point mask has the shape [B, N]
            radius: Spherical range filter radius
            target_num: Target number of sampling points, default is2048
        return:
            sampled_points: The sampled point cloud has a shape of [B, target_num, 3]
            sampled_img: The sampled image has a shape of [B, target_num, 3]
        """
        B = len(points)
        N, _ = points[0].shape
        device = points[0].device
        sampled_points = torch.zeros((B, target_num, 3), device=device)
        sampled_img = torch.zeros((B, target_num, 3), device=device)
        # else:

        for b in range(B):
            # 1. Apply initial effective mask
            current_points = points[b]  # [N_valid, 3]
            current_img = img[b]
            
            # 2. Spherical range filtering
            sphere_points = current_points
            sphere_img = current_img

            
            # 3. Check if there are enough points left
            #     raise ValueError(
            #         f"sample {b}: Insufficient points after filtering ({N_filtered} < {target_num})"
            #     )
            
            # 4. applicationFPSDownsampling
            xyz_batch = sphere_points.unsqueeze(0)  # Add batch dimension [1, N_filtered, 3]
            fps_idx = self.farthest_point_sample(xyz_batch, target_num)  # [1, target_num]
            fps_idx = fps_idx.squeeze(0)  # [target_num]
            
            # 5. Extract sampled point clouds and images
            sampled_points[b] = sphere_points[fps_idx]
            sampled_img[b] = sphere_img[fps_idx]
        
        return sampled_points, sampled_img

    def compute_xyz_stats(self, points, mask=None, mode="global"):
        """
        Statistical maximum and minimum values ​​of point cloud coordinates
        enter:
            points: Point cloud data, the shape is [B, N, 3] List[torch.tensor]
            mask:   Valid point mask with shape [B, N]（Truemeans valid)
            mode:   "global"（global statistics) or "per_batch"（Sample-by-sample statistics)
        return:
            like mode="global" -> Returns the global maximum and minimum values, with the shape of [2, 3]（[min, max]）
            like mode="per_batch" -> Returns the maximum and minimum values ​​of each sample, with shape [B, 2, 3]
        """
        device = points[0].device
        B = len(points)
        N, _ = points[0].shape
        
        # If no mask is provided, all points are valid by default
        # else:
        
        if mode == "global":
            # Global statistics: calculate extreme values ​​after merging all valid points
            all_points = []
            for b in range(B):
                valid_points = points[b]  # [N_valid, 3]
                all_points.append(valid_points)
            all_points = torch.cat(all_points, dim=0)  # [Total, 3]
            
            if all_points.size(0) == 0:
                return torch.tensor([[0.,0.,0.], [0.,0.,0.]], device=device)  # Default value when there is no valid point
            
            min_vals = torch.amin(all_points, dim=0)  # [3]
            max_vals = torch.amax(all_points, dim=0)  # [3]
            return torch.stack([min_vals, max_vals])   # [2, 3]
        
        elif mode == "per_batch":
            # Sample-by-sample statistics
            batch_stats = torch.zeros((B, 2, 3), device=device)
            for b in range(B):
                valid_points = points[b][mask[b]]  # [N_valid, 3]
                if valid_points.size(0) == 0:
                    batch_stats[b] = 0.0  # Fill when there are no valid points0
                    continue
                min_vals = torch.amin(valid_points, dim=0)  # [3]
                max_vals = torch.amax(valid_points, dim=0)  # [3]
                batch_stats[b] = torch.stack([min_vals, max_vals])
            return batch_stats  # [B, 2, 3]
        
        else:
            raise ValueError("`mode` must be 'global' or 'per_batch'")


    def env_step(self, observation, network_shadow):
        with torch.no_grad():
            if network_shadow is not None:
                loss_dict = network_shadow.forward_original(observation)
            else:
                loss_dict = self.forward_original(observation)
            new_loss_dict, policy_output = self.forward_original(observation, camera_policy=True, env=True)
            reward, rewards, unnorm_rewards = self.reward_calc.compute({
                'default_loss': loss_dict,
                'policy_loss': new_loss_dict,
                'heatmaps': policy_output['heatmaps'],
                'camera_params': policy_output['camera_params']
            })
            transition = {
                'obs': observation,
                'action': policy_output['actions'],
                'log_prob': policy_output['log_probs'],
                'value': policy_output['values'],
                'env_reward': reward,
                'done': None,
                # 'done': torch.ones(reward.shape[0], dtype=torch.bool, device=reward.device),
                'unnorm_rewards': unnorm_rewards
            }
            return transition
            
    def forward(self, observation, ppo_il_update=False, only_il_update=False, rank=-1):
        if only_il_update:
            loss_dict, _ = self.forward_original(observation, camera_policy=True, ppo_il_update=ppo_il_update, only_il_update=only_il_update, rank=rank)
            loss_dict.pop("rot_grip_log_probs", None)
            loss_dict.pop("stage2_log_probs_list", None)
            return loss_dict
        elif ppo_il_update:
            loss_dict, _ = self.forward_original(observation, camera_policy=True, ppo_il_update=ppo_il_update)
            loss_dict.pop("rot_grip_log_probs", None)
            loss_dict.pop("stage2_log_probs_list", None)
            return loss_dict
        else:
            if self.training:
                with torch.no_grad():
                    return self.forward_original(observation, camera_policy=True)
            else:
                continuous_action = self.forward_original(observation, camera_policy=True)
                return continuous_action

    def forward_old(self, observation):
        if self.training:
            with torch.no_grad():
                loss_dict = self.forward_original(observation)
                new_loss_dict, policy_output = self.forward_original(observation, camera_policy=True)
            reward, rewards, unnorm_rewards = self.reward_calc.compute({
                'default_loss': loss_dict,
                'policy_loss': new_loss_dict,
                'heatmaps': policy_output['heatmaps'],
                'camera_params': policy_output['camera_params']
            })
            values = policy_output['values']
            log_probs = policy_output['log_probs']
            
            # PPOrenew
            advantage = reward - values.detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            
            policy_loss = -(advantage * log_probs).mean()
            reward = reward.unsqueeze(1)
            value_loss = F.mse_loss(values, reward)
            entropy_bonus = policy_output['entropy_bonus']
            total_loss = policy_loss + 0.5*value_loss - 0.01*entropy_bonus
            loss_dict_output = {
                'total_loss': total_loss,
                'stat_dict': {
                    
                    "reward/total": reward.mean().item(),
                    # "reward/task": rewards['task'].item(),
                    "reward/confidence": rewards['confidence'].mean().item(),
                    "reward/diversity": rewards['diversity'].mean().item(),
                    "reward/stability": rewards['stability'].mean().item(),
                    "reward/task_stage2_hm1": rewards['task_stage2_hm1'].mean().item(),
                    "reward/task_stage2_hm2": rewards['task_stage2_hm2'].mean().item(),
                    "reward/task_stage2_hm3": rewards['task_stage2_hm3'].mean().item(),
                    "reward/rot_grip_log_probs_x": rewards['rot_grip_log_probs_x'].mean().item(),
                    "reward/rot_grip_log_probs_y": rewards['rot_grip_log_probs_y'].mean().item(),
                    "reward/rot_grip_log_probs_z": rewards['rot_grip_log_probs_z'].mean().item(),
                    "reward/rot_grip_log_probs_grip": rewards['rot_grip_log_probs_grip'].mean().item(),
                    "reward/rot_grip_log_probs_coll": rewards['rot_grip_log_probs_coll'].mean().item(),

                    "unnorm_reward/total": unnorm_rewards['total'].mean().item(),
                    # "unnorm_reward/task": unnorm_rewards['task'].item(),
                    "unnorm_reward/confidence": unnorm_rewards['confidence'].mean().item(),
                    "unnorm_reward/diversity": unnorm_rewards['diversity'].mean().item(),
                    "unnorm_reward/stability": unnorm_rewards['stability'].mean().item(),
                    "unnorm_reward/task_stage2_hm1": unnorm_rewards['task_stage2_hm1'].mean().item(),
                    "unnorm_reward/task_stage2_hm2": unnorm_rewards['task_stage2_hm2'].mean().item(),
                    "unnorm_reward/task_stage2_hm3": unnorm_rewards['task_stage2_hm3'].mean().item(),
                    "unnorm_reward/rot_grip_log_probs_x": unnorm_rewards['rot_grip_log_probs_x'].mean().item(),
                    "unnorm_reward/rot_grip_log_probs_y": unnorm_rewards['rot_grip_log_probs_y'].mean().item(),
                    "unnorm_reward/rot_grip_log_probs_z": unnorm_rewards['rot_grip_log_probs_z'].mean().item(),
                    "unnorm_reward/rot_grip_log_probs_grip": unnorm_rewards['rot_grip_log_probs_grip'].mean().item(),
                    "unnorm_reward/rot_grip_log_probs_coll": unnorm_rewards['rot_grip_log_probs_coll'].mean().item(),

                    "ppo_loss/total_loss": total_loss.item(),
                    "ppo_loss/policy_loss": policy_loss.item(),
                    "ppo_loss/value_loss": value_loss.item(),

                    "ppo_policy/entropy_bonus": entropy_bonus.mean().item(),
                    "ppo_policy/value_estimate": values.mean().item(),
                    "ppo_policy/advantage_mean": advantage.mean().item(),
                    "ppo_policy/advantage_std": advantage.std().item(),
                    "ppo_policy/log_probs_mean": log_probs.mean().item(),
                    "ppo_policy/log_probs_std": log_probs.std().item()

                }
            }
            return loss_dict_output
        else:
            continuous_action = self.forward_original(observation, camera_policy=True)
            return continuous_action

    def print_shape(self, waypoint_stage2):
        print("========")
        print(type(waypoint_stage2))
        if isinstance(waypoint_stage2, list):
                print(waypoint_stage2[0].shape)
        elif isinstance(waypoint_stage2, torch.Tensor):
            print(waypoint_stage2.shape)

    @timer_decorator
    def get_gt_translation_action_Dynamic(
        self,
        waypoint, # this is groundtruth 3d point
        dims,
        cam_look_at
    ): # note: will be called separately for stage 1 / 2
        bs, nc, h, w = dims
        wpt_img = self.renderer.points3d_to_screen2d_Dynamic(waypoint.unsqueeze(1), cam_look_at)
        assert wpt_img.shape[1] == 1
        wpt_img = wpt_img.squeeze(1)  # (bs, num_img, 2)
        action_trans = generate_heatmap_from_screen_pts(
            wpt_img.reshape(-1, 2), #! just the winning points
            (h, w),
            sigma=self.gt_hm_sigma,
            thres_sigma_times=3,
        )
        action_trans = action_trans.view(bs, nc, h * w).transpose(1, 2).clone()
        return action_trans, wpt_img


    @timer_decorator
    def com_stage1_xyz_predict_loss(self, hms, cam_look_at, waypoint_stage2):
        pred_pt = []
        for hm, pa in zip(hms, cam_look_at):
            pred_pt.append(self.renderer.get_most_likely_point_3d_Dynamic(hm, pa))
        waypoint_stage2_predict_only_il = torch.cat(pred_pt, 0) # bs, 3
        mean_loss = F.mse_loss(waypoint_stage2_predict_only_il, waypoint_stage2)
        return mean_loss

    @timer_decorator
    def forward_original(self, observation, camera_policy=False, env=False, ppo_il_update=False, only_il_update=False, rank=-1):
        loss_dicts = []
        policy_output = {}
        nc, h, w = len(self.cfg.mvt_cameras), self.img_size, self.img_size
        dev = observation["lang_goal_embs"].device
        if self.training:
            action_grip = observation["gripper_action"].int() # (b,) of int
            action_ignore_collisions = observation["ignore_collisions"].view(-1, 1).int()  # (b, 1) of int
            action_gripper_pose = observation["gripper_pose"]  # (b, 7)
            action_trans_con = action_gripper_pose[:, 0:3]  # (b, 3), translation in xyz
            action_rot = action_gripper_pose[:, 3:7]  # (b, 4), rotation in quaternion xyzw

        lang_goal_embs = observation["lang_goal_embs"].float()
        proprio = observation["low_dim_state"]
        task_id = observation.get("task_idx")
        if task_id is None:
            raise KeyError(
                "observation['task_idx'] is required by TaskMoE-enabled stage23 forward."
            )

        #region preprocess and augmentation

        obs, pcd = preprocess_images_in_batch(observation, self.cameras)
        pc, img_feat = flatten_img_pc_to_points(obs, pcd)

        with torch.no_grad():
            if self._transform_augmentation and self.training:
                action_trans_con, action_rot, pc = apply_se3_augmentation( #! where the gt really comes out (for SE3 trans)
                    pcd=pc,
                    action_gripper_pose=action_gripper_pose,
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range=torch.tensor(self._transform_augmentation_xyz),
                    rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                )
                action_trans_con = torch.tensor(action_trans_con).to(pc.device)
                action_rot = torch.tensor(action_rot).to(pc.device)
                action_rot = action_rot.cpu().numpy()
                for i, _action_rot in enumerate(action_rot):
                    _action_rot = math3d.normalize_quaternion(_action_rot)
                    if _action_rot[-1] < 0:
                        _action_rot = -_action_rot
                    action_rot[i] = _action_rot

        pc, img_feat = clamp_pc_in_bound(pc, img_feat, self.scene_bounds, skip=not self.move_pc_in_bound)
        pc_new, rev_trans_stage1, waypoint_stage1 = [], [], []

        for i, _pc in enumerate(pc):
            a, b = place_pc_in_cube(_pc,
                with_mean_or_bounds=self._place_with_mean,
                scene_bounds=None if self._place_with_mean else self.scene_bounds,
            )
            if self.training:
                waypoint_stage1.append(place_pc_in_cube(_pc, action_trans_con[i][:3],
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0].unsqueeze(0))
            pc_new.append(a)
            rev_trans_stage1.append(b)
        pc = pc_new
        bs = len(pc)

        if self.training:
            waypoint_stage1 = torch.cat(waypoint_stage1, axis=0).clone().detach()
            if self.point_augment_noise != 0:
                with torch.no_grad():
                    for x in img_feat:
                        stdv = self.point_augment_noise * torch.rand(1, device=x.device)
                        noise = stdv * ((2 * torch.rand(*x.shape, device=x.device)) - 1)
                        x += noise
        img = self.render(pc, img_feat, self.mvt1)
                    
        #endregion ###########################
        visual_featmap_1, loss1 = self.mvt1(img=img, task_id=task_id, proprio=proprio, lang_emb=lang_goal_embs)
        
        if self.training:
            smooth_spatial_label_stage1, screen_waypoint_stage1 = self.get_gt_translation_action(waypoint_stage1, dims=(bs, nc, h, w))
            stage1_chk_ids = torch.as_tensor([0], device=dev)[None, :]

            # the 0, 0 are dummy input
            seq = torch.as_tensor([0, 0, self.policy.token_name_2_ids['stage1-screen-pts']], device=dev).reshape(1, 1, 3).repeat(bs, 1, 1)
            tmp_loss_dict = defaultdict(list)
            for view_id in range(3):
                _loss_dict = self.policy.compute_loss(seq, stage1_chk_ids, match_layer='cross',
                            contexts={
                                'visual-tokens': visual_featmap_1[:, view_id].flatten(-2, -1).permute(0, 2, 1),
                                'visual-featmap': visual_featmap_1[:, view_id],
                                'smooth-heatmap': smooth_spatial_label_stage1[:, :, view_id]
                            })
                for k, v in _loss_dict.items():
                    tmp_loss_dict[k].append(v)
            
            loss_dicts.append({k: sum(v) / len(v) for k, v in tmp_loss_dict.items()})
        else:
            prompt_seq = torch.zeros([bs, 0, 3], device=dev, dtype=torch.float32)
            future_tk_chk_ids = [dict(chk_id=0, tk_id=self.policy.token_name_2_ids['stage1-screen-pts'])]

            assert len(self.spatial_logits_buffer) == 0
            for view_id in range(3):
                self.policy.generate(prompt_seq, future_tk_chk_ids, match_layer='cross', sample_function=self.sample_callback,
                                    contexts={
                                            'visual-tokens': visual_featmap_1[:, view_id].flatten(-2, -1).permute(0, 2, 1),
                                            'visual-featmap': visual_featmap_1[:, view_id],
                                    })
                assert len(self.spatial_logits_buffer) == (view_id + 1)
            hms = torch.cat([F.softmax(hm_logits.reshape(bs, -1), dim=1).reshape(bs, 1, 224, 224) 
                             for hm_logits in self.spatial_logits_buffer], dim=1)
            pred_pt = [self.renderer.get_most_likely_point_3d(hms[i : i + 1]) for i in range(bs)]
            waypoint_stage1 = torch.cat(pred_pt, 0) # bs, 3
            self.spatial_logits_buffer.clear()

        with torch.no_grad():
            if self.training:
                waypoint_stage1_noisy = add_uniform_noise(
                    waypoint_stage1.clone().detach(), 2 * self.stage2_waypoint_label_noise
                )
                pc, rev_trans_stage2 = transform_pc(pc, loc=waypoint_stage1_noisy, sca=self.stage2_zoom_scale)
                waypoint_stage2, _ = transform_pc(waypoint_stage1, loc=waypoint_stage1_noisy, sca=self.stage2_zoom_scale)
            else:
                pc, rev_trans_stage2 = transform_pc(pc, loc=waypoint_stage1, sca=self.stage2_zoom_scale)
                waypoint_stage1_noisy = waypoint_stage1
                waypoint_stage2 = None
        cam_look_at = None
        if camera_policy:
            if self.training and not ppo_il_update and not only_il_update:
                    #         min_max_coo: tensor([[-6.6552, -6.6940, -2.9346],
                    # [ 5.9213,  7.2347,  6.1286]], device='cuda:0')

                    # mu, std, values = self.camera_policy_net(pc)
                    # # Sampling action
                    # policy_output['log_probs'] = log_probs
                    # policy_output['values'] = values
                pc_new, img_feat_new = self.point_sampler.process_batch(pc, img_feat)
                if not env:
                    with torch.enable_grad():
                        mu, std, value = self.multiview_camera_policy(pc_new, img_feat_new)
                        return mu, std, value
                        # camera_params, log_probs, values, entropy_bonus = self.multiview_camera_policy.sample(pc_new, img_feat_new)
                else:
                    camera_params, log_probs, values, entropy_bonus = self.multiview_camera_policy.sample(pc_new, img_feat_new) 
                policy_output['log_probs'] = log_probs
                policy_output['values'] = values
                policy_output['entropy_bonus'] = entropy_bonus
                
                # 渲染策略perspective
                policy_output['actions'] = camera_params
                camera_params = [camera_params[i] for i in range(camera_params.shape[0])]
                img, cam_look_at = self.render(pc, img_feat, self.mvt2, camera_params)
                cam_eye = [row[0] for row in cam_look_at]
                policy_output['camera_params'] = torch.tensor(cam_eye, device=pc[0].device)
            else:
                if only_il_update and not self.cfg.freeze_camera_policy:
                    # with torch.amp.autocast(device_type='cuda', enabled=False):
                    pc_new, img_feat_new = self.point_sampler.process_batch(pc, img_feat)
                    camera_params = self.multiview_camera_policy.sample(pc_new, img_feat_new, deterministic=True)
                    img, cam_look_at = self.render(pc, img_feat, self.mvt2, camera_params)
                # elif ppo_il_update:
                #     pc_new, img_feat_new = self.point_sampler.process_batch(pc, img_feat)
                #     img, cam_look_at = self.render(pc, img_feat, self.mvt2, camera_params)
                else:
                    with torch.no_grad():
                        pc_new, img_feat_new = self.point_sampler.process_batch(pc, img_feat)
                        camera_params = self.multiview_camera_policy.sample(pc_new, img_feat_new, deterministic=True)
                        img, cam_look_at = self.render(pc, img_feat, self.mvt2, camera_params)
        else:
            img = self.render(pc, img_feat, self.mvt2)
        visual_featmap_2, loss2 = self.mvt2(img=img, task_id=task_id, proprio=proprio, lang_emb=lang_goal_embs)
        if loss1 and loss2:
            for i in loss1.keys():
                loss_dicts.append({i:loss1[i]+loss2[i]})
        #     "taskmoe_loss":loss1["taskmoe_loss"]+loss2["taskmoe_loss"],
        # })
        if self.training:
            (
                action_rot_x,
                action_rot_y,
                action_rot_z,
                action_grip,       # (bs)
                action_collision,  # (bs)
            ) = [v.argmax(-1) for v in self.get_gt_rot_grip_collision(bs, action_rot, action_grip, action_ignore_collisions, device=dev)]

            if self.rotation_aug:
                rotation_aug = torch.from_numpy(np.random.choice(self.rotation_aug[0], p=self.rotation_aug[1], size=(bs, 3))).to(dev)
                action_rot_aug_x = action_rot_x + rotation_aug[:, 0]
                action_rot_aug_y = action_rot_y + rotation_aug[:, 1]
                action_rot_aug_z = action_rot_z + rotation_aug[:, 2]
            else:
                action_rot_aug_x = action_rot_x
                action_rot_aug_y = action_rot_y
                action_rot_aug_z = action_rot_z
            action_rot_aug_x %= self._num_rotation_classes
            action_rot_aug_y %= self._num_rotation_classes
            action_rot_aug_z %= self._num_rotation_classes
            if camera_policy:
                # with torch.amp.autocast(device_type='cuda', enabled=False):
                smooth_spatial_label_stage2, screen_waypoint_stage2 = self.get_gt_translation_action_Dynamic(waypoint_stage2, dims=(bs, nc, h, w), cam_look_at=cam_look_at)
                # # print(waypoint_stage2)
                assert torch.all((screen_waypoint_stage2 >= 0) & (screen_waypoint_stage2 <= 224)), \
                            f"screen_waypoint_stage2 contains values outside the range [0, 224]. Min: {screen_waypoint_stage2.min()}, Max: {screen_waypoint_stage2.max()}"

                # AssertionError: screen_waypoint_stage2 contains values outside the range [0, 224]. Min: -54.755409240722656, Max: 240.7436065673828
            else:
                smooth_spatial_label_stage2, screen_waypoint_stage2 = self.get_gt_translation_action(waypoint_stage2, dims=(bs, nc, h, w))
            # with torch.amp.autocast(device_type='cuda', enabled=False):
            stage2_chk_ids = torch.as_tensor([0], device=dev)[None, :]
            seq = torch.as_tensor([0, 0, self.policy.token_name_2_ids['stage2-screen-pts']], device=dev).reshape(1, 1, 3).repeat(bs, 1, 1)
            tmp_loss_dict = defaultdict(list)
            stage2_log_probs_list = []
            if camera_policy:
                tmp_hms = []
            for view_id in range(3):
                if camera_policy:
                    # with autocast(enabled=True):
                    _loss_dict, log_probs, hm_logits = self.policy.compute_loss(seq, stage2_chk_ids, match_layer='cross',log_prob=True, log_hm=True, 
                        contexts={
                            'visual-tokens': visual_featmap_2[:, view_id].flatten(-2, -1).permute(0, 2, 1),
                            'visual-featmap': visual_featmap_2[:, view_id],
                            'smooth-heatmap': smooth_spatial_label_stage2[:, :, view_id]
                        })
                    hm_logits = F.softmax(hm_logits.reshape(bs, -1), dim=1).reshape(bs, 1, 224, 224) 
                    tmp_hms.append(hm_logits)
                else:
                    _loss_dict, log_probs = self.policy.compute_loss(seq, stage2_chk_ids, match_layer='cross',log_prob=True,
                            contexts={
                                'visual-tokens': visual_featmap_2[:, view_id].flatten(-2, -1).permute(0, 2, 1),
                                'visual-featmap': visual_featmap_2[:, view_id],
                                'smooth-heatmap': smooth_spatial_label_stage2[:, :, view_id]
                            })
                for k, v in _loss_dict.items():
                    tmp_loss_dict[k].append(v)
                stage2_log_probs_list.append(log_probs)
            loss_dicts.append({k: sum(v) / len(v) for k, v in tmp_loss_dict.items()})
            if not only_il_update:
                if camera_policy:
                    policy_output["heatmaps"] = torch.cat(tmp_hms, dim=1)
                loss_dicts.append({
                    "stage2_log_probs_list": stage2_log_probs_list
                })
            # ------------------------------------------- #
            #     # print("random_num:")
            #     # print(random_num)
            #         # print(random_num)
            #         # pred_pt = [self.renderer.get_most_likely_point_3d(hms[i : i + 1]) for i in range(bs)]
            #             'stage1_xyz_predict_loss': mean_loss
            #         })
            # with torch.amp.autocast(device_type='cuda', enabled=False):
            prompt_features = torch.cat([ # [bs, 6, 128]
                    grid_sample_from_heatmap(screen_waypoint_stage2.reshape(-1, 1, 2) / self.img_patch_size, 
                                            visual_featmap_2.flatten(0, 1))[0].reshape(bs, -1, 128),
                    visual_featmap_2.max(dim=-1)[0].max(dim=-1)[0]], dim=1)
            
            seq = torch.as_tensor([(i, self.policy.token_name_2_ids['prompt-features']) for i in range(6)], 
                                device=dev).reshape(1, 6, 2).repeat(bs, 1, 1)
            seq = torch.cat([seq, torch.cat([
                                    torch.cat([
                                        action_rot_aug_x[:, None, None],
                                        action_rot_aug_y[:, None, None],
                                        action_rot_aug_z[:, None, None],
                                        action_grip[:, None, None],
                                        action_collision[:, None, None]], dim=1), 
                                    torch.as_tensor([self.policy.token_name_2_ids[k] for k in ['rot-x', 'rot-y', 'rot-z', 'grip', 'collision']], 
                                                    device=dev)[None, :, None].repeat(bs, 1, 1)], dim=-1)
                            ], dim=1)
            chk_ids = torch.as_tensor(list(range(11)), device=dev)[None, :]
            loss_dict_gripper, log_probs = self.policy.compute_loss(seq, chk_ids,
                                                        block_attn_directions=self.block_attn_directions, log_prob=True,
                                                        match_layer='self', contexts={
                                                            'prompt-features': prompt_features,
                                                            'rot-x': action_rot_x[:, None],
                                                            'rot-y': action_rot_y[:, None], 'rot-z': action_rot_z[:, None]
                                                        })
            loss_dicts.append(loss_dict_gripper)
            if not only_il_update:
                loss_dicts.append({
                    "rot_grip_log_probs": log_probs
                })
        else:
            prompt_seq = torch.zeros([bs, 0, 3], device=dev, dtype=torch.float32)
            future_tk_chk_ids = [dict(chk_id=0, tk_id=self.policy.token_name_2_ids['stage2-screen-pts'])]
            for view_id in range(3):
                self.policy.generate(prompt_seq, future_tk_chk_ids, match_layer='cross', sample_function=self.sample_callback,
                                    contexts={
                                            'visual-tokens': visual_featmap_2[:, view_id].flatten(-2, -1).permute(0, 2, 1),
                                            'visual-featmap': visual_featmap_2[:, view_id],
                                    })
                assert len(self.spatial_logits_buffer) == (view_id + 1)

            hms = torch.cat([F.softmax(hm_logits.reshape(bs, -1), dim=1).reshape(bs, 1, 224, 224) 
                             for hm_logits in self.spatial_logits_buffer], dim=1)
            if camera_policy:
                pred_pt = []
                for hm, pa in zip(hms, cam_look_at):
                    pred_pt.append(self.renderer.get_most_likely_point_3d_Dynamic(hm, pa))
            else:
                pred_pt = [self.renderer.get_most_likely_point_3d(hms[i : i + 1]) for i in range(bs)]
            waypoint_stage2 = torch.cat(pred_pt, 0) # bs, 3
            self.spatial_logits_buffer.clear()
            if camera_policy:
                screen_waypoint_stage2 = self.renderer.points3d_to_screen2d_Dynamic(waypoint_stage2[:, None, :], cam_look_at)[:, 0]
            else:
                screen_waypoint_stage2 = self.renderer.points3d_to_screen2d(waypoint_stage2[:, None, :])[:, 0]

            prompt_features = torch.cat([ # [bs, 6, 128]
                    grid_sample_from_heatmap(screen_waypoint_stage2.reshape(-1, 1, 2) / self.img_patch_size, 
                                            visual_featmap_2.flatten(0, 1))[0].reshape(bs, -1, 128),
                    visual_featmap_2.max(dim=-1)[0].max(dim=-1)[0]], dim=1)
            
            prompt_seq = torch.as_tensor([(i, self.policy.token_name_2_ids['prompt-features']) for i in range(6)], 
                                  device=dev).reshape(1, 6, 2).repeat(bs, 1, 1)
            future_tk_chk_ids = [dict(chk_id=chk_id, tk_id=self.policy.token_name_2_ids[tk_name]) 
                                 for chk_id, tk_name in zip(range(6, 11), ['rot-x', 'rot-y', 'rot-z', 'grip', 'collision'])]
            
            result_seq_stage2 = self.policy.generate(prompt_seq, future_tk_chk_ids, match_layer='self', 
                                                    sample=False,  block_attn_directions=self.block_attn_directions,  
                                                    contexts={
                                                        'prompt-features': prompt_features
                                                    })

        if self.training:
            loss_dict = {}
            for d in loss_dicts: loss_dict.update(d)
            norm = lambda x: torch.norm(x.flatten(1), dim=1).mean().item()
            loss_dict['stat_dict'] = {
                    'v1_norm': norm(visual_featmap_1.flatten(0, 1)),
                    'v2_norm': norm(visual_featmap_2.flatten(0, 1))
            }
            if camera_policy:
                return loss_dict, policy_output
            else:
                return loss_dict

            
        else:
            final_waypoint = rev_trans_stage1[0](rev_trans_stage2(waypoint_stage2))
            pred_rot = result_seq_stage2[:, 6:9, 0]
            pred_rot_quat = math3d.discrete_euler_to_quaternion(pred_rot.cpu().numpy(), self._rotation_resolution)
            continuous_action = np.concatenate(
                (
                    final_waypoint[0].cpu().numpy(),
                    pred_rot_quat[0],
                    result_seq_stage2[:, 9, 0].cpu().numpy(),
                    result_seq_stage2[:, 10, 0].cpu().numpy(),
                )
            )
            return continuous_action

    @timer_decorator
    def render(self, pc, img_feat, mvt: TaskMoEMVT, camera_params=None):
        renderer = self.cpp_renderer if self.render_with_cpp else self.renderer
        # add_corr: true norm_corr: true
        if camera_params is not None:
            with torch.no_grad():
                with autocast(enabled=False):
                    if mvt.add_corr:
                        if mvt.norm_corr:
                            img, eyes = [], []
                            for _pc, _img_feat, _camera_params in zip(pc, img_feat, camera_params):
                                max_pc = 1.0 if len(_pc) == 0 else torch.max(torch.abs(_pc))
                                im, eye = renderer(_pc, torch.cat((_pc / max_pc, _img_feat), dim=-1), camera_params=_camera_params)
                                img.append(im.unsqueeze(0))
                                eyes.append(eye)
                        else:
                            img, eyes = [], []
                            for _pc, _img_feat, _camera_params in zip(pc, img_feat, camera_params):
                                im, eye = renderer(_pc, torch.cat((_pc, _img_feat), dim=-1), camera_params=_camera_params)
                                img.append(im.unsqueeze(0))
                                eyes.append(eye)
                    else:
                        img, eyes = [], []
                        for _pc, _img_feat, _camera_params in zip(pc, img_feat, camera_params):
                            im, eye = renderer(_pc, _img_feat, camera_params=_camera_params)
                            img.append(im.unsqueeze(0))
                            eyes.append(eye)
        else:
            with torch.no_grad():
                with autocast(enabled=False):
                    if mvt.add_corr:
                        if mvt.norm_corr:
                            img = []
                            for _pc, _img_feat in zip(pc, img_feat):
                                max_pc = 1.0 if len(_pc) == 0 else torch.max(torch.abs(_pc))
                                img.append(
                                    renderer(_pc, torch.cat((_pc / max_pc, _img_feat), dim=-1)).unsqueeze(0) # [3, 224, 224, 7], 3 -> views, 7 -> feats
                                )
                        else:
                            img = [renderer(_pc, torch.cat((_pc, _img_feat), dim=-1)).unsqueeze(0) for _pc, _img_feat in zip(pc, img_feat)]
                    else:
                        img = [renderer(_pc, _img_feat).unsqueeze(0) for _pc, _img_feat in zip(pc, img_feat)]

        img = torch.cat(img, 0)
        img = img.permute(0, 1, 4, 2, 3) # [1, 3, 7, 224, 224]

        if mvt.add_pixel_loc:
            bs = img.shape[0]
            pixel_loc = mvt.pixel_loc.to(img.device) # extra feature
            img = torch.cat(
                (img, pixel_loc.unsqueeze(0).repeat(bs, 1, 1, 1, 1)), dim=2
            )
        if camera_params is not None:
            return img, eyes
        else:
            return img

from collections import deque

# Add gradientNaNCheck method
def check_grad_nan(self, model):
    for param in model.parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            return True
    return False

class Policy:
    def __init__(self, network: PolicyNetwork, model_cfg: DictConfig, log_dir=""):
        self._optimizer_type = model_cfg.optimizer_type
        self.warmup_steps = model_cfg.warmup_steps
        self.lr_cos_dec = model_cfg.lr_cos_dec
        self.cos_dec_max_step = model_cfg.cos_dec_max_step

        self._resume = model_cfg.resume
        self._lr = model_cfg.lr
        self.il_lr = model_cfg.il_lr
        self._lambda_weight_l2 = model_cfg.lambda_weight_l2
        self.amp = model_cfg.amp
        self.bnb = model_cfg.bnb
        self.add_lang = model_cfg.add_lang
        self.proprio_dim = model_cfg.proprio_dim
        self.clip_grad_norm = model_cfg.clip_grad_norm
        self.model_cfg = model_cfg

        self._network = network
        self._network_shadow = None
        
        self.log_dir = log_dir
        
        self.scaler = GradScaler(enabled=self.amp)
        self.buf_size = 5
        self.buffer = deque(maxlen=self.buf_size)  # Experience replay buffer

    def build(self, training: bool, device: torch.device = 'cpu', only_il_update=False):
        self._training = training
        self._device = device
        # params += list(self._network.camera_policy_net2.parameters())
        # params += list(self._network.camera_policy_net3.parameters())
        model = self.get_model(self._network)
        if only_il_update:
            if self.model_cfg.freeze_camera_policy:
                params = [p for name, p in model.named_parameters() 
                  if 'multiview_camera_policy' not in name]
            else:
                params = list(model.parameters()) 
        else:
            params = list(model.multiview_camera_policy.parameters()) 
        
        if self._training:
            if self._optimizer_type == "lamb":
                if self.bnb:
                    import bitsandbytes as bnb
                    print("Using 8-Bit Optimizer")
                    self._optimizer = bnb.optim.LAMB(
                        params,
                        lr=self._lr,
                        weight_decay=self._lambda_weight_l2,
                        betas=(0.9, 0.999),
                    )
                else:
                    # From: https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
                    self._optimizer = Lamb(
                        params,
                        lr=self._lr,
                        weight_decay=self._lambda_weight_l2,
                        betas=(0.9, 0.999),
                        adam=False,
                    )
            elif self._optimizer_type == "adam":
                self._optimizer = torch.optim.Adam(
                    params,
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                )
            elif self._optimizer_type == "adamw":
                self._optimizer = torch.optim.AdamW(
                    params,
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999), # Standard Momentum Parameters
    				eps=1e-8
                )
            else:
                raise Exception("Unknown optimizer")

            if self.lr_cos_dec:
                after_scheduler = CosineAnnealingLR(
                    self._optimizer,
                    T_max=self.cos_dec_max_step,
                    eta_min=self._lr / 100,  # mininum lr
                )
            else:
                after_scheduler = None
            self._lr_sched = GradualWarmupScheduler(
                self._optimizer,
                multiplier=1,
                total_epoch=self.warmup_steps,
                after_scheduler=after_scheduler,
            )

    def load_clip(self):
        self.clip_model, self.clip_preprocess = clip.load("RN50", device=self._device)
        self.clip_model.eval()

    def unload_clip(self):
        del self.clip_model
        del self.clip_preprocess
        with torch.cuda.device(self._device):
            torch.cuda.empty_cache()

    def reset(self, **kwargs):
        self._network.renderer.reset()

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()

    def get_model(self, model):
        if isinstance(model, DistributedDataParallel):
            return model.module
        else:
            return model

    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        epoch = checkpoint.get("epoch", checkpoint.get("step", None))
        model = self._network
        if isinstance(model, DistributedDataParallel):
            model.module.load_state_dict(checkpoint["model_state"], strict=False)
        else:
            model.load_state_dict(checkpoint["model_state"], strict=False)


        if self._training:
            # Freeze all parameters
            model = self.get_model(model)
            for param in model.parameters():
                param.requires_grad = False
            
            # thawmultiview_camera_policyParameters
            for param in model.multiview_camera_policy.parameters():
                param.requires_grad = True

            #             param_group['lr'] = 1e-4

            if self._resume:
                try:
                    if self.model_cfg.freeze_camera_policy:
                        self._optimizer.load_state_dict(checkpoint["optimizer_state"])
                        for param_group in self._optimizer.param_groups:
                            param_group['lr'] = param_group['lr'] / 2.0
                except:
                    print("WARNING: Optimizer state not loaded. KNOW WHAT YOU ARE DOING!!")
                try:
                    if self.model_cfg.freeze_camera_policy:
                        self._lr_sched.load_state_dict(checkpoint["lr_sched_state"])

                except:
                    print("WARNING: No lr_sched_state in checkpoint" "KNOW WHAT YOU ARE DOING!!")
        return epoch

    def print_lr(self, checkpoint):
        # Print checkpoint 中保存of学习率调度器状态
        print("Checkpoint lr_sched_state:")
        print(checkpoint["lr_sched_state"])

        # Print the status of the current learning rate scheduler
        print("\nCurrent lr_sched_state:")
        print(self._lr_sched.state_dict())

        # Compare whether the two are consistent
        if checkpoint["lr_sched_state"] == self._lr_sched.state_dict():
            print("\nThe scheduler states are consistent.")
        else:
            print("\nThe scheduler states are NOT consistent. Differences:")
            import pprint
            # Print specific differences
            for key in checkpoint["lr_sched_state"]:
                if checkpoint["lr_sched_state"][key] != self._lr_sched.state_dict().get(key):
                    print(f"Key: {key}")
                    print("Checkpoint value:")
                    pprint.pprint(checkpoint["lr_sched_state"][key])
                    print("Current value:")
                    pprint.pprint(self._lr_sched.state_dict().get(key))

    def save(self, step):
        model_path = f"{self.log_dir}/model_{step}.pth"
        model = self._network
        optimizer = self._optimizer
        lr_sched = self._lr_sched
        if isinstance(model, DistributedDataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        if self.model_cfg.ppo_il_update:
            torch.save(
                {
                    "step": step,
                    "model_state": model_state,
                    "optimizer_state": self.il_optimizer.state_dict(),
                    "lr_sched_state": self.il_lr_sched.state_dict(),
                },
                model_path
            )
        else:
            torch.save(
                {
                    "step": step,
                    "model_state": model_state,
                    "optimizer_state": optimizer.state_dict(),
                    "lr_sched_state": lr_sched.state_dict(),
                },
                model_path
            )

    def check_grad_nan(self, model):
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"parameter {name} The gradient contains NaN！")
                return True
        return False

    def ppo(self, dataloader: DataLoader, logger, device: int, freq: int = 30, rank: int = 0, save_freq: int = 6000, start_step=0, use_wandb=False):
        self.start = time()
        self.run_stats = {}
        self.steps = 0
        self.clip_epsilon = 0.2
        self.logger = logger
        self.freq = freq
        self.save_freq = save_freq
        self.use_wandb = use_wandb
        self.rank = rank
        self.total_steps = len(dataloader)
        for i, batch in enumerate(tqdm(dataloader, disable=rank != 0)):
            batch = {k:v.to(device) for k,v in batch.items()}
            with autocast(enabled=self.amp):
                if len(self.buffer) >= 0 and len(self.buffer) < self.buf_size:
                    self.collect_experience(batch)
                elif len(self.buffer) == self.buf_size:
                    self.update_ppo()
        self.save(self.steps)

    def collect_experience(self, replay_sample):
        model = self.get_model(self._network)
        transition = model.env_step(replay_sample, self._network_shadow)
        self.buffer.append(transition)

    def compute_td0_advantage(self, rewards, values, dones):
        advantages = rewards - values
        returns = rewards
        return advantages, returns

    def get_optim_and_set_model_state_in_ppo_il_update(self):
        model = self.get_model(self._network)
        
        self._network_shadow = copy.deepcopy(model)
        self._network_shadow.train()
        for param in self._network_shadow.parameters():
            param.requires_grad = False

        params = []
        for name, param in model.named_parameters():
            if 'multiview_camera_policy' not in name:
                params.append(param)
                
        import bitsandbytes as bnb
        self.il_optimizer = bnb.optim.LAMB(
                        params,
                        lr=self.il_lr,
                        weight_decay=self._lambda_weight_l2,
                        betas=(0.9, 0.999),
                    )
        il_after_scheduler = CosineAnnealingLR(
                    self.il_optimizer,
                    T_max=self.cos_dec_max_step,
                    eta_min=self.il_lr / 100,  # mininum lr
                )
        self.il_lr_sched = GradualWarmupScheduler(
                self.il_optimizer,
                multiplier=1,
                total_epoch=self.warmup_steps,
                after_scheduler=il_after_scheduler,
            )
        self.il_scaler = GradScaler(enabled=self.amp)
        for name, param in model.named_parameters():
            param.requires_grad = True


    def only_il_update(self, dataloader: DataLoader, logger, device: int, freq: int = 30, rank: int = 0, save_freq: int = 6000, start_step=0, use_wandb=False):
    
        self.start = time()
        self.run_stats = {}
        self.il_run_stats = {}
        self.steps = start_step
        self.clip_epsilon = 0.2
        self.logger = logger
        self.freq = freq
        self.save_freq = save_freq
        self.use_wandb = use_wandb
        self.rank = rank
        self.total_steps = len(dataloader)

        model = self.get_model(self._network)
        if self.model_cfg.freeze_camera_policy:
            for name, param in model.named_parameters():
                if 'multiview_camera_policy' not in name:
                    param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = True

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = 1e-5

        for i, replay_sample in enumerate(tqdm(dataloader, disable=rank != 0)):
            replay_sample = {k: v.to(device) for k, v in replay_sample.items()}
            assert replay_sample["gripper_pose"].shape[1:] == (7, )
            assert replay_sample["lang_goal_embs"].shape[1:] == (77, 512)
            assert replay_sample["low_dim_state"].shape[1:] == (self.proprio_dim,)
            assert self._network.training
            with autocast(enabled=self.amp):
                loss_dict = self._network(replay_sample, only_il_update=True, rank=self.rank)
                stat_dict = loss_dict.pop('stat_dict',  {})
                stage1_xyz_predict_loss = loss_dict.pop('stage1_xyz_predict_loss', -1)

                total_loss = sum(loss_dict.values())

                self._optimizer.zero_grad(set_to_none=True)

                self.scaler.scale(total_loss).backward()
                #     shape, c = (param.grad.shape, param.grad.sum()) if param.grad is not None else (None, None)

                # Add gradient clipping logic
                if getattr(self, 'clip_grad_norm', None) and self.clip_grad_norm > 0:
                    # Gradient scaling must be deactivated in the zoom environment
                    self.scaler.unscale_(self._optimizer)
                    # Apply gradient norm clipping
                    torch.nn.utils.clip_grad_norm_(
                        self._network.parameters(),
                        max_norm=self.clip_grad_norm,
                        norm_type=2  # Used by defaultL2Norm, can be adjusted as needed
                    )

                self.scaler.step(self._optimizer)
                self.scaler.update()
                loss_log = {
                    **{k: v.item() for k, v in loss_dict.items()},
                    "lr": self._optimizer.param_groups[0]["lr"],
                    **stat_dict,
                    "stage1_xyz_predict_loss": stage1_xyz_predict_loss
                }
            loss_dict = loss_log
            for k, v in loss_dict.items():
                if 'loss' in k and k not in self.run_stats:
                    self.run_stats[k] = Statistics()
            stat_dict = copy.copy(loss_dict)
            if self.use_wandb and self.rank == 0: wandb.log(stat_dict, step=self.steps)
            for k in self.run_stats:
                self.run_stats[k].push(loss_dict[k])
                stat_dict[k] = self.run_stats[k].mean()
            if self.steps % self.freq == 0 and self.rank == 0:
                self.logger(f"[step:{str(self.steps).zfill(8)} time:{time()-self.start:.01f}s] " + " ".join([f"{k}:{v:.04f}" for k, v in sorted(stat_dict.items())]),
                    printer=tqdm.write)
            if self.rank == 0 and self.steps != 0 and self.steps % self.save_freq == 0:
                self.logger(f"checkpoint to {self.log_dir} at step {self.steps} and reset running metrics", printer=tqdm.write)
                self.save(self.steps)
                self.run_stats = {}
            self.steps += 1

        self.save(self.steps)


    def only_il_update1(self, dataloader: DataLoader, logger, device: int, freq: int = 30, 
                  rank: int = 0, save_freq: int = 6000, start_step=0, use_wandb=False):
        self.start = time()
        self.run_stats = {}
        self.il_run_stats = {}
        self.steps = start_step
        self.clip_epsilon = 0.2
        self.logger = logger
        self.freq = freq
        self.save_freq = save_freq
        self.use_wandb = use_wandb
        self.rank = rank
        self.total_steps = len(dataloader)

        model = self.get_model(self._network)
        if self.model_cfg.freeze_camera_policy:
            for name, param in model.named_parameters():
                if 'multiview_camera_policy' not in name:
                    param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = True
        max_retry = 5  # Maximum number of retries

        for i, replay_sample in enumerate(tqdm(dataloader, disable=rank != 0)):
            replay_sample = {k: v.to(device) for k, v in replay_sample.items()}
            assert replay_sample["gripper_pose"].shape[1:] == (7, )
            assert replay_sample["lang_goal_embs"].shape[1:] == (77, 512)
            assert replay_sample["low_dim_state"].shape[1:] == (self.proprio_dim,)
            assert self._network.training
            # Retry control variables
            retry_count = 0
            success = False
            
            while not success and retry_count <= max_retry:
                with autocast(enabled=self.amp):
                    loss_dict = self._network(replay_sample, only_il_update=True, rank=self.rank)
                    
                    # examineNaNvalue
                    nan_detected = False
                    for k, v in loss_dict.items():
                        if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                            logger(f"NaN detected in {k} at step {self.steps}, retry {retry_count}")
                            nan_detected = True
                            break
                        if isinstance(v, float) and math.isnan(v):
                            logger(f"NaN detected in {k} at step {self.steps}, retry {retry_count}")
                            nan_detected = True
                            break
                    
                    # detectedNaNprocessing flow
                    if nan_detected:
                        # Clear gradient
                        self._optimizer.zero_grad(set_to_none=True)
                        current_lr = self._optimizer.param_groups[0]["lr"]
                        new_ = max(current_lr * 0.85, 5e-6)
                        # learning rate halved (All parameter groups)
                        for param_group in self._optimizer.param_groups:
                            param_group['lr'] = new_
                        logger(f"Learning rate reduced to {new_:.7f}")
                        
                        retry_count += 1
                        continue  # Retry forward pass
                    
                    # Normal processing flow
                    stat_dict = loss_dict.pop('stat_dict',  {})
                    stage1_xyz_predict_loss = loss_dict.pop('stage1_xyz_predict_loss', -1)
                    total_loss = sum(loss_dict.values())
                    success = True  # Mark successful completion forward

             # Retry processing that exceeds the upper limit
            if not success:
                logger(f"WARNING: Failed after {max_retry} retries at step {self.steps}")
                continue  # skip currentbatch

            # ===== Backpropagation and optimization =====
            self._optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total_loss).backward()
            
            # gradientNaNexamine
            if self.check_grad_nan(self._network):
                logger(f"Gradient NaN detected at step {self.steps}, skipping update")
                continue
            
            # gradient crop (in zoom environment)
            if getattr(self, 'clip_grad_norm', None) and self.clip_grad_norm > 0:
                self.scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._network.parameters(),
                    max_norm=self.clip_grad_norm,
                    norm_type=2
                )
            
            self.scaler.step(self._optimizer)
            self.scaler.update()
            self._lr_sched.step()
                
            # ===== Logging and saving =====
            loss_log = {**{k: v.item() for k, v in loss_dict.items()},
                    "lr": self._optimizer.param_groups[0]["lr"],
                    **stat_dict,
                    "stage1_xyz_predict_loss": stage1_xyz_predict_loss}
            
            loss_dict = loss_log
            for k, v in loss_dict.items():
                if 'loss' in k and k not in self.run_stats:
                    self.run_stats[k] = Statistics()
            stat_dict = copy.copy(loss_dict)
            if self.use_wandb and self.rank == 0: wandb.log(stat_dict, step=self.steps)
            for k in self.run_stats:
                self.run_stats[k].push(loss_dict[k])
                stat_dict[k] = self.run_stats[k].mean()
            if self.steps % self.freq == 0 and self.rank == 0:
                self.logger(f"[step:{str(self.steps).zfill(8)} time:{time()-self.start:.01f}s] " + " ".join([f"{k}:{v:.04f}" for k, v in sorted(stat_dict.items())]),
                    printer=tqdm.write)
            if self.rank == 0 and self.steps != 0 and self.steps % self.save_freq == 0:
                self.logger(f"checkpoint to {self.log_dir} at step {self.steps} and reset running metrics", printer=tqdm.write)
                self.save(self.steps)
                self.run_stats = {}
            self.steps += 1

        self.save(self.steps)

    def ppo_il_update(self, dataloader: DataLoader, logger, device: int, freq: int = 30, rank: int = 0, save_freq: int = 6000, start_step=0, use_wandb=False):
        
        self.get_optim_and_set_model_state_in_ppo_il_update()
        
        self.start = time()
        self.run_stats = {}
        self.il_run_stats = {}
        self.steps = 0
        self.clip_epsilon = 0.2
        self.logger = logger
        self.freq = freq
        self.save_freq = save_freq
        self.use_wandb = use_wandb
        self.rank = rank
        self.total_steps = len(dataloader)


        self.ppo_update_times = 0
        self.ppo_update_max_times = 1
        self.il_train_times = 0
        self.il_train_max_times = 3 * 15

        for i, batch in enumerate(tqdm(dataloader, disable=rank != 0)):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Check if a round of alternating training is completed
            if self.ppo_update_times >= self.ppo_update_max_times and \
            self.il_train_times >= self.il_train_max_times:
                # Reset update counter
                self.ppo_update_times = 0
                self.il_train_times = 0

            with autocast(enabled=self.amp):
            # Alternating training logic
                if self.il_train_times < self.il_train_max_times:
                        # Imitation learning update stage
                    self.update_il_once(batch)
                elif self.ppo_update_times < self.ppo_update_max_times:
                        # PPOupdate stage
                    self.update_ppo_once(batch)
                    

            	#     # PPOupdate stage
            	# elif self.il_train_times < self.il_train_max_times:
            	#     # Imitation learning update stage

        self.save(self.steps)

    def update_il_once(self, replay_sample):
        assert replay_sample["gripper_pose"].shape[1:] == (7, )
        assert replay_sample["lang_goal_embs"].shape[1:] == (77, 512)
        assert replay_sample["low_dim_state"].shape[1:] == (self.proprio_dim,)
        assert self._network.training
        with autocast(enabled=self.amp):
            loss_dict = self._network(replay_sample, only_il_update=True, rank=self.rank)
            stat_dict = loss_dict.pop('stat_dict',  {})
            stage1_xyz_predict_loss = loss_dict.pop('stage1_xyz_predict_loss', -1)
            total_loss = sum(loss_dict.values())
            self.il_optimizer.zero_grad(set_to_none=True)
            self.il_scaler.scale(total_loss).backward()

            # Add gradient clipping logic
            if getattr(self, 'clip_grad_norm', None) and self.clip_grad_norm > 0:
                # Gradient scaling must be deactivated in the zoom environment
                self.il_scaler.unscale_(self.il_optimizer)
                # Apply gradient norm clipping
                torch.nn.utils.clip_grad_norm_(
                    self._network.parameters(),
                    max_norm=self.clip_grad_norm,
                    norm_type=2  # Used by defaultL2Norm, can be adjusted as needed
                )

            self.il_scaler.step(self.il_optimizer)
            self.il_scaler.update()
            self.il_lr_sched.step()
            loss_log = {
                **{k: v.item() for k, v in loss_dict.items()},
                "lr": self.il_optimizer.param_groups[0]["lr"],
                **stat_dict
            }
        loss_dict = loss_log
        for k, v in loss_dict.items():
            if 'loss' in k and k not in self.il_run_stats:
                self.il_run_stats[k] = Statistics()
        stat_dict = copy.copy(loss_dict)
        if self.use_wandb and self.rank == 0: wandb.log(stat_dict, step=self.steps)
        for k in self.il_run_stats:
            self.il_run_stats[k].push(loss_dict[k])
            stat_dict[k] = self.il_run_stats[k].mean()
        if self.steps % self.freq == 0 and self.rank == 0:
            self.logger(f"[step:{str(self.steps).zfill(8)} time:{time()-self.start:.01f}s] " + " ".join([f"{k}:{v:.04f}" for k, v in sorted(stat_dict.items())]),
                printer=tqdm.write)
        if self.rank == 0 and self.steps != 0 and self.steps % self.save_freq == 0:
            self.logger(f"checkpoint to {self.log_dir} at step {self.steps} and reset running metrics", printer=tqdm.write)
            self.save(self.steps)
            self.il_run_stats = {}
        self.steps += 1

        self.il_train_times += 1


    def update_ppo_once(self, batch):
        if len(self.buffer) >= 0 and len(self.buffer) < self.buf_size - 1:
                self.collect_experience(batch)
        elif len(self.buffer) == self.buf_size - 1:
            self.collect_experience(batch)
            self.update_ppo()
            self.ppo_update_times += 1



    def update_ppo(self, epochs=3):
        for _ in range(epochs):
            for data in self.buffer:
                with torch.no_grad():
                    advantages, returns = self.compute_td0_advantage(data['env_reward'], data['value'].squeeze(), data['done'])
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                with autocast(enabled=self.amp):
                    mu, std, values = self._network(data['obs'])
                    dist = Normal(mu, std)
                    log_probs = dist.log_prob(data['action']).sum(dim=[1,2])  # joint probability
                    entropy_bonus = dist.entropy().mean()
                    # Calculate ratio (probability ratio of old and new strategies)
                    ratio = torch.exp(log_probs - data['log_prob'].detach())
                    unnorm_rewards = data['unnorm_rewards']
                    # PPOclipping objective function
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # value function loss
                    # Modify value loss calculation
                    old_values = data['value'].squeeze()

                    # unclipped
                    value_loss_unclipped = (values.squeeze() - returns).pow(2)

                    # clipped value prediction
                    value_pred_clipped = old_values + (values.squeeze() - old_values).clamp(-self.clip_epsilon, self.clip_epsilon)
                    value_loss_clipped = (value_pred_clipped - returns).pow(2)

                    # take the max
                    value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()

                    # total loss
                    total_loss = policy_loss + 0.5 * value_loss - 0.03 * entropy_bonus
                    loss_dict = {
                        'total_loss': total_loss,
                        'stat_dict': {
                            "reward/total": returns.mean().item(),

                            "ppo_loss/total_loss": total_loss.item(),
                            "ppo_loss/policy_loss": policy_loss.item(),
                            "ppo_loss/value_loss": value_loss.item(),

                            "ppo_policy/entropy_bonus": entropy_bonus.mean().item(),
                            "ppo_policy/value_estimate": values.mean().item(),
                            "ppo_policy/advantage_mean": advantages.mean().item(),
                            "ppo_policy/advantage_std": advantages.std().item(),
                            "ppo_policy/log_probs_mean": log_probs.mean().item(),
                            "ppo_policy/log_probs_std": log_probs.std().item(),

                            "unnorm_reward/total": unnorm_rewards['total'].mean().item(),
                            "unnorm_reward/confidence": unnorm_rewards['confidence'].mean().item(),
                            "unnorm_reward/diversity": unnorm_rewards['diversity'].mean().item(),
                            # "unnorm_reward/stability": unnorm_rewards['stability'].mean().item(),
                            "unnorm_reward/task_stage2_hm1": unnorm_rewards['task_stage2_hm1'].mean().item(),
                            "unnorm_reward/task_stage2_hm2": unnorm_rewards['task_stage2_hm2'].mean().item(),
                            "unnorm_reward/task_stage2_hm3": unnorm_rewards['task_stage2_hm3'].mean().item(),
                            "unnorm_reward/rot_grip_log_probs_x": unnorm_rewards['rot_grip_log_probs_x'].mean().item(),
                            "unnorm_reward/rot_grip_log_probs_y": unnorm_rewards['rot_grip_log_probs_y'].mean().item(),
                            "unnorm_reward/rot_grip_log_probs_z": unnorm_rewards['rot_grip_log_probs_z'].mean().item(),
                            "unnorm_reward/rot_grip_log_probs_grip": unnorm_rewards['rot_grip_log_probs_grip'].mean().item(),
                            "unnorm_reward/rot_grip_log_probs_coll": unnorm_rewards['rot_grip_log_probs_coll'].mean().item(),

                            }
                    }
                    stat_dict = loss_dict.pop('stat_dict',  {})
                    total_loss = sum(loss_dict.values())
                    self._optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(total_loss).backward()

                    # Add gradient clipping logic
                    if getattr(self, 'clip_grad_norm', None) and self.clip_grad_norm > 0:
                        # Gradient scaling must be deactivated in the zoom environment
                        self.scaler.unscale_(self._optimizer)
                        # Apply gradient norm clipping
                        torch.nn.utils.clip_grad_norm_(
                            self._network.parameters(),
                            max_norm=self.clip_grad_norm,
                            norm_type=2  # Used by defaultL2Norm, can be adjusted as needed
                        )

                    self.scaler.step(self._optimizer)
                    self.scaler.update()
                    self._lr_sched.step()
                    loss_log = {
                        **{k: v.item() for k, v in loss_dict.items()},
                        "lr": self._optimizer.param_groups[0]["lr"],
                        **stat_dict
                    }
                loss_dict = loss_log
                for k, v in loss_dict.items():
                    if 'loss' in k and k not in self.run_stats:
                        self.run_stats[k] = Statistics()
                stat_dict = copy.copy(loss_dict)
                if self.use_wandb and self.rank == 0: wandb.log(stat_dict, step=self.steps)
                for k in self.run_stats:
                    self.run_stats[k].push(loss_dict[k])
                    stat_dict[k] = self.run_stats[k].mean()
                if self.steps % self.freq == 0 and self.rank == 0:
                    self.logger(f"[step:{str(self.steps).zfill(8)} time:{time()-self.start:.01f}s] " + " ".join([f"{k}:{v:.04f}" for k, v in sorted(stat_dict.items())]),
                        printer=tqdm.write)
                if self.rank == 0 and self.steps != 0 and self.steps % self.save_freq == 0:
                    self.logger(f"checkpoint to {self.log_dir} at step {self.steps} and reset running metrics", printer=tqdm.write)
                    self.save(self.steps)
                    self.run_stats = {}
                self.steps += 1
        self.buffer.clear()

    def update(
        self,
        replay_sample: dict,
    ) -> dict:
        assert replay_sample["gripper_pose"].shape[1:] == (7, )
        assert replay_sample["lang_goal_embs"].shape[1:] == (77, 512)
        assert replay_sample["low_dim_state"].shape[1:] == (self.proprio_dim,)
        assert self._network.training
        with autocast(enabled=self.amp):
            loss_dict = self._network(replay_sample)
            stat_dict = loss_dict.pop('stat_dict',  {})
            total_loss = sum(loss_dict.values())
            self._optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total_loss).backward()

            # Add gradient clipping logic
            if getattr(self, 'clip_grad_norm', None) and self.clip_grad_norm > 0:
                # Gradient scaling must be deactivated in the zoom environment
                self.scaler.unscale_(self._optimizer)
                # Apply gradient norm clipping
                torch.nn.utils.clip_grad_norm_(
                    self._network.parameters(),
                    max_norm=self.clip_grad_norm,
                    norm_type=2  # Used by defaultL2Norm, can be adjusted as needed
                )

            self.scaler.step(self._optimizer)
            self.scaler.update()
            self._lr_sched.step()
            loss_log = {
                **{k: v.item() for k, v in loss_dict.items()},
                "lr": self._optimizer.param_groups[0]["lr"],
                **stat_dict
            }
        return loss_log

    @torch.no_grad()
    def act(
        self, step: int, observation: dict
    ) -> ActResult:
        assert observation['left_shoulder_rgb'].size(0) == 1, "Only batch size 1 is supported for evaluation"
        if self.add_lang:
            lang_goal_tokens = observation.get("lang_goal_tokens", None).long()
            _, lang_goal_embs = clip_encode_text(self.clip_model, lang_goal_tokens)
            lang_goal_embs = lang_goal_embs.float()
            observation['lang_goal_embs'] = lang_goal_embs
        else:
            lang_goal_embs = (
                torch.zeros(observation["lang_goal_embs"].shape)
                .float()
                .to(self._device)
            )
            observation['lang_goal_embs'] = lang_goal_embs
        assert not self._network.training

        continuous_action = self._network(observation)
        return ActResult(continuous_action)


def find_out_of_range_values(tensor, cam_look_at, waypoint_stage2, min_val=0, max_val=224):
    # Create a mask to mark out-of-range elements
    mask = (tensor < min_val) | (tensor > max_val)
    
    # If there is a value out of range
    if mask.any():
        print(f"turn up {mask.sum().item()} values ​​out of range:")
        
        # Get out-of-range values ​​and their indexes
        indices = torch.nonzero(mask, as_tuple=True)
        values = tensor[indices]
        
        # Print each out-of-range value and its index
        for i in range(len(values)):
            idx = [idx[i].item() for idx in indices]
            val = values[i].item()
            cam = cam_look_at[idx[0]][0][idx[1]]
            waypoint = waypoint_stage2[idx[0]]
            print(f"index {idx}: value = {val} {'< 0' if val < 0 else '> 224'}, cam={cam}, waypoint={waypoint}")
        return True
    return False
