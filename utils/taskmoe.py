import json
import math
import os
from inspect import isfunction
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# constants
MIN_EXPERT_CAPACITY = 4

# helper functions
def default(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val

def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)

def top1(t):
    values, index = t.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index

def cumsum_exclusive(t, dim=-1):
    num_dims = len(t.shape)
    num_pad_dims = - dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice   = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]

def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]

def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)

# activations
class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

# FiLM layer for feature-wise linear modulation
class FiLM(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        
        # FiLM generates scaling and shifting parameters
        self.film_generator = nn.Sequential(
            nn.Linear(condition_dim, 2 * input_dim),
            nn.GELU()
        )
        
    def forward(self, x, condition):
        """
        x: [batch, seq_len, input_dim] or [batch, input_dim]
        condition: [batch, condition_dim]
        """
        film_params = self.film_generator(condition)  # [batch, 2 * input_dim]
        gamma, beta = torch.chunk(film_params, 2, dim=-1)  # each [batch, input_dim]
        
        # Add dimensions for broadcasting if needed
        if x.dim() == 3:
            gamma = gamma.unsqueeze(1)  # [batch, 1, input_dim]
            beta = beta.unsqueeze(1)    # [batch, 1, input_dim]
            
        return gamma * x + beta

# Cross-modal attention for instruction-scene fusion
class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, visual_features, instruction_features):
        """
        visual_features: [batch, seq_len, dim]
        instruction_features: [batch, dim]
        """
        batch_size, seq_len, _ = visual_features.shape
        
        # Project queries from visual features
        q = self.q_proj(visual_features).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Project keys and values from instruction features (expand to match sequence length)
        instruction_expanded = instruction_features.unsqueeze(1).expand(batch_size, seq_len, self.dim)
        k = self.k_proj(instruction_expanded).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(instruction_expanded).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        return self.out_proj(attn_output)

# Task-aware Experts
class TaskExperts(nn.Module):
    def __init__(self, dim, num_experts=16, hidden_dim=None, activation=GELU):
        super().__init__()
        hidden_dim = default(hidden_dim, dim * 4)
        num_experts = cast_tuple(num_experts)

        w1 = torch.zeros(*num_experts, dim, hidden_dim)
        w2 = torch.zeros(*num_experts, hidden_dim, dim)

        w1 = init_(w1)
        w2 = init_(w2)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.act = activation()

    def forward(self, x):
        hidden = torch.einsum('...nd,...dh->...nh', x, self.w1)
        hidden = self.act(hidden)
        out    = torch.einsum('...nh,...hd->...nd', hidden, self.w2)
        return out

# Task-aware Gating Network
class TaskAwareGating(nn.Module):
    def __init__(self, dim, num_gates, num_tasks, instruction_dim, num_experts,
                 eps=1e-9, expert_select_k=2,
                 second_policy_train='random', second_policy_eval='random',
                 second_threshold_train=0.2, second_threshold_eval=0.2,
                 capacity_factor_train=1.25, capacity_factor_eval=2.,
                 expert_capacity_factor_train=None, expert_capacity_factor_eval=None,
                 semantic_reg_weight=0.0, gate_entropy_weight=0.0, expert_entropy_weight=0.0):
        super().__init__()

        self.eps = eps
        self.num_gates = num_gates
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.expert_select_k = max(1, expert_select_k)
        self.semantic_reg_weight = semantic_reg_weight
        self.gate_entropy_weight = gate_entropy_weight
        self.expert_entropy_weight = expert_entropy_weight

        # Cross-modal attention for instruction-scene fusion
        self.cross_modal_attn = CrossModalAttention(dim)

        # Task embedding
        self.task_embedding = nn.Embedding(num_tasks, dim)

        # FiLM layer for adaptive conditioning
        self.film = FiLM(dim, dim * 2)

        # Gating network
        self.w_gating = nn.Parameter(torch.randn(dim, num_gates))

        # Expert router
        self.w_expert = nn.Parameter(torch.randn(dim, num_gates, num_experts))
        self.gate_to_expert_bias = nn.Parameter(torch.zeros(num_gates, num_experts))

        # Task-to-gate preference
        self.task_gate_mapping = nn.Parameter(torch.randn(num_tasks, num_gates))

        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.gate_capacity_factor_train = capacity_factor_train
        self.gate_capacity_factor_eval = capacity_factor_eval
        self.expert_capacity_factor_train = expert_capacity_factor_train or capacity_factor_train
        self.expert_capacity_factor_eval = expert_capacity_factor_eval or capacity_factor_eval

        self.collect_stats = False
        self.latest_stats = None

    def _compute_capacity(self, group_size, num_units, factor):
        capacity = min(group_size, int((group_size * factor) / num_units))
        capacity = max(capacity, MIN_EXPERT_CAPACITY)
        return float(capacity)

    def _semantic_alignment(self, task_id, instruction_embedding, expert_probs):
        if self.semantic_reg_weight <= 0 or instruction_embedding is None or expert_probs is None:
            return 0.0
        task_gate_pref = self.task_gate_mapping[task_id].softmax(dim=-1)
        task_gate_pref = F.normalize(task_gate_pref, dim=-1)
        instr_norm = F.normalize(instruction_embedding, dim=-1)
        sim_instr = torch.matmul(instr_norm, instr_norm.transpose(0, 1)).detach()
        sim_gate = torch.matmul(task_gate_pref, task_gate_pref.transpose(0, 1))

        expert_profile = expert_probs.mean(dim=1)
        expert_profile = F.normalize(expert_profile + self.eps, dim=-1)
        sim_expert = torch.matmul(expert_profile, expert_profile.transpose(0, 1))

        return 0.5 * (F.mse_loss(sim_gate, sim_instr) + F.mse_loss(sim_expert, sim_instr))

    def forward(self, x, instruction_embedding, task_id):
        """
        x: [batch, seq_len, dim]
        instruction_embedding: [batch, instruction_dim]
        task_id: [batch]
        """
        *_, batch_size, group_size, dim = x.shape
        num_gates = self.num_gates

        fused_features = self.cross_modal_attn(x, instruction_embedding)
        task_emb = self.task_embedding(task_id)
        condition_input = torch.cat([fused_features.mean(dim=1), task_emb], dim=-1)
        modulated_features = self.film(fused_features, condition_input)

        raw_gates = torch.einsum('bnd,dg->bng', modulated_features, self.w_gating)
        task_gate_weights = self.task_gate_mapping[task_id].unsqueeze(1)
        raw_gates = raw_gates * task_gate_weights
        gate_probs = raw_gates.softmax(dim=-1)

        gate_values, gate_indices = gate_probs.max(dim=-1)
        gate_mask = F.one_hot(gate_indices, num_gates).float()

        gates_without_top = gate_probs * (1.0 - gate_mask)
        gate_second_values, gate_second_idx = gates_without_top.max(dim=-1)
        gate_second_mask = F.one_hot(gate_second_idx, num_gates).float()

        if self.training:
            gate_capacity_factor = self.gate_capacity_factor_train
            expert_capacity_factor = self.expert_capacity_factor_train
            policy = self.second_policy_train
            threshold = self.second_threshold_train
        else:
            gate_capacity_factor = self.gate_capacity_factor_eval
            expert_capacity_factor = self.expert_capacity_factor_eval
            policy = self.second_policy_eval
            threshold = self.second_threshold_eval

        gate_capacity = self._compute_capacity(group_size, num_gates, gate_capacity_factor)
        position_gate = cumsum_exclusive(gate_mask, dim=-2) * gate_mask
        gate_mask = gate_mask * (position_gate < gate_capacity).float()
        gate_mask_flat = gate_mask.sum(dim=-1)
        gate_values = gate_values * gate_mask_flat

        density_gate = gate_mask.mean(dim=-2)
        density_gate_proxy = gate_probs.mean(dim=-2)
        gate_balance_loss = (density_gate_proxy * density_gate).mean() * float(num_gates ** 2)
        gate_entropy = -(gate_probs * (gate_probs.clamp_min(self.eps).log())).sum(dim=-1).mean()

        # Expert routing within selected gate
        expert_logits = torch.einsum('bnd,dge->bnge', modulated_features, self.w_expert)
        expert_logits = expert_logits + self.gate_to_expert_bias
        expert_logits = expert_logits * gate_mask.unsqueeze(-1)
        expert_logits = expert_logits.sum(dim=2)

        valid_tokens = gate_mask_flat > 0
        expert_logits = expert_logits.masked_fill(~valid_tokens.unsqueeze(-1), float('-inf'))
        expert_probs = F.softmax(expert_logits, dim=-1)
        expert_probs = torch.where(valid_tokens.unsqueeze(-1), expert_probs, torch.zeros_like(expert_probs))
        expert_probs = expert_probs * gate_values.unsqueeze(-1)

        expert_1, exp_index_1 = top1(expert_probs)
        exp_mask_1 = F.one_hot(exp_index_1, self.num_experts).float()
        exp_mask_1 = exp_mask_1 * valid_tokens.unsqueeze(-1).float()

        experts_without_top = expert_probs * (1.0 - exp_mask_1)
        expert_2, exp_index_2 = top1(experts_without_top)
        exp_mask_2 = F.one_hot(exp_index_2, self.num_experts).float()
        exp_mask_2 = exp_mask_2 * valid_tokens.unsqueeze(-1).float()

        if policy == "none":
            exp_mask_2 = torch.zeros_like(exp_mask_2)
        elif policy == "threshold":
            exp_mask_2 = exp_mask_2 * (expert_2 > threshold).float().unsqueeze(-1)
        elif policy == "random":
            probs = torch.zeros_like(expert_2).uniform_(0., 1.)
            exp_mask_2 = exp_mask_2 * (probs < (expert_2 / max(threshold, self.eps))).float().unsqueeze(-1)

        denom = expert_1 + expert_2 + self.eps
        expert_1 = (expert_1 / denom) * exp_mask_1.sum(dim=-1)
        expert_2 = (expert_2 / denom) * exp_mask_2.sum(dim=-1)

        expert_capacity = self._compute_capacity(group_size, self.num_experts, expert_capacity_factor)

        position_exp_1 = cumsum_exclusive(exp_mask_1, dim=-2) * exp_mask_1
        exp_mask_1 = exp_mask_1 * (position_exp_1 < expert_capacity).float()
        exp_mask_1_flat = exp_mask_1.sum(dim=-1)
        position_exp_1 = position_exp_1.sum(dim=-1)

        position_exp_2 = cumsum_exclusive(exp_mask_2, dim=-2) * exp_mask_2
        exp_mask_2 = exp_mask_2 * (position_exp_2 < expert_capacity).float()
        exp_mask_2_flat = exp_mask_2.sum(dim=-1)
        position_exp_2 = position_exp_2.sum(dim=-1)

        density_exp = (exp_mask_1 + exp_mask_2).mean(dim=-2)
        density_exp_proxy = expert_probs.mean(dim=-2)
        expert_balance_loss = (density_exp_proxy * density_exp).mean() * float(self.num_experts ** 2)
        expert_entropy = -(expert_probs * (expert_probs.clamp_min(self.eps).log())).sum(dim=-1).mean()

        combine_tensor = (
            expert_1[..., None, None]
            * exp_mask_1_flat[..., None, None]
            * F.one_hot(exp_index_1, self.num_experts)[..., None]
            * safe_one_hot(position_exp_1.long(), int(expert_capacity))[..., None, :]
            +
            expert_2[..., None, None]
            * exp_mask_2_flat[..., None, None]
            * F.one_hot(exp_index_2, self.num_experts)[..., None]
            * safe_one_hot(position_exp_2.long(), int(expert_capacity))[..., None, :]
        )

        dispatch_tensor = combine_tensor.bool().to(combine_tensor)

        semantic_loss = self._semantic_alignment(task_id, instruction_embedding, expert_probs if instruction_embedding is not None else None)
        total_loss = gate_balance_loss + expert_balance_loss
        if self.semantic_reg_weight > 0:
            total_loss = total_loss + self.semantic_reg_weight * semantic_loss
        if self.gate_entropy_weight > 0:
            total_loss = total_loss - self.gate_entropy_weight * gate_entropy
        if self.expert_entropy_weight > 0:
            total_loss = total_loss - self.expert_entropy_weight * expert_entropy

        metrics = {
            "gate_entropy": gate_entropy.detach(),
            "expert_entropy": expert_entropy.detach(),
            "gate_load_std": density_gate.std(dim=-1).mean().detach(),
            "expert_load_std": density_exp.std(dim=-1).mean().detach(),
            "semantic_loss": semantic_loss if isinstance(semantic_loss, torch.Tensor) else torch.tensor(semantic_loss),
        }
        self.last_step_metrics = {k: (v.detach() if torch.is_tensor(v) else torch.tensor(v)) for k, v in metrics.items()}

        if self.collect_stats and not self.training:
            self.latest_stats = {
                "gate_top1": gate_indices.detach(),
                "gate_top2": gate_second_idx.detach(),
                "gate_top1_mask": gate_mask_flat.detach(),
                "gate_top2_mask": gate_second_mask.sum(dim=-1).detach(),
                "gate_probs": gate_probs.detach(),
                "expert_top1": exp_index_1.detach(),
                "expert_top2": exp_index_2.detach(),
                "expert_top1_mask": exp_mask_1_flat.detach(),
                "expert_top2_mask": exp_mask_2_flat.detach(),
                "expert_probs": expert_probs.detach(),
            }
        else:
            self.latest_stats = None

        return dispatch_tensor, combine_tensor, total_loss

# Main TaskMoE module
class TaskMoE(nn.Module):
    def __init__(self, dim, num_experts=16, num_gates=8, num_tasks=10, 
                 instruction_dim=512, hidden_dim=None, activation=GELU,
                 second_policy_train='random', second_policy_eval='random',
                 second_threshold_train=0.2, second_threshold_eval=0.2,
                 capacity_factor_train=1.25, capacity_factor_eval=2.,
                 expert_capacity_factor_train=None, expert_capacity_factor_eval=None,
                 loss_coef=1e-2, experts=None, stats_recorder=None,
                 expert_select_k=2, semantic_reg_weight=0.0,
                 gate_entropy_weight=0.0, expert_entropy_weight=0.0):
        super().__init__()

        self.num_experts = num_experts
        self.num_gates = num_gates
        self.num_tasks = num_tasks

        # Task-aware gating with fewer gates than tasks (N_G < N_J)
        gating_kwargs = {
            'second_policy_train': second_policy_train,
            'second_policy_eval': second_policy_eval,
            'second_threshold_train': second_threshold_train,
            'second_threshold_eval': second_threshold_eval,
            'capacity_factor_train': capacity_factor_train,
            'capacity_factor_eval': capacity_factor_eval,
            'expert_capacity_factor_train': expert_capacity_factor_train,
            'expert_capacity_factor_eval': expert_capacity_factor_eval,
            'num_experts': num_experts,
            'expert_select_k': expert_select_k,
            'semantic_reg_weight': semantic_reg_weight,
            'gate_entropy_weight': gate_entropy_weight,
            'expert_entropy_weight': expert_entropy_weight,
        }

        self.gate = TaskAwareGating(
            dim, 
            num_gates=num_gates,
            num_tasks=num_tasks,
            instruction_dim=instruction_dim,
            **gating_kwargs
        )
        
        # Experts shared across all tasks
        self.experts = default(experts, lambda: TaskExperts(
            dim, 
            num_experts=num_experts, 
            hidden_dim=hidden_dim, 
            activation=activation
        ))
        
        self.loss_coef = loss_coef
        self.stats_recorder = None
        self.track_stats = False
        if stats_recorder is not None:
            self.enable_usage_logging(stats_recorder)

    def enable_usage_logging(self, recorder=None):
        self.stats_recorder = recorder
        self.track_stats = recorder is not None
        self.gate.collect_stats = self.track_stats
        if not self.track_stats:
            self.gate.latest_stats = None

    def _record_usage(self, task_id, instruction_embedding, dispatch_tensor):
        if self.stats_recorder is None:
            return
        stats = getattr(self.gate, "latest_stats", None)
        if not stats:
            return

        with torch.no_grad():
            task_tensor = task_id if torch.is_tensor(task_id) else torch.as_tensor(task_id, device=dispatch_tensor.device)
            task_tensor = task_tensor.detach().to("cpu", dtype=torch.long).view(-1)

            expert_counts = dispatch_tensor.detach().sum(dim=-1).sum(dim=1).to("cpu")
            gate_top1 = stats["gate_top1"].detach().to("cpu")
            gate_top2 = stats["gate_top2"].detach().to("cpu")
            gate_top1_mask = stats["gate_top1_mask"].detach().to("cpu")
            gate_top2_mask = stats["gate_top2_mask"].detach().to("cpu")
            gate_probs = stats["gate_probs"].detach().to("cpu")
            expert_top1 = stats["expert_top1"].detach().to("cpu")
            expert_top2 = stats["expert_top2"].detach().to("cpu")
            expert_top1_mask = stats["expert_top1_mask"].detach().to("cpu")
            expert_top2_mask = stats["expert_top2_mask"].detach().to("cpu")
            expert_probs = stats["expert_probs"].detach().to("cpu")
            instruction_cpu = (
                instruction_embedding.detach().to("cpu") if instruction_embedding is not None else None
            )

            self.stats_recorder.update(
                task_ids=task_tensor,
                gate_top1=gate_top1,
                gate_top2=gate_top2,
                gate_top1_mask=gate_top1_mask,
                gate_top2_mask=gate_top2_mask,
                gate_probs=gate_probs,
                expert_top1=expert_top1,
                expert_top2=expert_top2,
                expert_top1_mask=expert_top1_mask,
                expert_top2_mask=expert_top2_mask,
                expert_probs=expert_probs,
                expert_counts=expert_counts,
                instruction_embedding=instruction_cpu,
            )

        self.gate.latest_stats = None

    def forward(self, inputs, instruction_embedding, task_id, **kwargs):
        """
        inputs: [batch, seq_len, dim]
        instruction_embedding: [batch, instruction_dim]
        task_id: [batch] - task identifiers
        """
        b, n, d, e = *inputs.shape, self.num_experts
        
        # Get task-aware gating
        dispatch_tensor, combine_tensor, loss = self.gate(inputs, instruction_embedding, task_id)
        
        # Route inputs to experts
        # print("dispatch_tensor shape:", dispatch_tensor.shape)
        # print("inputs shape:", inputs.shape)
        expert_inputs = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor)

        # Process through experts
        orig_shape = expert_inputs.shape
        # print("expert_inputs shape before reshape:", expert_inputs.shape)
        expert_inputs = expert_inputs.reshape(e, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        # Combine expert outputs
        output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor)

        self.last_step_metrics = getattr(self.gate, "last_step_metrics", None)
        if self.track_stats and not self.training:
            self._record_usage(task_id, instruction_embedding, dispatch_tensor)

        return output, loss * self.loss_coef


class TaskMoEUsageRecorder:
    def __init__(
        self,
        name: str,
        num_tasks: int,
        num_gates: int,
        num_experts: int,
        task_name_lookup: Optional[Dict[int, str]] = None,
        max_hist_samples: int = 20000,
        max_tsne_samples: int = 2048,
    ):
        self.name = name
        self.num_tasks = num_tasks
        self.num_gates = num_gates
        self.num_experts = num_experts
        self.max_hist_samples = max_hist_samples
        self.max_tsne_samples = max_tsne_samples
        self.task_names = [
            (task_name_lookup.get(idx) if task_name_lookup and idx in task_name_lookup else f"task_{idx}")
            for idx in range(num_tasks)
        ]
        self.reset()

    def reset(self):
        dtype = torch.float64
        self.top1_counts = torch.zeros(self.num_tasks, self.num_gates, dtype=dtype)
        self.top2_counts = torch.zeros(self.num_tasks, self.num_gates, dtype=dtype)
        self.expert_counts = torch.zeros(self.num_tasks, self.num_experts, dtype=dtype)
        self.tokens_per_task = torch.zeros(self.num_tasks, dtype=dtype)
        self.gate_prob_samples: List[np.ndarray] = []
        self.expert_prob_samples: List[np.ndarray] = []
        self.instruction_embeddings: List[np.ndarray] = []
        self.instruction_labels: List[np.ndarray] = []
        self._hist_collected = 0
        self._tsne_collected = 0
        self.gate_expert_matrix = torch.zeros(self.num_gates, self.num_experts, dtype=dtype)
        self.task_gate_hits = torch.zeros(self.num_tasks, self.num_gates, dtype=dtype)
        self.task_expert_hits = torch.zeros(self.num_tasks, self.num_experts, dtype=dtype)

    def _ensure_expert_dim(self, new_dim: int):
        if new_dim <= 0:
            return
        if new_dim == self.num_experts:
            return
        dtype = self.expert_counts.dtype
        new_counts = torch.zeros(self.num_tasks, new_dim, dtype=dtype)
        min_dim = min(new_dim, self.expert_counts.shape[1] if self.expert_counts.ndim > 1 else 0)
        if min_dim > 0:
            new_counts[:, :min_dim] = self.expert_counts[:, :min_dim]
        self.expert_counts = new_counts
        self.num_experts = new_dim
        new_task_hits = torch.zeros(self.num_tasks, new_dim, dtype=dtype)
        if min_dim > 0:
            new_task_hits[:, :min_dim] = self.task_expert_hits[:, :min_dim]
        self.task_expert_hits = new_task_hits
        new_matrix = torch.zeros(self.num_gates, new_dim, dtype=dtype)
        if min_dim > 0:
            new_matrix[:, :min_dim] = self.gate_expert_matrix[:, :min_dim]
        self.gate_expert_matrix = new_matrix

    def update(
        self,
        task_ids: torch.Tensor,
        gate_top1: torch.Tensor,
        gate_top2: torch.Tensor,
        gate_top1_mask: torch.Tensor,
        gate_top2_mask: torch.Tensor,
        gate_probs: torch.Tensor,
        expert_top1: torch.Tensor,
        expert_top2: torch.Tensor,
        expert_top1_mask: torch.Tensor,
        expert_top2_mask: torch.Tensor,
        expert_probs: torch.Tensor,
        expert_counts: torch.Tensor,
        instruction_embedding: Optional[torch.Tensor] = None,
    ):
        self._ensure_expert_dim(expert_counts.shape[-1])
        task_ids = torch.as_tensor(task_ids, dtype=torch.long).view(-1)
        gate_top1 = gate_top1.long()
        gate_top2 = gate_top2.long()
        gate_top1_mask = gate_top1_mask > 0
        gate_top2_mask = gate_top2_mask > 0
        expert_top1 = expert_top1.long()
        expert_top2 = expert_top2.long()
        expert_top1_mask = expert_top1_mask > 0
        expert_top2_mask = expert_top2_mask > 0
        expert_counts = expert_counts.to(torch.float64)

        for idx, task in enumerate(task_ids.tolist()):
            if task < 0 or task >= self.num_tasks:
                continue
            valid_gate = gate_top1_mask[idx]
            if valid_gate.any():
                counts = torch.bincount(gate_top1[idx][valid_gate], minlength=self.num_gates).to(torch.float64)
                self.top1_counts[task] += counts
                self.tokens_per_task[task] += valid_gate.sum().item()
                self.task_gate_hits[task] += counts
            valid_gate2 = gate_top2_mask[idx]
            if valid_gate2.any():
                counts2 = torch.bincount(gate_top2[idx][valid_gate2], minlength=self.num_gates).to(torch.float64)
                self.top2_counts[task] += counts2
                self.task_gate_hits[task] += counts2

            valid_exp1 = valid_gate & expert_top1_mask[idx]
            if valid_exp1.any():
                g_ids = gate_top1[idx][valid_exp1].long()
                e_ids = expert_top1[idx][valid_exp1].long()
                values = torch.ones_like(g_ids, dtype=self.gate_expert_matrix.dtype)
                self.gate_expert_matrix.index_put_((g_ids, e_ids), values, accumulate=True)

            valid_exp2 = valid_gate & expert_top2_mask[idx]
            if valid_exp2.any():
                g_ids2 = gate_top1[idx][valid_exp2].long()
                e_ids2 = expert_top2[idx][valid_exp2].long()
                values2 = torch.ones_like(g_ids2, dtype=self.gate_expert_matrix.dtype)
                self.gate_expert_matrix.index_put_((g_ids2, e_ids2), values2, accumulate=True)

            self.expert_counts[task] += expert_counts[idx]
            self.task_expert_hits[task] += expert_counts[idx]

        self._append_gate_probs(gate_probs)
        self._append_expert_probs(expert_probs)
        self._append_instruction_embeddings(instruction_embedding, expert_counts, task_ids)

    def _append_gate_probs(self, gate_probs: torch.Tensor):
        if gate_probs.numel() == 0 or self._hist_collected >= self.max_hist_samples:
            return
        flat = gate_probs.reshape(-1, self.num_gates)
        available = self.max_hist_samples - self._hist_collected
        take = min(available, flat.shape[0])
        if take <= 0:
            return
        if flat.shape[0] > take:
            idx = torch.randperm(flat.shape[0])[:take]
            sample = flat[idx]
        else:
            sample = flat
        self.gate_prob_samples.append(sample.numpy())
        self._hist_collected += sample.shape[0]

    def _append_expert_probs(self, expert_probs: torch.Tensor):
        if expert_probs.numel() == 0:
            return
        flat = expert_probs.reshape(-1, self.num_experts)
        take = min(self.max_hist_samples, flat.shape[0])
        if take <= 0:
            return
        if flat.shape[0] > take:
            idx = torch.randperm(flat.shape[0])[:take]
            sample = flat[idx]
        else:
            sample = flat
        self.expert_prob_samples.append(sample.numpy())

    def _append_instruction_embeddings(
        self,
        instruction_embedding: Optional[torch.Tensor],
        expert_counts: torch.Tensor,
        task_ids: torch.Tensor,
    ):
        if instruction_embedding is None or instruction_embedding.numel() == 0:
            return
        if self._tsne_collected >= self.max_tsne_samples:
            return

        dom_expert = expert_counts.argmax(dim=1).long()
        available = self.max_tsne_samples - self._tsne_collected
        num_samples = instruction_embedding.shape[0]
        take = min(available, num_samples)
        if take <= 0:
            return

        if num_samples > take:
            idx = torch.randperm(num_samples)[:take]
            embeddings = instruction_embedding[idx]
            dom_expert = dom_expert[idx]
        else:
            embeddings = instruction_embedding

        self.instruction_embeddings.append(embeddings.numpy())
        self.instruction_labels.append(dom_expert.numpy())
        self._tsne_collected += embeddings.shape[0]

    def summary(self) -> Dict:
        stats = self.get_utilization_stats()
        return {
            "name": self.name,
            "num_tasks": self.num_tasks,
            "num_gates": self.num_gates,
            "num_experts": self.num_experts,
            "tokens_per_task": self.tokens_per_task.tolist(),
            "top1_counts": self.top1_counts.tolist(),
            "top2_counts": self.top2_counts.tolist(),
            "expert_counts": self.expert_counts.tolist(),
            "gate_imbalance": stats["gate_imbalance"],
            "expert_imbalance": stats["expert_imbalance"],
            "avg_specialization": stats["avg_specialization"],
        }

    def get_utilization_stats(self) -> Dict[str, float]:
        total_tokens = float(self.tokens_per_task.sum().clamp_min(1))
        gate_util = (self.task_gate_hits.sum(dim=0) / max(total_tokens, 1.0)).cpu()
        expert_util = (self.expert_counts.sum(dim=0) / max(total_tokens, 1.0)).cpu()
        specialization = torch.where(
            self.tokens_per_task.unsqueeze(1) > 0,
            (self.expert_counts > 0).float(),
            torch.zeros_like(self.expert_counts),
        )
        specialization = specialization.sum(dim=1) / max(float(self.num_experts), 1.0)
        return {
            "gate_utilization": gate_util.numpy(),
            "expert_utilization": expert_util.numpy(),
            "gate_imbalance": float(gate_util.std().item()) if gate_util.numel() else 0.0,
            "expert_imbalance": float(expert_util.std().item()) if expert_util.numel() else 0.0,
            "task_specialization": specialization.numpy(),
            "avg_specialization": float(specialization.mean().item()) if specialization.numel() else 0.0,
        }

    def _task_normalize(self, tensor: torch.Tensor) -> np.ndarray:
        denom = self.tokens_per_task.unsqueeze(1).clamp_min(1)
        normalized = tensor / denom
        return normalized.cpu().numpy()

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        safe_name = self.name.replace(".", "_")
        summary_path = os.path.join(output_dir, f"{safe_name}_summary.json")
        with open(summary_path, "w") as fp:
            json.dump(self.summary(), fp, indent=2)

        self._plot_heatmap(
            data=self._normalized(self.top1_counts),
            title=f"{self.name} - Top1 Gate Usage",
            filename=os.path.join(output_dir, f"{safe_name}_gate_top1_heatmap.png"),
            xlabel="Gate",
            ylabel="Task",
        )
        self._plot_heatmap(
            data=self._normalized(self.expert_counts),
            title=f"{self.name} - Expert Usage",
            filename=os.path.join(output_dir, f"{safe_name}_expert_heatmap.png"),
            xlabel="Expert",
            ylabel="Task",
        )
        self._plot_histogram(output_dir, safe_name)
        self._plot_tsne(output_dir, safe_name)
        self._plot_routing_dashboard(output_dir, safe_name)
        self._plot_task_clustering(output_dir, safe_name)
        self._export_latex_table(output_dir, safe_name)
        self._save_npz(output_dir, safe_name)

    def _normalized(self, tensor: torch.Tensor) -> np.ndarray:
        data = tensor.clone()
        denom = data.sum(dim=1, keepdim=True).clamp_min(1)
        return (data / denom).numpy()

    def _plot_heatmap(self, data: np.ndarray, title: str, filename: str, xlabel: str, ylabel: str):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(6, data.shape[1] * 0.6), max(4, data.shape[0] * 0.4)))
        im = ax.imshow(data, aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(data.shape[1]))
        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels(self.task_names)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(filename, dpi=200)
        plt.close(fig)

    def _plot_histogram(self, output_dir: str, safe_name: str):
        if not self.gate_prob_samples and not self.expert_prob_samples:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cols = 2 if self.expert_prob_samples else 1
        fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 4))
        axes = np.atleast_1d(axes)

        if self.gate_prob_samples:
            data_gate = np.concatenate(self.gate_prob_samples, axis=0)
            axes[0].hist(data_gate.flatten(), bins=30, color="#4477AA", alpha=0.85)
            axes[0].set_title(f"{self.name} - Gate Probability Histogram")
            axes[0].set_xlabel("Gate probability")
            axes[0].set_ylabel("Frequency")

        if self.expert_prob_samples:
            data_exp = np.concatenate(self.expert_prob_samples, axis=0)
            target_ax = axes[1] if cols == 2 else axes[0]
            target_ax.hist(data_exp.flatten(), bins=30, color="#66CC99", alpha=0.85)
            target_ax.set_title(f"{self.name} - Expert Probability Histogram")
            target_ax.set_xlabel("Expert probability")
            target_ax.set_ylabel("Frequency")

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{safe_name}_prob_hist.png"), dpi=200)
        plt.close(fig)

    def _plot_tsne(self, output_dir: str, safe_name: str):
        if not self.instruction_embeddings:
            return
        data = np.concatenate(self.instruction_embeddings, axis=0)
        if data.shape[0] < 5:
            return
        labels = np.concatenate(self.instruction_labels, axis=0)

        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, init="random", learning_rate="auto", perplexity=min(30, max(5, data.shape[0] // 3)))
        coords = tsne.fit_transform(data)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(6, 5))
        cmap = plt.cm.get_cmap("tab20", max(self.num_experts, int(labels.max() + 1)))
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap=cmap, s=12, alpha=0.8)
        ax.set_title(f"{self.name} - Instruction Embedding t-SNE")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        unique_labels = np.unique(labels.astype(int))
        legend_handles = [
            mpatches.Patch(color=cmap(int(lbl)), label=f"E{int(lbl)}") for lbl in unique_labels
        ]
        if legend_handles:
            ax.legend(handles=legend_handles, title="Expert", loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{safe_name}_instruction_tsne.png"), dpi=220)
        plt.close(fig)

    def _plot_routing_dashboard(self, output_dir: str, safe_name: str):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
        except Exception as exc:
            print(f"[TaskMoE] Skipping routing dashboard for {self.name}: {exc}")
            return

        gate_heat = self._task_normalize(self.task_gate_hits)
        expert_heat = self._task_normalize(self.task_expert_hits)
        total = self.gate_expert_matrix.sum().item()
        gate_expert_norm = (
            (self.gate_expert_matrix / max(total, 1.0)).cpu().numpy()
            if total > 0
            else self.gate_expert_matrix.cpu().numpy()
        )
        stats = self.get_utilization_stats()

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        sns.heatmap(
            gate_heat,
            ax=axes[0, 0],
            cmap="YlOrRd",
            xticklabels=[f"G{i}" for i in range(self.num_gates)],
            yticklabels=self.task_names,
        )
        axes[0, 0].set_title("Gate Usage by Task (normalized)")
        axes[0, 0].set_xlabel("Gates")
        axes[0, 0].set_ylabel("Tasks")

        sns.heatmap(
            expert_heat,
            ax=axes[0, 1],
            cmap="YlOrRd",
            xticklabels=[f"E{i}" for i in range(self.num_experts)],
            yticklabels=self.task_names,
        )
        axes[0, 1].set_title("Expert Usage by Task (normalized)")
        axes[0, 1].set_xlabel("Experts")
        axes[0, 1].set_ylabel("Tasks")

        sns.heatmap(
            gate_expert_norm,
            ax=axes[1, 0],
            cmap="Blues",
            xticklabels=[f"E{i}" for i in range(self.num_experts)],
            yticklabels=[f"G{i}" for i in range(self.num_gates)],
        )
        axes[1, 0].set_title("Gate ↔ Expert Co-occurrence")
        axes[1, 0].set_xlabel("Experts")
        axes[1, 0].set_ylabel("Gates")

        expert_color = "#3366CC"
        gate_color = "#CC3366"
        axes[1, 1].bar(
            np.arange(self.num_experts),
            stats["expert_utilization"],
            alpha=0.8,
            label="Experts",
            color=expert_color,
        )
        axes[1, 1].bar(
            np.arange(self.num_gates),
            stats["gate_utilization"][: self.num_gates],
            alpha=0.6,
            label="Gates",
            color=gate_color,
        )
        axes[1, 1].set_title("Utilization Distribution")
        axes[1, 1].set_xlabel("Index")
        axes[1, 1].set_ylabel("Normalized usage")
        axes[1, 1].set_ylim(0.0, 1.0)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        fig.suptitle(f"Routing Dashboard – {self.name}", y=0.98, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(output_dir, f"{safe_name}_routing_dashboard.png"), dpi=250)
        plt.close(fig)

    def _plot_task_clustering(self, output_dir: str, safe_name: str):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.cluster import KMeans
            from sklearn.manifold import TSNE
        except Exception as exc:
            print(f"[TaskMoE] Skipping task clustering for {self.name}: {exc}")
            return

        features = self.task_expert_hits.cpu().numpy()
        if features.shape[0] < 2:
            return
        perplexity = min(30, max(5, features.shape[0] - 1))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="random", learning_rate="auto")
        emb = tsne.fit_transform(features)

        n_clusters = min(4, features.shape[0])
        if n_clusters < 2:
            clusters = np.zeros(features.shape[0], dtype=int)
        else:
            clusters = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(features)

        fig, ax = plt.subplots(figsize=(9, 7))
        scatter = ax.scatter(emb[:, 0], emb[:, 1], c=clusters, cmap="tab10", s=70, alpha=0.8)
        for idx, (x, y) in enumerate(emb):
            ax.annotate(self.task_names[idx], (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
        if n_clusters >= 2:
            legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
            ax.add_artist(legend)
        ax.set_title(f"Task Clustering via Expert Usage – {self.name}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{safe_name}_task_clusters.png"), dpi=250)
        plt.close(fig)

    def _export_latex_table(self, output_dir: str, safe_name: str):
        stats = self.get_utilization_stats()
        lines = [
            "\\begin{table}[ht]",
            "\\centering",
            f"\\caption{{Routing statistics for {self.name}}}",
            "\\label{tab:routing_stats_%s}" % safe_name,
            "\\begin{tabular}{lcccc}",
            "\\hline",
            "Task & Gate Util. & Expert Util. & Specialization & Top-3 Experts \\\\",
            "\\hline",
        ]
        for idx, task_name in enumerate(self.task_names):
            gate_util = float(self.task_gate_hits[idx].sum().item() / max(self.tokens_per_task[idx].item(), 1.0))
            expert_util = float(self.task_expert_hits[idx].sum().item() / max(self.tokens_per_task[idx].item(), 1.0))
            specialization = stats["task_specialization"][idx] if idx < len(stats["task_specialization"]) else 0.0
            topk = torch.topk(self.task_expert_hits[idx], k=min(3, self.num_experts))
            primary = ", ".join([f"E{int(e)}" for e in topk.indices.tolist()])
            lines.append(
                f"{task_name} & {gate_util:.3f} & {expert_util:.3f} & {specialization:.3f} & {primary} \\\\"
            )
        lines.extend(
            [
                "\\hline",
                f"Average & {np.mean(stats['gate_utilization']):.3f} & "
                f"{np.mean(stats['expert_utilization']):.3f} & {stats['avg_specialization']:.3f} & - \\\\",
                "\\hline",
                "\\end{tabular}",
                "\\end{table}",
            ]
        )
        with open(os.path.join(output_dir, f"{safe_name}_routing_table.tex"), "w") as fp:
            fp.write("\n".join(lines))

    def _save_npz(self, output_dir: str, safe_name: str):
        stats = self.get_utilization_stats()
        np.savez(
            os.path.join(output_dir, f"{safe_name}_routing_stats.npz"),
            gate_usage=self.task_gate_hits.cpu().numpy(),
            expert_usage=self.task_expert_hits.cpu().numpy(),
            tokens_per_task=self.tokens_per_task.cpu().numpy(),
            gate_expert_matrix=self.gate_expert_matrix.cpu().numpy(),
            statistics=stats,
        )


class TaskMoEUsageAnalyzer:
    def __init__(
        self,
        output_dir: str,
        task_name_lookup: Optional[Dict[int, str]] = None,
        max_hist_samples: int = 20000,
        max_tsne_samples: int = 2048,
    ):
        self.output_dir = output_dir
        self.task_name_lookup = task_name_lookup or {}
        self.max_hist_samples = max_hist_samples
        self.max_tsne_samples = max_tsne_samples
        self.recorders: Dict[str, TaskMoEUsageRecorder] = {}

    def register(self, module_name: str, num_tasks: int, num_gates: int, num_experts: int) -> TaskMoEUsageRecorder:
        if module_name in self.recorders:
            return self.recorders[module_name]
        recorder = TaskMoEUsageRecorder(
            name=module_name,
            num_tasks=num_tasks,
            num_gates=num_gates,
            num_experts=num_experts,
            task_name_lookup=self.task_name_lookup,
            max_hist_samples=self.max_hist_samples,
            max_tsne_samples=self.max_tsne_samples,
        )
        self.recorders[module_name] = recorder
        return recorder

    def attach(self, model: nn.Module, num_tasks_=None):
        for name, module in model.named_modules():
            if isinstance(module, TaskMoE):
                if num_tasks_ is not None:
                    recorder = self.register(
                        name,
                        num_tasks_,
                        module.num_gates,
                        module.num_experts,
                    )
                    print(f"Attached recorder to {name}")
                    print(f"Number of tasks: {module.num_tasks}")
                else:
                    recorder = self.register(
                        name,
                        module.num_tasks,
                        module.num_gates,
                        module.num_experts,
                    )
                    print(f"Attached recorder to {name}")
                    print(f"Number of tasks: {module.num_tasks}")
                print(f"Number of gates: {module.num_gates}")
                print(f"Number of experts: {module.num_experts}")
                module.enable_usage_logging(recorder)

    def reset(self):
        for recorder in self.recorders.values():
            recorder.reset()

    def save_all(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for recorder in self.recorders.values():
            recorder.save(self.output_dir)

# Example usage and test
if __name__ == "__main__":
    # Test parameters matching paper
    dim = 512
    num_experts = 16
    num_gates = 8
    num_tasks = 18  # RLBench tasks
    instruction_dim = 512
    batch_size = 4
    seq_len = 256
    
    # Create TaskMoE instance
    task_moe = TaskMoE(
        dim=dim,
        num_experts=num_experts,
        num_gates=num_gates,
        num_tasks=num_tasks,
        instruction_dim=instruction_dim
    )
    
    # Test forward pass
    visual_features = torch.randn(batch_size, seq_len, dim)
    instruction_emb = torch.randn(batch_size, instruction_dim)
    task_ids = torch.randint(0, num_tasks, (batch_size,))
    print(task_ids)
    output, loss = task_moe(visual_features, instruction_emb, task_ids)
    
    print(f"Input shape: {visual_features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Routing loss: {loss.item()}")
    print("TaskMoE implementation successful!")
