from omegaconf import DictConfig
from torch import nn
from einops import rearrange
from typing import Optional
import torch
from preprocess import CubePointCloudRenderer
from utils.layers import Conv2DBlock, PreNorm, Attention, DenseBlock, FeedForward
from utils.taskmoe import TaskMoE as VisionTaskMoE


def _extract_taskmoe_kwargs(cfg: DictConfig, default_dim: int) -> dict:
    """
    Collect TaskMoE hyperparameters from config with sensible defaults.
    """
    default_experts = getattr(cfg, "taskmoe_num_experts", None)
    if default_experts is None:
        default_experts = 16

    default_gates = getattr(cfg, "taskmoe_num_gates", None)
    if default_gates is None:
        default_gates = 8

    return {
        "num_experts": default_experts,
        "num_gates": default_gates,
        "instruction_dim": getattr(cfg, "taskmoe_instruction_dim", default_dim),
        "hidden_dim": getattr(cfg, "taskmoe_hidden_dim", None),
        "second_policy_train": getattr(cfg, "taskmoe_second_policy_train", "random"),
        "second_policy_eval": getattr(cfg, "taskmoe_second_policy_eval", "random"),
        "second_threshold_train": getattr(cfg, "taskmoe_second_threshold_train", 0.2),
        "second_threshold_eval": getattr(cfg, "taskmoe_second_threshold_eval", 0.2),
        "capacity_factor_train": getattr(cfg, "taskmoe_capacity_factor_train", 1.25),
        "capacity_factor_eval": getattr(cfg, "taskmoe_capacity_factor_eval", 1.25),
        "semantic_reg_weight": getattr(cfg, "taskmoe_semantic_reg_weight", 1e-4),
        "gate_entropy_weight": getattr(cfg, "taskmoe_gate_entropy_weight", 0.01),
        "expert_entropy_weight": getattr(cfg, "taskmoe_expert_entropy_weight", 0.01),
        "loss_coef": getattr(cfg, "taskmoe_loss_coef", 1e-2),
    }


class TaskMoEFeedForwardBlock(nn.Module):
    """
    PreNorm-friendly wrapper around the TaskMoE implementation from taskmoe.py.
    """

    def __init__(self, dim: int, num_tasks: int, instruction_dim: Optional[int] = None, **taskmoe_kwargs):
        super().__init__()
        instruction_dim = instruction_dim if instruction_dim is not None else dim
        self.task_moe = VisionTaskMoE(
            dim=dim,
            num_tasks=num_tasks,
            instruction_dim=instruction_dim,
            **taskmoe_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
        instruction_embedding: Optional[torch.Tensor] = None,
        **_,
    ):
        batch_size = x.shape[0]
        device = x.device

        if task_id is None:
            task_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            task_id = task_id.to(device=device, dtype=torch.long)
            if task_id.ndim == 0:
                task_id = task_id.unsqueeze(0)
            if task_id.shape[0] == 1 and batch_size != 1:
                task_id = task_id.expand(batch_size)
            elif task_id.shape[0] != batch_size:
                raise ValueError(
                    f"Incompatible task_id batch ({task_id.shape[0]}) for input batch ({batch_size})."
                )

        if instruction_embedding is None:
            instruction_embedding = x.mean(dim=1)
        else:
            instruction_embedding = instruction_embedding.to(device=device, dtype=x.dtype)
            if instruction_embedding.ndim == 2 and instruction_embedding.shape[0] == 1 and batch_size != 1:
                instruction_embedding = instruction_embedding.expand(batch_size, -1)
            elif instruction_embedding.shape[0] != batch_size:
                raise ValueError(
                    f"Incompatible instruction_embedding batch ({instruction_embedding.shape[0]}) "
                    f"for input batch ({batch_size})."
                )

        output, loss = self.task_moe(x, instruction_embedding, task_id)
        return output, loss


class TaskMoEMVT(nn.Module):
    def __init__(self, cfg: DictConfig, renderer: Optional[CubePointCloudRenderer] = None):
        super().__init__()
        self.depth = cfg.depth
        self.img_feat_dim = cfg.img_feat_dim
        self.img_size = cfg.img_size
        self.add_proprio = cfg.add_proprio
        self.proprio_dim = cfg.proprio_dim
        self.add_lang = cfg.add_lang
        self.lang_dim = cfg.lang_dim
        self.lang_len = cfg.lang_len
        self.im_channels = cfg.im_channels  # 64
        self.img_patch_size = cfg.img_patch_size
        self.attn_dropout = cfg.attn_dropout
        self.add_corr = cfg.add_corr
        self.add_pixel_loc = cfg.add_pixel_loc
        self.add_depth = cfg.add_depth
        self.pe_fix = cfg.pe_fix
        self.attn_dim = cfg.attn_dim
        self.attn_heads = cfg.attn_heads
        self.attn_dim_head = cfg.attn_dim_head
        self.attn_dropout = cfg.attn_dropout
        self.use_xformers = cfg.use_xformers
        self.feat_dim = cfg.feat_dim
        self.n_tasks = cfg.n_tasks
        activation = "lrelu"

        self.norm_corr = cfg.norm_corr
        self.num_rot = cfg.num_rotation_classes
        self.cfg = cfg

        assert renderer is not None
        self.renderer = renderer
        self.num_cameras = self.renderer.num_cameras

        # patchified input dimensions
        spatial_size = self.img_size // self.img_patch_size  # 16

        if self.add_proprio:
            # 64 img features + 64 proprio features
            self.input_dim_before_seq = self.im_channels * 2
        else:
            self.input_dim_before_seq = self.im_channels

        # learnable positional encoding
        if self.add_lang:
            lang_emb_dim, lang_max_seq_len = self.lang_dim, self.lang_len
        else:
            lang_emb_dim, lang_max_seq_len = 0, 0
        self.lang_emb_dim = lang_emb_dim
        self.lang_max_seq_len = lang_max_seq_len

        if self.pe_fix:
            num_pe_token = spatial_size**2 * self.num_cameras
        else:
            num_pe_token = lang_max_seq_len + (spatial_size**2 * self.num_cameras)

        self.pos_encoding = nn.Parameter(torch.randn(1, num_pe_token, self.input_dim_before_seq))

        inp_img_feat_dim = self.img_feat_dim
        if self.add_corr:
            inp_img_feat_dim += 3
        if self.add_pixel_loc:
            inp_img_feat_dim += 3
            self.pixel_loc = torch.zeros((self.num_cameras, 3, self.img_size, self.img_size))
            self.pixel_loc[:, 0, :, :] = torch.linspace(-1, 1, self.num_cameras).unsqueeze(-1).unsqueeze(-1)
            self.pixel_loc[:, 1, :, :] = torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(-1)
            self.pixel_loc[:, 2, :, :] = torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(0)  # [3, 3, 224, 224]

        if self.add_depth:
            inp_img_feat_dim += 1

        if self.add_proprio:
            # proprio preprocessing encoder
            self.proprio_preprocess = DenseBlock(
                self.proprio_dim,
                self.im_channels,
                norm="group",
                activation=activation,
            )

        self.patchify = Conv2DBlock(
            inp_img_feat_dim,
            self.im_channels,
            kernel_sizes=self.img_patch_size,
            strides=self.img_patch_size,
            norm="group",
            activation=activation,
            padding=0,
        )

        # lang preprocess
        if self.add_lang:
            self.lang_preprocess = DenseBlock(
                lang_emb_dim,
                self.im_channels * 2,
                norm="group",
                activation=activation,
            )

        self.fc_bef_attn = DenseBlock(
            self.input_dim_before_seq,
            self.attn_dim,
            norm=None,
            activation=None,
        )
        self.fc_aft_attn = DenseBlock(
            self.attn_dim,
            self.input_dim_before_seq,
            norm=None,
            activation=None,
        )

        get_attn_attn = lambda: PreNorm(
            self.attn_dim,
            Attention(
                self.attn_dim,
                heads=self.attn_heads,
                dim_head=self.attn_dim_head,
                dropout=self.attn_dropout,
                use_fast=self.use_xformers,
            ),
        )

        taskmoe_kwargs = _extract_taskmoe_kwargs(cfg, self.attn_dim)
        instruction_dim = taskmoe_kwargs.pop("instruction_dim")
        get_attn_moe = lambda: PreNorm(
            self.attn_dim,
            TaskMoEFeedForwardBlock(
                self.attn_dim,
                num_tasks=self.n_tasks,
                instruction_dim=instruction_dim,
                **taskmoe_kwargs,
            ),
        )
        get_attn_ff = lambda: PreNorm(self.attn_dim, FeedForward(self.attn_dim))
        self.layers = nn.ModuleList([])
        attn_depth = self.depth
        for _ in range(attn_depth):
            self.layers.append(nn.ModuleList([get_attn_attn(), get_attn_ff()]))
        
        self.layers_moe_x = nn.ModuleList([])
        for _ in range(self.cfg.moe_x_depth):
            self.layers_moe_x.append(nn.ModuleList([get_attn_attn(), get_attn_moe()]))


    def _compute_instruction_embedding(
        self, language_tokens: torch.Tensor, image_tokens: torch.Tensor, num_lang_tokens: int
    ) -> torch.Tensor:
        """
        Build a per-sample instruction embedding for TaskMoE conditioning.
        """
        if num_lang_tokens > 0 and language_tokens.numel() > 0:
            return language_tokens.mean(dim=1)
        if image_tokens.numel() > 0:
            return image_tokens.mean(dim=1)
        device = image_tokens.device
        return torch.zeros(image_tokens.shape[0], image_tokens.shape[-1], device=device, dtype=image_tokens.dtype)

    def forward(
        self,
        img,
        task_id,
        proprio=None,
        lang_emb=None,
        history_mem=None,
    ):
        """
        :param img: tensor of shape (bs, num_cameras, img_feat_dim, h, w)
        :param proprio: tensor of shape (bs, priprio_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        """

        bs, num_cameras, img_feat_dim, h, w = img.shape
        num_pat_img = h // self.img_patch_size
        assert num_cameras == self.num_cameras
        assert h == w == self.img_size

        img = img.view(bs * num_cameras, img_feat_dim, h, w)
        ins = self.patchify(img)
        ins = (
            ins.view(
                bs,
                num_cameras,
                self.im_channels,
                num_pat_img,
                num_pat_img,
            )
            .transpose(1, 2)
            .clone()
        )

        _, _, _d, _h, _w = ins.shape
        if self.add_proprio:
            p = self.proprio_preprocess(proprio)
            p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, _d, _h, _w)
            ins = torch.cat([ins, p], dim=1)

        ins = rearrange(ins, "b d ... -> b ... d")
        ins_orig_shape = ins.shape
        ins = rearrange(ins, "b ... d -> b (...) d")

        if self.pe_fix:
            ins += self.pos_encoding

        num_lang_tok = 0
        if self.add_lang and lang_emb is not None:
            l = self.lang_preprocess(lang_emb.view(bs * self.lang_max_seq_len, self.lang_emb_dim))
            l = l.view(bs, self.lang_max_seq_len, -1)
            num_lang_tok = l.shape[1]
            ins = torch.cat((l, ins), dim=1)

        if not self.pe_fix:
            ins = ins + self.pos_encoding

        x = self.fc_bef_attn(ins)
        lx, img_tokens = x[:, :num_lang_tok], x[:, num_lang_tok:]

        requires_taskmoe = True
        if requires_taskmoe:
            if task_id is None:
                raise ValueError("task_id must be provided when TaskMoE layers are enabled.")
            task_id = task_id.to(x.device).long()
            if task_id.ndim == 0:
                task_id = task_id.unsqueeze(0)
            if task_id.shape[0] != bs:
                raise ValueError(f"Expected task_id batch {bs}, received {task_id.shape[0]}.")
            instruction_embedding = self._compute_instruction_embedding(lx, img_tokens, num_lang_tok)
            instruction_embedding_img = instruction_embedding.repeat_interleave(self.num_cameras, dim=0)
            task_id_img = task_id.repeat_interleave(self.num_cameras)
        else:
            instruction_embedding = None
            instruction_embedding_img = None
            task_id_img = None
            task_id = task_id.to(x.device).long() if task_id is not None else None

        imgx = img_tokens.reshape(bs * num_cameras, num_pat_img * num_pat_img, -1)
        total_loss = {}

        for self_attn, self_ff in self.layers[: len(self.layers) // 2]:
            imgx = self_attn(imgx) + imgx
            imgx = self_ff(imgx) + imgx

        moe_x_loss = torch.zeros((), device=imgx.device)
        for self_attn, self_ff in self.layers_moe_x:
            x = self_attn(x) + x

            def _moe_x_forward(t):
                return self_ff(
                    t,
                    task_id=task_id,
                    instruction_embedding=instruction_embedding,
                )
            x_out, loss = _moe_x_forward(x)
            x = x_out + x
            moe_x_loss = moe_x_loss + loss
        total_loss["taskmoe_x_loss"] = moe_x_loss

        imgx = imgx.reshape(bs, num_cameras * num_pat_img * num_pat_img, -1)
        x = torch.cat((lx, imgx), dim=1)

        for self_attn, self_ff in self.layers[len(self.layers) // 2 :]:
            x = self_attn(x) + x
            x = self_ff(x) + x
            
        if self.add_lang:
            x = x[:, num_lang_tok:]
        x = self.fc_aft_attn(x)

        x = x.reshape(bs, *ins_orig_shape[1:-1], x.shape[-1])
        x = rearrange(x, "b ... d -> b d ...")
        x = x.transpose(1, 2)

        return x, total_loss
