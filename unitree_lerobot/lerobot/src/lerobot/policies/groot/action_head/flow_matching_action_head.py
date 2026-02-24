# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import math
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.distributions import Beta

from lerobot.utils.import_utils import _transformers_available

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers import PretrainedConfig
    from transformers.feature_extraction_utils import BatchFeature
else:
    PretrainedConfig = object
    BatchFeature = None

from lerobot.policies.groot.action_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from .cross_attention_dit import DiT, SelfAttentionTransformer


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_w = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_w) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        b, t, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == b:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, t)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


def _generate_2d_positional_embedding(height: int, width: int, d_model: int) -> torch.Tensor:
    if d_model % 2 != 0:
        raise ValueError(f"d_model should be even, got {d_model}")

    pe = torch.zeros(height, width, d_model, dtype=torch.float32)
    d_model_half = d_model // 2
    div_term = torch.exp(
        torch.arange(0, d_model_half, 2, dtype=torch.float32) * (-(math.log(10000.0) / d_model_half))
    )
    pos_w = torch.arange(0, width, dtype=torch.float32).unsqueeze(1)
    pos_h = torch.arange(0, height, dtype=torch.float32).unsqueeze(1)
    pe[:, :, 0:d_model_half:2] = torch.sin(pos_w * div_term).unsqueeze(0).repeat(height, 1, 1)
    pe[:, :, 1:d_model_half:2] = torch.cos(pos_w * div_term).unsqueeze(0).repeat(height, 1, 1)
    pe[:, :, d_model_half::2] = torch.sin(pos_h * div_term).unsqueeze(1).repeat(1, width, 1)
    pe[:, :, d_model_half + 1 :: 2] = torch.cos(pos_h * div_term).unsqueeze(1).repeat(1, width, 1)
    return pe


def _to_mask_bhw(mask: torch.Tensor) -> torch.Tensor:
    """Canonicalize mask tensor to (B, H, W), float in [0, 1]."""
    mask_t = torch.as_tensor(mask)

    if mask_t.ndim == 5:
        # (B, T, C, H, W) or (B, T, H, W, C): use current frame only
        mask_t = mask_t[:, 0]

    if mask_t.ndim == 4:
        # BCHW
        if mask_t.shape[1] in (1, 3):
            mask_t = mask_t[:, 0]
        # BHWC
        elif mask_t.shape[-1] in (1, 3):
            mask_t = mask_t[..., 0]
        else:
            raise ValueError(f"Unsupported mask tensor shape: {tuple(mask_t.shape)}")
    elif mask_t.ndim == 3:
        # HWC without batch
        if mask_t.shape[-1] in (1, 3) and mask_t.shape[0] > 4 and mask_t.shape[1] > 4:
            mask_t = mask_t[..., 0].unsqueeze(0)
        # CHW without batch
        elif mask_t.shape[0] in (1, 3) and mask_t.shape[1] > 4 and mask_t.shape[2] > 4:
            mask_t = mask_t[0].unsqueeze(0)
        # Else assume BHW
    elif mask_t.ndim == 2:
        mask_t = mask_t.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported mask tensor shape: {tuple(mask_t.shape)}")

    mask_t = mask_t.to(dtype=torch.float32)
    if mask_t.numel() > 0 and mask_t.max() > 1:
        mask_t = mask_t / 255.0
    mask_t = torch.nan_to_num(mask_t, nan=0.0, posinf=1.0, neginf=0.0)
    return mask_t.clamp_(0.0, 1.0)


class ObjectCentricPool2d(nn.Module):
    def __init__(self, n_emb: int, height: int, width: int, threshold: float = 0.5):
        super().__init__()
        self.height = height
        self.width = width
        self.threshold = threshold
        self.register_buffer(
            "positional_embedding",
            _generate_2d_positional_embedding(height, width, n_emb),
            persistent=False,
        )
        self.global_object_embedding = nn.Parameter(torch.randn(n_emb))
        self.empty_object_embedding = nn.Parameter(torch.randn(n_emb))

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        mask = _to_mask_bhw(mask)
        if mask.shape[-2:] != (self.height, self.width):
            mask = F.interpolate(mask.unsqueeze(1), size=(self.height, self.width), mode="nearest").squeeze(1)

        bs = mask.shape[0]
        mask_bool = mask > self.threshold
        mask_float = mask_bool.to(dtype=torch.float32)

        y_coords = torch.arange(self.height, device=mask.device).view(1, -1, 1).expand(bs, -1, self.width)
        x_coords = torch.arange(self.width, device=mask.device).view(1, 1, -1).expand(bs, self.height, -1)
        true_y_sum = (y_coords * mask_float).sum(dim=(1, 2))
        true_x_sum = (x_coords * mask_float).sum(dim=(1, 2))
        true_count = mask_float.sum(dim=(1, 2))
        safe_count = torch.where(true_count > 0, true_count, torch.ones_like(true_count))

        true_y = (true_y_sum / safe_count).long().clamp_(0, self.height - 1)
        true_x = (true_x_sum / safe_count).long().clamp_(0, self.width - 1)

        pos_emb = self.positional_embedding[true_y, true_x].to(mask.device)
        global_emb = self.global_object_embedding.unsqueeze(0).expand(bs, -1)
        empty_emb = self.empty_object_embedding.unsqueeze(0).expand(bs, -1)
        non_empty = (true_count > 0).unsqueeze(1)
        return torch.where(non_empty, global_emb + pos_emb, empty_emb)


class GrayScaleFeatureExtractor(nn.Module):
    def __init__(
        self,
        n_dim: int,
        height: int,
        width: int,
        target_size: tuple[int, int] = (32, 32),
        threshold: float = 0.5,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.target_size = target_size
        self.threshold = threshold
        self.register_buffer(
            "positional_embedding",
            _generate_2d_positional_embedding(height, width, n_dim),
            persistent=False,
        )

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(64, n_dim)

    def _get_batch_bounding_boxes(self, masks: torch.Tensor) -> list[tuple[int, int, int, int]]:
        bboxes: list[tuple[int, int, int, int]] = []
        for i in range(masks.shape[0]):
            binary = masks[i] > self.threshold
            nonzero = torch.nonzero(binary, as_tuple=False)
            if nonzero.numel() == 0:
                bboxes.append((0, 0, 1, 1))
                continue

            ymin = int(nonzero[:, 0].min().item())
            xmin = int(nonzero[:, 1].min().item())
            ymax = int(nonzero[:, 0].max().item()) + 1
            xmax = int(nonzero[:, 1].max().item()) + 1

            if ymax <= ymin:
                ymax = min(self.height, ymin + 1)
            if xmax <= xmin:
                xmax = min(self.width, xmin + 1)
            bboxes.append((xmin, ymin, xmax, ymax))
        return bboxes

    def _crop_and_resize(self, masks: torch.Tensor, bboxes: list[tuple[int, int, int, int]]) -> torch.Tensor:
        resized = []
        for i, (xmin, ymin, xmax, ymax) in enumerate(bboxes):
            crop = masks[i : i + 1, ymin:ymax, xmin:xmax].unsqueeze(1)
            if crop.numel() == 0:
                crop = torch.zeros((1, 1, 1, 1), dtype=masks.dtype, device=masks.device)
            crop = F.interpolate(crop, size=self.target_size, mode="bilinear", align_corners=False)
            resized.append(crop)
        return torch.cat(resized, dim=0)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        mask = _to_mask_bhw(mask)
        if mask.shape[-2:] != (self.height, self.width):
            mask = F.interpolate(mask.unsqueeze(1), size=(self.height, self.width), mode="nearest").squeeze(1)

        bboxes = self._get_batch_bounding_boxes(mask)
        x = self._crop_and_resize(mask, bboxes)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.global_max_pool(x).flatten(1)
        x = self.fc(x)

        bbox_h = torch.tensor(
            [min(self.height - 1, max(0, b[3] - b[1])) for b in bboxes], device=x.device, dtype=torch.long
        )
        bbox_w = torch.tensor(
            [min(self.width - 1, max(0, b[2] - b[0])) for b in bboxes], device=x.device, dtype=torch.long
        )
        pos_emb = self.positional_embedding[bbox_h, bbox_w].to(x.device)
        return x + pos_emb


class PickMaskControlEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        height: int = 224,
        width: int = 224,
        local_feature: str | None = None,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.object_pool = ObjectCentricPool2d(
            n_emb=embedding_dim,
            height=height,
            width=width,
            threshold=threshold,
        )
        if local_feature is None:
            self.local_feature = None
        elif local_feature == "graycnn":
            self.local_feature = GrayScaleFeatureExtractor(
                n_dim=embedding_dim,
                height=height,
                width=width,
                threshold=threshold,
            )
        else:
            raise ValueError(f"Unsupported local_feature={local_feature}. Expected None or 'graycnn'.")

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        tokens = [self.object_pool(mask)]
        if self.local_feature is not None:
            tokens.append(self.local_feature(mask))
        return torch.stack(tokens, dim=1)


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(default=True, metadata={"help": "Whether to add positional embedding"})
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(default=None, metadata={"help": "Diffusion model configuration."})
    input_embedding_dim: int = field(default=1536, metadata={"help": "Input embedding channel dimension."})
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maximum Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(default=0.999, metadata={"help": "Flow matching noise Beta distribution s."})
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(default=32, metadata={"help": "Number of target vision tokens."})
    use_pick_mask_control: bool = field(
        default=True,
        metadata={"help": "Enable ControlVLA-style pick-mask control branch in DiT cross-attention."},
    )
    pick_mask_key: str = field(default="pick_mask", metadata={"help": "Batch key for pick mask tensor."})
    pick_mask_height: int = field(default=224, metadata={"help": "Mask resize height for control encoder."})
    pick_mask_width: int = field(default=224, metadata={"help": "Mask resize width for control encoder."})
    pick_mask_threshold: float = field(default=0.5, metadata={"help": "Foreground threshold for mask binarization."})
    pick_mask_local_feature: str | None = field(
        default="graycnn",
        metadata={"help": "Optional local mask feature branch. Supported: None, 'graycnn'."},
    )
    zero_init_pick_mask_control: bool = field(
        default=True,
        metadata={"help": "Zero-init control cross-attention output projection to preserve pretrained behavior at step 0."},
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        diffusion_model_cfg = dict(config.diffusion_model_cfg or {})
        if config.use_pick_mask_control:
            diffusion_model_cfg.setdefault("enable_control_cross_attention", True)
            diffusion_model_cfg.setdefault("zero_init_control", config.zero_init_pick_mask_control)
        self.model = DiT(**diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.vlln = nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        self.vl_self_attention = (
            SelfAttentionTransformer(**(config.vl_self_attention_cfg or {}))
            if config.use_vlln
            else nn.Identity()
        )
        self._last_control_token_abs_mean: float = 0.0
        self._last_control_token_norm: float = 0.0
        self.pick_mask_encoder = None
        if config.use_pick_mask_control:
            self.pick_mask_encoder = PickMaskControlEncoder(
                embedding_dim=config.backbone_embedding_dim,
                height=config.pick_mask_height,
                width=config.pick_mask_width,
                local_feature=config.pick_mask_local_feature,
                threshold=config.pick_mask_threshold,
            )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.reset_pick_mask_control_parameters()
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.pick_mask_encoder is not None:
                self.pick_mask_encoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.pick_mask_encoder is not None:
                    self.pick_mask_encoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    @torch.no_grad()
    def reset_pick_mask_control_parameters(self) -> None:
        if not self.config.use_pick_mask_control:
            return

        # Reinitialize mask encoder parameters with stable finite values.
        if self.pick_mask_encoder is not None:
            nn.init.normal_(self.pick_mask_encoder.object_pool.global_object_embedding, mean=0.0, std=0.02)
            nn.init.normal_(self.pick_mask_encoder.object_pool.empty_object_embedding, mean=0.0, std=0.02)

            if self.pick_mask_encoder.local_feature is not None:
                local = self.pick_mask_encoder.local_feature
                nn.init.normal_(local.conv1.weight, mean=0.0, std=0.02)
                nn.init.zeros_(local.conv1.bias)
                nn.init.normal_(local.conv2.weight, mean=0.0, std=0.02)
                nn.init.zeros_(local.conv2.bias)
                nn.init.normal_(local.fc.weight, mean=0.0, std=0.02)
                nn.init.zeros_(local.fc.bias)
                nn.init.ones_(local.bn1.weight)
                nn.init.zeros_(local.bn1.bias)
                nn.init.ones_(local.bn2.weight)
                nn.init.zeros_(local.bn2.bias)
                local.bn1.running_mean.zero_()
                local.bn1.running_var.fill_(1.0)
                local.bn2.running_mean.zero_()
                local.bn2.running_var.fill_(1.0)

        # Zero-init only control output projection so branch starts as no-op but K/V can learn immediately.
        if self.config.zero_init_pick_mask_control:
            for block in self.model.transformer_blocks:
                attn_control = getattr(block, "attn_control", None)
                if attn_control is None:
                    continue
                to_out = attn_control.to_out[0] if isinstance(attn_control.to_out, (list, tuple, nn.ModuleList)) else None
                if to_out is not None and hasattr(to_out, "weight") and to_out.weight is not None:
                    nn.init.zeros_(to_out.weight)
                if to_out is not None and hasattr(to_out, "bias") and to_out.bias is not None:
                    nn.init.zeros_(to_out.bias)

        # Last-resort sanitization in case loader left any non-finite values.
        for p in self.parameters():
            if p.requires_grad and p.dtype.is_floating_point:
                p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1.0, neginf=-1.0)

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(
        self, backbone_output: BatchFeature, action_input: BatchFeature | None = None
    ) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features

        control_features = None
        if self.config.use_pick_mask_control and self.pick_mask_encoder is not None and action_input is not None:
            pick_mask = action_input.get(self.config.pick_mask_key)
            if isinstance(pick_mask, torch.Tensor):
                # Keep mask encoder in fp32 for numerical stability under AMP/bf16.
                with torch.autocast(device_type=backbone_features.device.type, enabled=False):
                    control_features = self.pick_mask_encoder(
                        pick_mask.to(device=backbone_features.device, dtype=torch.float32)
                    )
                control_features = control_features.to(
                    device=backbone_features.device,
                    dtype=backbone_features.dtype,
                )
                det = control_features.detach().float()
                self._last_control_token_abs_mean = float(det.abs().mean().item())
                self._last_control_token_norm = float(det.norm().item())
            elif pick_mask is not None:
                raise ValueError(
                    f"{self.config.pick_mask_key} must be a torch.Tensor, got {type(pick_mask)}"
                )

        if control_features is not None:
            backbone_output["backbone_control_features"] = control_features
        else:
            backbone_output.pop("backbone_control_features", None)
            self._last_control_token_abs_mean = 0.0
            self._last_control_token_norm = 0.0
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output, action_input)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        control_vl_embs = backbone_output.get("backbone_control_features")
        device = vl_embs.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        vl_attn_mask = backbone_output.backbone_attention_mask

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            control_encoder_hidden_states=control_vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()
        output_dict = {
            "loss": loss,
        }
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        backbone_output = self.process_backbone_output(backbone_output, action_input)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        control_vl_embs = backbone_output.get("backbone_control_features")
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                control_encoder_hidden_states=control_vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return BatchFeature(data={"action_pred": actions})

    @torch.no_grad()
    def get_pick_mask_debug_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if not self.config.use_pick_mask_control:
            return metrics

        metrics["pickmask_ctrl_token_abs_mean"] = float(self._last_control_token_abs_mean)
        metrics["pickmask_ctrl_token_norm"] = float(self._last_control_token_norm)

        if self.pick_mask_encoder is not None:
            metrics["pickmask_obj_emb_norm"] = float(
                self.pick_mask_encoder.object_pool.global_object_embedding.detach().float().norm().item()
            )
            metrics["pickmask_empty_emb_norm"] = float(
                self.pick_mask_encoder.object_pool.empty_object_embedding.detach().float().norm().item()
            )
            if self.pick_mask_encoder.local_feature is not None:
                metrics["pickmask_cnn_fc_weight_abs_mean"] = float(
                    self.pick_mask_encoder.local_feature.fc.weight.detach().float().abs().mean().item()
                )

        first_control_block = next(
            (blk for blk in self.model.transformer_blocks if getattr(blk, "attn_control", None) is not None),
            None,
        )
        if first_control_block is not None:
            attn_control = first_control_block.attn_control
            metrics["pickmask_ctrl_k_abs_mean"] = float(attn_control.to_k.weight.detach().float().abs().mean().item())
            metrics["pickmask_ctrl_v_abs_mean"] = float(attn_control.to_v.weight.detach().float().abs().mean().item())
            to_out = attn_control.to_out[0] if isinstance(attn_control.to_out, (list, tuple, nn.ModuleList)) else None
            if to_out is not None and hasattr(to_out, "weight") and to_out.weight is not None:
                metrics["pickmask_ctrl_out_abs_mean"] = float(to_out.weight.detach().float().abs().mean().item())

        return metrics

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
