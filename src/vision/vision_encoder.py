"""MoonViT, hierarchical, and Swin vision encoder components."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .outputs import VisionEncoderOutput
from .patch_embedding import VisionPatchEmbedding
from .positional_embedding import LearnedAbsolutePositionEmbedding
from .utils import build_vision_norm, initialize_vision_module, to_2tuple
from .vision_block import VisionTransformerBlock


@dataclass(frozen=True)
class VisionEncoderConfig:
    """Configuration for the standard global-attention MoonViT encoder."""

    image_size: int | tuple[int, int] = 224
    patch_size: int | tuple[int, int] = 14
    in_channels: int = 3
    embed_dim: int = 192
    depth: int = 6
    num_heads: int = 12
    mlp_ratio: float = 4.0
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-6
    position_embedding_type: str = "learned_absolute"
    use_cls_token: bool = False
    patch_bias: bool = False
    qkv_bias: bool = False
    proj_bias: bool = False
    mlp_bias: bool = False
    dropout: float = 0.0
    attention_dropout: float = 0.0
    drop_path_rate: float = 0.0
    initializer_std: float = 0.02

    def __post_init__(self):
        image = to_2tuple(self.image_size, "image_size")
        patch = to_2tuple(self.patch_size, "patch_size")
        if image[0] % patch[0] or image[1] % patch[1]:
            raise ValueError("configured image_size must be divisible by patch_size")
        if self.in_channels <= 0 or self.embed_dim <= 0:
            raise ValueError("in_channels and embed_dim must be > 0")
        if self.depth <= 0 or self.num_heads <= 0:
            raise ValueError("depth and num_heads must be > 0")
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be > 0")
        if self.position_embedding_type not in ("none", "learned_absolute"):
            raise ValueError(
                "position_embedding_type must be 'none' or 'learned_absolute'"
            )
        for name in ("dropout", "attention_dropout", "drop_path_rate"):
            value = getattr(self, name)
            upper_inclusive = name == "drop_path_rate"
            valid = 0 <= value <= 1 if upper_inclusive else 0 <= value < 1
            if not valid:
                raise ValueError(f"{name} has invalid probability {value}")
        if self.drop_path_rate >= 1:
            raise ValueError("drop_path_rate must be < 1")
        if self.initializer_std <= 0:
            raise ValueError("initializer_std must be > 0")


class MoonViTEncoder(nn.Module):
    """Flat global MoonViT proxy; it intentionally has no classifier head."""

    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.config = config
        image = to_2tuple(config.image_size, "image_size")
        patch = to_2tuple(config.patch_size, "patch_size")
        self.patch_embedding = VisionPatchEmbedding(
            config.in_channels,
            config.embed_dim,
            patch,
            bias=config.patch_bias,
        )
        self.cls_token = (
            nn.Parameter(torch.empty(1, 1, config.embed_dim))
            if config.use_cls_token
            else None
        )
        base_grid = (image[0] // patch[0], image[1] // patch[1])
        self.position_embedding = (
            LearnedAbsolutePositionEmbedding(
                base_grid, config.embed_dim, use_cls_token=config.use_cls_token
            )
            if config.position_embedding_type == "learned_absolute"
            else None
        )
        rates = torch.linspace(0, config.drop_path_rate, config.depth).tolist()
        self.blocks = nn.ModuleList(
            VisionTransformerBlock(
                config.embed_dim,
                config.num_heads,
                mlp_ratio=config.mlp_ratio,
                norm_type=config.norm_type,
                norm_eps=config.norm_eps,
                qkv_bias=config.qkv_bias,
                proj_bias=config.proj_bias,
                mlp_bias=config.mlp_bias,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                drop_path=rates[index],
            )
            for index in range(config.depth)
        )
        self.final_norm = build_vision_norm(
            config.norm_type, config.embed_dim, config.norm_eps
        )
        self.apply(
            lambda module: initialize_vision_module(module, config.initializer_std)
        )
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=config.initializer_std)

    def forward(
        self,
        images: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> VisionEncoderOutput:
        x, grid = self.patch_embedding(images)
        batch, patch_count, _ = x.shape
        if padding_mask is not None:
            if padding_mask.shape != (batch, patch_count):
                raise ValueError(
                    f"padding_mask must have shape {(batch, patch_count)}, "
                    f"got {tuple(padding_mask.shape)}"
                )
            if padding_mask.dtype != torch.bool:
                raise TypeError("padding_mask must be boolean")
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(batch, -1, -1), x), dim=1)
            if padding_mask is not None:
                cls_mask = torch.ones(
                    batch, 1, dtype=torch.bool, device=padding_mask.device
                )
                padding_mask = torch.cat((cls_mask, padding_mask), dim=1)
        if self.position_embedding is not None:
            x = x + self.position_embedding(
                grid, dtype=x.dtype, device=x.device
            )

        hidden_states = [x] if output_hidden_states else None
        attentions = [] if output_attentions else None
        for block in self.blocks:
            x, weights = block(
                x,
                padding_mask,
                output_attentions=output_attentions,
            )
            if hidden_states is not None:
                hidden_states.append(x)
            if attentions is not None:
                attentions.append(weights)
        x = self.final_norm(x)
        if hidden_states is not None:
            hidden_states[-1] = x
        return VisionEncoderOutput(
            last_hidden_state=x,
            grid_size=grid,
            hidden_states=tuple(hidden_states) if hidden_states is not None else None,
            attentions=tuple(attentions) if attentions is not None else None,
        )


VisionEncoder = MoonViTEncoder
