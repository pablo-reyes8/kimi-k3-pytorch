"""MoonViT, hierarchical, and Swin vision encoder components."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .outputs import VisionEncoderOutput
from .patch_embedding import VisionPatchEmbedding
from .positional_embedding import LearnedAbsolutePositionEmbedding
from .utils import (
    build_vision_norm,
    initialize_vision_module,
    to_2tuple,
    validate_token_grid,
)
from .vision_block import VisionTransformerBlock


class HierarchicalTokenPool(nn.Module):
    """PiT-style depthwise spatial pooling followed by channel projection."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be > 0")
        self.input_dim = input_dim
        self.depthwise = nn.Conv2d(
            input_dim,
            input_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=input_dim,
            bias=bias,
        )
        self.projection = nn.Conv2d(input_dim, output_dim, 1, bias=bias)
        self.norm = build_vision_norm(norm_type, output_dim, norm_eps)

    def forward(
        self,
        tokens: torch.Tensor,
        grid_size: tuple[int, int],
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[int, int], torch.Tensor | None]:
        batch, height, width = validate_token_grid(
            tokens, grid_size, self.input_dim
        )
        features = tokens.reshape(batch, height, width, self.input_dim)
        features = features.permute(0, 3, 1, 2)
        features = self.projection(self.depthwise(features))
        new_grid = (features.shape[-2], features.shape[-1])
        output = features.flatten(2).transpose(1, 2)
        output = self.norm(output)
        new_mask = None
        if padding_mask is not None:
            if padding_mask.shape != (batch, height * width):
                raise ValueError("padding_mask does not match the token grid")
            mask_map = padding_mask.reshape(batch, 1, height, width).float()
            new_mask = F.max_pool2d(
                mask_map, kernel_size=3, stride=2, padding=1
            ).flatten(1).bool()
        return output, new_grid, new_mask


@dataclass(frozen=True)
class HierarchicalVisionConfig:
    """Configuration for the hierarchical MoonViT ablation."""

    image_size: int | tuple[int, int] = 224
    patch_size: int | tuple[int, int] = 14
    in_channels: int = 3
    embed_dims: tuple[int, ...] = (96, 192, 384)
    depths: tuple[int, ...] = (2, 2, 4)
    num_heads: tuple[int, ...] = (6, 12, 12)
    mlp_ratio: float = 4.0
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-6
    position_embedding_type: str = "learned_absolute"
    patch_bias: bool = False
    qkv_bias: bool = False
    proj_bias: bool = False
    mlp_bias: bool = False
    pool_bias: bool = False
    dropout: float = 0.0
    attention_dropout: float = 0.0
    drop_path_rate: float = 0.0
    initializer_std: float = 0.02

    def __post_init__(self):
        image = to_2tuple(self.image_size, "image_size")
        patch = to_2tuple(self.patch_size, "patch_size")
        if image[0] % patch[0] or image[1] % patch[1]:
            raise ValueError("configured image_size must be divisible by patch_size")
        stages = len(self.embed_dims)
        if stages == 0 or len(self.depths) != stages or len(self.num_heads) != stages:
            raise ValueError("embed_dims, depths, and num_heads must have equal nonzero length")
        if self.in_channels <= 0 or any(value <= 0 for value in self.embed_dims):
            raise ValueError("channels and embedding dimensions must be > 0")
        if any(value <= 0 for value in self.depths + self.num_heads):
            raise ValueError("depths and num_heads must be > 0")
        if any(dim % heads for dim, heads in zip(self.embed_dims, self.num_heads)):
            raise ValueError("every embed_dim must be divisible by its num_heads")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be > 0")
        if self.position_embedding_type not in ("none", "learned_absolute"):
            raise ValueError("unsupported position_embedding_type")
        for name in ("dropout", "attention_dropout", "drop_path_rate"):
            value = getattr(self, name)
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")


class HierarchicalMoonViTEncoder(nn.Module):
    """Global-attention MoonViT ablation with PiT-style stage pooling."""

    def __init__(self, config: HierarchicalVisionConfig):
        super().__init__()
        self.config = config
        image = to_2tuple(config.image_size, "image_size")
        patch = to_2tuple(config.patch_size, "patch_size")
        self.patch_embedding = VisionPatchEmbedding(
            config.in_channels,
            config.embed_dims[0],
            patch,
            bias=config.patch_bias,
        )
        grids = []
        current = (image[0] // patch[0], image[1] // patch[1])
        for stage_index in range(len(config.embed_dims)):
            grids.append(current)
            if stage_index < len(config.embed_dims) - 1:
                current = ((current[0] + 1) // 2, (current[1] + 1) // 2)
        self.position_embeddings = nn.ModuleList(
            LearnedAbsolutePositionEmbedding(grid, dim)
            for grid, dim in zip(grids, config.embed_dims)
        ) if config.position_embedding_type == "learned_absolute" else nn.ModuleList()

        total_depth = sum(config.depths)
        rates = torch.linspace(0, config.drop_path_rate, total_depth).tolist()
        cursor = 0
        self.stages = nn.ModuleList()
        for dim, depth, heads in zip(
            config.embed_dims, config.depths, config.num_heads
        ):
            blocks = nn.ModuleList(
                VisionTransformerBlock(
                    dim,
                    heads,
                    mlp_ratio=config.mlp_ratio,
                    norm_type=config.norm_type,
                    norm_eps=config.norm_eps,
                    qkv_bias=config.qkv_bias,
                    proj_bias=config.proj_bias,
                    mlp_bias=config.mlp_bias,
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout,
                    drop_path=rates[cursor + index],
                )
                for index in range(depth)
            )
            cursor += depth
            self.stages.append(blocks)
        self.pools = nn.ModuleList(
            HierarchicalTokenPool(
                config.embed_dims[index],
                config.embed_dims[index + 1],
                norm_type=config.norm_type,
                norm_eps=config.norm_eps,
                bias=config.pool_bias,
            )
            for index in range(len(config.embed_dims) - 1)
        )
        self.final_norm = build_vision_norm(
            config.norm_type, config.embed_dims[-1], config.norm_eps
        )
        self.apply(
            lambda module: initialize_vision_module(module, config.initializer_std)
        )

    def forward(
        self,
        images: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> VisionEncoderOutput:
        x, grid = self.patch_embedding(images)
        if padding_mask is not None:
            if padding_mask.shape != x.shape[:2] or padding_mask.dtype != torch.bool:
                raise ValueError("padding_mask must be boolean and match patch tokens")
        hidden_states = [] if output_hidden_states else None
        attentions = [] if output_attentions else None
        for stage_index, blocks in enumerate(self.stages):
            if self.position_embeddings:
                x = x + self.position_embeddings[stage_index](
                    grid, dtype=x.dtype, device=x.device
                )
            if hidden_states is not None:
                hidden_states.append(x)
            for block in blocks:
                x, weights = block(
                    x, padding_mask, output_attentions=output_attentions
                )
                if hidden_states is not None:
                    hidden_states.append(x)
                if attentions is not None:
                    attentions.append(weights)
            if stage_index < len(self.pools):
                x, grid, padding_mask = self.pools[stage_index](
                    x, grid, padding_mask
                )
        x = self.final_norm(x)
        if hidden_states is not None:
            hidden_states[-1] = x
        return VisionEncoderOutput(
            last_hidden_state=x,
            grid_size=grid,
            hidden_states=tuple(hidden_states) if hidden_states is not None else None,
            attentions=tuple(attentions) if attentions is not None else None,
        )
