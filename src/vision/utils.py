"""MoonViT, hierarchical, and Swin vision encoder components."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from src.transformer_modules.rms_norm import RMSNorm


def to_2tuple(value: int | Sequence[int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        pair = (value, value)
    elif len(value) == 2:
        pair = (int(value[0]), int(value[1]))
    else:
        raise ValueError(f"{name} must be an int or a sequence of length 2")
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"{name} values must be > 0, got {pair}")
    return pair


def build_vision_norm(
    norm_type: str, dim: int, eps: float = 1e-6
) -> nn.Module:
    normalized = norm_type.lower()
    if normalized == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    if normalized == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(
        f"norm_type must be 'rmsnorm' or 'layernorm', got {norm_type!r}"
    )


def initialize_vision_module(module: nn.Module, std: float = 0.02) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        if module.elementwise_affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def validate_token_grid(
    tokens: torch.Tensor, grid_size: tuple[int, int], dim: int | None = None
) -> tuple[int, int, int]:
    if tokens.ndim != 3:
        raise ValueError(
            f"tokens must have shape [B, N, D], got {tuple(tokens.shape)}"
        )
    batch, count, width = tokens.shape
    height, grid_width = to_2tuple(grid_size, "grid_size")
    if count != height * grid_width:
        raise ValueError(
            f"token count N={count} does not match grid {grid_size} "
            f"(expected {height * grid_width})"
        )
    if dim is not None and width != dim:
        raise ValueError(f"token width must be {dim}, got {width}")
    return batch, height, grid_width

