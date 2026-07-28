"""MoonViT, hierarchical, and Swin vision encoder components."""

from dataclasses import dataclass

import torch


@dataclass
class VisionEncoderOutput:
    """Vision token sequence, grid metadata, mask, and optional attentions."""

    last_hidden_state: torch.Tensor
    grid_size: tuple[int, int]
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


@dataclass
class PixelShuffleOutput:
    """Spatially packed visual tokens and their reduced grid metadata."""

    last_hidden_state: torch.Tensor
    grid_size: tuple[int, int]
