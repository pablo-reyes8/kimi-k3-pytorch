from __future__ import annotations

import torch
import torch.nn as nn

from .utils import to_2tuple


class VisionPatchEmbedding(nn.Module):
    """Strict, non-overlapping convolutional image patchification."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int | tuple[int, int],
        *,
        bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = to_2tuple(patch_size, "patch_size")
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if images.ndim != 4:
            raise ValueError(
                f"images must have shape [B, C, H, W], got {tuple(images.shape)}"
            )
        if images.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {images.shape[1]}"
            )
        height, width = images.shape[-2:]
        patch_h, patch_w = self.patch_size
        if height % patch_h or width % patch_w:
            raise ValueError(
                f"image size {(height, width)} must be divisible by patch size "
                f"{self.patch_size}"
            )
        features = self.projection(images)
        grid = (features.shape[-2], features.shape[-1])
        tokens = features.flatten(2).transpose(1, 2)
        return tokens, grid

