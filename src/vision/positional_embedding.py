"""MoonViT, hierarchical, and Swin vision encoder components."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import to_2tuple


class LearnedAbsolutePositionEmbedding(nn.Module):
    """Learned 2D positions with bicubic interpolation to the runtime grid."""

    def __init__(
        self,
        base_grid_size: int | tuple[int, int],
        dim: int,
        *,
        use_cls_token: bool = False,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be > 0")
        self.base_grid_size = to_2tuple(base_grid_size, "base_grid_size")
        self.dim = dim
        self.use_cls_token = use_cls_token
        count = self.base_grid_size[0] * self.base_grid_size[1]
        self.patch_positions = nn.Parameter(torch.empty(1, count, dim))
        self.cls_position = (
            nn.Parameter(torch.empty(1, 1, dim)) if use_cls_token else None
        )
        nn.init.trunc_normal_(self.patch_positions, std=0.02)
        if self.cls_position is not None:
            nn.init.trunc_normal_(self.cls_position, std=0.02)

    def forward(
        self,
        grid_size: tuple[int, int],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        height, width = to_2tuple(grid_size, "grid_size")
        base_h, base_w = self.base_grid_size
        positions = self.patch_positions
        if (height, width) != self.base_grid_size:
            positions = positions.reshape(1, base_h, base_w, self.dim)
            positions = positions.permute(0, 3, 1, 2)
            positions = F.interpolate(
                positions.float(),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            ).to(self.patch_positions.dtype)
            positions = positions.permute(0, 2, 3, 1).reshape(
                1, height * width, self.dim
            )
        if self.cls_position is not None:
            positions = torch.cat((self.cls_position, positions), dim=1)
        return positions.to(dtype=dtype, device=device)

