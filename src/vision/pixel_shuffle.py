"""MoonViT, hierarchical, and Swin vision encoder components."""

import torch
import torch.nn as nn

from .outputs import PixelShuffleOutput
from .utils import validate_token_grid


class SpatialTokenPixelShuffle(nn.Module):
    """Pack 2x2 spatial tokens in TL, TR, BL, BR channel order."""

    def forward(
        self, tokens: torch.Tensor, grid_size: tuple[int, int]
    ) -> PixelShuffleOutput:
        batch, height, width = validate_token_grid(tokens, grid_size)
        if height % 2 or width % 2:
            raise ValueError(
                f"pixel packing requires an even grid, got {(height, width)}"
            )
        dim = tokens.shape[-1]
        spatial = tokens.reshape(batch, height, width, dim)
        packed = torch.cat(
            (
                spatial[:, 0::2, 0::2],
                spatial[:, 0::2, 1::2],
                spatial[:, 1::2, 0::2],
                spatial[:, 1::2, 1::2],
            ),
            dim=-1,
        )
        new_grid = (height // 2, width // 2)
        return PixelShuffleOutput(
            last_hidden_state=packed.reshape(batch, -1, 4 * dim),
            grid_size=new_grid,
        )

