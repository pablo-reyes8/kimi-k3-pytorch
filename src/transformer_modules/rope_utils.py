# ============================================================
# Baseline-only RoPE utilities
# Rotary Positional Embedding — standalone utility
# ============================================================

from typing import Optional

import torch


# ============================================================
# rotate_half
# ============================================================

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate last dimension by splitting it into two halves.

    Input:
        x: [..., rotary_dim]

    Operation:
        x1 = x[..., :rotary_dim // 2]
        x2 = x[..., rotary_dim // 2:]
        return concat(-x2, x1, dim=-1)

    Output:
        rotated: [..., rotary_dim]

    Preserves:
        - shape
        - dtype
        - device
    """

    rotary_dim = x.shape[-1]

    if rotary_dim % 2 != 0:
        raise ValueError(
            f"rotate_half requires an even last dimension, got {rotary_dim}"
        )

    half = rotary_dim // 2

    x1 = x[..., :half]
    x2 = x[..., half:]

    return torch.cat((-x2, x1), dim=-1)




