"""Reusable neural-network primitives shared by Kimi attention implementations."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ShortConvState:
    """Incremental convolution history with shape ``[B, K-1, C]``."""

    buffer: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.buffer, torch.Tensor):
            raise TypeError("buffer must be a torch.Tensor")
        if self.buffer.ndim != 3:
            raise ValueError(
                "ShortConvState.buffer must have shape [B, K-1, C], "
                f"got {tuple(self.buffer.shape)}"
            )

    def clone(self) -> "ShortConvState":
        """Clone storage without detaching it from autograd."""
        return ShortConvState(self.buffer.clone())

