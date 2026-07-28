"""Stable LatentMoE routing, expert dispatch, and load-balancing components."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.kimi_primitives import SiTUGLU


class _SiTUExpert(nn.Module):
    def __init__(
        self,
        width: int,
        hidden_dim: int,
        beta_gate: float,
        beta_up: float,
        bias: bool,
        init_std: float,
    ):
        super().__init__()
        self.width = width
        self.hidden_dim = hidden_dim
        self.transform = SiTUGLU(
            width,
            hidden_dim,
            beta_gate=beta_gate,
            beta_up=beta_up,
            bias=bias,
            output_bias=bias,
            init_std=init_std,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim < 1 or inputs.shape[-1] != self.width:
            raise ValueError(
                f"expert input must have shape [...,{self.width}]"
            )
        return self.transform(inputs)


class SharedExpert(_SiTUExpert):
    """Always-active full-width SiTU-GLU expert."""


class RoutedExpert(_SiTUExpert):
    """Sparse SiTU-GLU expert operating only in latent width."""
