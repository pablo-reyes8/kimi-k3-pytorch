"""Reusable neural-network primitives shared by Kimi attention implementations."""

from __future__ import annotations

import torch
import torch.nn as nn


def softcap(x: torch.Tensor, beta: float) -> torch.Tensor:
    """Smoothly bound ``x`` as ``beta * tanh(x / beta)``."""
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")
    return torch.tanh(x / beta) * beta


def situ_glu_activation(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta_gate: float = 4.0,
    beta_up: float = 25.0,
) -> torch.Tensor:
    """Apply ``softcap(g,b1) * sigmoid(g) * softcap(u,b2)``."""
    if gate.shape != up.shape:
        raise ValueError(
            f"gate and up must have identical shapes, got {gate.shape} and {up.shape}"
        )
    if beta_gate <= 0 or beta_up <= 0:
        raise ValueError("beta_gate and beta_up must be > 0")
    return (
        softcap(gate, beta_gate)
        * torch.sigmoid(gate)
        * softcap(up, beta_up)
    )


class SiTUGLU(nn.Module):
    """Kimi K3 SiTU-GLU feed-forward transformation.

    ``down(softcap(gate(x),4) * sigmoid(gate(x)) * softcap(up(x),25))``.
    The module contains no normalization or residual connection.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        beta_gate: float = 4.0,
        beta_up: float = 25.0,
        bias: bool = False,
        output_bias: bool | None = None,
        init_std: float = 0.02,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_model <= 0 or hidden_dim <= 0:
            raise ValueError("d_model and hidden_dim must be > 0")
        if beta_gate <= 0 or beta_up <= 0:
            raise ValueError("beta_gate and beta_up must be > 0")
        if init_std <= 0:
            raise ValueError("init_std must be > 0")
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.beta_gate = float(beta_gate)
        self.beta_up = float(beta_up)
        output_bias = bias if output_bias is None else output_bias
        factory_kwargs = {"device": device, "dtype": dtype}
        self.gate_proj = nn.Linear(
            d_model, hidden_dim, bias=bias, **factory_kwargs
        )
        self.up_proj = nn.Linear(
            d_model, hidden_dim, bias=bias, **factory_kwargs
        )
        self.down_proj = nn.Linear(
            hidden_dim, d_model, bias=output_bias, **factory_kwargs
        )
        self.init_std = float(init_std)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in (self.gate_proj, self.up_proj, self.down_proj):
            nn.init.normal_(projection.weight, mean=0.0, std=self.init_std)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"x must have shape [...,{self.d_model}], got {tuple(x.shape)}"
            )
        hidden = situ_glu_activation(
            self.gate_proj(x),
            self.up_proj(x),
            self.beta_gate,
            self.beta_up,
        )
        return self.down_proj(hidden)

