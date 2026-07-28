"""Baseline transformer modules retained for controls and training infrastructure."""

# ============================================================
# Dense SwiGLU baseline (Kimi's SiTU-GLU is intentionally not implemented yet)
# ============================================================

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utilities
# ============================================================

def round_up_to_multiple(value: int, multiple_of: int) -> int:
    """
    Round value up to the nearest multiple of multiple_of.
    """
    if multiple_of <= 0:
        raise ValueError(f"multiple_of must be > 0, got {multiple_of}")

    return ((value + multiple_of - 1) // multiple_of) * multiple_of


# ============================================================
# CONFIG
# ============================================================

@dataclass
class SwiGLUMLPConfig:
    d_model: int

    hidden_dim: Optional[int] = None
    expansion_factor: float = 4.0
    multiple_of: int = 1

    dropout: float = 0.0
    use_bias: bool = False
    init_std: float = 0.02

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")

        if self.hidden_dim is not None and self.hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be > 0 when provided, got {self.hidden_dim}"
            )

        if self.expansion_factor <= 0:
            raise ValueError(
                f"expansion_factor must be > 0, got {self.expansion_factor}"
            )

        if self.multiple_of <= 0:
            raise ValueError(f"multiple_of must be > 0, got {self.multiple_of}")

        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(
                f"dropout must satisfy 0 <= dropout < 1, got {self.dropout}"
            )

        if self.init_std <= 0:
            raise ValueError(f"init_std must be > 0, got {self.init_std}")

    def resolved_hidden_dim(self) -> int:
        """
        Resolve hidden_dim from explicit hidden_dim or expansion_factor.

        If hidden_dim is None:
            hidden_dim = int(expansion_factor * d_model)

        Then round up to multiple_of.
        """
        self.validate()

        if self.hidden_dim is None:
            hidden_dim = int(self.expansion_factor * self.d_model)
        else:
            hidden_dim = self.hidden_dim

        hidden_dim = round_up_to_multiple(hidden_dim, self.multiple_of)

        return hidden_dim


# ============================================================
# SwiGLU MLP
# ============================================================

class SwiGLUFeedForward(nn.Module):
    """
    Dense feed-forward baseline using SwiGLU.

    Input:
        x: [B, T, d_model]

    Forward:
        gate = gate_proj(x)
        up   = up_proj(x)
        hidden = silu(gate) * up
        out = down_proj(hidden)
        out = dropout(out)

    Output:
        out: [B, T, d_model]

    This module intentionally does NOT include:
        - residual connection
        - RMSNorm
        - MoE routing
        - shared experts
        - top-k experts
        - attention
    """

    def __init__(self, config: SwiGLUMLPConfig):
        super().__init__()

        config.validate()

        self.config = config
        self.d_model = config.d_model
        self.hidden_dim = config.resolved_hidden_dim()

        self.gate_proj = nn.Linear(
            self.d_model,
            self.hidden_dim,
            bias=config.use_bias)

        self.up_proj = nn.Linear(
            self.d_model,
            self.hidden_dim,
            bias=config.use_bias)

        self.down_proj = nn.Linear(
            self.hidden_dim,
            self.d_model,
            bias=config.use_bias)

        self.dropout = nn.Dropout(config.dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize all linear weights with Normal(0, init_std).
        Biases, if present, are initialized to zero.
        """
        for module in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.init_std,)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                Hidden states [B, T, d_model].

        Returns:
            out:
                MLP output [B, T, d_model].
        """

        if x.dim() != 3:
            raise ValueError(
                f"SwiGLUFeedForward expects x with shape [B, T, d_model], "
                f"got {tuple(x.shape)}")

        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected x.shape[-1] == d_model={self.d_model}, "
                f"got {x.shape[-1]}")

        gate = self.gate_proj(x)
        up = self.up_proj(x)

        hidden = F.silu(gate) * up

        out = self.down_proj(hidden)
        out = self.dropout(out)

        return out


SwiGLUMLP = SwiGLUFeedForward
