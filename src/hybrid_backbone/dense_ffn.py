from __future__ import annotations

import torch
import torch.nn as nn

from src.kimi_primitives import SiTUGLU


class DenseKimiFFN(nn.Module):
    """Dense SiTU-GLU channel-mixing ablation."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
        init_std: float = 0.02,
    ):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= p < 1")
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.transform = SiTUGLU(
            d_model,
            hidden_dim,
            bias=bias,
            output_bias=bias,
            init_std=init_std,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            hidden_states.ndim != 3
            or hidden_states.shape[-1] != self.d_model
        ):
            raise ValueError(
                f"hidden_states must have shape [B,T,{self.d_model}]"
            )
        return self.dropout(self.transform(hidden_states))
