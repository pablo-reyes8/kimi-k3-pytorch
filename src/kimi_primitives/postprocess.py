"""Reusable neural-network primitives shared by Kimi attention implementations."""

import torch
import torch.nn as nn

from .head_utils import combine_heads
from .headwise_rmsnorm import HeadwiseRMSNorm
from .output_gate import FullRankOutputGate


class PrimitiveAttentionPostprocess(nn.Module):
    """Compose head-wise normalization, exact head combine, and output gating."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        eps: float = 1e-6,
        norm_affine: bool = True,
        per_head_affine: bool = True,
        gate_bias: bool = False,
        output_bias: bool = False,
    ):
        super().__init__()
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError("num_heads and head_dim must be > 0")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_model = num_heads * head_dim
        self.norm = HeadwiseRMSNorm(
            num_heads,
            head_dim,
            eps=eps,
            elementwise_affine=norm_affine,
            per_head_affine=per_head_affine,
        )
        self.output_gate = FullRankOutputGate(
            self.d_model, bias=gate_bias, output_bias=output_bias
        )

    def forward(
        self,
        head_outputs: torch.Tensor,
        residual_input: torch.Tensor,
        *,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        normalized = self.norm(head_outputs)
        combined = combine_heads(normalized)
        return self.output_gate(
            combined, residual_input, return_gate=return_gate
        )

