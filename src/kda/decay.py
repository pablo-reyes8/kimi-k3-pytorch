"""Kimi Delta Attention operators, projections, states, and diagnostics."""

import torch
import torch.nn as nn


class LowerBoundedDecay(nn.Module):
    """Kimi K3 decay ``g = g_min * sigmoid(exp(A) * z)``."""

    def __init__(
        self,
        num_heads: int,
        g_min: float = -5.0,
        *,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if g_min >= 0:
            raise ValueError("g_min must be negative")
        self.num_heads = num_heads
        self.g_min = float(g_min)
        self.A_log = nn.Parameter(
            torch.zeros(num_heads, dtype=torch.float32, device=device)
        )

    def forward(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if logits.ndim != 4 or logits.shape[2] != self.num_heads:
            raise ValueError(
                f"logits must have shape [B,T,{self.num_heads},K], "
                f"got {tuple(logits.shape)}"
            )
        scale = self.A_log.exp().to(dtype=logits.dtype)[None, None, :, None]
        probability = torch.sigmoid(scale * logits)
        # Preserve the report's open intervals under finite-precision sigmoid
        # saturation (large finite logits can otherwise become exactly 0/1).
        epsilon = torch.finfo(probability.dtype).eps
        probability = probability.clamp(epsilon, 1.0 - epsilon)
        log_decay = self.g_min * probability
        return log_decay, torch.exp(log_decay)
