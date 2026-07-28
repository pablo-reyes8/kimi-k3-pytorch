"""High-level next-token cross-entropy for textual and multimodal pretraining."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .common.types import TokenCrossEntropyOutput
from .common.validation import validate_zero_policy
from .pretraining.cross_entropy import compute_aligned_cross_entropy


class NextTokenCrossEntropyLoss(nn.Module):
    """Compute canonical autoregressive CE with explicit alignment and masks.

    Visual inputs need no separate contrastive objective: placeholder targets
    are masked by the collator while later text targets propagate gradients
    through the shared multimodal forward into MoonViT and its projector.
    """

    def __init__(
        self,
        *,
        ignore_index: int = -100,
        alignment: Literal["causal_shift", "already_aligned"] = "causal_shift",
        label_smoothing: float = 0.0,
        zero_valid_policy: Literal["raise", "connected_zero"] = "raise",
    ) -> None:
        super().__init__()
        if alignment not in ("causal_shift", "already_aligned"):
            raise ValueError("unsupported alignment")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.ignore_index = ignore_index
        self.alignment = alignment
        self.label_smoothing = label_smoothing
        self.zero_valid_policy = validate_zero_policy(zero_valid_policy)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        boundary_mask: torch.Tensor | None = None,
        token_weights: torch.Tensor | None = None,
        return_per_token: bool = False,
    ) -> TokenCrossEntropyOutput:
        """Return FP32 token-weighted CE and its unreduced accounting."""

        return compute_aligned_cross_entropy(
            logits,
            labels,
            alignment=self.alignment,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            zero_valid_policy=self.zero_valid_policy,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            boundary_mask=boundary_mask,
            token_weights=token_weights,
            return_per_token=return_per_token,
        )


__all__ = ["NextTokenCrossEntropyLoss"]

