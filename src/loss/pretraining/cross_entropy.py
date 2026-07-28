"""Reusable aligned cross-entropy computation for NTP and SFT."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..common.reduction import reduce_token_losses
from ..common.types import TokenCrossEntropyOutput
from ..common.validation import align_token_inputs, validate_token_ids


def compute_aligned_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    alignment: str,
    ignore_index: int,
    label_smoothing: float,
    zero_valid_policy: str,
    attention_mask: torch.Tensor | None = None,
    loss_mask: torch.Tensor | None = None,
    boundary_mask: torch.Tensor | None = None,
    token_weights: torch.Tensor | None = None,
    return_per_token: bool = False,
) -> TokenCrossEntropyOutput:
    """Compute FP32 CE after one explicit alignment and semantic masking step."""

    aligned_logits, aligned_labels, valid, weights = align_token_inputs(
        logits,
        labels,
        alignment=alignment,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        boundary_mask=boundary_mask,
        token_weights=token_weights,
        ignore_index=ignore_index,
    )
    validate_token_ids(aligned_labels, valid, logits.shape[-1])
    safe_labels = aligned_labels.masked_fill(~valid, 0).long()
    flattened = F.cross_entropy(
        aligned_logits.float().reshape(-1, logits.shape[-1]),
        safe_labels.reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
    )
    per_token = flattened.reshape_as(safe_labels)
    if torch.any(valid & ~torch.isfinite(per_token)):
        raise FloatingPointError("cross-entropy produced non-finite valid values")
    return reduce_token_losses(
        per_token,
        valid,
        weights,
        source=logits,
        zero_valid_policy=zero_valid_policy,
        return_per_token=return_per_token,
    )

