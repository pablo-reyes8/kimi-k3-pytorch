"""High-level CE over phase-9 multi-token prediction targets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common.reduction import connected_zero
from .common.types import MTPLossOutput
from .common.validation import validate_token_ids, validate_zero_policy
from .pretraining.mtp_components import canonicalize_mtp_inputs


class MultiTokenPredictionLoss(nn.Module):
    """Reduce already-aligned MTP targets without reconstructing future shifts."""

    def __init__(
        self,
        *,
        ignore_index: int = -100,
        depth_weights: Sequence[float] | None = None,
        zero_valid_policy: Literal[
            "raise", "connected_zero"
        ] = "connected_zero",
    ) -> None:
        super().__init__()
        if depth_weights is not None and (
            not depth_weights
            or any(not torch.isfinite(torch.tensor(x)) or x < 0 for x in depth_weights)
        ):
            raise ValueError("depth_weights must be finite and non-negative")
        self.ignore_index = ignore_index
        self.depth_weights = (
            None if depth_weights is None else tuple(float(x) for x in depth_weights)
        )
        self.zero_valid_policy = validate_zero_policy(zero_valid_policy)

    def forward(
        self,
        mtp_logits: torch.Tensor,
        mtp_labels: torch.Tensor,
        *,
        mtp_loss_mask: torch.Tensor | None = None,
        future_offsets: torch.Tensor | Sequence[int] | None = None,
        return_per_depth: bool = False,
    ) -> MTPLossOutput:
        """Compute an FP32 weighted mean over all valid future targets."""

        logits, labels, mask, offsets, _ = canonicalize_mtp_inputs(
            mtp_logits, mtp_labels, mtp_loss_mask, future_offsets
        )
        valid = mask & labels.ne(self.ignore_index)
        validate_token_ids(labels, valid, logits.shape[-1], name="mtp_labels")
        safe_labels = labels.masked_fill(~valid, 0).long()
        per_token = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            safe_labels.reshape(-1),
            reduction="none",
        ).reshape_as(labels)
        if torch.any(valid & ~torch.isfinite(per_token)):
            raise FloatingPointError("MTP cross-entropy produced non-finite values")
        depth = logits.shape[1]
        if self.depth_weights is not None and len(self.depth_weights) != depth:
            raise ValueError("depth_weights must match the number of MTP depths")
        depth_weights = torch.tensor(
            self.depth_weights or (1.0,) * depth,
            dtype=torch.float32,
            device=logits.device,
        )
        weighted_mask = valid.float() * depth_weights.view(1, depth, 1)
        weighted_losses = torch.where(
            valid,
            per_token * depth_weights.view(1, depth, 1),
            torch.zeros_like(per_token),
        )
        per_depth_sum = weighted_losses.sum(dim=(0, 2))
        per_depth_normalizer = weighted_mask.sum(dim=(0, 2))
        loss_sum = per_depth_sum.sum()
        normalizer = per_depth_normalizer.sum()
        num_valid = valid.sum().to(torch.float32)
        if normalizer.item() == 0:
            if self.zero_valid_policy == "raise":
                raise ValueError("MTP batch contains no valid future targets")
            loss = connected_zero(mtp_logits)
            loss_sum = loss
        else:
            loss = loss_sum / normalizer
        return MTPLossOutput(
            loss=loss,
            loss_sum=loss_sum,
            normalizer=normalizer.detach(),
            num_valid_tokens=num_valid.detach(),
            per_token_nll=per_token.detach() if return_per_depth else None,
            per_sample_loss_sum=weighted_losses.sum((1, 2)).detach(),
            per_sample_num_tokens=valid.sum((1, 2)).detach(),
            per_depth_loss_sum=per_depth_sum.detach() if return_per_depth else None,
            per_depth_normalizer=(
                per_depth_normalizer.detach() if return_per_depth else None
            ),
            future_offsets=offsets,
        )


__all__ = ["MultiTokenPredictionLoss"]
