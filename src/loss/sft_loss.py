"""High-level assistant-only cross-entropy for supervised trajectories."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common.reduction import connected_zero
from .common.types import SFTLossOutput
from .common.validation import (
    align_token_inputs,
    validate_token_ids,
    validate_zero_policy,
)
from .sft.components import resolve_component_weights


class SFTTrajectoryCrossEntropyLoss(nn.Module):
    """Optimize only model-authored tokens in complex supervised trajectories.

    The required ``assistant_mask`` keeps system/user/tool observations as
    context while allowing reasoning, tool calls, arguments, final answers,
    and end-of-turn tokens to be selected explicitly by the collator.
    """

    def __init__(
        self,
        *,
        ignore_index: int = -100,
        alignment: Literal["causal_shift", "already_aligned"] = "causal_shift",
        reduction: Literal["token_mean", "sequence_mean"] = "token_mean",
        zero_valid_policy: Literal["raise", "connected_zero"] = "raise",
    ) -> None:
        super().__init__()
        if alignment not in ("causal_shift", "already_aligned"):
            raise ValueError("unsupported alignment")
        if reduction not in ("token_mean", "sequence_mean"):
            raise ValueError("unsupported SFT reduction")
        self.ignore_index = ignore_index
        self.alignment = alignment
        self.reduction = reduction
        self.zero_valid_policy = validate_zero_policy(zero_valid_policy)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        assistant_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        component_ids: torch.Tensor | None = None,
        component_weights: Mapping[int, float] | torch.Tensor | None = None,
        boundary_mask: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
        return_per_component: bool = False,
    ) -> SFTLossOutput:
        """Return assistant-target CE using token- or sequence-level reduction."""

        if assistant_mask.shape != labels.shape:
            raise ValueError("assistant_mask must match labels")
        assistant_mask = assistant_mask.to(torch.bool)
        weights = torch.ones_like(labels, dtype=torch.float32)
        if component_ids is not None:
            if component_ids.shape != labels.shape:
                raise ValueError("component_ids must match labels")
            weights = resolve_component_weights(component_ids, component_weights)
            assistant_mask = assistant_mask & component_ids.ne(0)
        elif component_weights is not None:
            raise ValueError("component_weights require component_ids")
        aligned_logits, aligned_labels, valid, aligned_weights = align_token_inputs(
            logits,
            labels,
            alignment=self.alignment,
            attention_mask=attention_mask,
            loss_mask=assistant_mask,
            boundary_mask=boundary_mask,
            token_weights=weights,
            ignore_index=self.ignore_index,
        )
        validate_token_ids(aligned_labels, valid, logits.shape[-1])
        safe_labels = aligned_labels.masked_fill(~valid, 0).long()
        per_token = F.cross_entropy(
            aligned_logits.float().reshape(-1, logits.shape[-1]),
            safe_labels.reshape(-1),
            reduction="none",
        ).reshape_as(aligned_labels)
        if torch.any(valid & ~torch.isfinite(per_token)):
            raise FloatingPointError("SFT cross-entropy produced non-finite values")
        token_weights = aligned_weights * valid.float()
        weighted = torch.where(
            valid,
            per_token * aligned_weights,
            torch.zeros_like(per_token),
        )
        token_count = valid.sum().to(torch.float32)
        per_sample_sum = weighted.sum(-1)
        per_sample_denominator = token_weights.sum(-1)
        if sample_weights is None:
            samples = torch.ones(
                logits.shape[0], dtype=torch.float32, device=logits.device
            )
        else:
            if sample_weights.shape != (logits.shape[0],):
                raise ValueError("sample_weights must have shape [B]")
            samples = sample_weights.float().to(logits.device)
            if not torch.isfinite(samples).all() or torch.any(samples < 0):
                raise ValueError("sample_weights must be finite and non-negative")
        if self.reduction == "token_mean":
            sample_matrix = samples[:, None]
            loss_sum = (weighted * sample_matrix).sum()
            normalizer = (token_weights * sample_matrix).sum()
        else:
            active = per_sample_denominator > 0
            sequence_means = per_sample_sum / per_sample_denominator.clamp_min(1)
            loss_sum = (sequence_means * samples * active).sum()
            normalizer = (samples * active).sum()
        if normalizer.item() == 0:
            if self.zero_valid_policy == "raise":
                raise ValueError("SFT batch contains no assistant targets")
            loss = connected_zero(logits)
            loss_sum = loss
        else:
            loss = loss_sum / normalizer
        component_sums = None
        component_counts = None
        if return_per_component:
            if component_ids is None:
                raise ValueError("component diagnostics require component_ids")
            aligned_components = (
                component_ids[:, 1:]
                if self.alignment == "causal_shift"
                else component_ids
            )
            component_sums, component_counts = {}, {}
            for identifier in torch.unique(aligned_components[valid]).tolist():
                selected = valid & aligned_components.eq(identifier)
                component_sums[int(identifier)] = per_token[selected].sum().detach()
                component_counts[int(identifier)] = selected.sum().detach()
        return SFTLossOutput(
            loss=loss,
            loss_sum=loss_sum,
            normalizer=normalizer.detach(),
            num_valid_tokens=token_count.detach(),
            per_token_nll=None,
            per_sample_loss_sum=per_sample_sum.detach(),
            per_sample_num_tokens=valid.sum(-1).detach(),
            per_component_loss_sum=component_sums,
            per_component_num_tokens=component_counts,
            reduction=self.reduction,
        )


__all__ = ["SFTTrajectoryCrossEntropyLoss"]
