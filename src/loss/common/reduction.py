"""Numerically stable token reductions with explicit global accounting."""

from __future__ import annotations

import torch

from .types import TokenCrossEntropyOutput


def connected_zero(source: torch.Tensor) -> torch.Tensor:
    """Return an FP32 scalar zero that remains connected to ``source``."""

    return source.float().reshape(-1)[:0].sum()


def reduce_token_losses(
    per_token_loss: torch.Tensor,
    valid_mask: torch.Tensor,
    weights: torch.Tensor,
    *,
    source: torch.Tensor,
    zero_valid_policy: str,
    return_per_token: bool,
) -> TokenCrossEntropyOutput:
    """Reduce valid weighted token losses and expose sums for future DDP."""

    effective_weights = weights.float() * valid_mask.float()
    normalizer = effective_weights.sum(dtype=torch.float32)
    num_valid = valid_mask.sum().to(dtype=torch.float32)
    weighted = torch.where(
        valid_mask,
        per_token_loss.float() * weights.float(),
        torch.zeros_like(per_token_loss, dtype=torch.float32),
    )
    loss_sum = weighted.sum(dtype=torch.float32)
    if normalizer.item() == 0:
        if zero_valid_policy == "raise":
            raise ValueError("loss batch contains no valid tokens")
        loss = connected_zero(source)
        loss_sum = loss
    else:
        loss = loss_sum / normalizer
    return TokenCrossEntropyOutput(
        loss=loss,
        loss_sum=loss_sum,
        normalizer=normalizer.detach(),
        num_valid_tokens=num_valid.detach(),
        per_token_nll=per_token_loss.detach() if return_per_token else None,
        per_sample_loss_sum=weighted.sum(dim=-1).detach(),
        per_sample_num_tokens=valid_mask.sum(dim=-1).detach(),
    )
