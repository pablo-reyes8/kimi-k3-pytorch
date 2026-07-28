"""Multi-token prediction components used as an optional KimiK3 output head."""

from __future__ import annotations

import torch
from src.loss import MultiTokenPredictionLoss


def masked_mtp_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    return MultiTokenPredictionLoss(
        zero_valid_policy="connected_zero"
    )(
        logits,
        target_ids,
        mtp_loss_mask=valid_mask,
    ).loss


def combine_ntp_mtp_losses(
    ntp_loss: torch.Tensor,
    mtp_loss: torch.Tensor | None,
    loss_weight: float,
) -> torch.Tensor:
    if loss_weight < 0:
        raise ValueError("loss_weight must be >= 0")
    if mtp_loss is None:
        return ntp_loss
    return ntp_loss + loss_weight * mtp_loss
