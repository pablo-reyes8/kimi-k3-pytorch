from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mtp_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [B,T,V]")
    if target_ids.shape != logits.shape[:2]:
        raise ValueError("target_ids shape must match logits [B,T]")
    if valid_mask.shape != target_ids.shape:
        raise ValueError("valid_mask shape must match target_ids")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean")
    if torch.any(valid_mask):
        return F.cross_entropy(logits[valid_mask].float(), target_ids[valid_mask])
    return logits.sum() * 0.0


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
