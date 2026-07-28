"""Composition of canonical NTP and configurable auxiliary MTP objectives."""

from __future__ import annotations

import torch.nn as nn

from ..common.types import KimiPretrainingLossOutput
from ..multi_token_prediction_loss import MultiTokenPredictionLoss
from ..next_token_loss import NextTokenCrossEntropyLoss


class KimiPretrainingLoss(nn.Module):
    """Combine NTP and MTP without introducing any MoE auxiliary loss."""

    def __init__(
        self,
        *,
        lambda_mtp: float = 0.0,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if lambda_mtp < 0:
            raise ValueError("lambda_mtp must be >= 0")
        self.lambda_mtp = float(lambda_mtp)
        self.ntp_loss = NextTokenCrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        self.mtp_loss = MultiTokenPredictionLoss(ignore_index=ignore_index)

    def forward(
        self,
        *,
        logits,
        labels,
        attention_mask=None,
        loss_mask=None,
        boundary_mask=None,
        token_weights=None,
        mtp_logits=None,
        mtp_labels=None,
        mtp_loss_mask=None,
        future_offsets=None,
        moe_diagnostics=None,
    ) -> KimiPretrainingLossOutput:
        """Return ``NTP + lambda_mtp * MTP`` with no router loss term."""

        ntp = self.ntp_loss(
            logits,
            labels,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            boundary_mask=boundary_mask,
            token_weights=token_weights,
        )
        supplied = (mtp_logits is not None, mtp_labels is not None)
        if any(supplied) and not all(supplied):
            raise ValueError("mtp_logits and mtp_labels must be supplied together")
        mtp = (
            self.mtp_loss(
                mtp_logits,
                mtp_labels,
                mtp_loss_mask=mtp_loss_mask,
                future_offsets=future_offsets,
            )
            if all(supplied)
            else None
        )
        total = ntp.loss if mtp is None else ntp.loss + self.lambda_mtp * mtp.loss
        return KimiPretrainingLossOutput(
            loss=total,
            ntp=ntp,
            mtp=mtp,
            lambda_mtp=self.lambda_mtp,
            moe_diagnostics=moe_diagnostics,
        )
