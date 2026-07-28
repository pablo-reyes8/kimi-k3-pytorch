"""Canonical NTP/MTP convergence metrics."""

from __future__ import annotations

import math


def compute_loss_metrics(
    *,
    total_loss: float,
    ntp_loss: float,
    mtp_loss: float | None,
    ntp_tokens: int | float,
    mtp_tokens: int | float,
    lambda_mtp: float = 0.0,
) -> dict[str, float]:
    metrics = {
        "train/loss_total": float(total_loss),
        "train/loss_ntp": float(ntp_loss),
        "train/perplexity_ntp_clipped": math.exp(min(float(ntp_loss), 20.0)),
        "train/valid_ntp_tokens": float(ntp_tokens),
        "train/valid_mtp_tokens": float(mtp_tokens),
        "mtp/loss_weight": float(lambda_mtp),
    }
    if mtp_loss is not None and math.isfinite(float(mtp_loss)):
        metrics.update(
            {
                "train/loss_mtp": float(mtp_loss),
                "mtp/loss": float(mtp_loss),
                "mtp/perplexity_clipped": math.exp(
                    min(float(mtp_loss), 20.0)
                ),
                "mtp/valid_tokens": float(mtp_tokens),
                "mtp/main_to_mtp_loss_ratio": float(ntp_loss)
                / max(float(mtp_loss), 1e-12),
            }
        )
    return metrics
