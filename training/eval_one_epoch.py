"""Side-effect-free, token-weighted evaluation."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from data.batch import normalize_lm_batch

from .autocast import autocast_ctx, move_batch_to_device
from .loss_accounting import combine_window_loss, extract_loss_contribution
from .model_call import call_model


@torch.no_grad()
def eval_one_epoch(
    model,
    dataloader,
    *,
    device: str | torch.device = "cpu",
    amp_enabled: bool = False,
    amp_dtype: str = "bf16",
    max_batches: Optional[int] = None,
    use_mtp: bool | None = None,
    ema=None,
    use_ema: bool = False,
) -> Dict[str, float]:
    if max_batches is not None and max_batches < 0:
        raise ValueError("max_batches must be None or non-negative")
    if use_ema and ema is None:
        raise ValueError("use_ema=True requires an EMA instance")

    device = torch.device(device)
    was_training = model.training
    model.to(device).eval()
    contributions = []
    num_samples = 0

    def evaluate_batches() -> None:
        nonlocal num_samples
        for batch_index, raw_batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch_to_device(normalize_lm_batch(raw_batch), device)
            num_samples += int(batch["input_ids"].shape[0])
            with autocast_ctx(
                device, enabled=amp_enabled, amp_dtype=amp_dtype
            ):
                output = call_model(model, batch, use_mtp=use_mtp)
            contributions.append(extract_loss_contribution(output, batch))

    try:
        if use_ema:
            with ema.average_parameters(model):
                evaluate_batches()
        else:
            evaluate_batches()
    finally:
        model.train(was_training)

    if not contributions:
        return {
            "loss": 0.0,
            "ntp_loss": 0.0,
            "mtp_loss": float("nan"),
            "perplexity": 1.0,
            "ntp_perplexity": 1.0,
            "ntp_tokens": 0.0,
            "mtp_tokens": 0.0,
            "tokens": 0.0,
            "num_batches": 0.0,
            "num_samples": 0.0,
            "used_ema": float(use_ema),
        }

    _, stats = combine_window_loss(contributions)
    ntp_perplexity = math.exp(min(stats["ntp_loss"], 50.0))
    return {
        **stats,
        # Kept as a compatibility alias. It is NTP perplexity, never the
        # mixed NTP+MTP objective perplexity.
        "perplexity": ntp_perplexity,
        "ntp_perplexity": ntp_perplexity,
        "num_batches": float(len(contributions)),
        "num_samples": float(num_samples),
        "used_ema": float(use_ema),
    }
