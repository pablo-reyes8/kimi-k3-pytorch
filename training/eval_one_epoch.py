"""Generic evaluation loop."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from data.batch import normalize_lm_batch
from .autocast import autocast_ctx, move_batch_to_device


@torch.no_grad()
def eval_one_epoch(
    model,
    dataloader,
    *,
    device: str | torch.device = "cpu",
    amp_enabled: bool = False,
    amp_dtype: str = "bf16",
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    device = torch.device(device)
    model.to(device).eval()
    losses = []
    for batch_index, raw_batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch_to_device(normalize_lm_batch(raw_batch), device)
        with autocast_ctx(device, enabled=amp_enabled, amp_dtype=amp_dtype):
            output = model(**batch)
        if output.loss is None:
            raise ValueError("model output must contain a loss")
        losses.append(float(output.loss.item()))
    loss = sum(losses) / max(1, len(losses))
    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 50.0)),
        "num_batches": float(len(losses)),
    }
