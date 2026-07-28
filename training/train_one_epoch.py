"""Generic one-epoch training loop."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from data.batch import normalize_lm_batch
from .autocast import autocast_ctx, move_batch_to_device


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    *,
    device: str | torch.device = "cpu",
    scheduler=None,
    scaler=None,
    amp_enabled: bool = False,
    amp_dtype: str = "bf16",
    grad_accum_steps: int = 1,
    grad_clip: Optional[float] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive")
    device = torch.device(device)
    model.to(device).train()
    optimizer.zero_grad(set_to_none=True)
    losses, steps = [], 0

    for batch_index, raw_batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch_to_device(normalize_lm_batch(raw_batch), device)
        with autocast_ctx(device, enabled=amp_enabled, amp_dtype=amp_dtype):
            output = model(**batch)
            if output.loss is None:
                raise ValueError("model output must contain a loss")
            scaled_loss = output.loss / grad_accum_steps
        if scaler is None:
            scaled_loss.backward()
        else:
            scaler.scale(scaled_loss).backward()

        should_step = (batch_index + 1) % grad_accum_steps == 0
        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            steps += 1
        losses.append(float(output.loss.detach().item()))

    return {
        "loss": sum(losses) / max(1, len(losses)),
        "num_batches": float(len(losses)),
        "optimizer_steps": float(steps),
    }
