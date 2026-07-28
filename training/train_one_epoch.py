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
    accumulated_batches = 0

    def optimizer_step(window_size: int) -> None:
        nonlocal steps
        if scaler is not None:
            scaler.unscale_(optimizer)

        # Every loss is divided by grad_accum_steps. If the final window is
        # partial, rescale its gradients so it still represents the mean over
        # the batches actually present instead of an artificially smaller step.
        if window_size != grad_accum_steps:
            correction = grad_accum_steps / window_size
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)

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

        accumulated_batches += 1
        if accumulated_batches == grad_accum_steps:
            optimizer_step(accumulated_batches)
            accumulated_batches = 0
        losses.append(float(output.loss.detach().item()))

    if accumulated_batches:
        optimizer_step(accumulated_batches)

    return {
        "loss": sum(losses) / max(1, len(losses)),
        "num_batches": float(len(losses)),
        "optimizer_steps": float(steps),
    }
