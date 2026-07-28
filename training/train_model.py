"""Minimal architecture-neutral orchestration."""

from __future__ import annotations

from typing import Dict

from .eval_one_epoch import eval_one_epoch
from .train_one_epoch import train_one_epoch


def train_model(
    model,
    train_loader,
    optimizer,
    *,
    epochs: int,
    device="cpu",
    val_loader=None,
    scheduler=None,
    max_batches=None,
) -> Dict[str, list]:
    history = {"train": [], "validation": []}
    for _ in range(epochs):
        history["train"].append(
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                device=device,
                scheduler=scheduler,
                max_batches=max_batches,
            )
        )
        if val_loader is not None:
            history["validation"].append(
                eval_one_epoch(model, val_loader, device=device, max_batches=max_batches)
            )
    return history
