"""Small causal-LM metric helpers."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


def compute_lm_metrics(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> Dict[str, float]:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    valid = flat_labels.ne(ignore_index)
    if not valid.any():
        return {"loss": float("nan"), "perplexity": float("nan"), "token_accuracy": float("nan")}
    loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index)
    predictions = flat_logits.argmax(dim=-1)
    accuracy = predictions[valid].eq(flat_labels[valid]).float().mean()
    return {
        "loss": float(loss.item()),
        "perplexity": float(math.exp(min(loss.item(), 50.0))),
        "token_accuracy": float(accuracy.item()),
    }
