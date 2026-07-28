"""Canonical sampled-token log-probability gathering for RL and MOPD."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gather_token_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Gather FP32 log-probabilities for selected tokens without CUDA kernels."""

    if logits.ndim != 3 or not logits.dtype.is_floating_point:
        raise ValueError("logits must be floating point with shape [B,T,V]")
    if token_ids.ndim != 2 or token_ids.shape != logits.shape[:2]:
        raise ValueError("token_ids must match logits shape [B,T]")
    if token_ids.device != logits.device:
        raise ValueError("token_ids and logits must share device")
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("token_ids must use int32 or int64")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if torch.any((token_ids < 0) | (token_ids >= logits.shape[-1])):
        raise ValueError("token_ids contain out-of-vocabulary IDs")
    log_probs = F.log_softmax(logits.float() / temperature, dim=-1)
    selected = log_probs.gather(-1, token_ids.long().unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(selected).all():
        raise FloatingPointError("gather_token_logprobs produced non-finite values")
    return selected

