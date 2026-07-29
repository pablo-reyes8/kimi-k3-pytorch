"""Numerically defensive greedy, top-k and nucleus sampling."""

from __future__ import annotations

import torch

from .config import GenerationConfig


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    penalty: float | None,
) -> torch.Tensor:
    if penalty is None or penalty == 1:
        return logits
    if penalty <= 0:
        raise ValueError("repetition penalty must be positive")
    adjusted = logits.clone()
    for batch_index in range(logits.shape[0]):
        seen = torch.unique(token_ids[batch_index].long()).to(logits.device)
        selected = adjusted[batch_index, seen]
        adjusted[batch_index, seen] = torch.where(
            selected < 0, selected * penalty, selected / penalty
        )
    return adjusted


def top_k_filter(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None or top_k >= logits.shape[-1]:
        return logits
    cutoff = torch.topk(logits, int(top_k), dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < cutoff, float("-inf"))


def top_p_filter(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    if top_p is None or top_p >= 1:
        return logits
    sorted_logits, indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.empty_like(logits)
    return filtered.scatter(-1, indices, sorted_logits)


def _safe_argmax(logits: torch.Tensor) -> torch.Tensor:
    safe = torch.nan_to_num(
        logits, nan=float("-inf"), posinf=1e30, neginf=-1e30
    )
    return safe.argmax(dim=-1, keepdim=True)


def sample_next_token(
    logits: torch.Tensor,
    config: GenerationConfig,
    *,
    previous_token_ids: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("sampling logits must have shape [B,V]")
    if previous_token_ids is not None:
        logits = apply_repetition_penalty(
            logits, previous_token_ids, config.repetition_penalty
        )
    if not torch.isfinite(logits).all() or not config.do_sample:
        return _safe_argmax(logits)
    filtered = top_p_filter(
        top_k_filter(logits / config.temperature, config.top_k),
        config.top_p,
    )
    probabilities = torch.softmax(filtered, dim=-1)
    valid = (
        torch.isfinite(probabilities).all(dim=-1)
        & probabilities.sum(dim=-1).gt(0)
    )
    if not bool(valid.all()):
        return _safe_argmax(filtered)
    return torch.multinomial(
        probabilities, num_samples=1, generator=generator
    )


__all__ = [
    "apply_repetition_penalty",
    "sample_next_token",
    "top_k_filter",
    "top_p_filter",
]
