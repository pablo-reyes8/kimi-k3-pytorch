"""Gated Multi-head Latent Attention components and cache utilities."""

from __future__ import annotations

import torch


def accumulation_dtype(
    dtype: torch.dtype, keep_output_fp32: bool
) -> torch.dtype:
    if keep_output_fp32 and dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def validate_right_padding_mask(
    attention_mask: torch.Tensor | None,
    batch: int,
    tokens: int,
    *,
    require_nonempty: bool = True,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.shape != (batch, tokens):
        raise ValueError(
            f"attention_mask must have shape {(batch, tokens)}, "
            f"got {tuple(attention_mask.shape)}"
        )
    if attention_mask.dtype != torch.bool:
        raise TypeError("attention_mask must be boolean (True means valid)")
    if tokens > 1 and torch.any((~attention_mask[:, :-1]) & attention_mask[:, 1:]):
        raise ValueError("attention_mask must be monotonic right-padding")
    if require_nonempty and torch.any(attention_mask.sum(dim=1) == 0):
        raise ValueError("all-padding sequences are not supported")
    return attention_mask


def validate_hidden_states(
    hidden_states: torch.Tensor, d_model: int
) -> tuple[int, int]:
    if hidden_states.ndim != 3 or hidden_states.shape[-1] != d_model:
        raise ValueError(
            f"hidden_states must have shape [B,T,{d_model}], "
            f"got {tuple(hidden_states.shape)}"
        )
    if hidden_states.shape[1] == 0:
        raise ValueError("Gated MLA does not support empty sequences")
    if not hidden_states.dtype.is_floating_point:
        raise TypeError("hidden_states must use a floating-point dtype")
    return hidden_states.shape[:2]
