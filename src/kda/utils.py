"""Kimi Delta Attention operators, projections, states, and diagnostics."""

from __future__ import annotations

import torch


def validate_attention_mask(
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
    # Public KDA initially supports monotonic right-padding.
    if tokens > 1 and torch.any((~attention_mask[:, :-1]) & attention_mask[:, 1:]):
        raise ValueError("attention_mask must be monotonic right-padding")
    if require_nonempty and torch.any(attention_mask.sum(dim=1) == 0):
        raise ValueError("all-padding sequences are not supported")
    return attention_mask


def validate_core_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
) -> tuple[int, int, int, int, int]:
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B,T,H,K], got {tuple(q.shape)}")
    batch, tokens, heads, key_dim = q.shape
    if tokens == 0:
        raise ValueError("KDA does not support empty sequences")
    if k.shape != q.shape or g.shape != q.shape:
        raise ValueError("k and g must have the same shape as q")
    if v.ndim != 4 or v.shape[:3] != (batch, tokens, heads):
        raise ValueError("v must have shape [B,T,H,V] matching q")
    value_dim = v.shape[-1]
    if beta.shape != (batch, tokens, heads):
        raise ValueError(
            f"beta must have shape {(batch, tokens, heads)}, "
            f"got {tuple(beta.shape)}"
        )
    tensors = (q, k, v, g, beta)
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("all KDA inputs must be on the same device")
    if any(not tensor.dtype.is_floating_point for tensor in tensors):
        raise TypeError("all KDA inputs must use floating-point dtypes")
    if initial_state is not None:
        expected = (batch, heads, key_dim, value_dim)
        if initial_state.shape != expected:
            raise ValueError(
                f"initial_state must have shape {expected}, "
                f"got {tuple(initial_state.shape)}"
            )
        if initial_state.device != q.device:
            raise ValueError("initial_state and inputs must be on the same device")
    validate_attention_mask(attention_mask, batch, tokens)
    return batch, tokens, heads, key_dim, value_dim


def accumulation_dtype(
    dtype: torch.dtype, accumulate_state_in_fp32: bool
) -> torch.dtype:
    if accumulate_state_in_fp32 and dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def apply_operator_mask(
    q: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if attention_mask is None:
        return q, g, beta
    valid = attention_mask[:, :, None]
    q = q * valid[..., None].to(q.dtype)
    g = torch.where(valid[..., None], g, torch.zeros_like(g))
    beta = torch.where(valid, beta, torch.zeros_like(beta))
    return q, g, beta


def tiled_causal_decay_dot(
    left: torch.Tensor,
    right: torch.Tensor,
    left_log_decay: torch.Tensor,
    right_log_decay: torch.Tensor,
    tile_size: int,
    *,
    include_diagonal: bool,
) -> torch.Tensor:
    """Stable causal dot products using dense secondary tiles.

    Inputs use internal layout ``[B,H,C,K]``. For row ``i`` and column ``j``,
    computes ``sum_k left_i[k] * exp(l_i[k]-l_j[k]) * right_j[k]`` only
    where ``j <= i`` (or ``j < i``).
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if (
        left.shape != left_log_decay.shape
        or right.shape != right_log_decay.shape
        or left.shape != right.shape
    ):
        raise ValueError("left/right tensors and log-decays must share shape")
    batch, heads, count, _ = left.shape
    output = left.new_zeros(batch, heads, count, count)
    for row_start in range(0, count, tile_size):
        row_end = min(row_start + tile_size, count)
        for col_start in range(0, row_end, tile_size):
            col_end = min(col_start + tile_size, count)
            left_tile = left[:, :, row_start:row_end]
            right_tile = right[:, :, col_start:col_end]
            log_difference = (
                left_log_decay[:, :, row_start:row_end, None, :]
                - right_log_decay[:, :, None, col_start:col_end, :]
            )
            block = torch.einsum(
                "bhrck,bhck->bhrc",
                left_tile[:, :, :, None, :] * torch.exp(log_difference),
                right_tile,
            )
            row_ids = torch.arange(row_start, row_end, device=left.device)[:, None]
            col_ids = torch.arange(col_start, col_end, device=left.device)[None, :]
            causal = row_ids >= col_ids if include_diagonal else row_ids > col_ids
            output[:, :, row_start:row_end, col_start:col_end] = (
                block * causal.to(block.dtype)
            )
    return output

