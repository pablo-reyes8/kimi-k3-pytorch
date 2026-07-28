from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F

from .utils import accumulation_dtype


def _validate_inputs(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[int, int, int, int]:
    if q.ndim != 4:
        raise ValueError("q must have shape [B,Tq,H,Q]")
    batch, query_tokens, heads, query_dim = q.shape
    if k.ndim != 4 or k.shape[0] != batch or k.shape[2:] != (heads, query_dim):
        raise ValueError("k must have shape [B,Tk,H,Q] matching q")
    if v.ndim != 4 or v.shape[:3] != k.shape[:3]:
        raise ValueError("v must have shape [B,Tk,H,V] matching k")
    if query_tokens == 0 or k.shape[1] == 0:
        raise ValueError("attention does not support empty sequences")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must share device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q, k, and v must share dtype")
    if not q.dtype.is_floating_point:
        raise TypeError("q, k, and v must be floating point")
    return batch, query_tokens, heads, k.shape[1]


def _allowed_attention_mask(
    q: torch.Tensor,
    key_tokens: int,
    attention_mask: torch.Tensor | None,
    query_mask: torch.Tensor | None,
    is_causal: bool,
    query_positions: torch.Tensor | None,
) -> torch.Tensor:
    batch, query_tokens = q.shape[:2]
    if attention_mask is None:
        key_valid = torch.ones(
            batch, key_tokens, dtype=torch.bool, device=q.device
        )
    else:
        if attention_mask.shape != (batch, key_tokens):
            raise ValueError(
                f"attention_mask must have shape {(batch, key_tokens)}"
            )
        if attention_mask.dtype != torch.bool:
            raise TypeError("attention_mask must be boolean")
        key_valid = attention_mask
    if query_mask is None:
        query_valid = torch.ones(
            batch, query_tokens, dtype=torch.bool, device=q.device
        )
    else:
        if query_mask.shape != (batch, query_tokens):
            raise ValueError(f"query_mask must have shape {(batch, query_tokens)}")
        if query_mask.dtype != torch.bool:
            raise TypeError("query_mask must be boolean")
        query_valid = query_mask
    allowed = key_valid[:, None, :].expand(batch, query_tokens, key_tokens)
    if is_causal:
        if query_positions is None:
            offset = key_tokens - query_tokens
            query_positions = (
                torch.arange(query_tokens, device=q.device) + offset
            )[None, :].expand(batch, -1)
        elif query_positions.shape != (batch, query_tokens):
            raise ValueError(
                f"query_positions must have shape {(batch, query_tokens)}"
            )
        key_indices = torch.arange(key_tokens, device=q.device)
        allowed = allowed & (
            key_indices[None, None, :] <= query_positions[:, :, None]
        )
    return allowed & query_valid[:, :, None]


def manual_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    is_causal: bool = True,
    dropout_p: float = 0.0,
    keep_output_fp32: bool = True,
    *,
    query_mask: torch.Tensor | None = None,
    query_positions: torch.Tensor | None = None,
    training: bool = False,
    return_attentions: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    batch, _, _, key_tokens = _validate_inputs(q, k, v)
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must satisfy 0 <= p < 1")
    allowed = _allowed_attention_mask(
        q,
        key_tokens,
        attention_mask,
        query_mask,
        is_causal,
        query_positions,
    )
    compute_dtype = accumulation_dtype(q.dtype, keep_output_fp32)
    query = q.transpose(1, 2).to(compute_dtype)
    key = k.transpose(1, 2).to(compute_dtype)
    value = v.transpose(1, 2).to(compute_dtype)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    expanded_allowed = allowed[:, None, :, :]
    scores = scores.masked_fill(~expanded_allowed, -torch.inf)
    valid_rows = expanded_allowed.any(dim=-1, keepdim=True)
    safe_scores = torch.where(valid_rows, scores, torch.zeros_like(scores))
    probabilities = torch.softmax(safe_scores, dim=-1)
    probabilities = torch.where(
        expanded_allowed, probabilities, torch.zeros_like(probabilities)
    )
    probabilities = F.dropout(
        probabilities, p=dropout_p, training=training
    )
    output = torch.matmul(probabilities, value).transpose(1, 2)
    if not keep_output_fp32 and output.dtype != q.dtype:
        output = output.to(q.dtype)
    if return_attentions:
        return output, probabilities
    return output


def mla_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    is_causal: bool = True,
    dropout_p: float = 0.0,
    keep_output_fp32: bool = True,
    *,
    backend: Literal["manual", "sdpa"] = "sdpa",
    query_mask: torch.Tensor | None = None,
    query_positions: torch.Tensor | None = None,
    training: bool = False,
    return_attentions: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if backend not in ("manual", "sdpa"):
        raise ValueError("backend must be 'manual' or 'sdpa'")
    if backend == "manual" or return_attentions:
        return manual_causal_attention(
            q,
            k,
            v,
            attention_mask,
            is_causal,
            dropout_p,
            keep_output_fp32,
            query_mask=query_mask,
            query_positions=query_positions,
            training=training,
            return_attentions=return_attentions,
        )
    _, _, _, key_tokens = _validate_inputs(q, k, v)
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must satisfy 0 <= p < 1")
    allowed = _allowed_attention_mask(
        q,
        key_tokens,
        attention_mask,
        query_mask,
        is_causal,
        query_positions,
    )
    compute_dtype = accumulation_dtype(q.dtype, keep_output_fp32)
    query = q.transpose(1, 2).to(compute_dtype)
    key = k.transpose(1, 2).to(compute_dtype)
    value = v.transpose(1, 2).to(compute_dtype)
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=allowed[:, None, :, :],
        dropout_p=dropout_p if training else 0.0,
        is_causal=False,
    ).transpose(1, 2)
    output = output * allowed.any(dim=-1)[:, :, None, None].to(output.dtype)
    if not keep_output_fp32 and output.dtype != q.dtype:
        output = output.to(q.dtype)
    return output
