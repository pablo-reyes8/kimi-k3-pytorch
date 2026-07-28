from __future__ import annotations

from typing import Literal

import torch

from src.hybrid_backbone import HybridBackboneCache


def validate_and_build_attention_mask(
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    *,
    d_model: int,
    pad_token_id: int | None,
) -> tuple[int, int, torch.Tensor]:
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError(
            "exactly one of input_ids or inputs_embeds must be provided"
        )
    if input_ids is not None:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B,T]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must use an integer dtype")
        batch, tokens = input_ids.shape
        source = input_ids
    else:
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != d_model:
            raise ValueError(f"inputs_embeds must have shape [B,T,{d_model}]")
        batch, tokens = inputs_embeds.shape[:2]
        source = inputs_embeds
    if tokens == 0:
        raise ValueError("empty token sequences are not supported")
    if attention_mask is None:
        if input_ids is not None and pad_token_id is not None:
            attention_mask = input_ids.ne(pad_token_id)
        else:
            attention_mask = torch.ones(
                batch, tokens, dtype=torch.bool, device=source.device
            )
    if attention_mask.shape != (batch, tokens):
        raise ValueError(f"attention_mask must have shape {(batch, tokens)}")
    if attention_mask.dtype != torch.bool:
        raise TypeError("attention_mask must be boolean")
    if attention_mask.device != source.device:
        raise ValueError("attention_mask and text inputs must share device")
    if tokens > 1 and torch.any(
        (~attention_mask[:, :-1]) & attention_mask[:, 1:]
    ):
        raise ValueError(
            "left padding is not supported by the current hybrid backbone"
        )
    if torch.any(attention_mask.sum(dim=1) == 0):
        raise ValueError("all-padding sequences are not supported")
    return batch, tokens, attention_mask


def resolve_execution_mode(
    cache: HybridBackboneCache | None,
    use_cache: bool,
    tokens: int,
) -> Literal["full", "prefill", "decode"]:
    if cache is not None and not use_cache:
        raise ValueError("a cache requires use_cache=True")
    if cache is not None and tokens != 1:
        raise ValueError("cached decode accepts exactly one new token")
    if cache is not None:
        return "decode"
    return "prefill" if use_cache else "full"
