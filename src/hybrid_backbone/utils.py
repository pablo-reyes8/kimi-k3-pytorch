"""Hybrid KDA/MLA backbone components and cache structures."""

from __future__ import annotations

import torch


VALID_MODES = ("full", "prefill", "decode")


def validate_backbone_inputs(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    d_model: int,
    mode: str,
) -> tuple[int, int, torch.Tensor]:
    if (
        hidden_states.ndim != 3
        or hidden_states.shape[-1] != d_model
    ):
        raise ValueError(
            f"hidden_states must have shape [B,T,{d_model}], "
            f"got {tuple(hidden_states.shape)}"
        )
    batch, tokens, _ = hidden_states.shape
    if tokens == 0:
        raise ValueError("hybrid backbone does not support empty sequences")
    if not hidden_states.dtype.is_floating_point:
        raise TypeError("hidden_states must be floating point")
    if mode not in VALID_MODES:
        raise ValueError(f"unsupported execution mode {mode!r}")
    if mode == "decode" and tokens != 1:
        raise ValueError("decode mode requires exactly one token")
    if attention_mask is None:
        mask = torch.ones(
            batch, tokens, dtype=torch.bool, device=hidden_states.device
        )
    else:
        if attention_mask.shape != (batch, tokens):
            raise ValueError(
                f"attention_mask must have shape {(batch, tokens)}"
            )
        if attention_mask.dtype != torch.bool:
            raise TypeError("attention_mask must be boolean")
        if tokens > 1 and torch.any(
            (~attention_mask[:, :-1]) & attention_mask[:, 1:]
        ):
            raise ValueError("attention_mask must be monotonic right-padding")
        if torch.any(attention_mask.sum(dim=1) == 0):
            raise ValueError("all-padding sequences are not supported")
        mask = attention_mask
    return batch, tokens, mask


def masked_norm_mean(
    hidden_states: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return hidden_states.float().norm(dim=-1)[mask].mean()


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
