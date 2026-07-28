"""Shape, mask, alignment, and numerical validation for Kimi losses."""

from __future__ import annotations

from typing import Literal

import torch


Alignment = Literal["causal_shift", "already_aligned"]
ZeroValidPolicy = Literal["raise", "connected_zero"]


def boolean_mask(
    name: str,
    value: torch.Tensor | None,
    shape: tuple[int, ...],
    device: torch.device,
    *,
    default: bool = True,
) -> torch.Tensor:
    """Normalize an optional mask to a boolean tensor on the expected device."""

    if value is None:
        return torch.full(shape, default, dtype=torch.bool, device=device)
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.device != device:
        raise ValueError(f"{name} must share the logits device")
    return value.to(torch.bool)


def require_finite(
    name: str,
    values: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    """Raise when any valid position contains NaN or infinity."""

    invalid = valid_mask & ~torch.isfinite(values)
    if torch.any(invalid):
        raise FloatingPointError(
            f"{name} contains {int(invalid.sum().item())} non-finite valid values"
        )


def validate_token_ids(
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    vocab_size: int,
    *,
    name: str = "labels",
) -> None:
    """Ensure valid target IDs lie inside the vocabulary."""

    if labels.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must use int32 or int64")
    bad = valid_mask & ((labels < 0) | (labels >= vocab_size))
    if torch.any(bad):
        raise ValueError(f"{name} contains IDs outside [0, {vocab_size})")


def validate_zero_policy(value: str) -> ZeroValidPolicy:
    """Validate and narrow a zero-valid-token policy."""

    if value not in ("raise", "connected_zero"):
        raise ValueError("zero_valid_policy must be 'raise' or 'connected_zero'")
    return value


def align_token_inputs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    alignment: Alignment,
    attention_mask: torch.Tensor | None,
    loss_mask: torch.Tensor | None,
    boundary_mask: torch.Tensor | None,
    token_weights: torch.Tensor | None,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align token logits, labels, semantic masks, and optional weights."""

    if logits.ndim != 3 or not logits.dtype.is_floating_point:
        raise ValueError("logits must be floating point with shape [B,T,V]")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must match logits shape [B,T]")
    if labels.device != logits.device:
        raise ValueError("labels and logits must share device")
    if alignment not in ("causal_shift", "already_aligned"):
        raise ValueError("unsupported alignment")
    batch, tokens = labels.shape
    source_mask = boolean_mask(
        "attention_mask", attention_mask, (batch, tokens), logits.device
    )
    semantic_mask = boolean_mask(
        "loss_mask", loss_mask, (batch, tokens), logits.device
    )
    boundary = boolean_mask(
        "boundary_mask", boundary_mask, (batch, tokens), logits.device
    )
    if token_weights is None:
        weights = torch.ones((batch, tokens), device=logits.device)
    else:
        if token_weights.shape != (batch, tokens):
            raise ValueError(f"token_weights must have shape {(batch, tokens)}")
        if token_weights.device != logits.device:
            raise ValueError("token_weights and logits must share device")
        weights = token_weights.float()
        if not torch.isfinite(weights).all() or torch.any(weights < 0):
            raise ValueError("token_weights must be finite and non-negative")
    if alignment == "causal_shift":
        aligned_logits = logits[:, :-1]
        aligned_labels = labels[:, 1:]
        valid = (
            source_mask[:, :-1]
            & source_mask[:, 1:]
            & semantic_mask[:, 1:]
            & boundary[:, 1:]
        )
        weights = weights[:, 1:]
    else:
        aligned_logits = logits
        aligned_labels = labels
        valid = source_mask & semantic_mask & boundary
    valid = valid & aligned_labels.ne(ignore_index)
    return aligned_logits, aligned_labels, valid, weights

