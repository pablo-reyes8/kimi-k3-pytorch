"""Multi-token prediction components used as an optional KimiK3 output head."""

from __future__ import annotations

from dataclasses import dataclass

import torch


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


@dataclass(frozen=True)
class MTPTrainingView:
    """Aligned context, future targets, and validity mask for one MTP depth."""

    source_hidden: torch.Tensor
    future_input_ids: torch.Tensor
    target_ids: torch.Tensor
    valid_mask: torch.Tensor


def _validate_token_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, int],
    device: torch.device,
) -> None:
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if tensor.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must use an integer dtype")
    if tensor.device != device:
        raise ValueError(f"{name} and last_hidden_state must share device")


def build_mtp_training_view(
    last_hidden_state: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    segment_ids: torch.Tensor | None = None,
    *,
    ignore_index: int = -100,
) -> MTPTrainingView:
    """Build the sole canonical alignment: h[t] + x[t+1] predicts x[t+2]."""

    if last_hidden_state.ndim != 3:
        raise ValueError("last_hidden_state must have shape [B,T,D]")
    if not last_hidden_state.dtype.is_floating_point:
        raise TypeError("last_hidden_state must be floating point")
    batch, tokens, _ = last_hidden_state.shape
    shape = (batch, tokens)
    _validate_token_tensor(
        "input_ids", input_ids, shape, last_hidden_state.device
    )
    if attention_mask is None:
        mask = torch.ones(
            shape, dtype=torch.bool, device=last_hidden_state.device
        )
    else:
        if attention_mask.shape != shape:
            raise ValueError(f"attention_mask must have shape {shape}")
        if attention_mask.dtype != torch.bool:
            raise TypeError("attention_mask must be boolean")
        if attention_mask.device != last_hidden_state.device:
            raise ValueError(
                "attention_mask and last_hidden_state must share device"
            )
        mask = attention_mask
    targets = input_ids if labels is None else labels
    if labels is not None:
        _validate_token_tensor(
            "labels", labels, shape, last_hidden_state.device
        )
    if segment_ids is not None:
        _validate_token_tensor(
            "segment_ids", segment_ids, shape, last_hidden_state.device
        )

    source_hidden = last_hidden_state[:, :-2]
    future_input_ids = input_ids[:, 1:-1]
    target_ids = targets[:, 2:]
    valid = mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:]
    valid = valid & target_ids.ne(ignore_index)
    if segment_ids is not None:
        valid = (
            valid
            & segment_ids[:, :-2].eq(segment_ids[:, 1:-1])
            & segment_ids[:, 1:-1].eq(segment_ids[:, 2:])
        )
    return MTPTrainingView(
        source_hidden=source_hidden,
        future_input_ids=future_input_ids,
        target_ids=target_ids,
        valid_mask=valid,
    )


def build_mtp_feature_mask(
    attention_mask: torch.Tensor | None,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Mask structural triplets without letting ignored labels alter features."""

    if attention_mask is None:
        mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        mask = attention_mask
    return mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:]
