"""Shape normalization and depth weighting for phase-9 MTP targets."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def canonicalize_mtp_inputs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor | None,
    future_offsets: torch.Tensor | Sequence[int] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], bool]:
    """Normalize single- and multi-depth MTP inputs to ``[B,H,T,...]``."""

    single_depth = logits.ndim == 3
    if single_depth:
        logits = logits.unsqueeze(1)
        labels = labels.unsqueeze(1)
        if loss_mask is not None:
            loss_mask = loss_mask.unsqueeze(1)
    if logits.ndim != 4 or not logits.dtype.is_floating_point:
        raise ValueError("mtp_logits must have shape [B,T,V] or [B,H,T,V]")
    if labels.shape != logits.shape[:-1]:
        raise ValueError("mtp_labels must match mtp_logits without vocabulary")
    if labels.device != logits.device:
        raise ValueError("MTP labels and logits must share device")
    if labels.dtype not in (torch.int32, torch.int64):
        raise TypeError("mtp_labels must use int32 or int64")
    shape = logits.shape[:-1]
    if loss_mask is None:
        mask = torch.ones(shape, dtype=torch.bool, device=logits.device)
    else:
        if loss_mask.shape != shape:
            raise ValueError("mtp_loss_mask must match MTP labels")
        mask = loss_mask.to(torch.bool)
    depth = logits.shape[1]
    if future_offsets is None:
        offsets = tuple(range(2, depth + 2))
    elif isinstance(future_offsets, torch.Tensor):
        if future_offsets.ndim != 1:
            raise ValueError("future_offsets must be one-dimensional")
        offsets = tuple(int(x) for x in future_offsets.tolist())
    else:
        offsets = tuple(int(x) for x in future_offsets)
    if len(offsets) != depth or any(x <= 0 for x in offsets):
        raise ValueError("future_offsets must contain one positive value per depth")
    return logits, labels, mask, offsets, single_depth

