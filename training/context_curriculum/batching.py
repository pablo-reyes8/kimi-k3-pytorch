"""Safe truncation and dynamic-padding helpers for PCC batches."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
from torch.utils.data._utils.collate import default_collate


_SEQUENCE_KEYS = {
    "input_ids",
    "labels",
    "attention_mask",
    "segment_ids",
    "loss_mask",
    "boundary_mask",
    "token_weights",
    "inputs_embeds",
}


def _validate_packed_segments(batch: dict, mask: torch.Tensor) -> None:
    segments = batch.get("segment_ids")
    if segments is None:
        return
    if segments.shape != mask.shape:
        raise ValueError("segment_ids must align with input_ids")
    for row_segments, row_mask in zip(segments, mask):
        if torch.unique(row_segments[row_mask]).numel() > 1:
            raise ValueError(
                "packed documents are rejected: KDA/MLA currently have no "
                "segment-isolated attention kernel"
            )


def truncate_batch_to_context(
    batch: Mapping[str, object],
    max_seq_len: int,
    *,
    ignore_index: int = -100,
    image_token_id: int | None = None,
    video_token_id: int | None = None,
) -> tuple[dict, dict[str, float]]:
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if "input_ids" not in batch or not torch.is_tensor(batch["input_ids"]):
        raise ValueError("context batching requires tensor input_ids")
    input_ids = batch["input_ids"]
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B,T]")
    batch_size, original_length = input_ids.shape
    result = dict(batch)
    target = min(original_length, int(max_seq_len))
    original_ids = input_ids
    for name in _SEQUENCE_KEYS:
        value = result.get(name)
        if value is None:
            continue
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a tensor")
        if value.shape[:2] != (batch_size, original_length):
            raise ValueError(f"{name} must align with input_ids")
        result[name] = value[:, :target].contiguous()
    mask = result.get("attention_mask")
    if mask is None:
        mask = torch.ones(
            batch_size,
            target,
            dtype=torch.bool,
            device=input_ids.device,
        )
        result["attention_mask"] = mask
    elif mask.dtype != torch.bool:
        raise TypeError("attention_mask must be boolean")
    if torch.any(mask.sum(dim=1) == 0):
        raise ValueError("context truncation produced an all-padding sample")
    if mask.shape[1] > 1 and torch.any((~mask[:, :-1]) & mask[:, 1:]):
        raise ValueError(
            "context batching requires right-padded attention_mask rows"
        )
    necessary = int(mask.sum(dim=1).max().item())
    necessary = max(1, min(necessary, target))
    if necessary < target:
        for name in _SEQUENCE_KEYS:
            value = result.get(name)
            if torch.is_tensor(value):
                result[name] = value[:, :necessary].contiguous()
        target = necessary
        mask = result["attention_mask"]
    _validate_packed_segments(result, mask)
    segments = result.get("segment_ids")
    if segments is not None and "boundary_mask" not in result:
        boundary = torch.ones_like(mask)
        boundary[:, 1:] = segments[:, 1:].eq(segments[:, :-1])
        result["boundary_mask"] = boundary & mask

    for token_id, values_name in (
        (image_token_id, "pixel_values"),
        (video_token_id, "video_values"),
    ):
        if (
            token_id is not None
            and result.get(values_name) is not None
            and not torch.equal(
                original_ids.eq(token_id).sum(dim=1),
                result["input_ids"].eq(token_id).sum(dim=1),
            )
        ):
            raise ValueError(
                f"context truncation would detach {values_name} from visual "
                "placeholder tokens"
            )
    valid_tokens = mask.sum()
    labels = result.get("labels")
    if torch.is_tensor(labels):
        valid_tokens = (mask & labels.ne(ignore_index)).sum()
    capacity = mask.numel()
    return result, {
        "valid_tokens": float(valid_tokens.item()),
        "padding_fraction": float(
            1.0 - mask.sum().item() / max(capacity, 1)
        ),
        "sequence_length": float(target),
    }


def _dynamic_collate(samples: Sequence[Mapping[str, object]], pad_token_id: int, ignore_index: int):
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    lengths = [int(sample["input_ids"].shape[0]) for sample in samples]
    width = max(lengths)
    output = {}
    keys = set().union(*(sample.keys() for sample in samples))
    padding = {
        "input_ids": pad_token_id,
        "labels": ignore_index,
        "attention_mask": False,
        "segment_ids": -1,
        "loss_mask": False,
        "boundary_mask": False,
        "token_weights": 0.0,
    }
    for name in keys:
        values = [sample.get(name) for sample in samples]
        if name in _SEQUENCE_KEYS and all(torch.is_tensor(value) for value in values):
            padded = []
            for value in values:
                clipped = value[:width]
                pad_shape = (width - clipped.shape[0],) + clipped.shape[1:]
                if pad_shape[0]:
                    fill = padding.get(name, 0)
                    clipped = torch.cat(
                        (clipped, clipped.new_full(pad_shape, fill)), dim=0
                    )
                padded.append(clipped)
            output[name] = torch.stack(padded)
        elif all(value is not None for value in values):
            output[name] = default_collate(values)
    if "attention_mask" not in output:
        output["attention_mask"] = torch.arange(width)[None, :] < torch.tensor(
            lengths
        )[:, None]
    return output


class ProgressiveContextCollator:
    """Apply an optional base collator, then enforce the active context."""

    def __init__(
        self,
        max_seq_len: int,
        *,
        base_collator: Callable | None = None,
        pad_token_id: int = 0,
        ignore_index: int = -100,
        image_token_id: int | None = None,
        video_token_id: int | None = None,
    ):
        self.max_seq_len = int(max_seq_len)
        self.base_collator = base_collator
        self.pad_token_id = int(pad_token_id)
        self.ignore_index = int(ignore_index)
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

    def __call__(self, samples):
        clipped_samples = []
        for sample in samples:
            copied = dict(sample)
            input_ids = copied.get("input_ids")
            for token_id, values_name in (
                (self.image_token_id, "pixel_values"),
                (self.video_token_id, "video_values"),
            ):
                if (
                    token_id is not None
                    and copied.get(values_name) is not None
                    and torch.is_tensor(input_ids)
                    and input_ids.eq(token_id).sum()
                    != input_ids[: self.max_seq_len].eq(token_id).sum()
                ):
                    raise ValueError(
                        f"context truncation would detach {values_name} from "
                        "visual placeholder tokens"
                    )
            for name in _SEQUENCE_KEYS:
                value = copied.get(name)
                if torch.is_tensor(value) and value.ndim >= 1:
                    copied[name] = value[: self.max_seq_len]
            clipped_samples.append(copied)
        batch = (
            self.base_collator(clipped_samples)
            if self.base_collator is not None
            else _dynamic_collate(
                clipped_samples, self.pad_token_id, self.ignore_index
            )
        )
        return truncate_batch_to_context(
            batch,
            self.max_seq_len,
            ignore_index=self.ignore_index,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
        )[0]


__all__ = ["ProgressiveContextCollator", "truncate_batch_to_context"]
