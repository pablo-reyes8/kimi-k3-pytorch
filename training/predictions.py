"""Qualitative next-token previews for epoch-level inspection."""

from __future__ import annotations

from typing import Any, Callable

import torch

from data.batch import normalize_lm_batch

from .autocast import autocast_ctx, move_batch_to_device
from .model_call import call_model


def decode_token_ids(
    token_ids,
    *,
    tokenizer=None,
    id_to_text: Callable[[list[int]], str] | None = None,
) -> str:
    if torch.is_tensor(token_ids):
        token_ids = token_ids.detach().cpu().tolist()
    values = [int(value) for value in token_ids]
    if id_to_text is not None:
        return str(id_to_text(values))
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        return str(tokenizer.decode(values))
    return " ".join(str(value) for value in values)


def _last_valid_index(batch: dict[str, Any], sample_index: int) -> int:
    attention_mask = batch.get("attention_mask")
    if torch.is_tensor(attention_mask):
        positions = attention_mask[sample_index].bool().nonzero().flatten()
        if positions.numel() == 0:
            raise ValueError("preview sample contains no valid tokens")
        return int(positions[-1].item())
    return int(batch["input_ids"].shape[1] - 1)


@torch.no_grad()
def next_token_preview(
    model,
    raw_batch,
    *,
    device: str | torch.device = "cpu",
    amp_enabled: bool = False,
    amp_dtype: str = "bf16",
    use_mtp: bool | None = None,
    sample_index: int = 0,
    max_tokens: int = 32,
    tokenizer=None,
    id_to_text: Callable[[list[int]], str] | None = None,
) -> dict[str, Any]:
    """Return teacher-forced predictions and the final next-token decision."""

    device = torch.device(device)
    batch = move_batch_to_device(normalize_lm_batch(raw_batch), device)
    batch_size = int(batch["input_ids"].shape[0])
    if not 0 <= sample_index < batch_size:
        raise IndexError("preview sample_index is outside the batch")

    was_training = model.training
    model.eval()
    try:
        with autocast_ctx(
            device, enabled=amp_enabled, amp_dtype=amp_dtype
        ):
            output = call_model(model, batch, use_mtp=use_mtp)
        logits = (
            output["logits"]
            if isinstance(output, dict)
            else getattr(output, "logits", None)
        )
        if logits is None:
            raise ValueError("model output must contain logits for preview")
        predictions = logits.argmax(dim=-1)
        last = _last_valid_index(batch, sample_index)
        start = max(0, last + 1 - max_tokens)
        labels = batch.get("labels")
        input_slice = batch["input_ids"][sample_index, start : last + 1]
        prediction_slice = predictions[sample_index, start : last + 1]
        reference_slice = (
            None
            if not torch.is_tensor(labels)
            else labels[sample_index, start : last + 1]
        )
        return {
            "sample_index": int(sample_index),
            "position": last,
            "input_ids": input_slice.detach().cpu().tolist(),
            "predicted_ids": prediction_slice.detach().cpu().tolist(),
            "reference_ids": (
                None
                if reference_slice is None
                else reference_slice.detach().cpu().tolist()
            ),
            "next_token_id": int(predictions[sample_index, last].item()),
            "reference_next_token_id": (
                None
                if reference_slice is None
                else int(labels[sample_index, last].item())
            ),
            "context": decode_token_ids(
                input_slice, tokenizer=tokenizer, id_to_text=id_to_text
            ),
            "prediction": decode_token_ids(
                prediction_slice, tokenizer=tokenizer, id_to_text=id_to_text
            ),
            "reference": (
                None
                if reference_slice is None
                else decode_token_ids(
                    reference_slice,
                    tokenizer=tokenizer,
                    id_to_text=id_to_text,
                )
            ),
        }
    finally:
        model.train(was_training)


def print_next_token_preview(
    preview: dict[str, Any],
    *,
    title: str = "Kimi K3 next-token preview",
) -> None:
    print(f"\n{title}")
    print(f"  context    : {preview['context']!r}")
    if preview.get("reference") is not None:
        print(f"  reference  : {preview['reference']!r}")
    print(f"  prediction : {preview['prediction']!r}")
    print(
        "  next token : "
        f"pred={preview['next_token_id']} "
        f"ref={preview.get('reference_next_token_id')}"
    )
