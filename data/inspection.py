"""Inspection helpers for LM dataloaders and tensor batches."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from data.batch import normalize_lm_batch


def summarize_tensor(tensor: torch.Tensor) -> Dict[str, Any]:
    tensor = tensor.detach()
    summary: Dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
    }

    if tensor.numel() == 0:
        return summary

    numeric = tensor.float()
    summary.update(
        {
            "min": float(numeric.min().item()),
            "max": float(numeric.max().item()),
            "mean": float(numeric.mean().item()),
        }
    )
    return summary


def summarize_lm_batch(batch: Any) -> Dict[str, Any]:
    batch = normalize_lm_batch(batch)
    return {
        key: summarize_tensor(value)
        for key, value in batch.items()
        if torch.is_tensor(value)
    }


def decode_preview(ids: torch.Tensor, tokenizer: Any, *, max_tokens: int = 48) -> str:
    ids_list = [int(x) for x in ids.detach().cpu().flatten()[:max_tokens].tolist()]

    if hasattr(tokenizer, "decode"):
        try:
            return tokenizer.decode(ids_list)
        except Exception:
            pass

    if hasattr(tokenizer, "idx_to_token"):
        return " ".join(tokenizer.idx_to_token.get(i, f"<{i}>") for i in ids_list)

    return " ".join(str(i) for i in ids_list)


def inspect_lm_dataloader(
    dataloader,
    *,
    tokenizer: Optional[Any] = None,
    num_batches: int = 1,
    max_preview_tokens: int = 48,
) -> Dict[str, Any]:
    batches = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        batch = normalize_lm_batch(batch)
        item: Dict[str, Any] = {
            "batch_idx": int(batch_idx),
            "tensors": summarize_lm_batch(batch),
        }

        if tokenizer is not None and "input_ids" in batch:
            item["input_preview"] = decode_preview(
                batch["input_ids"][0],
                tokenizer,
                max_tokens=max_preview_tokens,
            )

        if tokenizer is not None and "labels" in batch:
            item["label_preview"] = decode_preview(
                batch["labels"][0],
                tokenizer,
                max_tokens=max_preview_tokens,
            )

        batches.append(item)

    return {
        "num_batches_inspected": len(batches),
        "batches": batches,
    }
