"""Small exactness audit for full forward versus cached Kimi decode."""

from __future__ import annotations

import torch

from .cache import cache_summary
from .decode import decode_one_token
from .prefill import prefill_prompt


@torch.inference_mode()
def compare_full_vs_cached_logits(
    model,
    input_ids: torch.Tensor,
    *,
    split: int | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict:
    if input_ids.ndim != 2 or input_ids.shape[1] < 2:
        raise ValueError("audit requires input_ids [B,T] with T >= 2")
    split = int(split or max(1, input_ids.shape[1] // 2))
    if not 1 <= split < input_ids.shape[1]:
        raise ValueError("split must lie inside the sequence")
    device = next(model.parameters()).device
    ids = input_ids.to(device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    was_training = model.training
    model.eval()
    try:
        full = model(ids, mask, return_dict=True).logits
        prefix = prefill_prompt(model, ids[:, :split], mask[:, :split])
        pieces = [prefix.logits]
        cache = prefix.cache
        for index in range(split, ids.shape[1]):
            decoded = decode_one_token(
                model,
                ids[:, index : index + 1],
                cache,
                mask[:, index : index + 1],
            )
            pieces.append(decoded.logits)
            cache = decoded.cache
        incremental = torch.cat(pieces, dim=1)
        difference = (incremental - full).abs().max().item()
        return {
            "allclose": bool(
                torch.allclose(
                    incremental, full, atol=atol, rtol=rtol
                )
            ),
            "max_abs_diff": float(difference),
            "atol": float(atol),
            "rtol": float(rtol),
            "cache_stats": cache_summary(cache),
        }
    finally:
        if was_training:
            model.train()


__all__ = ["compare_full_vs_cached_logits"]
