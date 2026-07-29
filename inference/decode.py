"""One-token decode through KDA/MLA native cached state."""

from __future__ import annotations

import torch

from src.hybrid_backbone import HybridBackboneCache

from .cache import validate_kimi_cache


@torch.inference_mode()
def decode_one_token(
    model,
    token_ids: torch.Tensor,
    cache: HybridBackboneCache,
    attention_mask: torch.Tensor | None = None,
    *,
    validate_cache: bool = True,
):
    if token_ids.ndim != 2 or token_ids.shape[1] != 1:
        raise ValueError("decode token_ids must have shape [B,1]")
    if attention_mask is not None and (
        attention_mask.shape != token_ids.shape
        or attention_mask.dtype != torch.bool
    ):
        raise ValueError(
            "decode attention_mask must be boolean with shape [B,1]"
        )
    output = model.decode_step(
        token_ids,
        cache,
        attention_mask,
        return_dict=True,
    )
    if output.cache is None:
        raise RuntimeError("Kimi decode did not return an updated cache")
    if validate_cache:
        validate_kimi_cache(model, output.cache)
    return output


__all__ = ["decode_one_token"]
