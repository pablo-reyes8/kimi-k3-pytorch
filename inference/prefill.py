"""Prompt prefill through Kimi's native full-prefix cache path."""

from __future__ import annotations

import torch

from .cache import validate_kimi_cache


@torch.inference_mode()
def prefill_prompt(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    validate_cache: bool = True,
    **multimodal_inputs,
):
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("prefill input_ids must have shape [B,T], T > 0")
    if attention_mask is not None:
        if (
            attention_mask.shape != input_ids.shape
            or attention_mask.dtype != torch.bool
        ):
            raise ValueError(
                "attention_mask must be boolean and align with input_ids"
            )
    output = model.prefill(
        input_ids,
        attention_mask,
        return_dict=True,
        **multimodal_inputs,
    )
    if output.cache is None:
        raise RuntimeError("Kimi prefill did not return a cache")
    if validate_cache:
        validate_kimi_cache(model, output.cache)
    return output


__all__ = ["prefill_prompt"]
