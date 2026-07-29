"""Inspection helpers for Kimi's native heterogeneous backbone cache."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import torch

from src.hybrid_backbone import HybridBackboneCache


def _unique_tensors(value: Any, seen_objects: set[int]):
    if torch.is_tensor(value):
        if id(value) not in seen_objects:
            seen_objects.add(id(value))
            yield value
        return
    if is_dataclass(value):
        for field in fields(value):
            yield from _unique_tensors(
                getattr(value, field.name), seen_objects
            )
    elif isinstance(value, dict):
        for item in value.values():
            yield from _unique_tensors(item, seen_objects)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _unique_tensors(item, seen_objects)


def cache_memory_bytes(cache: HybridBackboneCache | None) -> int:
    if cache is None:
        return 0
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in _unique_tensors(cache, set())
    )


def cache_summary(cache: HybridBackboneCache | None) -> dict[str, Any]:
    if cache is None:
        return {}
    if not isinstance(cache, HybridBackboneCache):
        raise TypeError("expected Kimi HybridBackboneCache")
    return {
        "sequence_length": int(cache.sequence_length),
        "sequence_lengths": cache.sequence_lengths.detach().cpu().tolist(),
        "num_layers": len(cache.layer_caches),
        "num_kda_layers": sum(
            layer.attention_type == "kda" for layer in cache.layer_caches
        ),
        "num_mla_layers": sum(
            layer.attention_type == "gated_mla"
            for layer in cache.layer_caches
        ),
        "kda_elements": int(cache.kda_elements),
        "mla_elements": int(cache.mla_elements),
        "total_elements": int(cache.total_elements),
        "memory_bytes": cache_memory_bytes(cache),
    }


def validate_kimi_cache(model, cache: HybridBackboneCache) -> None:
    if not isinstance(cache, HybridBackboneCache):
        raise TypeError("model did not return HybridBackboneCache")
    # ``gated_mla_final`` is a topology label used by the backbone for its
    # final global layer.  The runtime cache deliberately stores the
    # underlying state kind, which is the same ``gated_mla`` state.
    expected = tuple(
        "gated_mla" if name == "gated_mla_final" else name
        for name in model.backbone.attention_types
    )
    actual = tuple(layer.attention_type for layer in cache.layer_caches)
    if actual != expected:
        raise ValueError("cache attention layout does not match Kimi backbone")
    if cache.sequence_lengths.numel() == 0:
        raise ValueError("cache must contain at least one sequence")


__all__ = [
    "cache_memory_bytes",
    "cache_summary",
    "validate_kimi_cache",
]
