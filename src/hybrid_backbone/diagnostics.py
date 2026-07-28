from __future__ import annotations

import torch
import torch.nn as nn

from .cache import HybridBackboneCache
from .utils import count_parameters


def parameter_counts(
    groups: nn.ModuleList,
    final_global_layer: nn.Module,
    final_norm: nn.Module,
) -> dict[str, int]:
    layers = [
        layer
        for group in groups
        for layer in group.layers
    ] + [final_global_layer]
    return {
        "kda": sum(
            count_parameters(layer.attention)
            for layer in layers
            if layer.attention_type == "kda"
        ),
        "gated_mla": sum(
            count_parameters(layer.attention)
            for layer in layers
            if layer.attention_type == "gated_mla"
        ),
        "ffn": sum(
            count_parameters(layer.ffn)
            for layer in layers
            if layer.ffn is not None
        ),
        "norms": sum(
            count_parameters(layer.attention_norm)
            + (
                count_parameters(layer.ffn_norm)
                if layer.ffn_norm is not None
                else 0
            )
            for layer in layers
        )
        + count_parameters(final_norm),
    }


def build_backbone_diagnostics(
    layer_diagnostics: tuple[dict[str, object], ...],
    groups: nn.ModuleList,
    final_global_layer: nn.Module,
    final_norm: nn.Module,
    cache: HybridBackboneCache | None,
    device: torch.device,
) -> dict[str, object]:
    counts = parameter_counts(groups, final_global_layer, final_norm)
    counts["total"] = sum(counts.values())
    return {
        "layers": layer_diagnostics,
        "num_kda_layers": sum(
            item["attention_type"] == "kda" for item in layer_diagnostics
        ),
        "num_mla_layers": sum(
            item["attention_type"] in ("gated_mla", "gated_mla_final")
            for item in layer_diagnostics
        ),
        "num_parameters_by_component": counts,
        "kda_cache_elements": torch.tensor(
            0 if cache is None else cache.kda_elements,
            dtype=torch.long,
            device=device,
        ),
        "mla_cache_elements": torch.tensor(
            0 if cache is None else cache.mla_elements,
            dtype=torch.long,
            device=device,
        ),
        "total_cache_elements": torch.tensor(
            0 if cache is None else cache.total_elements,
            dtype=torch.long,
            device=device,
        ),
    }
