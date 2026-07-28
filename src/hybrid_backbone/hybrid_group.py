from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn

from .attention_layer import HybridAttentionLayer
from .cache import HybridLayerCache
from .outputs import HybridGroupOutput


class HybridAttentionGroup(nn.Module):
    def __init__(
        self,
        layers: Sequence[HybridAttentionLayer],
        *,
        group_index: int,
    ):
        super().__init__()
        if not layers:
            raise ValueError("hybrid group must contain at least one layer")
        self.group_index = group_index
        self.layers = nn.ModuleList(layers)

    @property
    def attention_types(self) -> tuple[str, ...]:
        return tuple(layer.attention_type for layer in self.layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: Sequence[HybridLayerCache] | None = None,
        *,
        use_cache: bool = False,
        mode: Literal["full", "prefill", "decode"] = "full",
        output_hidden_states: bool = False,
        output_diagnostics: bool = False,
    ) -> HybridGroupOutput:
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError("group cache count must match group layer count")
        next_caches = []
        layer_states = []
        diagnostics = []
        output = hidden_states
        for index, layer in enumerate(self.layers):
            layer_output = layer(
                output,
                attention_mask,
                None if cache is None else cache[index],
                use_cache=use_cache,
                mode=mode,
                output_diagnostics=output_diagnostics,
            )
            output = layer_output.hidden_states
            if use_cache:
                next_caches.append(layer_output.cache)
            if output_hidden_states:
                layer_states.append(output)
            if output_diagnostics:
                diagnostics.append(layer_output.diagnostics)
        return HybridGroupOutput(
            hidden_states=output,
            layer_caches=tuple(next_caches) if use_cache else None,
            hidden_states_by_layer=(
                tuple(layer_states) if output_hidden_states else None
            ),
            diagnostics=tuple(diagnostics) if output_diagnostics else None,
        )
