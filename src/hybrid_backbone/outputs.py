from __future__ import annotations

from dataclasses import dataclass

import torch

from .cache import HybridBackboneCache, HybridLayerCache


@dataclass
class HybridLayerOutput:
    hidden_states: torch.Tensor
    cache: HybridLayerCache | None = None
    diagnostics: dict[str, object] | None = None


@dataclass
class HybridGroupOutput:
    hidden_states: torch.Tensor
    layer_caches: tuple[HybridLayerCache, ...] | None = None
    hidden_states_by_layer: tuple[torch.Tensor, ...] | None = None
    diagnostics: tuple[dict[str, object], ...] | None = None


@dataclass
class HybridBackboneOutput:
    last_hidden_state: torch.Tensor
    cache: HybridBackboneCache | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    diagnostics: dict[str, object] | None = None
