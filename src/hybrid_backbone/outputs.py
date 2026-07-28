"""Hybrid KDA/MLA backbone components and cache structures."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.attention_residuals import AttentionResidualBackboneOutput

from .cache import HybridBackboneCache, HybridLayerCache


@dataclass
class HybridLayerOutput:
    """Output of one hybrid attention and feed-forward layer."""

    hidden_states: torch.Tensor
    cache: HybridLayerCache | None = None
    diagnostics: dict[str, object] | None = None
    pre_attention_state: torch.Tensor | None = None
    attention_output: torch.Tensor | None = None
    pre_ffn_state: torch.Tensor | None = None
    ffn_output: torch.Tensor | None = None


@dataclass
class HybridGroupOutput:
    """Output of one configured KDA/MLA layer group."""

    hidden_states: torch.Tensor
    layer_caches: tuple[HybridLayerCache, ...] | None = None
    hidden_states_by_layer: tuple[torch.Tensor, ...] | None = None
    diagnostics: tuple[dict[str, object], ...] | None = None


@dataclass
class HybridBackboneOutput:
    """Final hidden state, cache, and diagnostics from the hybrid backbone."""

    last_hidden_state: torch.Tensor
    cache: HybridBackboneCache | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    hidden_state_trace: "BackboneHiddenStateTrace | None" = None
    depth_outputs: AttentionResidualBackboneOutput | None = None
    diagnostics: dict[str, object] | None = None


@dataclass
class BackboneHiddenStateTrace:
    """Named hidden states captured while executing the backbone."""

    embedding: torch.Tensor
    pre_attention: tuple[torch.Tensor, ...]
    attention_outputs: tuple[torch.Tensor, ...]
    pre_ffn: tuple[torch.Tensor, ...]
    ffn_outputs: tuple[torch.Tensor, ...]
    final_mixed: torch.Tensor
