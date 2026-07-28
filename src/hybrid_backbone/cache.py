"""Hybrid KDA/MLA backbone components and cache structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from src.kda import KDAState
from src.mla import MLACache


AttentionType = Literal["kda", "gated_mla"]


def state_elements(state: KDAState | MLACache) -> int:
    """Count tensor elements stored by one attention-layer cache."""
    if isinstance(state, MLACache):
        return state.latent_kv.numel()
    return sum(
        tensor.numel()
        for tensor in (
            state.recurrent_state,
            state.q_conv_state.buffer,
            state.k_conv_state.buffer,
            state.v_conv_state.buffer,
        )
    )


@dataclass
class HybridLayerCache:
    """Pair one layer's attention type with its concrete recurrent cache."""

    attention_type: AttentionType
    state: KDAState | MLACache

    def __post_init__(self) -> None:
        if self.attention_type not in ("kda", "gated_mla"):
            raise ValueError("unsupported cache attention_type")
        expected = KDAState if self.attention_type == "kda" else MLACache
        if not isinstance(self.state, expected):
            raise TypeError(
                f"{self.attention_type} cache requires {expected.__name__}"
            )

    @property
    def sequence_offsets(self) -> torch.Tensor:
        return self.state.sequence_offset

    @property
    def num_elements(self) -> int:
        return state_elements(self.state)

    def clone(self) -> "HybridLayerCache":
        return HybridLayerCache(self.attention_type, self.state.clone())

    def reorder(self, indices: torch.Tensor) -> "HybridLayerCache":
        return HybridLayerCache(
            self.attention_type, self.state.reorder(indices)
        )


@dataclass
class HybridBackboneCache:
    """Synchronized collection of KDA and MLA caches for incremental decoding."""

    layer_caches: tuple[HybridLayerCache, ...]
    sequence_length: int

    def __post_init__(self) -> None:
        self.layer_caches = tuple(self.layer_caches)
        if not self.layer_caches:
            raise ValueError("layer_caches must not be empty")
        if self.sequence_length < 0:
            raise ValueError("sequence_length must be >= 0")
        reference = self.layer_caches[0].sequence_offsets
        for layer_cache in self.layer_caches[1:]:
            if not torch.equal(layer_cache.sequence_offsets, reference):
                raise ValueError("all hybrid cache offsets must be synchronized")
        expected_length = int(reference.max().item()) if reference.numel() else 0
        if self.sequence_length != expected_length:
            raise ValueError(
                "sequence_length must equal the largest per-sample offset"
            )

    @property
    def sequence_lengths(self) -> torch.Tensor:
        return self.layer_caches[0].sequence_offsets

    @property
    def kda_elements(self) -> int:
        return sum(
            cache.num_elements
            for cache in self.layer_caches
            if cache.attention_type == "kda"
        )

    @property
    def mla_elements(self) -> int:
        return sum(
            cache.num_elements
            for cache in self.layer_caches
            if cache.attention_type == "gated_mla"
        )

    @property
    def total_elements(self) -> int:
        return self.kda_elements + self.mla_elements

    def clone(self) -> "HybridBackboneCache":
        return HybridBackboneCache(
            tuple(cache.clone() for cache in self.layer_caches),
            self.sequence_length,
        )

    def reorder(self, indices: torch.Tensor) -> "HybridBackboneCache":
        reordered = tuple(
            cache.reorder(indices) for cache in self.layer_caches
        )
        lengths = reordered[0].sequence_offsets
        return HybridBackboneCache(
            reordered,
            int(lengths.max().item()) if lengths.numel() else 0,
        )
