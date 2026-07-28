"""Hybrid KDA/MLA backbone components and cache structures."""

from .attention_layer import HybridAttentionLayer
from .backbone import HybridAttentionBackbone
from .cache import HybridBackboneCache, HybridLayerCache
from .config import CANONICAL_ATTENTION_PATTERN, HybridBackboneConfig
from .dense_ffn import DenseKimiFFN
from .diagnostics import parameter_counts
from .hybrid_group import HybridAttentionGroup
from .outputs import (
    BackboneHiddenStateTrace,
    HybridBackboneOutput,
    HybridGroupOutput,
    HybridLayerOutput,
)

__all__ = [
    "CANONICAL_ATTENTION_PATTERN",
    "BackboneHiddenStateTrace",
    "DenseKimiFFN",
    "HybridAttentionBackbone",
    "HybridAttentionGroup",
    "HybridAttentionLayer",
    "HybridBackboneCache",
    "HybridBackboneConfig",
    "HybridBackboneOutput",
    "HybridGroupOutput",
    "HybridLayerCache",
    "HybridLayerOutput",
    "parameter_counts",
]
