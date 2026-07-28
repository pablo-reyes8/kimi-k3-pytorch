"""Attention-residual components for mixing hidden-state streams across model depth."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .metadata import DepthSiteMetadata


@dataclass
class DepthSoftmaxStats:
    """Online-softmax statistics accumulated across residual sources."""

    max_logit: torch.Tensor
    exp_sum: torch.Tensor
    weighted_sum: torch.Tensor
    logits: torch.Tensor | None = None


@dataclass
class DepthAttentionStats:
    """Diagnostics describing attention weights over model depth."""

    metadata: DepthSiteMetadata
    source_count: int
    weight_entropy: torch.Tensor
    normalized_entropy: torch.Tensor
    max_weight: torch.Tensor
    min_weight: torch.Tensor
    embedding_weight: torch.Tensor
    most_recent_source_weight: torch.Tensor
    dominant_source_index: torch.Tensor
    dominant_source_weight: torch.Tensor
    mean_weighted_source_index: torch.Tensor
    mean_retrieval_distance: torch.Tensor
    completed_block_weight_total: torch.Tensor | None = None
    current_partial_weight: torch.Tensor | None = None
    current_depth_block_index: int | None = None
    number_of_completed_blocks: int | None = None
    mean_weights: torch.Tensor | None = None


@dataclass
class AttentionResidualMixOutput:
    """Result of mixing the hidden states available at one residual site."""

    mixed_state: torch.Tensor
    weights: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    stats: DepthAttentionStats | None = None


@dataclass
class AttentionResidualBackboneOutput:
    """Backbone-level attention-residual outputs and optional diagnostics."""

    mode: str
    site_stats: tuple[DepthAttentionStats, ...]
    final_output_stats: DepthAttentionStats
    averaged_weight_matrix: torch.Tensor | None = None
    source_mask: torch.Tensor | None = None
    source_labels: tuple[tuple[str, ...], ...] | None = None
    source_tensor_count: int = 0
    source_elements: int = 0
    peak_source_count: int = 0
    num_depth_blocks: int = 0
    partial_final_block_size: int = 0
    inter_block_scan_count: int = 0
