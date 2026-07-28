"""Attention-residual components for mixing hidden-state streams across model depth."""

from __future__ import annotations

import math

import torch

from .metadata import DepthSiteMetadata
from .outputs import DepthAttentionStats


def build_depth_attention_stats(
    weights: torch.Tensor,
    metadata: DepthSiteMetadata,
    *,
    mode: str,
    number_of_completed_blocks: int = 0,
    has_current_partial: bool = False,
    retain_mean_weights: bool = False,
) -> DepthAttentionStats:
    probabilities = weights.float()
    source_count = probabilities.shape[-1]
    entropy = -(
        probabilities
        * probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
    ).sum(dim=-1).mean()
    normalized = (
        entropy / math.log(source_count)
        if source_count > 1
        else entropy.new_zeros(())
    )
    mean_weights = probabilities.mean(dim=(0, 1))
    dominant_index = mean_weights.argmax()
    source_indices = torch.arange(
        source_count, dtype=torch.float32, device=weights.device
    )
    weighted_index = (mean_weights * source_indices).sum()
    destination_index = torch.tensor(
        float(metadata.site_index), device=weights.device
    )
    completed_total = None
    partial_weight = None
    if mode == "block":
        completed_end = 1 + number_of_completed_blocks
        completed_total = mean_weights[1:completed_end].sum()
        partial_weight = (
            mean_weights[-1] if has_current_partial else mean_weights.new_zeros(())
        )
    return DepthAttentionStats(
        metadata=metadata,
        source_count=source_count,
        weight_entropy=entropy,
        normalized_entropy=normalized,
        max_weight=mean_weights.max(),
        min_weight=mean_weights.min(),
        embedding_weight=mean_weights[0],
        most_recent_source_weight=mean_weights[-1],
        dominant_source_index=dominant_index,
        dominant_source_weight=mean_weights[dominant_index],
        mean_weighted_source_index=weighted_index,
        mean_retrieval_distance=(destination_index - weighted_index).abs(),
        completed_block_weight_total=completed_total,
        current_partial_weight=partial_weight,
        current_depth_block_index=metadata.depth_block_index,
        number_of_completed_blocks=(
            number_of_completed_blocks if mode == "block" else None
        ),
        mean_weights=mean_weights if retain_mean_weights else None,
    )


def padded_weight_matrix(
    stats: tuple[DepthAttentionStats, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not stats:
        raise ValueError("stats must not be empty")
    rows = [item.mean_weights for item in stats]
    if any(row is None for row in rows):
        raise ValueError("mean weights were not retained")
    max_sources = max(row.numel() for row in rows)
    matrix = rows[0].new_zeros(len(rows), max_sources)
    mask = torch.zeros(
        len(rows), max_sources, dtype=torch.bool, device=matrix.device
    )
    for index, row in enumerate(rows):
        matrix[index, : row.numel()] = row
        mask[index, : row.numel()] = True
    return matrix, mask
