"""Attention Residual depth-mixture reductions."""

from __future__ import annotations

import math

import torch

from .reducers import scalar


def compute_attnres_metrics(depth_output, prefix: str = "attnres") -> dict[str, float]:
    if depth_output is None:
        return {}
    stats = tuple(depth_output.site_stats) + (
        depth_output.final_output_stats,
    )
    if not stats:
        return {}

    def mean(attribute: str, default=0.0):
        values = [
            scalar(getattr(item, attribute))
            for item in stats
            if getattr(item, attribute, None) is not None
        ]
        return sum(values) / len(values) if values else default

    normalized_entropy = mean("normalized_entropy")
    effective_sources = [
        math.exp(scalar(item.weight_entropy)) for item in stats
    ]
    mean_weights = [
        item.mean_weights.detach().float()
        for item in stats
        if item.mean_weights is not None
    ]
    cv = 0.0
    oldest = 0.0
    if mean_weights:
        coefficients = []
        oldest_values = []
        weighted_stats = [
            item for item in stats if item.mean_weights is not None
        ]
        for item, weights in zip(weighted_stats, mean_weights):
            coefficients.append(
                scalar(weights.std(unbiased=False) / weights.mean().clamp_min(1e-12))
            )
            # Source 0 is the embedding; source 1 is the oldest block.
            completed = getattr(
                item,
                "number_of_completed_blocks",
                max(weights.numel() - 1, 0),
            )
            oldest_values.append(
                scalar(weights[1])
                if completed is not None and completed > 0
                else 0.0
            )
        cv = sum(coefficients) / len(coefficients)
        oldest = sum(oldest_values) / len(oldest_values)
    return {
        f"{prefix}/source_entropy_normalized": normalized_entropy,
        f"{prefix}/effective_num_sources": sum(effective_sources)
        / len(effective_sources),
        f"{prefix}/max_source_weight": mean("max_weight"),
        f"{prefix}/embedding_source_weight": mean("embedding_weight"),
        f"{prefix}/current_block_partial_weight": mean(
            "current_partial_weight"
        ),
        f"{prefix}/oldest_block_weight": oldest,
        f"{prefix}/top1_source_index_mean": mean(
            "dominant_source_index"
        ),
        f"{prefix}/source_weight_cv": cv,
    }
