"""Stable LatentMoE routing and branch-utility metrics."""

from __future__ import annotations

import math

from .reducers import scalar


def compute_moe_metrics(diagnostics, prefix: str = "moe") -> dict[str, float]:
    if diagnostics is None:
        return {}
    mean_load = scalar(diagnostics.mean_load)
    min_load = scalar(diagnostics.min_load)
    max_load = scalar(diagnostics.max_load)
    num_experts = int(diagnostics.expert_load.numel())
    top_k = max(
        int(diagnostics.num_assignments / max(diagnostics.num_tokens, 1)),
        1,
    )
    entropy = scalar(diagnostics.routing_entropy_over_selected)
    return {
        f"{prefix}/load_mean": mean_load,
        f"{prefix}/load_std": scalar(diagnostics.std_load),
        f"{prefix}/load_cv": scalar(diagnostics.coefficient_of_variation),
        f"{prefix}/load_max_to_mean": max_load / max(mean_load, 1e-12),
        f"{prefix}/load_min_to_mean": min_load / max(mean_load, 1e-12),
        f"{prefix}/dead_expert_fraction_batch": scalar(
            diagnostics.zero_load_experts
        )
        / max(num_experts, 1),
        f"{prefix}/router_entropy_normalized": entropy
        / (math.log(top_k) if top_k > 1 else 1.0),
        f"{prefix}/top1_share": scalar(diagnostics.selected_weight_max),
        f"{prefix}/topk_weight_entropy": entropy,
        f"{prefix}/qb_bias_mean": scalar(diagnostics.routing_bias_mean),
        f"{prefix}/qb_bias_std": scalar(diagnostics.routing_bias_std),
        f"{prefix}/qb_bias_absmax": max(
            abs(scalar(diagnostics.routing_bias_min)),
            abs(scalar(diagnostics.routing_bias_max)),
        ),
        f"{prefix}/qb_update_rms": scalar(diagnostics.qb_update_rms),
        f"{prefix}/qb_quantile_error_estimate": scalar(
            diagnostics.qb_quantile_error_estimate
        ),
        f"{prefix}/shared_output_rms": scalar(
            diagnostics.shared_output_rms
        ),
        f"{prefix}/routed_pre_norm_rms": scalar(
            diagnostics.routed_aggregate_rms_before_norm
        ),
        f"{prefix}/routed_post_norm_rms": scalar(
            diagnostics.routed_aggregate_rms_after_norm
        ),
        f"{prefix}/routed_up_projection_rms": scalar(
            diagnostics.routed_output_rms
        ),
        f"{prefix}/shared_to_total_ratio": scalar(
            diagnostics.shared_to_total_ratio
        ),
        f"{prefix}/routed_to_total_ratio": scalar(
            diagnostics.routed_to_total_ratio
        ),
        f"{prefix}/shared_routed_cosine": scalar(
            diagnostics.shared_routed_cosine
        ),
        f"{prefix}/output_rms": scalar(diagnostics.output_rms),
    }
