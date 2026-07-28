"""Stable LatentMoE routing, expert dispatch, and load-balancing components."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QuantileBalanceUpdate:
    """Candidate router-bias update produced by quantile balancing."""

    next_bias: torch.Tensor
    target_quantile: float
    quantiles: torch.Tensor
    margin_min: torch.Tensor
    margin_max: torch.Tensor
    num_tokens: int


@dataclass
class RouterOutput:
    """Selected experts, routing weights, logits, and balancing margins."""

    selected_experts: torch.Tensor
    selected_raw_scores: torch.Tensor
    selected_weights: torch.Tensor
    expert_load: torch.Tensor
    cutoff_k_plus_one: torch.Tensor | None = None
    routing_bias_before: torch.Tensor | None = None
    routing_bias_after: torch.Tensor | None = None
    raw_logits: torch.Tensor | None = None
    raw_scores: torch.Tensor | None = None
    biased_scores: torch.Tensor | None = None


@dataclass
class MoEDiagnostics:
    """Load, routing, and expert-utilization diagnostics for one MoE call."""

    num_tokens: int
    num_assignments: int
    shared_token_evaluations: int
    expert_load: torch.Tensor
    expert_load_fraction: torch.Tensor
    mean_load: torch.Tensor
    std_load: torch.Tensor
    min_load: torch.Tensor
    max_load: torch.Tensor
    coefficient_of_variation: torch.Tensor
    zero_load_experts: torch.Tensor
    selected_weight_mean: torch.Tensor
    selected_weight_min: torch.Tensor
    selected_weight_max: torch.Tensor
    routing_entropy_over_selected: torch.Tensor
    routing_bias_mean: torch.Tensor
    routing_bias_std: torch.Tensor
    routing_bias_min: torch.Tensor
    routing_bias_max: torch.Tensor
    qb_update_rms: torch.Tensor
    qb_quantile_error_estimate: torch.Tensor
    shared_output_rms: torch.Tensor
    routed_output_rms: torch.Tensor
    routed_aggregate_rms_before_norm: torch.Tensor
    routed_aggregate_rms_after_norm: torch.Tensor
    shared_to_total_ratio: torch.Tensor
    routed_to_total_ratio: torch.Tensor
    shared_routed_cosine: torch.Tensor
    output_rms: torch.Tensor


@dataclass
class StableLatentMoEOutput:
    """Combined shared/routed expert output and optional router diagnostics."""

    hidden_states: torch.Tensor
    shared_output: torch.Tensor | None = None
    routed_output: torch.Tensor | None = None
    router_output: RouterOutput | None = None
    diagnostics: MoEDiagnostics | None = None
