"""Stable LatentMoE routing, expert dispatch, and load-balancing components."""

from __future__ import annotations

import torch

from .outputs import MoEDiagnostics, RouterOutput
from .utils import rms


def build_moe_diagnostics(
    router: RouterOutput,
    num_shared_experts: int,
    shared_output: torch.Tensor,
    routed_output: torch.Tensor,
    aggregate: torch.Tensor,
    normalized_aggregate: torch.Tensor,
) -> MoEDiagnostics:
    loads = router.expert_load
    tokens, top_k = router.selected_experts.shape
    assignments = tokens * top_k
    loads_float = loads.float()
    mean_load = loads_float.mean()
    std_load = loads_float.std(unbiased=False)
    weights = router.selected_weights.float()
    entropy = -(
        weights
        * weights.clamp_min(torch.finfo(torch.float32).tiny).log()
    ).sum(dim=-1).mean()
    bias = router.routing_bias_before.float()
    return MoEDiagnostics(
        num_tokens=tokens,
        num_assignments=assignments,
        shared_token_evaluations=num_shared_experts * tokens,
        expert_load=loads,
        expert_load_fraction=loads_float / max(assignments, 1),
        mean_load=mean_load,
        std_load=std_load,
        min_load=loads_float.min(),
        max_load=loads_float.max(),
        coefficient_of_variation=std_load / mean_load.clamp_min(1e-12),
        zero_load_experts=(loads == 0).sum(),
        selected_weight_mean=weights.mean(),
        selected_weight_min=weights.min(),
        selected_weight_max=weights.max(),
        routing_entropy_over_selected=entropy,
        routing_bias_mean=bias.mean(),
        routing_bias_std=bias.std(unbiased=False),
        routing_bias_min=bias.min(),
        routing_bias_max=bias.max(),
        shared_output_rms=rms(shared_output),
        routed_output_rms=rms(routed_output),
        routed_aggregate_rms_before_norm=rms(aggregate),
        routed_aggregate_rms_after_norm=rms(normalized_aggregate),
    )
