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
    bias_after = router.routing_bias_after
    qb_update_rms = (
        bias.new_zeros(())
        if bias_after is None
        else rms(bias_after.float() - bias)
    )
    target_fraction = 1.0 / max(loads.numel(), 1)
    load_fraction = loads_float / max(assignments, 1)
    shared_rms = rms(shared_output)
    routed_rms = rms(routed_output)
    total = shared_output.float() + routed_output.float()
    total_rms = rms(total)
    flat_shared = shared_output.float().reshape(-1)
    flat_routed = routed_output.float().reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(
        flat_shared, flat_routed, dim=0, eps=1e-12
    )
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
        qb_update_rms=qb_update_rms,
        qb_quantile_error_estimate=(
            load_fraction - target_fraction
        ).abs().mean(),
        shared_output_rms=shared_rms,
        routed_output_rms=routed_rms,
        routed_aggregate_rms_before_norm=rms(aggregate),
        routed_aggregate_rms_after_norm=rms(normalized_aggregate),
        shared_to_total_ratio=shared_rms / total_rms.clamp_min(1e-12),
        routed_to_total_ratio=routed_rms / total_rms.clamp_min(1e-12),
        shared_routed_cosine=cosine,
        output_rms=total_rms,
    )
