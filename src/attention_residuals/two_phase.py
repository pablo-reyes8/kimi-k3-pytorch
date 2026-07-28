from __future__ import annotations

import torch

from .online_softmax import depth_softmax_stats
from .outputs import DepthSoftmaxStats
from .site import AttentionResidualSite
from .utils import accumulation_dtype, validate_sources


def precompute_inter_block_stats(
    sources: torch.Tensor,
    sites: tuple[AttentionResidualSite, ...],
) -> dict[int, DepthSoftmaxStats]:
    """Vectorize the fixed completed-block scan for all sites in one block."""
    if not sites:
        raise ValueError("sites must not be empty")
    d_model = sites[0].d_model
    validate_sources(sources, d_model)
    if any(site.d_model != d_model for site in sites):
        raise ValueError("all sites must share d_model")
    eps = sites[0].key_norm.eps
    if any(site.key_norm.eps != eps for site in sites):
        raise ValueError("two-phase sites must share RMSNorm epsilon")
    logits_in_fp32 = sites[0].logits_in_fp32
    weighted_sum_in_fp32 = sites[0].weighted_sum_in_fp32
    if any(site.logits_in_fp32 != logits_in_fp32 for site in sites):
        raise ValueError("two-phase sites must share the logits dtype policy")
    if any(
        site.weighted_sum_in_fp32 != weighted_sum_in_fp32 for site in sites
    ):
        raise ValueError(
            "two-phase sites must share the weighted-sum dtype policy"
        )
    logit_dtype = accumulation_dtype(sources.dtype, logits_in_fp32)
    sum_dtype = accumulation_dtype(sources.dtype, weighted_sum_in_fp32)
    norm_dtype = (
        torch.float32
        if sources.dtype in (torch.float16, torch.bfloat16)
        else sources.dtype
    )
    norm_inputs = sources.to(norm_dtype)
    base_keys = norm_inputs * torch.rsqrt(
        norm_inputs.square().mean(dim=-1, keepdim=True) + eps
    )
    base_keys = base_keys.to(sources.dtype)
    queries = torch.stack(
        [site.pseudo_query.to(logit_dtype) for site in sites]
    )
    norm_weights = torch.stack(
        [site.key_norm.weight.to(sources.dtype) for site in sites]
    )
    keys = (
        base_keys.unsqueeze(0)
        * norm_weights[:, None, None, None, :]
    ).to(logit_dtype)
    logits = torch.einsum(
        "pd,pbtsd->pbts", queries, keys
    )
    result = {}
    for index, site in enumerate(sites):
        result[site.metadata.site_index] = depth_softmax_stats(
            logits[index],
            sources,
            weighted_sum_dtype=sum_dtype,
        )
    return result


def score_single_partial(
    partial: torch.Tensor, site: AttentionResidualSite
) -> torch.Tensor:
    if partial.ndim != 3 or partial.shape[-1] != site.d_model:
        raise ValueError("partial source must have shape [B,T,D]")
    keys = site.key_norm(partial)
    compute_dtype = accumulation_dtype(
        partial.dtype, site.logits_in_fp32
    )
    return torch.einsum(
        "d,btd->bt",
        site.pseudo_query.to(compute_dtype),
        keys.to(compute_dtype),
    )
