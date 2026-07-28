from __future__ import annotations

import torch

from .outputs import DepthSoftmaxStats


def depth_softmax_stats(
    logits: torch.Tensor,
    values: torch.Tensor,
    *,
    weighted_sum_dtype: torch.dtype | None = None,
) -> DepthSoftmaxStats:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [B,T,S]")
    if values.ndim != 4 or values.shape[:3] != logits.shape:
        raise ValueError("values must have shape [B,T,S,D] matching logits")
    if logits.shape[-1] == 0:
        raise ValueError("stats require at least one source")
    maximum = logits.max(dim=-1).values
    exponentials = torch.exp(logits - maximum[..., None])
    exp_sum = exponentials.sum(dim=-1)
    sum_dtype = weighted_sum_dtype or exponentials.dtype
    weighted_sum = torch.einsum(
        "bts,btsd->btd",
        exponentials.to(sum_dtype),
        values.to(sum_dtype),
    )
    return DepthSoftmaxStats(maximum, exp_sum, weighted_sum, logits)


def single_source_stats(
    logit: torch.Tensor,
    value: torch.Tensor,
    *,
    weighted_sum_dtype: torch.dtype | None = None,
) -> DepthSoftmaxStats:
    if logit.ndim != 2 or value.ndim != 3 or value.shape[:2] != logit.shape:
        raise ValueError("single source expects logit [B,T] and value [B,T,D]")
    sum_dtype = weighted_sum_dtype or logit.dtype
    return DepthSoftmaxStats(
        max_logit=logit,
        exp_sum=torch.ones_like(logit),
        weighted_sum=value.to(sum_dtype),
        logits=logit[..., None],
    )


def merge_depth_softmax_stats(
    first: DepthSoftmaxStats,
    second: DepthSoftmaxStats,
) -> DepthSoftmaxStats:
    if first.max_logit.shape != second.max_logit.shape:
        raise ValueError("softmax stats must have matching [B,T] shapes")
    maximum = torch.maximum(first.max_logit, second.max_logit)
    first_scale = torch.exp(first.max_logit - maximum)
    second_scale = torch.exp(second.max_logit - maximum)
    exp_sum = (
        first_scale * first.exp_sum + second_scale * second.exp_sum
    )
    weighted_sum = (
        first_scale[..., None] * first.weighted_sum
        + second_scale[..., None] * second.weighted_sum
    )
    logits = (
        None
        if first.logits is None or second.logits is None
        else torch.cat((first.logits, second.logits), dim=-1)
    )
    return DepthSoftmaxStats(maximum, exp_sum, weighted_sum, logits)


def normalize_depth_softmax_stats(
    stats: DepthSoftmaxStats, output_dtype: torch.dtype
) -> torch.Tensor:
    return (stats.weighted_sum / stats.exp_sum[..., None]).to(output_dtype)


def weights_from_stats(stats: DepthSoftmaxStats) -> torch.Tensor:
    if stats.logits is None:
        raise ValueError("logits were not retained in these stats")
    return torch.exp(stats.logits - stats.max_logit[..., None]) / (
        stats.exp_sum[..., None]
    )
