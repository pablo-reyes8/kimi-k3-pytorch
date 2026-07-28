"""Stable LatentMoE routing, expert dispatch, and load-balancing components."""

from __future__ import annotations

import torch

from .outputs import QuantileBalanceUpdate


class ExactQuantileBalancer:
    def __init__(self, num_experts: int, top_k: int):
        if num_experts <= 1 or not 1 <= top_k < num_experts:
            raise ValueError("exact balancing requires 1 <= k < n")
        self.num_experts = num_experts
        self.top_k = top_k
        self.target_quantile = 1.0 - top_k / num_experts

    @torch.no_grad()
    def compute_next_bias(
        self,
        raw_scores: torch.Tensor,
        biased_cutoffs: torch.Tensor,
        old_bias: torch.Tensor,
    ) -> QuantileBalanceUpdate:
        if (
            raw_scores.ndim != 2
            or raw_scores.shape[1] != self.num_experts
        ):
            raise ValueError("raw_scores must have shape [M,E]")
        if biased_cutoffs.shape != (raw_scores.shape[0],):
            raise ValueError("biased_cutoffs must have shape [M]")
        if old_bias.shape != (self.num_experts,):
            raise ValueError("old_bias must have shape [E]")
        compute_dtype = (
            torch.float64
            if raw_scores.dtype == torch.float64
            else torch.float32
        )
        margins = raw_scores.to(compute_dtype) - biased_cutoffs.to(
            compute_dtype
        )[:, None]
        quantiles = torch.quantile(
            margins,
            self.target_quantile,
            dim=0,
            interpolation="linear",
        )
        provisional = -quantiles
        next_bias = provisional - provisional.mean()
        return QuantileBalanceUpdate(
            next_bias=next_bias,
            target_quantile=self.target_quantile,
            quantiles=quantiles,
            margin_min=margins.amin(dim=0),
            margin_max=margins.amax(dim=0),
            num_tokens=raw_scores.shape[0],
        )
