"""Stable LatentMoE routing, expert dispatch, and load-balancing components."""

from __future__ import annotations

import torch
import torch.nn as nn

from .outputs import QuantileBalanceUpdate


class HistogramQuantileBalancer(nn.Module):
    """Single-process, bounded-memory per-expert margin histograms."""

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        num_bins: int,
        min_margin: float,
        max_margin: float,
    ):
        super().__init__()
        if num_experts <= 1 or not 1 <= top_k < num_experts:
            raise ValueError("histogram balancing requires 1 <= k < n")
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2")
        if max_margin <= min_margin:
            raise ValueError("max_margin must exceed min_margin")
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_bins = num_bins
        self.min_margin = float(min_margin)
        self.max_margin = float(max_margin)
        self.target_quantile = 1.0 - top_k / num_experts
        self.register_buffer(
            "counts",
            torch.zeros(num_experts, num_bins, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "underflow",
            torch.zeros(num_experts, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "overflow",
            torch.zeros(num_experts, dtype=torch.int64),
            persistent=False,
        )

    @property
    def bin_width(self) -> float:
        return (self.max_margin - self.min_margin) / self.num_bins

    @torch.no_grad()
    def reset(self) -> None:
        self.counts.zero_()
        self.underflow.zero_()
        self.overflow.zero_()

    @torch.no_grad()
    def accumulate(
        self,
        raw_scores: torch.Tensor,
        biased_cutoffs: torch.Tensor,
    ) -> None:
        if raw_scores.ndim != 2 or raw_scores.shape[1] != self.num_experts:
            raise ValueError("raw_scores must have shape [M,E]")
        if biased_cutoffs.shape != (raw_scores.shape[0],):
            raise ValueError("biased_cutoffs must have shape [M]")
        margins = raw_scores.float() - biased_cutoffs.float()[:, None]
        self.underflow.add_((margins < self.min_margin).sum(dim=0))
        self.overflow.add_((margins >= self.max_margin).sum(dim=0))
        indices = torch.floor(
            (margins - self.min_margin) / self.bin_width
        ).long()
        indices.clamp_(0, self.num_bins - 1)
        expert_offsets = (
            torch.arange(self.num_experts, device=margins.device)
            * self.num_bins
        )
        flat = indices + expert_offsets[None, :]
        additions = torch.bincount(
            flat.reshape(-1),
            minlength=self.num_experts * self.num_bins,
        ).reshape(self.num_experts, self.num_bins)
        self.counts.add_(additions.to(self.counts.device))

    @torch.no_grad()
    def compute_next_bias(
        self, old_bias: torch.Tensor
    ) -> QuantileBalanceUpdate:
        if old_bias.shape != (self.num_experts,):
            raise ValueError("old_bias must have shape [E]")
        totals = self.counts.sum(dim=-1)
        if torch.any(totals == 0):
            raise ValueError("each expert histogram requires at least one margin")
        ranks = self.target_quantile * (totals.float() - 1)
        lower_ranks = ranks.floor().long()
        upper_ranks = ranks.ceil().long()
        cumulative = self.counts.cumsum(dim=-1)
        lower_bins = (
            cumulative
            > lower_ranks[:, None]
        ).long().argmax(dim=-1)
        upper_bins = (
            cumulative
            > upper_ranks[:, None]
        ).long().argmax(dim=-1)
        compute_dtype = (
            torch.float64
            if old_bias.dtype == torch.float64
            else torch.float32
        )
        lower_values = (
            self.min_margin
            + (lower_bins.float() + 0.5) * self.bin_width
        )
        upper_values = (
            self.min_margin
            + (upper_bins.float() + 0.5) * self.bin_width
        )
        fraction = ranks - lower_ranks
        quantiles = (
            lower_values + fraction * (upper_values - lower_values)
        ).to(compute_dtype)
        provisional = -quantiles
        next_bias = provisional - provisional.mean()
        lower = torch.full_like(quantiles, self.min_margin)
        upper = torch.full_like(quantiles, self.max_margin)
        return QuantileBalanceUpdate(
            next_bias=next_bias,
            target_quantile=self.target_quantile,
            quantiles=quantiles,
            margin_min=lower,
            margin_max=upper,
            num_tokens=int(totals[0].item()),
        )
