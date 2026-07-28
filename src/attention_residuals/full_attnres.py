"""Attention-residual components for mixing hidden-state streams across model depth."""

from __future__ import annotations

import torch
import torch.nn as nn

from .full_state import FullAttentionResidualState
from .site import AttentionResidualSite


class FullAttentionResidualController(nn.Module):
    def initialize(
        self, embedding: torch.Tensor
    ) -> FullAttentionResidualState:
        if embedding.ndim != 3:
            raise ValueError("embedding must have shape [B,T,D]")
        return FullAttentionResidualState([embedding], 0)

    def mix_for_site(
        self,
        site: AttentionResidualSite,
        state: FullAttentionResidualState,
        *,
        return_weights: bool = False,
        return_logits: bool = False,
        return_stats: bool = False,
    ):
        return site(
            state.available_sources(),
            return_weights=return_weights,
            return_logits=return_logits,
            return_stats=return_stats,
            mode="full",
        )

    def append_output(
        self,
        state: FullAttentionResidualState,
        output: torch.Tensor,
    ) -> None:
        state.append_output(output)

    def finalize(
        self,
        state: FullAttentionResidualState,
        final_site: AttentionResidualSite,
        *,
        return_weights: bool = False,
        return_logits: bool = False,
        return_stats: bool = False,
    ):
        return self.mix_for_site(
            final_site,
            state,
            return_weights=return_weights,
            return_logits=return_logits,
            return_stats=return_stats,
        )
