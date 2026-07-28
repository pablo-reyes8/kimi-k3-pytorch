from __future__ import annotations

import torch
import torch.nn as nn

from src.transformer_modules.rms_norm import RMSNorm

from .diagnostics import build_depth_attention_stats
from .metadata import DepthSiteMetadata
from .outputs import AttentionResidualMixOutput
from .utils import accumulation_dtype, validate_sources


def depth_softmax_mix_reference(
    sources: torch.Tensor,
    pseudo_query: torch.Tensor,
    key_norm: nn.Module,
    *,
    logits_in_fp32: bool = True,
    weighted_sum_in_fp32: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pseudo_query.ndim != 1:
        raise ValueError("pseudo_query must have shape [D]")
    _, _, _ = validate_sources(sources, pseudo_query.shape[0])
    if not torch.isfinite(pseudo_query).all():
        raise ValueError("pseudo_query must be finite")
    keys = key_norm(sources)
    logit_dtype = accumulation_dtype(sources.dtype, logits_in_fp32)
    logits = torch.einsum(
        "d,btsd->bts",
        pseudo_query.to(logit_dtype),
        keys.to(logit_dtype),
    )
    weights = torch.softmax(logits, dim=-1)
    sum_dtype = accumulation_dtype(sources.dtype, weighted_sum_in_fp32)
    mixed = torch.einsum(
        "bts,btsd->btd",
        weights.to(sum_dtype),
        sources.to(sum_dtype),
    ).to(sources.dtype)
    return mixed, weights, logits


class AttentionResidualSite(nn.Module):
    """Canonical single-query attention over preceding depth outputs."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-6,
        logits_in_fp32: bool = True,
        weighted_sum_in_fp32: bool = True,
        *,
        metadata: DepthSiteMetadata | None = None,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        self.d_model = d_model
        self.logits_in_fp32 = logits_in_fp32
        self.weighted_sum_in_fp32 = weighted_sum_in_fp32
        self.metadata = metadata
        self.pseudo_query = nn.Parameter(torch.zeros(d_model))
        self.key_norm = RMSNorm(d_model, eps=eps)

    def forward(
        self,
        sources: torch.Tensor,
        *,
        return_weights: bool = False,
        return_logits: bool = False,
        return_stats: bool = False,
        mode: str = "full",
        number_of_completed_blocks: int = 0,
        has_current_partial: bool = False,
    ) -> AttentionResidualMixOutput:
        validate_sources(sources, self.d_model)
        if not torch.isfinite(self.key_norm.weight).all():
            raise ValueError("AttnRes key RMSNorm parameters must be finite")
        mixed, weights, logits = depth_softmax_mix_reference(
            sources,
            self.pseudo_query,
            self.key_norm,
            logits_in_fp32=self.logits_in_fp32,
            weighted_sum_in_fp32=self.weighted_sum_in_fp32,
        )
        stats = None
        if return_stats:
            if self.metadata is None:
                raise ValueError("metadata is required to build site diagnostics")
            stats = build_depth_attention_stats(
                weights,
                self.metadata,
                mode=mode,
                number_of_completed_blocks=number_of_completed_blocks,
                has_current_partial=has_current_partial,
                retain_mean_weights=return_weights,
            )
        return AttentionResidualMixOutput(
            mixed,
            weights if return_weights else None,
            logits if return_logits else None,
            stats,
        )
