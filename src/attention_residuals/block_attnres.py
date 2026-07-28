from __future__ import annotations

import torch
import torch.nn as nn

from .block_state import BlockAttentionResidualState
from .diagnostics import build_depth_attention_stats
from .online_softmax import (
    merge_depth_softmax_stats,
    normalize_depth_softmax_stats,
    single_source_stats,
    weights_from_stats,
)
from .outputs import AttentionResidualMixOutput
from .site import AttentionResidualSite
from .two_phase import precompute_inter_block_stats, score_single_partial
from .utils import accumulation_dtype


class BlockAttentionResidualController(nn.Module):
    def __init__(
        self,
        sublayers_per_depth_block: int,
        backend: str = "eager",
    ):
        super().__init__()
        if sublayers_per_depth_block <= 0:
            raise ValueError("sublayers_per_depth_block must be > 0")
        if backend not in ("eager", "two_phase"):
            raise ValueError("backend must be 'eager' or 'two_phase'")
        self.sublayers_per_depth_block = sublayers_per_depth_block
        self.backend = backend

    def initialize(
        self, embedding: torch.Tensor
    ) -> BlockAttentionResidualState:
        return BlockAttentionResidualState(
            embedding, self.sublayers_per_depth_block
        )

    def prepare_depth_block(
        self,
        state: BlockAttentionResidualState,
        sites: tuple[AttentionResidualSite, ...],
    ) -> None:
        state.prepare_for_site()
        if self.backend != "two_phase":
            return
        block_index = state.current_depth_block_index
        if state.prepared_depth_block_index == block_index:
            return
        state.phase_stats = precompute_inter_block_stats(
            state.completed_sources(), sites
        )
        state.prepared_depth_block_index = block_index
        state.inter_block_scan_count += 1

    def mix_for_site(
        self,
        site: AttentionResidualSite,
        state: BlockAttentionResidualState,
        *,
        return_weights: bool = False,
        return_logits: bool = False,
        return_stats: bool = False,
    ) -> AttentionResidualMixOutput:
        state.prepare_for_site()
        completed_count = len(state.completed_blocks)
        has_partial = state.partial_block is not None
        if self.backend == "eager":
            return site(
                state.available_sources(),
                return_weights=return_weights,
                return_logits=return_logits,
                return_stats=return_stats,
                mode="block",
                number_of_completed_blocks=completed_count,
                has_current_partial=has_partial,
            )
        site_index = site.metadata.site_index
        if site_index not in state.phase_stats:
            raise RuntimeError(
                "two-phase depth block must be prepared before mixing"
            )
        merged = state.phase_stats[site_index]
        if has_partial:
            partial_logit = score_single_partial(state.partial_block, site)
            partial_stats = single_source_stats(
                partial_logit,
                state.partial_block,
                weighted_sum_dtype=accumulation_dtype(
                    state.partial_block.dtype,
                    site.weighted_sum_in_fp32,
                ),
            )
            merged = merge_depth_softmax_stats(merged, partial_stats)
        mixed = normalize_depth_softmax_stats(
            merged, state.embedding.dtype
        )
        need_weights = return_weights or return_stats
        weights = weights_from_stats(merged) if need_weights else None
        stats = None
        if return_stats:
            stats = build_depth_attention_stats(
                weights,
                site.metadata,
                mode="block",
                number_of_completed_blocks=completed_count,
                has_current_partial=has_partial,
                retain_mean_weights=return_weights,
            )
        return AttentionResidualMixOutput(
            mixed_state=mixed,
            weights=weights if return_weights else None,
            logits=merged.logits if return_logits else None,
            stats=stats,
        )

    def append_output(
        self,
        state: BlockAttentionResidualState,
        output: torch.Tensor,
    ) -> None:
        state.append_output(output)

    def finalize(
        self,
        state: BlockAttentionResidualState,
        final_site: AttentionResidualSite,
        *,
        return_weights: bool = False,
        return_logits: bool = False,
        return_stats: bool = False,
    ) -> AttentionResidualMixOutput:
        state.finalize()
        return final_site(
            state.available_sources(),
            return_weights=return_weights,
            return_logits=return_logits,
            return_stats=return_stats,
            mode="block",
            number_of_completed_blocks=len(state.completed_blocks),
            has_current_partial=False,
        )
