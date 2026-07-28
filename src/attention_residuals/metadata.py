"""Attention-residual components for mixing hidden-state streams across model depth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DepthSiteMetadata:
    """Identify one residual-mixing site in the backbone depth topology."""

    site_index: int
    transformer_layer_index: int | None
    site_kind: Literal["pre_attention", "pre_ffn", "final_output"]
    attention_type: Literal["kda", "gated_mla"] | None
    hybrid_group_index: int | None
    position_in_hybrid_group: int | None
    depth_block_index: int | None
    position_in_depth_block: int | None
