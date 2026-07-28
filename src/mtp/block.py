"""Multi-token prediction components used as an optional KimiK3 output head."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from src.hybrid_backbone import (
    CANONICAL_ATTENTION_PATTERN,
    HybridAttentionBackbone,
    HybridBackboneCache,
    HybridBackboneConfig,
    HybridBackboneOutput,
)
from src.stable_latent_moe import StableLatentMoE

from .config import KimiMTPConfig


class KimiMTPBlock(nn.Module):
    """One independent 3×KDA + 1×Gated-MLA group, with local AttnRes."""

    def __init__(self, config: KimiMTPConfig):
        super().__init__()
        if not config.enabled:
            raise ValueError("cannot instantiate KimiMTPBlock when MTP is disabled")
        self.config = config
        backbone_config = HybridBackboneConfig(
            d_model=config.d_model,
            num_hybrid_groups=1,
            attention_pattern=CANONICAL_ATTENTION_PATTERN,
            enforce_canonical_pattern=False,
            add_final_gated_mla=False,
            add_ffn_after_final_global=True,
            rms_norm_eps=config.rms_norm_eps,
            residual_dropout=config.residual_dropout,
            ffn_dropout=0.0,
            kda_config=config.kda_config,
            mla_config=config.mla_config,
            use_dense_ffn=False,
            channel_mixer_type="stable_latent_moe",
            stable_latent_moe_config=config.stable_latent_moe_config,
            init_std=config.init_std,
            attention_residual_config=config.attention_residual_config,
        )
        self.backbone = HybridAttentionBackbone(backbone_config)

    @property
    def layers(self):
        return self.backbone.layers

    @property
    def attention_types(self) -> tuple[str, ...]:
        return self.backbone.attention_types

    @property
    def moe_layers(self) -> tuple[StableLatentMoE, ...]:
        return tuple(layer.ffn for layer in self.layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache: HybridBackboneCache | None = None,
        use_cache: bool = False,
        mode: Literal["full", "prefill", "decode"] = "full",
        output_hidden_states: bool = False,
        output_depth_weights: bool = False,
        output_diagnostics: bool = False,
        update_routing_bias: bool = False,
    ) -> HybridBackboneOutput:
        return self.backbone(
            hidden_states,
            attention_mask=attention_mask,
            cache=cache,
            use_cache=use_cache,
            mode=mode,
            output_hidden_states=output_hidden_states,
            output_depth_weights=output_depth_weights,
            output_diagnostics=output_diagnostics,
            update_routing_bias=update_routing_bias,
        )
