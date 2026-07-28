"""Top-level model components and public APIs for the research Kimi K3 implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn as nn

from src.attention_residuals import AttentionResidualConfig
from src.hybrid_backbone import (
    CANONICAL_ATTENTION_PATTERN,
    HybridAttentionBackbone,
    HybridBackboneCache,
    HybridBackboneConfig,
    HybridBackboneOutput,
)
from src.kda import KDAConfig
from src.mla import GatedMLAConfig
from src.stable_latent_moe import StableLatentMoE, StableLatentMoEConfig


@dataclass(frozen=True)
class KimiBlockConfig:
    """High-level configurable KDA/MLA + Stable LatentMoE text block."""

    d_model: int
    num_pattern_repeats: int
    kda_config: KDAConfig
    mla_config: GatedMLAConfig
    stable_latent_moe_config: StableLatentMoEConfig
    attention_residual_config: AttentionResidualConfig
    attention_pattern: tuple[str, ...] = CANONICAL_ATTENTION_PATTERN
    add_final_gated_mla: bool = True
    rms_norm_eps: float = 1e-6
    residual_dropout: float = 0.0
    init_std: float = 0.02

    def __post_init__(self) -> None:
        object.__setattr__(self, "attention_pattern", tuple(self.attention_pattern))
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if self.num_pattern_repeats <= 0:
            raise ValueError("num_pattern_repeats must be > 0")
        if not self.attention_pattern:
            raise ValueError("attention_pattern must not be empty")
        if not self.add_final_gated_mla:
            raise ValueError("KimiBlock requires the final global Gated MLA")
        unknown = set(self.attention_pattern) - {"kda", "gated_mla"}
        if unknown:
            raise ValueError(f"unsupported attention types: {sorted(unknown)}")
        for name, config in (
            ("kda_config", self.kda_config),
            ("mla_config", self.mla_config),
            ("stable_latent_moe_config", self.stable_latent_moe_config),
            ("attention_residual_config", self.attention_residual_config),
        ):
            if config.d_model != self.d_model:
                raise ValueError(f"{name}.d_model must match d_model")
        if self.attention_residual_config.mode == "standard":
            raise ValueError(
                "canonical KimiBlock requires Full or Block AttnRes"
            )

    @property
    def num_pattern_layers(self) -> int:
        return self.num_pattern_repeats * len(self.attention_pattern)

    @property
    def num_transformer_layers(self) -> int:
        return self.num_pattern_layers + int(self.add_final_gated_mla)

    @property
    def num_kda_layers(self) -> int:
        return self.num_pattern_repeats * self.attention_pattern.count("kda")

    @property
    def num_mla_layers(self) -> int:
        return (
            self.num_pattern_repeats
            * self.attention_pattern.count("gated_mla")
            + int(self.add_final_gated_mla)
        )

    @property
    def num_moe_layers(self) -> int:
        return self.num_transformer_layers

    def to_backbone_config(self) -> HybridBackboneConfig:
        return HybridBackboneConfig(
            d_model=self.d_model,
            num_hybrid_groups=self.num_pattern_repeats,
            attention_pattern=self.attention_pattern,
            enforce_canonical_pattern=False,
            add_final_gated_mla=self.add_final_gated_mla,
            add_ffn_after_final_global=True,
            rms_norm_eps=self.rms_norm_eps,
            residual_dropout=self.residual_dropout,
            ffn_dropout=0.0,
            kda_config=self.kda_config,
            mla_config=self.mla_config,
            use_dense_ffn=False,
            channel_mixer_type="stable_latent_moe",
            stable_latent_moe_config=self.stable_latent_moe_config,
            init_std=self.init_std,
            attention_residual_config=self.attention_residual_config,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "KimiBlockConfig":
        values = dict(values)
        values["kda_config"] = KDAConfig.from_dict(values["kda_config"])
        values["mla_config"] = GatedMLAConfig.from_dict(values["mla_config"])
        values["stable_latent_moe_config"] = (
            StableLatentMoEConfig.from_dict(
                values["stable_latent_moe_config"]
            )
        )
        values["attention_residual_config"] = (
            AttentionResidualConfig.from_dict(
                values["attention_residual_config"]
            )
        )
        values["attention_pattern"] = tuple(values["attention_pattern"])
        return cls(**values)


class KimiBlock(nn.Module):
    """Text-backbone block without embeddings, LM head, loss or generation."""

    def __init__(self, config: KimiBlockConfig):
        super().__init__()
        self.config = config
        self.backbone = HybridAttentionBackbone(
            config.to_backbone_config()
        )

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
