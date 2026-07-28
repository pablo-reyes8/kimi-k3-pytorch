"""Hybrid KDA/MLA backbone components and cache structures."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from src.attention_residuals import (
    AttentionResidualSite,
    BlockAttentionResidualController,
    BlockAttentionResidualState,
    FullAttentionResidualController,
    FullAttentionResidualState,
)
from src.kda import KDAState, KimiDeltaAttention
from src.mla import GatedMLA, MLACache
from src.stable_latent_moe import StableLatentMoE, StableLatentMoEOutput
from src.transformer_modules.rms_norm import RMSNorm

from .cache import HybridLayerCache
from .outputs import HybridLayerOutput
from .utils import masked_norm_mean


class HybridAttentionLayer(nn.Module):
    """One pre-norm attention residual and one replaceable FFN residual."""

    def __init__(
        self,
        attention_type: Literal["kda", "gated_mla"],
        attention_module: nn.Module,
        ffn_module: nn.Module | None,
        d_model: int,
        norm_eps: float,
        residual_dropout: float = 0.0,
        *,
        layer_index: int,
        group_index: int | None,
        position_in_group: int | None,
        is_final_global: bool = False,
        pre_attention_attnres: AttentionResidualSite | None = None,
        pre_ffn_attnres: AttentionResidualSite | None = None,
    ):
        super().__init__()
        if attention_type not in ("kda", "gated_mla"):
            raise ValueError("unsupported attention_type")
        expected = KimiDeltaAttention if attention_type == "kda" else GatedMLA
        if not isinstance(attention_module, expected):
            raise TypeError(
                f"{attention_type} requires {expected.__name__}"
            )
        if not 0.0 <= residual_dropout < 1.0:
            raise ValueError("residual_dropout must satisfy 0 <= p < 1")
        if attention_module.config.d_model != d_model:
            raise ValueError("attention d_model must match layer d_model")
        if is_final_global and attention_type != "gated_mla":
            raise ValueError("final global layer must use Gated MLA")
        self.attention_type = attention_type
        self.d_model = d_model
        self.layer_index = layer_index
        self.group_index = group_index
        self.position_in_group = position_in_group
        self.is_final_global = is_final_global
        self.attention_norm = RMSNorm(d_model, eps=norm_eps)
        self.attention = attention_module
        self.ffn_norm = (
            RMSNorm(d_model, eps=norm_eps) if ffn_module is not None else None
        )
        self.ffn = ffn_module
        self.residual_dropout = nn.Dropout(residual_dropout)
        self.pre_attention_attnres = pre_attention_attnres
        self.pre_ffn_attnres = pre_ffn_attnres

    @property
    def attention_label(self) -> str:
        return "gated_mla_final" if self.is_final_global else self.attention_type

    @property
    def channel_mixer(self) -> nn.Module | None:
        return self.ffn

    @property
    def pre_moe_attnres(self) -> AttentionResidualSite | None:
        return self.pre_ffn_attnres

    @property
    def moe(self) -> StableLatentMoE | None:
        return self.ffn if isinstance(self.ffn, StableLatentMoE) else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: HybridLayerCache | None = None,
        *,
        use_cache: bool = False,
        mode: Literal["full", "prefill", "decode"] = "full",
        output_diagnostics: bool = False,
        update_routing_bias: bool = False,
    ) -> HybridLayerOutput:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.d_model:
            raise ValueError(
                f"hidden_states must have shape [B,T,{self.d_model}]"
            )
        if cache is not None and cache.attention_type != self.attention_type:
            raise ValueError("cache type does not match attention layer")
        normalized = self.attention_norm(hidden_states)
        state = None if cache is None else cache.state
        if self.attention_type == "kda":
            if state is not None and not isinstance(state, KDAState):
                raise TypeError("KDA layer requires KDAState")
            kda_mode = "decode" if mode == "decode" else "chunkwise"
            attention_result = self.attention(
                normalized,
                attention_mask=attention_mask,
                state=state,
                mode=kda_mode,
                output_final_state=use_cache,
                output_diagnostics=output_diagnostics,
            )
            attention_output = attention_result.hidden_states
            next_state = attention_result.state
            mechanism_diagnostics = attention_result.diagnostics
        else:
            if state is not None and not isinstance(state, MLACache):
                raise TypeError("MLA layer requires MLACache")
            attention_result = self.attention(
                normalized,
                attention_mask=attention_mask,
                cache=state,
                use_cache=use_cache,
                output_diagnostics=output_diagnostics,
            )
            attention_output = attention_result.hidden_states
            next_state = attention_result.cache
            mechanism_diagnostics = attention_result.diagnostics

        post_attention = hidden_states + self.residual_dropout(attention_output)
        ffn_output = torch.zeros_like(post_attention)
        channel_mixer_diagnostics = None
        if self.ffn is not None:
            ffn_output, channel_mixer_diagnostics = (
                self._channel_mixer_transform(
                    self.ffn_norm(post_attention),
                    update_routing_bias=update_routing_bias,
                    output_diagnostics=output_diagnostics,
                )
            )
        output = post_attention + self.residual_dropout(ffn_output)

        next_cache = (
            HybridLayerCache(self.attention_type, next_state)
            if use_cache
            else None
        )
        diagnostics = None
        if output_diagnostics:
            diagnostics = {
                "layer_index": self.layer_index,
                "group_index": self.group_index,
                "position_in_group": self.position_in_group,
                "attention_type": self.attention_label,
                "input_norm": masked_norm_mean(hidden_states, attention_mask),
                "attention_output_norm": masked_norm_mean(
                    attention_output, attention_mask
                ),
                "post_attention_residual_norm": masked_norm_mean(
                    post_attention, attention_mask
                ),
                "ffn_output_norm": masked_norm_mean(ffn_output, attention_mask),
                "post_ffn_residual_norm": masked_norm_mean(
                    output, attention_mask
                ),
                "mechanism": mechanism_diagnostics,
                "channel_mixer": channel_mixer_diagnostics,
            }
        return HybridLayerOutput(output, next_cache, diagnostics)

    def _attention_transform(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: HybridLayerCache | None,
        *,
        use_cache: bool,
        mode: str,
        output_diagnostics: bool,
    ) -> tuple[torch.Tensor, object, dict[str, torch.Tensor] | None]:
        if cache is not None and cache.attention_type != self.attention_type:
            raise ValueError("cache type does not match attention layer")
        state = None if cache is None else cache.state
        if self.attention_type == "kda":
            if state is not None and not isinstance(state, KDAState):
                raise TypeError("KDA layer requires KDAState")
            result = self.attention(
                self.attention_norm(hidden_states),
                attention_mask=attention_mask,
                state=state,
                mode="decode" if mode == "decode" else "chunkwise",
                output_final_state=use_cache,
                output_diagnostics=output_diagnostics,
            )
            return result.hidden_states, result.state, result.diagnostics
        if state is not None and not isinstance(state, MLACache):
            raise TypeError("MLA layer requires MLACache")
        result = self.attention(
            self.attention_norm(hidden_states),
            attention_mask=attention_mask,
            cache=state,
            use_cache=use_cache,
            output_diagnostics=output_diagnostics,
        )
        return result.hidden_states, result.cache, result.diagnostics

    def _channel_mixer_transform(
        self,
        hidden_states: torch.Tensor,
        *,
        update_routing_bias: bool,
        output_diagnostics: bool,
    ) -> tuple[torch.Tensor, object | None]:
        if isinstance(self.ffn, StableLatentMoE):
            result = self.ffn(
                hidden_states,
                update_routing_bias=update_routing_bias,
                return_router_diagnostics=output_diagnostics,
            )
            if isinstance(result, StableLatentMoEOutput):
                return result.hidden_states, result.diagnostics
            return result, None
        return self.ffn(hidden_states), None

    def forward_attnres(
        self,
        depth_controller: (
            FullAttentionResidualController
            | BlockAttentionResidualController
        ),
        depth_state: (
            FullAttentionResidualState
            | BlockAttentionResidualState
        ),
        attention_mask: torch.Tensor,
        cache: HybridLayerCache | None = None,
        *,
        depth_block_sites: dict[int, tuple[AttentionResidualSite, ...]] | None = None,
        use_cache: bool = False,
        mode: Literal["full", "prefill", "decode"] = "full",
        output_depth_weights: bool = False,
        output_depth_stats: bool = False,
        output_diagnostics: bool = False,
        update_routing_bias: bool = False,
    ) -> tuple[HybridLayerOutput, tuple[object, object]]:
        if self.pre_attention_attnres is None or self.pre_ffn_attnres is None:
            raise RuntimeError("AttnRes sites are not configured on this layer")

        def prepare(site: AttentionResidualSite) -> None:
            if isinstance(depth_controller, BlockAttentionResidualController):
                block_index = site.metadata.depth_block_index
                depth_controller.prepare_depth_block(
                    depth_state, depth_block_sites[block_index]
                )

        prepare(self.pre_attention_attnres)
        attention_mix = depth_controller.mix_for_site(
            self.pre_attention_attnres,
            depth_state,
            return_weights=output_depth_weights,
            return_stats=output_depth_stats,
        )
        attention_output, next_state, mechanism_diagnostics = (
            self._attention_transform(
                attention_mix.mixed_state,
                attention_mask,
                cache,
                use_cache=use_cache,
                mode=mode,
                output_diagnostics=output_diagnostics,
            )
        )
        attention_output = self.residual_dropout(attention_output)
        depth_controller.append_output(depth_state, attention_output)

        prepare(self.pre_ffn_attnres)
        ffn_mix = depth_controller.mix_for_site(
            self.pre_ffn_attnres,
            depth_state,
            return_weights=output_depth_weights,
            return_stats=output_depth_stats,
        )
        if self.ffn is None or self.ffn_norm is None:
            raise RuntimeError(
                "canonical AttnRes requires an FFN at every transformer layer"
            )
        raw_ffn_output, channel_mixer_diagnostics = (
            self._channel_mixer_transform(
                self.ffn_norm(ffn_mix.mixed_state),
                update_routing_bias=update_routing_bias,
                output_diagnostics=output_diagnostics,
            )
        )
        ffn_output = self.residual_dropout(raw_ffn_output)
        depth_controller.append_output(depth_state, ffn_output)
        next_cache = (
            HybridLayerCache(self.attention_type, next_state)
            if use_cache
            else None
        )
        diagnostics = None
        if output_diagnostics:
            diagnostics = {
                "layer_index": self.layer_index,
                "group_index": self.group_index,
                "position_in_group": self.position_in_group,
                "attention_type": self.attention_label,
                "pre_attention_depth_norm": masked_norm_mean(
                    attention_mix.mixed_state, attention_mask
                ),
                "attention_output_norm": masked_norm_mean(
                    attention_output, attention_mask
                ),
                "pre_ffn_depth_norm": masked_norm_mean(
                    ffn_mix.mixed_state, attention_mask
                ),
                "ffn_output_norm": masked_norm_mean(
                    ffn_output, attention_mask
                ),
                "mechanism": mechanism_diagnostics,
                "channel_mixer": channel_mixer_diagnostics,
            }
        return (
            HybridLayerOutput(
                hidden_states=ffn_output,
                cache=next_cache,
                diagnostics=diagnostics,
                pre_attention_state=attention_mix.mixed_state,
                attention_output=attention_output,
                pre_ffn_state=ffn_mix.mixed_state,
                ffn_output=ffn_output,
            ),
            (attention_mix, ffn_mix),
        )
