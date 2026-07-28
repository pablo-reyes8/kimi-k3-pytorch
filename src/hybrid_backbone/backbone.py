from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from src.kda import KimiDeltaAttention
from src.mla import GatedMLA
from src.transformer_modules.rms_norm import RMSNorm

from .attention_layer import HybridAttentionLayer
from .cache import HybridBackboneCache, HybridLayerCache
from .config import HybridBackboneConfig
from .dense_ffn import DenseKimiFFN
from .diagnostics import build_backbone_diagnostics
from .hybrid_group import HybridAttentionGroup
from .outputs import HybridBackboneOutput
from .utils import validate_backbone_inputs


class HybridAttentionBackbone(nn.Module):
    """Kimi K3's 3 KDA : 1 MLA sequence mixer with temporary dense FFNs."""

    def __init__(self, config: HybridBackboneConfig):
        super().__init__()
        self.config = config
        groups = []
        layer_index = 0
        for group_index in range(config.num_hybrid_groups):
            layers = []
            for position, attention_type in enumerate(
                config.attention_pattern
            ):
                layers.append(
                    self._make_layer(
                        attention_type,
                        layer_index,
                        group_index,
                        position,
                        include_ffn=True,
                    )
                )
                layer_index += 1
            groups.append(
                HybridAttentionGroup(layers, group_index=group_index)
            )
        self.groups = nn.ModuleList(groups)
        self.final_global_layer = self._make_layer(
            "gated_mla",
            layer_index,
            None,
            None,
            include_ffn=config.add_ffn_after_final_global,
            is_final_global=True,
        )
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def _make_ffn(self) -> DenseKimiFFN:
        return DenseKimiFFN(
            self.config.d_model,
            self.config.resolved_mlp_hidden_dim,
            dropout=self.config.ffn_dropout,
            bias=self.config.ffn_bias,
            init_std=self.config.init_std,
        )

    def _make_layer(
        self,
        attention_type: str,
        layer_index: int,
        group_index: int | None,
        position_in_group: int | None,
        *,
        include_ffn: bool,
        is_final_global: bool = False,
    ) -> HybridAttentionLayer:
        attention = (
            KimiDeltaAttention(self.config.kda_config)
            if attention_type == "kda"
            else GatedMLA(self.config.mla_config)
        )
        return HybridAttentionLayer(
            attention_type,
            attention,
            self._make_ffn() if include_ffn else None,
            self.config.d_model,
            self.config.rms_norm_eps,
            self.config.residual_dropout,
            layer_index=layer_index,
            group_index=group_index,
            position_in_group=position_in_group,
            is_final_global=is_final_global,
        )

    @property
    def layers(self) -> tuple[HybridAttentionLayer, ...]:
        return tuple(
            layer for group in self.groups for layer in group.layers
        ) + (self.final_global_layer,)

    @property
    def attention_types(self) -> tuple[str, ...]:
        return tuple(layer.attention_label for layer in self.layers)

    def _validate_cache(
        self,
        cache: HybridBackboneCache | None,
        batch: int,
        hidden_states: torch.Tensor,
        mode: str,
    ) -> None:
        if mode == "full" and cache is not None:
            raise ValueError("full mode does not accept an external cache")
        if mode == "decode" and cache is None:
            raise ValueError("decode mode requires a non-empty cache")
        if cache is None:
            return
        if len(cache.layer_caches) != len(self.layers):
            raise ValueError("cache layer count must match backbone layer count")
        for layer, layer_cache in zip(self.layers, cache.layer_caches):
            if layer_cache.attention_type != layer.attention_type:
                raise ValueError("cache order/type does not match backbone")
            offsets = layer_cache.sequence_offsets
            if offsets.shape != (batch,):
                raise ValueError("cache batch size must match hidden_states")
            if offsets.device != hidden_states.device:
                raise ValueError("cache and hidden_states must share device")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache: HybridBackboneCache | None = None,
        use_cache: bool = False,
        mode: Literal["full", "prefill", "decode"] = "full",
        output_hidden_states: bool = False,
        output_diagnostics: bool = False,
    ) -> HybridBackboneOutput:
        batch, _, mask = validate_backbone_inputs(
            hidden_states,
            attention_mask,
            self.config.d_model,
            mode,
        )
        self._validate_cache(cache, batch, hidden_states, mode)
        return_cache = use_cache
        cache_cursor = 0
        next_caches: list[HybridLayerCache] = []
        recorded_states = [hidden_states] if output_hidden_states else []
        layer_diagnostics = []
        output = hidden_states

        for group in self.groups:
            group_size = len(group.layers)
            group_cache = (
                None
                if cache is None
                else cache.layer_caches[cache_cursor : cache_cursor + group_size]
            )
            group_output = group(
                output,
                mask,
                group_cache,
                use_cache=return_cache,
                mode=mode,
                output_hidden_states=output_hidden_states,
                output_diagnostics=output_diagnostics,
            )
            output = group_output.hidden_states
            cache_cursor += group_size
            if return_cache:
                next_caches.extend(group_output.layer_caches)
            if output_hidden_states:
                recorded_states.extend(group_output.hidden_states_by_layer)
            if output_diagnostics:
                layer_diagnostics.extend(group_output.diagnostics)

        final_cache = (
            None if cache is None else cache.layer_caches[cache_cursor]
        )
        final_output = self.final_global_layer(
            output,
            mask,
            final_cache,
            use_cache=return_cache,
            mode=mode,
            output_diagnostics=output_diagnostics,
        )
        output = final_output.hidden_states
        if return_cache:
            next_caches.append(final_output.cache)
        if output_hidden_states:
            recorded_states.append(output)
        if output_diagnostics:
            layer_diagnostics.append(final_output.diagnostics)

        output = self.final_norm(output)
        if output_hidden_states:
            recorded_states.append(output)

        backbone_cache = None
        if return_cache:
            offsets = next_caches[0].sequence_offsets
            backbone_cache = HybridBackboneCache(
                tuple(next_caches),
                int(offsets.max().item()),
            )
        diagnostics = None
        if output_diagnostics:
            diagnostics = build_backbone_diagnostics(
                tuple(layer_diagnostics),
                self.groups,
                self.final_global_layer,
                self.final_norm,
                backbone_cache,
                hidden_states.device,
            )
        return HybridBackboneOutput(
            last_hidden_state=output,
            cache=backbone_cache,
            hidden_states=(
                tuple(recorded_states) if output_hidden_states else None
            ),
            diagnostics=diagnostics,
        )
