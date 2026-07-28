from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from src.attention_residuals import (
    AttentionResidualBackboneOutput,
    AttentionResidualSite,
    BlockAttentionResidualController,
    BlockAttentionResidualState,
    DepthSiteMetadata,
    FullAttentionResidualController,
    FullAttentionResidualState,
    padded_weight_matrix,
)
from src.kda import KimiDeltaAttention
from src.mla import GatedMLA
from src.transformer_modules.rms_norm import RMSNorm

from .attention_layer import HybridAttentionLayer
from .cache import HybridBackboneCache, HybridLayerCache
from .config import HybridBackboneConfig
from .dense_ffn import DenseKimiFFN
from .diagnostics import build_backbone_diagnostics
from .hybrid_group import HybridAttentionGroup
from .outputs import BackboneHiddenStateTrace, HybridBackboneOutput
from .utils import validate_backbone_inputs


class HybridAttentionBackbone(nn.Module):
    """Kimi K3's 3 KDA : 1 MLA sequence mixer with temporary dense FFNs."""

    def __init__(self, config: HybridBackboneConfig):
        super().__init__()
        self.config = config
        self.attnres_config = config.attention_residual_config
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
        if config.depth_mixing == "full":
            self.depth_controller = FullAttentionResidualController()
        elif config.depth_mixing == "block":
            self.depth_controller = BlockAttentionResidualController(
                self.attnres_config.resolved_sublayers_per_depth_block,
                backend=self.attnres_config.backend,
            )
        else:
            self.depth_controller = None
        self.final_output_attnres = (
            AttentionResidualSite(
                config.d_model,
                eps=self.attnres_config.rms_norm_eps,
                logits_in_fp32=self.attnres_config.logits_in_fp32,
                weighted_sum_in_fp32=(
                    self.attnres_config.weighted_sum_in_fp32
                ),
                metadata=self._site_metadata(
                    config.num_attention_layers,
                    "final_output",
                ),
            )
            if self.attnres_config is not None
            and self.attnres_config.mode != "standard"
            else None
        )

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
        pre_attention_site = None
        pre_ffn_site = None
        if (
            self.attnres_config is not None
            and self.attnres_config.mode != "standard"
        ):
            site_kwargs = dict(
                d_model=self.config.d_model,
                eps=self.attnres_config.rms_norm_eps,
                logits_in_fp32=self.attnres_config.logits_in_fp32,
                weighted_sum_in_fp32=(
                    self.attnres_config.weighted_sum_in_fp32
                ),
            )
            pre_attention_site = AttentionResidualSite(
                **site_kwargs,
                metadata=self._site_metadata(
                    layer_index,
                    "pre_attention",
                    attention_type,
                    group_index,
                    position_in_group,
                ),
            )
            pre_ffn_site = AttentionResidualSite(
                **site_kwargs,
                metadata=self._site_metadata(
                    layer_index,
                    "pre_ffn",
                    attention_type,
                    group_index,
                    position_in_group,
                ),
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
            pre_attention_attnres=pre_attention_site,
            pre_ffn_attnres=pre_ffn_site,
        )

    def _site_metadata(
        self,
        transformer_layer_index: int,
        site_kind: str,
        attention_type: str | None = None,
        group_index: int | None = None,
        position_in_group: int | None = None,
    ) -> DepthSiteMetadata:
        if site_kind == "final_output":
            site_index = 2 * self.config.num_attention_layers
            return DepthSiteMetadata(
                site_index,
                None,
                "final_output",
                None,
                None,
                None,
                None,
                None,
            )
        offset = 0 if site_kind == "pre_attention" else 1
        site_index = 2 * transformer_layer_index + offset
        block_index = None
        position_in_block = None
        if (
            self.attnres_config is not None
            and self.attnres_config.mode == "block"
        ):
            block_size = (
                self.attnres_config.resolved_sublayers_per_depth_block
            )
            block_index = site_index // block_size
            position_in_block = site_index % block_size
        return DepthSiteMetadata(
            site_index,
            transformer_layer_index,
            site_kind,
            attention_type,
            group_index,
            position_in_group,
            block_index,
            position_in_block,
        )

    @property
    def layers(self) -> tuple[HybridAttentionLayer, ...]:
        return tuple(
            layer for group in self.groups for layer in group.layers
        ) + (self.final_global_layer,)

    @property
    def attention_types(self) -> tuple[str, ...]:
        return tuple(layer.attention_label for layer in self.layers)

    @property
    def depth_site_metadata(self) -> tuple[DepthSiteMetadata, ...]:
        if self.final_output_attnres is None:
            return ()
        metadata = [
            site.metadata
            for layer in self.layers
            for site in (
                layer.pre_attention_attnres,
                layer.pre_ffn_attnres,
            )
        ]
        metadata.append(self.final_output_attnres.metadata)
        return tuple(metadata)

    @property
    def _depth_block_sites(
        self,
    ) -> dict[int, tuple[AttentionResidualSite, ...]]:
        grouped: dict[int, list[AttentionResidualSite]] = {}
        for layer in self.layers:
            for site in (
                layer.pre_attention_attnres,
                layer.pre_ffn_attnres,
            ):
                if site is not None:
                    grouped.setdefault(
                        site.metadata.depth_block_index, []
                    ).append(site)
        return {
            index: tuple(sites) for index, sites in grouped.items()
        }

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
        output_depth_weights: bool = False,
        output_diagnostics: bool = False,
    ) -> HybridBackboneOutput:
        if self.config.depth_mixing == "standard":
            return self._forward_standard(
                hidden_states,
                attention_mask,
                cache,
                use_cache,
                mode,
                output_hidden_states,
                output_diagnostics,
            )
        return self._forward_attnres(
            hidden_states,
            attention_mask,
            cache,
            use_cache,
            mode,
            output_hidden_states,
            output_depth_weights,
            output_diagnostics,
        )

    def _forward_standard(
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
                None,
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

    def _forward_attnres(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache: HybridBackboneCache | None,
        use_cache: bool,
        mode: Literal["full", "prefill", "decode"],
        output_hidden_states: bool,
        output_depth_weights: bool,
        output_diagnostics: bool,
    ) -> HybridBackboneOutput:
        batch, _, mask = validate_backbone_inputs(
            hidden_states,
            attention_mask,
            self.config.d_model,
            mode,
        )
        self._validate_cache(cache, batch, hidden_states, mode)
        controller = self.depth_controller
        depth_state = controller.initialize(hidden_states)
        block_sites = (
            self._depth_block_sites
            if isinstance(controller, BlockAttentionResidualController)
            else None
        )
        return_weights = (
            output_depth_weights
            or self.attnres_config.return_depth_weights
        )
        return_stats = (
            return_weights
            or output_diagnostics
            or self.attnres_config.return_depth_stats
        )
        next_caches: list[HybridLayerCache] = []
        layer_diagnostics = []
        site_stats = []
        legacy_states = [hidden_states] if output_hidden_states else []
        pre_attention_states = []
        attention_outputs = []
        pre_ffn_states = []
        ffn_outputs = []

        for index, layer in enumerate(self.layers):
            layer_cache = (
                None if cache is None else cache.layer_caches[index]
            )
            layer_output, mixes = layer.forward_attnres(
                controller,
                depth_state,
                mask,
                layer_cache,
                depth_block_sites=block_sites,
                use_cache=use_cache,
                mode=mode,
                output_depth_weights=return_weights,
                output_depth_stats=return_stats,
                output_diagnostics=output_diagnostics,
            )
            if use_cache:
                next_caches.append(layer_output.cache)
            if output_diagnostics:
                layer_diagnostics.append(layer_output.diagnostics)
            if return_stats:
                site_stats.extend(
                    (mixes[0].stats, mixes[1].stats)
                )
            if output_hidden_states:
                pre_attention_states.append(
                    layer_output.pre_attention_state
                )
                attention_outputs.append(layer_output.attention_output)
                pre_ffn_states.append(layer_output.pre_ffn_state)
                ffn_outputs.append(layer_output.ffn_output)
                legacy_states.extend(
                    (
                        layer_output.attention_output,
                        layer_output.ffn_output,
                    )
                )

        final_mix = controller.finalize(
            depth_state,
            self.final_output_attnres,
            return_weights=return_weights,
            return_stats=return_stats,
        )
        final_mixed = final_mix.mixed_state
        output = self.final_norm(final_mixed)
        if output_hidden_states:
            legacy_states.extend((final_mixed, output))

        backbone_cache = None
        if use_cache:
            offsets = next_caches[0].sequence_offsets
            backbone_cache = HybridBackboneCache(
                tuple(next_caches), int(offsets.max().item())
            )

        depth_outputs = None
        if return_stats:
            all_stats = tuple(site_stats) + (final_mix.stats,)
            weight_matrix = None
            source_mask = None
            source_labels = None
            if return_weights:
                weight_matrix, source_mask = padded_weight_matrix(all_stats)
                labels = []
                for item in all_stats:
                    if self.config.depth_mixing == "full":
                        labels.append(
                            ("embedding",)
                            + tuple(
                                f"sublayer_{source}"
                                for source in range(1, item.source_count)
                            )
                        )
                    else:
                        completed = item.number_of_completed_blocks or 0
                        row = ["embedding"] + [
                            f"depth_block_{block}"
                            for block in range(completed)
                        ]
                        if item.source_count > 1 + completed:
                            row.append("current_partial")
                        labels.append(tuple(row))
                source_labels = tuple(labels)
            if isinstance(depth_state, FullAttentionResidualState):
                source_tensor_count = len(depth_state.sources)
                source_elements = depth_state.source_elements
                peak_source_count = len(depth_state.sources)
                num_depth_blocks = 0
                partial_final_size = 0
                scan_count = 0
            else:
                source_tensor_count = 1 + len(depth_state.completed_blocks)
                source_elements = depth_state.source_elements
                peak_source_count = source_tensor_count
                num_depth_blocks = len(depth_state.completed_blocks)
                last_size = (
                    depth_state.block_sizes[-1]
                    if depth_state.block_sizes
                    else 0
                )
                partial_final_size = (
                    last_size
                    if last_size
                    < depth_state.sublayers_per_depth_block
                    else 0
                )
                scan_count = depth_state.inter_block_scan_count
            depth_outputs = AttentionResidualBackboneOutput(
                mode=self.config.depth_mixing,
                site_stats=tuple(site_stats),
                final_output_stats=final_mix.stats,
                averaged_weight_matrix=weight_matrix,
                source_mask=source_mask,
                source_labels=source_labels,
                source_tensor_count=source_tensor_count,
                source_elements=source_elements,
                peak_source_count=peak_source_count,
                num_depth_blocks=num_depth_blocks,
                partial_final_block_size=partial_final_size,
                inter_block_scan_count=scan_count,
            )

        diagnostics = None
        if output_diagnostics:
            diagnostics = build_backbone_diagnostics(
                tuple(layer_diagnostics),
                self.groups,
                self.final_global_layer,
                self.final_norm,
                self.final_output_attnres,
                backbone_cache,
                hidden_states.device,
            )
            depth_stats = tuple(site_stats) + (final_mix.stats,)
            diagnostics["depth_mixing"] = self.config.depth_mixing
            diagnostics["mean_embedding_weight"] = torch.stack(
                [item.embedding_weight for item in depth_stats]
            ).mean()
            diagnostics["mean_depth_entropy"] = torch.stack(
                [item.weight_entropy for item in depth_stats]
            ).mean()
            diagnostics["mean_retrieval_distance"] = torch.stack(
                [item.mean_retrieval_distance for item in depth_stats]
            ).mean()
            diagnostics["fraction_dominated_by_embedding"] = torch.stack(
                [
                    (item.dominant_source_index == 0).float()
                    for item in depth_stats
                ]
            ).mean()
            diagnostics["fraction_dominated_by_most_recent"] = torch.stack(
                [
                    (
                        item.dominant_source_index
                        == item.source_count - 1
                    ).float()
                    for item in depth_stats
                ]
            ).mean()
            diagnostics["num_depth_blocks"] = (
                depth_outputs.num_depth_blocks
            )
            diagnostics["partial_final_block_size"] = (
                depth_outputs.partial_final_block_size
            )

        trace = None
        if output_hidden_states:
            trace = BackboneHiddenStateTrace(
                embedding=hidden_states,
                pre_attention=tuple(pre_attention_states),
                attention_outputs=tuple(attention_outputs),
                pre_ffn=tuple(pre_ffn_states),
                ffn_outputs=tuple(ffn_outputs),
                final_mixed=final_mixed,
            )
        return HybridBackboneOutput(
            last_hidden_state=output,
            cache=backbone_cache,
            hidden_states=(
                tuple(legacy_states) if output_hidden_states else None
            ),
            hidden_state_trace=trace,
            depth_outputs=depth_outputs,
            diagnostics=diagnostics,
        )
