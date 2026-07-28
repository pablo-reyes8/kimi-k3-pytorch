"""CPU microbenchmarks and decode accounting for Attention Residuals.

The first CSV isolates depth mixing at exactly 2/4/8/16 transformer layers.
The second CSV exercises the integrated hybrid backbone and its persistent
KDA/MLA cache. Timings are correctness-oriented, not kernel claims.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attention_residuals import (
    AttentionResidualConfig,
    AttentionResidualSite,
    BlockAttentionResidualController,
    BlockAttentionResidualState,
    DepthSiteMetadata,
    FullAttentionResidualController,
    FullAttentionResidualState,
)
from src.hybrid_backbone import HybridAttentionBackbone, HybridBackboneConfig
from src.kda import KDAConfig
from src.mla import GatedMLAConfig


def measure(function, repeats: int):
    with torch.inference_mode():
        result = function()
        durations = []
        for _ in range(repeats):
            started = time.perf_counter()
            result = function()
            durations.append(time.perf_counter() - started)
    return min(durations), result


class DepthMixMicrostack(nn.Module):
    def __init__(
        self,
        d_model: int,
        transformer_layers: int,
        mode: str,
        backend: str,
        block_size: int,
    ):
        super().__init__()
        self.mode = mode
        self.backend = backend
        self.block_size = block_size
        self.transforms = nn.ModuleList(
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(2 * transformer_layers)
        )
        if mode == "standard":
            self.sites = nn.ModuleList()
            self.final_site = None
            self.controller = None
            return
        self.sites = nn.ModuleList(
            AttentionResidualSite(
                d_model,
                metadata=DepthSiteMetadata(
                    site_index=index,
                    transformer_layer_index=index // 2,
                    site_kind=(
                        "pre_attention" if index % 2 == 0 else "pre_ffn"
                    ),
                    attention_type="kda",
                    hybrid_group_index=None,
                    position_in_hybrid_group=None,
                    depth_block_index=(
                        index // block_size if mode == "block" else None
                    ),
                    position_in_depth_block=(
                        index % block_size if mode == "block" else None
                    ),
                ),
            )
            for index in range(2 * transformer_layers)
        )
        self.final_site = AttentionResidualSite(
            d_model,
            metadata=DepthSiteMetadata(
                2 * transformer_layers,
                None,
                "final_output",
                None,
                None,
                None,
                None,
                None,
            ),
        )
        self.controller = (
            FullAttentionResidualController()
            if mode == "full"
            else BlockAttentionResidualController(block_size, backend)
        )

    def forward(self, embedding: torch.Tensor):
        if self.mode == "standard":
            hidden = embedding
            for transform in self.transforms:
                hidden = hidden + torch.tanh(transform(hidden))
            source_elements = 2 * embedding.numel()
            return hidden, source_elements, 2 * len(self.transforms), 0

        state = self.controller.initialize(embedding)
        sites_by_block = {}
        for site in self.sites:
            if self.mode == "block":
                sites_by_block.setdefault(
                    site.metadata.depth_block_index, []
                ).append(site)
        sites_by_block = {
            key: tuple(value) for key, value in sites_by_block.items()
        }
        peak_elements = state.source_elements
        conceptual_reads = 0
        for site, transform in zip(self.sites, self.transforms):
            if self.mode == "block":
                self.controller.prepare_depth_block(
                    state,
                    sites_by_block[site.metadata.depth_block_index],
                )
            source_count = (
                len(state.sources)
                if isinstance(state, FullAttentionResidualState)
                else 1
                + len(state.completed_blocks)
                + int(state.partial_block is not None)
            )
            conceptual_reads += source_count
            mixed = self.controller.mix_for_site(site, state).mixed_state
            self.controller.append_output(
                state, torch.tanh(transform(mixed))
            )
            peak_elements = max(peak_elements, state.source_elements)
        final_source_count = (
            len(state.sources)
            if isinstance(state, FullAttentionResidualState)
            else 1 + math.ceil(len(self.sites) / self.block_size)
        )
        conceptual_reads += final_source_count
        output = self.controller.finalize(state, self.final_site).mixed_state
        peak_elements = max(peak_elements, state.source_elements)
        scans = (
            state.inter_block_scan_count
            if isinstance(state, BlockAttentionResidualState)
            else 0
        )
        return output, peak_elements, conceptual_reads, scans


def make_hybrid_model(mode: str, backend: str) -> HybridAttentionBackbone:
    d_model, heads = 32, 4
    attnres = None
    if mode == "full":
        attnres = AttentionResidualConfig(
            d_model,
            mode="full",
            sublayers_per_depth_block=None,
        )
    elif mode == "block":
        attnres = AttentionResidualConfig(
            d_model,
            mode="block",
            sublayers_per_depth_block=4,
            backend=backend,
        )
    config = HybridBackboneConfig(
        d_model=d_model,
        num_hybrid_groups=1,
        mlp_hidden_dim=64,
        kda_config=KDAConfig(
            d_model=d_model,
            num_heads=heads,
            key_head_dim=4,
            value_head_dim=8,
            short_conv_kernel_size=4,
            chunk_size=64,
            secondary_tile_size=16,
        ),
        mla_config=GatedMLAConfig(
            d_model=d_model,
            num_heads=heads,
            q_head_dim=8,
            v_head_dim=8,
            kv_latent_dim=16,
        ),
        attention_residual_config=attnres,
    )
    return HybridAttentionBackbone(config).eval()


def cache_bytes(cache) -> int:
    total = 0
    for layer in cache.layer_caches:
        tensors = (
            (
                layer.state.recurrent_state,
                layer.state.q_conv_state.buffer,
                layer.state.k_conv_state.buffer,
                layer.state.v_conv_state.buffer,
            )
            if layer.attention_type == "kda"
            else (layer.state.latent_kv,)
        )
        total += sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--lengths", nargs="+", type=int, default=[1, 64, 256])
    parser.add_argument("--decode-contexts", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    torch.manual_seed(0)

    print(
        "kind,mode,backend,layers,length,seconds,peak_source_bytes,"
        "conceptual_source_reads,inter_block_scans"
    )
    modes = (
        ("standard", "eager"),
        ("full", "eager"),
        ("block", "eager"),
        ("block", "two_phase"),
    )
    for layers in args.layers:
        for length in args.lengths:
            hidden = torch.randn(1, length, args.d_model)
            for mode, backend in modes:
                stack = DepthMixMicrostack(
                    args.d_model,
                    layers,
                    mode,
                    backend,
                    args.block_size,
                ).eval()
                seconds, result = measure(
                    lambda: stack(hidden), args.repeats
                )
                _, peak_elements, reads, scans = result
                print(
                    f"depth,{mode},{backend},{layers},{length},"
                    f"{seconds:.6f},{peak_elements * hidden.element_size()},"
                    f"{reads},{scans}"
                )

    print(
        "kind,mode,backend,context,decode_ms,persistent_cache_elements,"
        "persistent_cache_bytes,attnres_cache_entries"
    )
    for mode, backend in modes:
        model = make_hybrid_model(mode, backend)
        for context in args.decode_contexts:
            hidden = torch.randn(1, context, model.config.d_model)
            with torch.inference_mode():
                prefill = model(hidden, mode="prefill", use_cache=True)
            token = torch.randn(1, 1, model.config.d_model)
            seconds, decoded = measure(
                lambda: model(
                    token,
                    cache=prefill.cache,
                    mode="decode",
                    use_cache=True,
                ),
                args.repeats,
            )
            print(
                f"decode,{mode},{backend},{context},{seconds * 1000:.3f},"
                f"{decoded.cache.total_elements},{cache_bytes(decoded.cache)},0"
            )


if __name__ == "__main__":
    main()
