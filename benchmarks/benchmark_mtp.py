"""Small CPU benchmark for the standalone Phase-9 MTP head."""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from src.attention_residuals import AttentionResidualConfig
from src.kda import KDAConfig
from src.kimi_block import KimiBlock, KimiBlockConfig
from src.mla import GatedMLAConfig
from src.mtp import KimiMTPConfig, KimiMTPHead, mtp_parameter_counts
from src.stable_latent_moe import StableLatentMoEConfig


def build_modules(
    vocab_size: int = 128,
    d_model: int = 16,
) -> tuple[nn.Embedding, KimiBlock, nn.Linear, KimiMTPHead]:
    kda_config = KDAConfig(
        d_model=d_model,
        num_heads=2,
        key_head_dim=4,
        value_head_dim=d_model // 2,
        short_conv_kernel_size=2,
        chunk_size=8,
        secondary_tile_size=4,
        decay_initializer="zeros",
    )
    mla_config = GatedMLAConfig(
        d_model=d_model,
        num_heads=2,
        q_head_dim=4,
        v_head_dim=d_model // 2,
        kv_latent_dim=8,
        attention_backend="manual",
    )
    moe_config = StableLatentMoEConfig(
        d_model=d_model,
        latent_dim=d_model // 2,
        num_shared_experts=2,
        num_routed_experts=4,
        routed_experts_per_token=2,
        shared_expert_hidden_dim=12,
        routed_expert_hidden_dim=10,
    )
    attnres_config = AttentionResidualConfig(
        d_model,
        mode="block",
        sublayers_per_depth_block=4,
    )
    config = KimiMTPConfig(
        d_model=d_model,
        vocab_size=vocab_size,
        kda_config=kda_config,
        mla_config=mla_config,
        stable_latent_moe_config=moe_config,
        attention_residual_config=attnres_config,
    )
    embedding = nn.Embedding(vocab_size, d_model)
    lm_head = nn.Linear(d_model, vocab_size, bias=False)
    lm_head.weight = embedding.weight
    backbone = KimiBlock(
        KimiBlockConfig(
            d_model=d_model,
            num_pattern_repeats=1,
            kda_config=kda_config,
            mla_config=mla_config,
            stable_latent_moe_config=moe_config,
            attention_residual_config=attnres_config,
        )
    )
    return (
        embedding.eval(),
        backbone.eval(),
        lm_head.eval(),
        KimiMTPHead(config, embedding, lm_head).eval(),
    )


def median_seconds(function, iterations: int) -> float:
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return sorted(samples)[len(samples) // 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(1)
    embedding, backbone, lm_head, head = build_modules()
    ids = torch.randint(0, head.config.vocab_size, (args.batch_size, args.tokens))
    def main_only():
        main_hidden = backbone(embedding(ids)).last_hidden_state
        return lm_head(main_hidden)

    def main_with_mtp():
        main_hidden = backbone(embedding(ids)).last_hidden_state
        main_logits = lm_head(main_hidden)
        mtp_output = head(main_hidden, ids)
        return main_logits, mtp_output.logits

    with torch.inference_mode():
        main_only()
        main_with_mtp()
        main_seconds = median_seconds(main_only, args.iterations)
        combined_seconds = median_seconds(
            main_with_mtp, args.iterations
        )
    counts = mtp_parameter_counts(head)
    overhead = 100.0 * (combined_seconds / main_seconds - 1.0)
    print(f"main_only_median_ms={1e3 * main_seconds:.3f}")
    print(f"main_plus_mtp_median_ms={1e3 * combined_seconds:.3f}")
    print(f"mtp_latency_overhead_percent={overhead:.2f}")
    print(f"unique_mtp_parameters={counts['unique_mtp_total']}")


if __name__ == "__main__":
    main()
