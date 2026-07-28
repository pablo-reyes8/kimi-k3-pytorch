"""CPU validation/benchmark for the pure-PyTorch Kimi K3 Gated MLA."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mla import GatedMLA, GatedMLAConfig


def timed_forward(model, hidden_states, repeats):
    durations = []
    with torch.inference_mode():
        model(hidden_states, use_cache=True)
        for _ in range(repeats):
            started = time.perf_counter()
            output = model(hidden_states, use_cache=True)
            durations.append(time.perf_counter() - started)
    return min(durations), output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", nargs="+", type=int, default=[64, 256, 1024])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    torch.manual_seed(0)
    config = GatedMLAConfig(
        d_model=64,
        num_heads=4,
        q_head_dim=16,
        v_head_dim=16,
        kv_latent_dim=32,
    )
    model = GatedMLA(config).eval()
    print(
        "length,seconds,decode_ms,latent_elements,full_kv_elements,"
        "compression_ratio"
    )
    for length in args.lengths:
        hidden = torch.randn(1, length, config.d_model)
        duration, prefill = timed_forward(model, hidden, args.repeats)
        token = torch.randn(1, 1, config.d_model)
        started = time.perf_counter()
        with torch.inference_mode():
            model(token, cache=prefill.cache, use_cache=True)
        decode_ms = (time.perf_counter() - started) * 1000
        full_kv_elements = length * config.full_kv_width
        print(
            f"{length},{duration:.6f},{decode_ms:.3f},"
            f"{prefill.cache.cache_elements},{full_kv_elements},"
            f"{config.cache_compression_ratio:.3f}"
        )


if __name__ == "__main__":
    main()
