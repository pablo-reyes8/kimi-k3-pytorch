"""Small CPU benchmark for the pure-PyTorch KDA reference implementation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.kda import KDAConfig, KimiDeltaAttention


def state_elements(state) -> int:
    return sum(
        tensor.numel()
        for tensor in (
            state.recurrent_state,
            state.q_conv_state.buffer,
            state.k_conv_state.buffer,
            state.v_conv_state.buffer,
            state.sequence_offset,
        )
    )


def measure(model, hidden_states, mode, repeats):
    durations = []
    with torch.inference_mode():
        model(hidden_states, mode=mode, output_final_state=True)
        for _ in range(repeats):
            started = time.perf_counter()
            output = model(
                hidden_states, mode=mode, output_final_state=True
            )
            durations.append(time.perf_counter() - started)
    return min(durations), output.state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", nargs="+", type=int, default=[64, 256, 1024, 4096])
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    torch.manual_seed(0)
    model = KimiDeltaAttention(
        KDAConfig(
            d_model=16,
            num_heads=2,
            key_head_dim=4,
            value_head_dim=8,
            chunk_size=64,
            secondary_tile_size=16,
        )
    ).eval()
    print("length,recurrent_seconds,chunkwise_seconds,decode_ms,state_elements")
    for length in args.lengths:
        hidden = torch.randn(1, length, 16)
        recurrent_time, recurrent_state = measure(
            model, hidden, "recurrent", args.repeats
        )
        chunk_time, chunk_state = measure(
            model, hidden, "chunkwise", args.repeats
        )
        token = hidden[:, :1]
        started = time.perf_counter()
        with torch.inference_mode():
            model(token, state=chunk_state, mode="decode")
        decode_ms = (time.perf_counter() - started) * 1000
        assert state_elements(recurrent_state) == state_elements(chunk_state)
        print(
            f"{length},{recurrent_time:.6f},{chunk_time:.6f},"
            f"{decode_ms:.3f},{state_elements(chunk_state)}"
        )


if __name__ == "__main__":
    main()
