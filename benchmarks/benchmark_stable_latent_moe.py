"""Correctness-oriented CPU benchmark for Stable LatentMoE backends."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stable_latent_moe import StableLatentMoE, StableLatentMoEConfig


def process_peak_memory_bytes() -> int:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        return int(counters.PeakWorkingSetSize)
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def measure_forward(model, inputs, repeats):
    with torch.inference_mode():
        output = model(inputs, return_router_diagnostics=True)
        durations = []
        for _ in range(repeats):
            started = time.perf_counter()
            output = model(inputs, return_router_diagnostics=True)
            durations.append(time.perf_counter() - started)
    return min(durations), output


def measure_backward(model, inputs, repeats):
    durations = []
    for _ in range(repeats):
        value = inputs.detach().clone().requires_grad_()
        model.zero_grad(set_to_none=True)
        started = time.perf_counter()
        model(value).square().mean().backward()
        durations.append(time.perf_counter() - started)
    return min(durations)


def make_model(backend, top_k, experts, d_model, latent_dim):
    return StableLatentMoE(
        StableLatentMoEConfig(
            d_model=d_model,
            latent_dim=latent_dim,
            num_shared_experts=2,
            num_routed_experts=experts,
            routed_experts_per_token=top_k,
            shared_expert_hidden_dim=2 * d_model,
            routed_expert_hidden_dim=2 * latent_dim,
            routing_backend=backend,
            enable_quantile_balancing=top_k < experts,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 64, 256])
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    torch.manual_seed(0)
    print(
        "backend,routing,tokens,top_k,forward_ms,forward_backward_ms,"
        "assignments,zero_load_experts,load_cv,process_peak_bytes"
    )
    for top_k in args.top_k:
        for tokens in args.tokens:
            inputs = torch.randn(tokens, args.d_model)
            for routing in ("natural", "imbalanced"):
                reference = make_model(
                    "reference",
                    top_k,
                    args.experts,
                    args.d_model,
                    args.latent_dim,
                )
                vectorized = make_model(
                    "vectorized",
                    top_k,
                    args.experts,
                    args.d_model,
                    args.latent_dim,
                )
                vectorized.load_state_dict(reference.state_dict())
                for backend, model in (
                    ("reference", reference),
                    ("vectorized", vectorized),
                ):
                    if routing == "imbalanced":
                        with torch.no_grad():
                            model.routing_bias.copy_(
                                torch.linspace(
                                    4.0,
                                    -4.0,
                                    args.experts,
                                )
                            )
                    forward, output = measure_forward(
                        model, inputs, args.repeats
                    )
                    backward = measure_backward(
                        model, inputs, args.repeats
                    )
                    diagnostics = output.diagnostics
                    print(
                        f"{backend},{routing},{tokens},{top_k},"
                        f"{forward * 1000:.3f},{backward * 1000:.3f},"
                        f"{diagnostics.num_assignments},"
                        f"{int(diagnostics.zero_load_experts)},"
                        f"{float(diagnostics.coefficient_of_variation):.4f},"
                        f"{process_peak_memory_bytes()}"
                    )


if __name__ == "__main__":
    main()
