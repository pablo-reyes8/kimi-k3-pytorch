"""Basic CPU benchmark for the pre-MoE Kimi K3 hybrid backbone."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hybrid_backbone import HybridAttentionBackbone, HybridBackboneConfig
from src.kda import KDAConfig
from src.mla import GatedMLAConfig


def process_peak_memory_bytes() -> int:
    """Return process peak RSS without adding a benchmark dependency."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
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

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE
        handle = get_current_process()
        get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_memory_info.restype = wintypes.BOOL
        succeeded = get_memory_info(
            handle, ctypes.byref(counters), counters.cb
        )
        if not succeeded:
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def make_model(groups: int) -> HybridAttentionBackbone:
    d_model, heads = 32, 4
    return HybridAttentionBackbone(
        HybridBackboneConfig(
            d_model=d_model,
            num_hybrid_groups=groups,
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
        )
    ).eval()


def measure(callable_, repeats: int):
    durations = []
    result = None
    with torch.inference_mode():
        result = callable_()
        for _ in range(repeats):
            started = time.perf_counter()
            result = callable_()
            durations.append(time.perf_counter() - started)
    return min(durations), result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--lengths", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    torch.manual_seed(0)
    print(
        "groups,length,full_seconds,prefill_seconds,decode_ms,"
        "cache_elements,cache_bytes,process_peak_bytes"
    )
    for groups in args.groups:
        model = make_model(groups)
        for length in args.lengths:
            hidden = torch.randn(1, length, model.config.d_model)
            full_time, _ = measure(
                lambda: model(hidden, mode="full"), args.repeats
            )
            prefill_time, prefill = measure(
                lambda: model(
                    hidden, mode="prefill", use_cache=True
                ),
                args.repeats,
            )
            token = torch.randn(1, 1, model.config.d_model)
            decode_time, _ = measure(
                lambda: model(
                    token,
                    cache=prefill.cache,
                    mode="decode",
                    use_cache=True,
                ),
                args.repeats,
            )
            cache_bytes = sum(
                tensor.numel() * tensor.element_size()
                for layer in prefill.cache.layer_caches
                for tensor in (
                    (
                        layer.state.recurrent_state,
                        layer.state.q_conv_state.buffer,
                        layer.state.k_conv_state.buffer,
                        layer.state.v_conv_state.buffer,
                    )
                    if layer.attention_type == "kda"
                    else (layer.state.latent_kv,)
                )
            )
            print(
                f"{groups},{length},{full_time:.6f},{prefill_time:.6f},"
                f"{decode_time * 1000:.3f},{prefill.cache.total_elements},"
                f"{cache_bytes},{process_peak_memory_bytes()}"
            )


if __name__ == "__main__":
    main()
