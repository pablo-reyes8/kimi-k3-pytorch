"""Manual A/B benchmark for Kimi training diagnostic levels.

Examples:
    PYTHONPATH=. python scripts/benchmark_training_diagnostics.py \
        --device cuda --contexts 512 2048 4096 --steps 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
from torch.utils.data import DataLoader

from src import KimiK3, kimi_k3_cpu_tiny_config
from training import DiagnosticsConfig, KimiDiagnosticCollector, train_one_epoch
from training.optimizer import KimiOptimizerConfig, build_kimi_optimizer


def percentile90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]


def run_mode(mode: str, context: int, steps: int, device: torch.device) -> dict:
    torch.manual_seed(17)
    model = KimiK3(
        kimi_k3_cpu_tiny_config(enable_vision=False)
    ).to(device)
    optimizer, registry = build_kimi_optimizer(
        model,
        KimiOptimizerConfig(qk_clip_enabled=True),
    )
    intervals = {
        "cheap": (steps + 1, steps + 1),
        "standard": (1, steps + 1),
        "deep": (steps + 1, 1),
    }
    collector = None
    if mode != "off":
        standard, deep = intervals[mode]
        collector = KimiDiagnosticCollector(
            model,
            DiagnosticsConfig(
                cheap_every_steps=1,
                standard_every_steps=standard,
                deep_every_steps=deep,
            ),
            parameter_specs=registry.specs,
        )
    generator = torch.Generator().manual_seed(123)
    batches = [
        {
            "input_ids": torch.randint(
                5, 128, (context,), generator=generator
            ),
            "labels": torch.randint(
                5, 128, (context,), generator=generator
            ),
            "attention_mask": torch.ones(context, dtype=torch.bool),
        }
        for _ in range(steps)
    ]
    step_metrics = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    train_one_epoch(
        model,
        DataLoader(batches, batch_size=1),
        optimizer,
        device=device,
        diagnostics=collector,
        max_batches=steps,
        use_mtp=True,
        on_optimizer_step=lambda metrics: step_metrics.append(dict(metrics)),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    times = [item["train/step_time_ms"] for item in step_metrics]
    tokens = sum(item["tokens"] for item in step_metrics)
    return {
        "mode": mode,
        "context": context,
        "steps": steps,
        "median_step_time_ms": statistics.median(times),
        "p90_step_time_ms": percentile90(times),
        "tokens_per_second": tokens / elapsed,
        "max_allocated_gpu_mb": (
            torch.cuda.max_memory_allocated(device) / 1024 ** 2
            if device.type == "cuda" else 0.0
        ),
        "persistent_diagnostic_bytes": max(
            (
                item.get("diagnostics/persistent_gpu_bytes", 0.0)
                for item in step_metrics
            ),
            default=0.0,
        ),
        "scalars": max(
            (
                item.get("diagnostics/scalars_emitted", 0.0)
                for item in step_metrics
            ),
            default=0.0,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    device = torch.device(args.device)
    rows = []
    for context in args.contexts:
        baseline = run_mode("off", context, args.steps, device)
        rows.append({**baseline, "overhead_vs_off": 0.0})
        for mode in ("cheap", "standard", "deep"):
            result = run_mode(mode, context, args.steps, device)
            result["overhead_vs_off"] = (
                result["median_step_time_ms"]
                / max(baseline["median_step_time_ms"], 1e-12)
                - 1.0
            )
            rows.append(result)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
