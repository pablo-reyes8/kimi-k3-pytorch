"""Token-weighted reductions and one-rank distributed presentation."""

from __future__ import annotations

import math
from typing import Mapping

import torch

from .collectives import all_reduce_sum
from .environment import DistributedContext


def reduce_weighted_mean(
    value: float | torch.Tensor,
    weight: float | torch.Tensor,
    *,
    context: DistributedContext,
    group=None,
) -> tuple[float, float]:
    pair = torch.stack(
        [
            torch.as_tensor(value, device=context.device, dtype=torch.float64)
            * torch.as_tensor(
                weight, device=context.device, dtype=torch.float64
            ),
            torch.as_tensor(weight, device=context.device, dtype=torch.float64),
        ]
    )
    all_reduce_sum(pair, context.dp_group if group is None else group)
    denominator = pair[1].clamp_min(1.0)
    return float((pair[0] / denominator).item()), float(pair[1].item())


def reduce_counter(
    value: int | float,
    *,
    context: DistributedContext,
    group=None,
) -> float:
    tensor = torch.tensor(float(value), device=context.device)
    all_reduce_sum(tensor, context.dp_group if group is None else group)
    return float(tensor.item())


def topology_lines(
    context: DistributedContext,
    *,
    wrapper_mode: str,
    checkpoint_format: str,
    global_batch_size: int | None = None,
) -> list[str]:
    batch = (
        "unknown"
        if global_batch_size is None
        else str(global_batch_size)
    )
    return [
        "Distributed topology",
        f"  backend={context.backend} world_size={context.world_size}",
        (
            f"  rank={context.global_rank} local_rank={context.local_rank} "
            f"coordinates=({context.dp_rank},{context.tp_rank},{context.ep_rank})"
        ),
        (
            f"  DP={context.dp_size} TP={context.tp_size} "
            f"EP={context.ep_size} wrapper={wrapper_mode}"
        ),
        f"  global_batch={batch} checkpoint={checkpoint_format}",
    ]


def print_topology(
    context: DistributedContext,
    *,
    log_rank: int,
    per_rank_debug: bool,
    **kwargs,
) -> None:
    if context.global_rank == log_rank or per_rank_debug:
        print("\n".join(topology_lines(context, **kwargs)))


def reduce_scalar_metrics(
    metrics: Mapping[str, object],
    *,
    context: DistributedContext,
) -> dict[str, object]:
    """Make scalar diagnostics rank-global while preserving non-scalars."""
    if not context.initialized or context.world_size == 1:
        return dict(metrics)
    reduced = dict(metrics)
    for name, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        tensor = torch.tensor(
            numeric, dtype=torch.float64, device=context.device
        )
        leaf = name.rsplit("/", 1)[-1].lower()
        if "max" in leaf or "peak" in leaf:
            torch.distributed.all_reduce(
                tensor, op=torch.distributed.ReduceOp.MAX
            )
        elif "min" in leaf:
            torch.distributed.all_reduce(
                tensor, op=torch.distributed.ReduceOp.MIN
            )
        else:
            torch.distributed.all_reduce(tensor)
            tensor.div_(context.world_size)
        reduced[name] = float(tensor.item())
    return reduced


__all__ = [
    "print_topology",
    "reduce_counter",
    "reduce_scalar_metrics",
    "reduce_weighted_mean",
    "topology_lines",
]
