"""Deterministic DP×TP×EP rank mapping and process groups."""

from __future__ import annotations

from itertools import product

import torch
import torch.distributed as dist

from .config import DistributedConfig
from .environment import DistributedContext, with_topology


def rank_from_coordinates(
    dp_rank: int,
    tp_rank: int,
    ep_rank: int,
    *,
    tp_size: int,
    ep_size: int,
) -> int:
    return (dp_rank * tp_size + tp_rank) * ep_size + ep_rank


def coordinates_from_rank(
    rank: int, *, tp_size: int, ep_size: int
) -> tuple[int, int, int]:
    dp_rank, remainder = divmod(rank, tp_size * ep_size)
    tp_rank, ep_rank = divmod(remainder, ep_size)
    return dp_rank, tp_rank, ep_rank


def _all_axis_groups(dp_size: int, tp_size: int, ep_size: int):
    dp_groups = [
        [
            rank_from_coordinates(dp, tp, ep, tp_size=tp_size, ep_size=ep_size)
            for dp in range(dp_size)
        ]
        for tp, ep in product(range(tp_size), range(ep_size))
    ]
    tp_groups = [
        [
            rank_from_coordinates(dp, tp, ep, tp_size=tp_size, ep_size=ep_size)
            for tp in range(tp_size)
        ]
        for dp, ep in product(range(dp_size), range(ep_size))
    ]
    ep_groups = [
        [
            rank_from_coordinates(dp, tp, ep, tp_size=tp_size, ep_size=ep_size)
            for ep in range(ep_size)
        ]
        for dp, tp in product(range(dp_size), range(tp_size))
    ]
    return dp_groups, tp_groups, ep_groups


def build_device_mesh(
    context: DistributedContext, config: DistributedConfig
) -> DistributedContext:
    """Create every subgroup in identical order on all ranks."""
    if not context.initialized:
        return context
    sizes = (
        config.data_parallel.size,
        config.tensor_parallel.size,
        config.expert_parallel.size,
    )
    coordinates = coordinates_from_rank(
        context.global_rank, tp_size=sizes[1], ep_size=sizes[2]
    )
    all_groups = _all_axis_groups(*sizes)
    selected: list[object | None] = [None, None, None]
    for axis, axis_groups in enumerate(all_groups):
        for ranks in axis_groups:
            group = None if len(ranks) == 1 else dist.new_group(ranks=ranks)
            if context.global_rank in ranks:
                selected[axis] = group

    mesh = None
    try:
        from torch.distributed.device_mesh import DeviceMesh

        rank_grid = torch.arange(context.world_size).reshape(sizes)
        mesh = DeviceMesh(
            context.device.type,
            rank_grid,
            mesh_dim_names=("dp", "tp", "ep"),
            _init_backend=False,
        )
    except (ImportError, TypeError, RuntimeError):
        # Process groups remain the compatibility contract on older PyTorch.
        mesh = None
    return with_topology(
        context,
        mesh=mesh,
        dp_group=selected[0],
        tp_group=selected[1],
        ep_group=selected[2],
        dp_rank=coordinates[0],
        tp_rank=coordinates[1],
        ep_rank=coordinates[2],
    )


__all__ = [
    "build_device_mesh",
    "coordinates_from_rank",
    "rank_from_coordinates",
]
