"""Small, size-one-safe distributed collective helpers."""

from __future__ import annotations

import torch
import torch.distributed as dist


def group_size(group=None) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size(group)


def group_rank(group=None) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank(group)


def all_reduce_sum(tensor: torch.Tensor, group=None) -> torch.Tensor:
    if group_size(group) > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor


def all_reduce_mean(tensor: torch.Tensor, group=None) -> torch.Tensor:
    all_reduce_sum(tensor, group)
    if group_size(group) > 1:
        tensor.div_(group_size(group))
    return tensor


def all_reduce_max(tensor: torch.Tensor, group=None) -> torch.Tensor:
    if group_size(group) > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    return tensor


def all_ranks_true(value: bool, *, device: torch.device, group=None) -> bool:
    flag = torch.tensor(int(value), dtype=torch.int32, device=device)
    if group_size(group) > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def any_rank_true(value: bool, *, device: torch.device, group=None) -> bool:
    flag = torch.tensor(int(value), dtype=torch.int32, device=device)
    if group_size(group) > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
    return bool(flag.item())


def all_gather_variable(
    tensor: torch.Tensor, *, dim: int = 0, group=None
) -> tuple[torch.Tensor, list[int]]:
    """Gather a tensor whose selected dimension may differ by rank."""
    size = group_size(group)
    if size == 1:
        return tensor, [tensor.shape[dim]]
    local = torch.tensor(
        tensor.shape[dim], dtype=torch.long, device=tensor.device
    )
    gathered_sizes = [torch.zeros_like(local) for _ in range(size)]
    dist.all_gather(gathered_sizes, local, group=group)
    lengths = [int(value.item()) for value in gathered_sizes]
    maximum = max(lengths)
    if tensor.shape[dim] < maximum:
        padding_shape = list(tensor.shape)
        padding_shape[dim] = maximum - tensor.shape[dim]
        tensor = torch.cat(
            [tensor, tensor.new_zeros(padding_shape)], dim=dim
        )
    gathered = [torch.empty_like(tensor) for _ in range(size)]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    pieces = [
        value.narrow(dim, 0, length)
        for value, length in zip(gathered, lengths)
    ]
    return torch.cat(pieces, dim=dim), lengths


__all__ = [
    "all_gather_variable",
    "all_ranks_true",
    "all_reduce_max",
    "all_reduce_mean",
    "all_reduce_sum",
    "any_rank_true",
    "group_rank",
    "group_size",
]
