"""Group baselines and action-token broadcasting for Kimi policy training."""

from __future__ import annotations

import torch


def group_mean_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    expected_group_size: int | None,
    require_complete_groups: bool,
) -> torch.Tensor:
    """Subtract the exact reward mean within each prompt group."""

    if rewards.ndim != 1 or group_ids.shape != rewards.shape:
        raise ValueError("rewards and prompt_group_ids must have shape [B]")
    if group_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("prompt_group_ids must use int32 or int64")
    if rewards.device != group_ids.device:
        raise ValueError("rewards and prompt_group_ids must share device")
    if not torch.isfinite(rewards).all():
        raise FloatingPointError("rewards contain non-finite values")
    unique, inverse, counts = torch.unique(
        group_ids, sorted=False, return_inverse=True, return_counts=True
    )
    del unique
    if require_complete_groups and expected_group_size is not None:
        if expected_group_size <= 0:
            raise ValueError("expected_group_size must be > 0")
        if torch.any(counts != expected_group_size):
            raise ValueError("prompt group is incomplete")
    sums = torch.zeros(counts.numel(), device=rewards.device, dtype=torch.float32)
    sums.scatter_add_(0, inverse, rewards.detach().float())
    means = sums / counts.float()
    return rewards.detach().float() - means[inverse]

