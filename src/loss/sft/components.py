"""Component weights and assistant-target masks for SFT trajectories."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def resolve_component_weights(
    component_ids: torch.Tensor,
    component_weights: Mapping[int, float] | torch.Tensor | None,
) -> torch.Tensor:
    """Map component IDs to non-negative per-token weights."""

    if component_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("component_ids must use int32 or int64")
    if torch.any(component_ids < 0):
        raise ValueError("component_ids must be non-negative")
    if component_weights is None:
        return torch.ones_like(component_ids, dtype=torch.float32)
    if isinstance(component_weights, torch.Tensor):
        table = component_weights.float().to(component_ids.device)
        if table.ndim != 1:
            raise ValueError("component weight tensor must be one-dimensional")
    else:
        if not component_weights:
            raise ValueError("component_weights mapping must not be empty")
        largest = max(int(key) for key in component_weights)
        table = torch.ones(largest + 1, device=component_ids.device)
        for key, value in component_weights.items():
            table[int(key)] = float(value)
    if not torch.isfinite(table).all() or torch.any(table < 0):
        raise ValueError("component weights must be finite and non-negative")
    if torch.any(component_ids >= table.numel()):
        raise ValueError("component_ids reference an unspecified component weight")
    return table[component_ids.long()]

