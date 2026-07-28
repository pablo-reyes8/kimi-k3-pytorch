"""Bounded activation-health statistics for sampled hidden states."""

from __future__ import annotations

import torch

from .reducers import rms, scalar


def compute_activation_metrics(
    tensor: torch.Tensor,
    *,
    max_elements: int = 8192,
    prefix: str = "activation",
) -> dict[str, float]:
    if max_elements <= 0:
        raise ValueError("max_elements must be positive")
    values = tensor.detach().reshape(-1)[:max_elements].float()
    if values.numel() == 0:
        return {}
    finite_mask = torch.isfinite(values)
    finite = torch.where(finite_mask, values, torch.zeros_like(values))
    return {
        f"{prefix}/rms": rms(finite),
        f"{prefix}/mean": scalar(finite.mean()),
        f"{prefix}/std": scalar(finite.std(unbiased=False)),
        f"{prefix}/absmax": scalar(finite.abs().max()),
        f"{prefix}/zero_fraction": scalar((finite == 0).float().mean()),
        f"{prefix}/nonfinite_fraction": scalar(
            (~finite_mask).float().mean()
        ),
    }


__all__ = ["compute_activation_metrics"]
