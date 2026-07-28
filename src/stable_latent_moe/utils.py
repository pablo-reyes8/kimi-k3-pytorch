"""Stable LatentMoE routing, expert dispatch, and load-balancing components."""

from __future__ import annotations

import torch


def policy_dtype(input_dtype: torch.dtype, policy: str) -> torch.dtype:
    if policy == "input":
        return input_dtype
    if policy == "float32":
        return (
            torch.float64
            if input_dtype == torch.float64
            else torch.float32
        )
    raise ValueError(f"unknown dtype policy: {policy}")


def rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()
