"""Pure Newton-Schulz and update-RMS helpers."""

from __future__ import annotations

from typing import Literal

import torch


@torch.no_grad()
def zeropower_via_newton_schulz(
    matrix: torch.Tensor,
    *,
    steps: int = 5,
    eps: float = 1e-7,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("Newton-Schulz input must be a 2D matrix")
    if steps <= 0 or eps <= 0:
        raise ValueError("steps and eps must be positive")
    work = matrix.detach().float().clone()
    if not torch.isfinite(work).all():
        raise FloatingPointError("Newton-Schulz input contains NaN or Inf")
    if float(work.norm().item()) <= eps:
        return torch.zeros_like(
            matrix, dtype=output_dtype or matrix.dtype
        )
    transposed = work.shape[0] > work.shape[1]
    if transposed:
        work = work.T
    work.div_(work.norm().clamp_min(eps))
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = work @ work.T
        work = a * work + b * (gram @ work) + c * ((gram @ gram) @ work)
    if transposed:
        work = work.T
    return work.to(
        device=matrix.device,
        dtype=output_dtype or matrix.dtype,
    )


@torch.no_grad()
def match_update_rms(
    orthogonal_update: torch.Tensor,
    reference: torch.Tensor | None = None,
    *,
    mode: Literal["shape", "reference_rms"] = "shape",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Make update RMS independent of matrix aspect ratio.

    ``shape`` scales a semi-orthogonal matrix by ``sqrt(max(rows, cols))``,
    giving unit RMS for an exact semi-orthogonal update. ``reference_rms``
    matches the RMS of the supplied raw momentum/update.
    """

    if orthogonal_update.ndim != 2:
        raise ValueError("orthogonal_update must be 2D")
    if mode == "shape":
        return orthogonal_update * orthogonal_update.new_tensor(
            max(orthogonal_update.shape) ** 0.5
        )
    if mode != "reference_rms":
        raise ValueError("unknown update RMS matching mode")
    if reference is None or reference.shape != orthogonal_update.shape:
        raise ValueError("reference_rms mode requires a matching reference")
    target = reference.detach().float().square().mean().sqrt()
    current = orthogonal_update.detach().float().square().mean().sqrt()
    if float(current.item()) <= eps:
        return torch.zeros_like(orthogonal_update)
    return orthogonal_update * (target / current).to(orthogonal_update.dtype)
