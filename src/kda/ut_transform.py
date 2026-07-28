"""Kimi Delta Attention operators, projections, states, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .utils import tiled_causal_decay_dot


@dataclass
class UTTransformOutput:
    """Upper-triangular transformed tensors used by chunkwise KDA."""

    M: torch.Tensor
    U: torch.Tensor
    W: torch.Tensor
    log_gamma: torch.Tensor
    k_gamma: torch.Tensor


def _ut_transform_internal(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    secondary_tile_size: int,
) -> UTTransformOutput:
    """UT transform for internal ``[B,H,C,*]`` tensors."""
    if k.ndim != 4 or g.shape != k.shape:
        raise ValueError("k and g must have shape [B,H,C,K]")
    if v.ndim != 4 or v.shape[:3] != k.shape[:3]:
        raise ValueError("v must have shape [B,H,C,V] matching k")
    if beta.shape != k.shape[:3]:
        raise ValueError("beta must have shape [B,H,C]")
    count = k.shape[2]
    log_gamma = torch.cumsum(g, dim=2)
    strict_pair = tiled_causal_decay_dot(
        k,
        k,
        log_gamma,
        log_gamma,
        secondary_tile_size,
        include_diagonal=False,
    )
    identity = torch.eye(count, dtype=k.dtype, device=k.device)
    lower = identity + beta[..., None] * strict_pair
    diagonal_beta = torch.diag_embed(beta)
    M = torch.linalg.solve_triangular(
        lower, diagonal_beta, upper=False, unitriangular=True
    )
    k_gamma = torch.exp(log_gamma) * k
    W = torch.matmul(M, k_gamma)
    U = torch.matmul(M, v)
    return UTTransformOutput(
        M=M, U=U, W=W, log_gamma=log_gamma, k_gamma=k_gamma
    )


def ut_transform(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    secondary_tile_size: int = 16,
) -> UTTransformOutput:
    """Public UT transform for canonical ``[B,C,H,*]`` inputs.

    ``M`` is returned as ``[B,H,C,C]``; U/W/log-gamma retain canonical
    ``[B,C,H,*]`` layout.
    """
    if k.ndim != 4 or g.shape != k.shape:
        raise ValueError("k and g must have shape [B,C,H,K]")
    if v.ndim != 4 or v.shape[:3] != k.shape[:3]:
        raise ValueError("v must have shape [B,C,H,V]")
    if beta.shape != k.shape[:3]:
        raise ValueError("beta must have shape [B,C,H]")
    internal = _ut_transform_internal(
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
        g.permute(0, 2, 1, 3),
        beta.permute(0, 2, 1),
        secondary_tile_size,
    )
    return UTTransformOutput(
        M=internal.M,
        U=internal.U.permute(0, 2, 1, 3),
        W=internal.W.permute(0, 2, 1, 3),
        log_gamma=internal.log_gamma.permute(0, 2, 1, 3),
        k_gamma=internal.k_gamma.permute(0, 2, 1, 3),
    )
