"""Stable LatentMoE routing, expert dispatch, and load-balancing components."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


def _validate_dispatch(
    latent: torch.Tensor,
    experts: Sequence[nn.Module],
    selected_experts: torch.Tensor,
    selected_weights: torch.Tensor,
) -> tuple[int, int, int]:
    if latent.ndim != 2:
        raise ValueError("latent must have shape [M,L]")
    if selected_experts.ndim != 2:
        raise ValueError("selected_experts must have shape [M,k]")
    if selected_weights.shape != selected_experts.shape:
        raise ValueError("selected weights and indices must have equal shape")
    if selected_experts.shape[0] != latent.shape[0]:
        raise ValueError("routing token count must match latent token count")
    if selected_experts.dtype != torch.long:
        raise TypeError("selected_experts must use int64")
    if not experts:
        raise ValueError("at least one routed expert is required")
    if selected_experts.numel() and (
        selected_experts.min() < 0
        or selected_experts.max() >= len(experts)
    ):
        raise ValueError("selected expert index out of range")
    return latent.shape[0], latent.shape[1], selected_experts.shape[1]


def reference_sparse_dispatch(
    latent: torch.Tensor,
    experts: Sequence[nn.Module],
    selected_experts: torch.Tensor,
    selected_weights: torch.Tensor,
    *,
    accumulation_dtype: torch.dtype,
) -> torch.Tensor:
    tokens, latent_dim, top_k = _validate_dispatch(
        latent, experts, selected_experts, selected_weights
    )
    aggregate = torch.zeros(
        tokens,
        latent_dim,
        device=latent.device,
        dtype=accumulation_dtype,
    )
    for token in range(tokens):
        for slot in range(top_k):
            expert_index = int(selected_experts[token, slot])
            expert_output = experts[expert_index](latent[token : token + 1])
            weighted = (
                expert_output.to(accumulation_dtype)
                * selected_weights[token, slot].to(accumulation_dtype)
            )
            token_index = torch.tensor(
                [token], device=latent.device, dtype=torch.long
            )
            aggregate = aggregate.index_add(0, token_index, weighted)
    return aggregate


def vectorized_sparse_dispatch(
    latent: torch.Tensor,
    experts: Sequence[nn.Module],
    selected_experts: torch.Tensor,
    selected_weights: torch.Tensor,
    *,
    accumulation_dtype: torch.dtype,
) -> torch.Tensor:
    tokens, latent_dim, top_k = _validate_dispatch(
        latent, experts, selected_experts, selected_weights
    )
    flat_experts = selected_experts.reshape(-1)
    flat_tokens = (
        torch.arange(tokens, device=latent.device)[:, None]
        .expand(tokens, top_k)
        .reshape(-1)
    )
    flat_weights = selected_weights.reshape(-1)
    order = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts[order]
    sorted_tokens = flat_tokens[order]
    sorted_weights = flat_weights[order]
    aggregate = torch.zeros(
        tokens,
        latent_dim,
        device=latent.device,
        dtype=accumulation_dtype,
    )
    for expert_index, expert in enumerate(experts):
        assignment_mask = sorted_experts == expert_index
        if not assignment_mask.any():
            continue
        token_ids = sorted_tokens[assignment_mask]
        expert_output = expert(latent.index_select(0, token_ids))
        weighted = expert_output.to(accumulation_dtype) * sorted_weights[
            assignment_mask, None
        ].to(accumulation_dtype)
        aggregate = aggregate.index_add(0, token_ids, weighted)
    return aggregate
