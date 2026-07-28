"""Kimi Delta Attention operators, projections, states, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .utils import (
    accumulation_dtype,
    apply_operator_mask,
    validate_core_inputs,
)


@dataclass
class KDAOperatorOutput:
    """Output and updated recurrent matrix from a KDA operator backend."""

    hidden_states: torch.Tensor
    final_state: torch.Tensor | None = None
    states_per_token: tuple[torch.Tensor, ...] | None = None


def recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    attention_mask: torch.Tensor | None = None,
    *,
    accumulate_state_in_fp32: bool = True,
    return_states_per_token: bool = False,
) -> KDAOperatorOutput:
    """Exact KDA recurrence with decay, delta update, then read-after-write."""
    batch, tokens, heads, key_dim, value_dim = validate_core_inputs(
        q, k, v, g, beta, initial_state, attention_mask
    )
    activation_dtype = v.dtype
    work_dtype = accumulation_dtype(activation_dtype, accumulate_state_in_fp32)
    q_work, k_work, v_work, g_work, beta_work = (
        tensor.to(work_dtype) for tensor in (q, k, v, g, beta)
    )
    q_work, g_work, beta_work = apply_operator_mask(
        q_work, g_work, beta_work, attention_mask
    )
    state = torch.zeros(
        batch,
        heads,
        key_dim,
        value_dim,
        dtype=work_dtype,
        device=q.device,
    )
    if initial_state is not None:
        state = state + initial_state.to(work_dtype)
    outputs = []
    state_history = [] if return_states_per_token else None
    for index in range(tokens):
        q_t = q_work[:, index]
        k_t = k_work[:, index]
        v_t = v_work[:, index]
        state = state * torch.exp(g_work[:, index])[..., None]
        recalled = torch.einsum("bhkv,bhk->bhv", state, k_t)
        error = v_t - recalled
        state = state + torch.einsum(
            "bhk,bhv->bhkv", beta_work[:, index, :, None] * k_t, error
        )
        outputs.append(torch.einsum("bhkv,bhk->bhv", state, q_t))
        if state_history is not None:
            state_history.append(state)
    hidden_states = torch.stack(outputs, dim=1).to(activation_dtype)
    return KDAOperatorOutput(
        hidden_states=hidden_states,
        final_state=state if output_final_state else None,
        states_per_token=tuple(state_history) if state_history is not None else None,
    )
