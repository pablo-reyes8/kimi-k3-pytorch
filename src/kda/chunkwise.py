"""Kimi Delta Attention operators, projections, states, and diagnostics."""

from __future__ import annotations

import torch

from .recurrent import KDAOperatorOutput
from .ut_transform import _ut_transform_internal
from .utils import (
    accumulation_dtype,
    apply_operator_mask,
    tiled_causal_decay_dot,
    validate_core_inputs,
)


def chunkwise_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    attention_mask: torch.Tensor | None = None,
    *,
    chunk_size: int = 64,
    secondary_tile_size: int = 16,
    accumulate_state_in_fp32: bool = True,
) -> KDAOperatorOutput:
    
    """Exact chunkwise KDA using UT/WY parallel computation within each chunk."""

    if chunk_size <= 0 or secondary_tile_size <= 0:
        raise ValueError("chunk_size and secondary_tile_size must be > 0")
    if secondary_tile_size > chunk_size:
        raise ValueError("secondary_tile_size must be <= chunk_size")
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

    output_chunks = []
    for start in range(0, tokens, chunk_size):
        end = min(start + chunk_size, tokens)
        q_chunk = q_work[:, start:end].permute(0, 2, 1, 3)
        k_chunk = k_work[:, start:end].permute(0, 2, 1, 3)
        v_chunk = v_work[:, start:end].permute(0, 2, 1, 3)
        g_chunk = g_work[:, start:end].permute(0, 2, 1, 3)
        beta_chunk = beta_work[:, start:end].permute(0, 2, 1)

        transformed = _ut_transform_internal(
            k_chunk,
            v_chunk,
            g_chunk,
            beta_chunk,
            secondary_tile_size,
        )
        pseudo_values = transformed.U - torch.matmul(transformed.W, state)
        causal_qk = tiled_causal_decay_dot(
            q_chunk,
            k_chunk,
            transformed.log_gamma,
            transformed.log_gamma,
            secondary_tile_size,
            include_diagonal=True,
        )
        inter_chunk = torch.matmul(
            torch.exp(transformed.log_gamma) * q_chunk, state
        )
        intra_chunk = torch.matmul(causal_qk, pseudo_values)
        output_chunks.append((inter_chunk + intra_chunk).permute(0, 2, 1, 3))

        final_log_gamma = transformed.log_gamma[:, :, -1]
        state = state * torch.exp(final_log_gamma)[..., None]
        end_relative_keys = (
            torch.exp(
                final_log_gamma[:, :, None, :]
                - transformed.log_gamma
            )
            * k_chunk
        )
        state = state + torch.matmul(
            end_relative_keys.transpose(-2, -1), pseudo_values
        )

    return KDAOperatorOutput(
        hidden_states=torch.cat(output_chunks, dim=1).to(activation_dtype),
        final_state=state if output_final_state else None,
    )

