"""Kimi Delta Attention operators, projections, states, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn as nn

from src.kimi_primitives import (
    FullRankOutputGate,
    HeadwiseRMSNorm,
    ShortConvState,
    combine_heads,
)

from .chunkwise import chunkwise_kda
from .config import KDAConfig
from .decay import LowerBoundedDecay
from .diagnostics import build_kda_diagnostics
from .projections import KDAProjections
from .recurrent import recurrent_kda
from .state import KDAState
from .utils import accumulation_dtype, validate_attention_mask


@dataclass
class KDAOutput:
    """Container returned by KDA with output states, cache, and diagnostics."""

    hidden_states: torch.Tensor
    state: KDAState | None = None
    diagnostics: dict[str, torch.Tensor] | None = None


class KimiDeltaAttention(nn.Module):
    """Pure-PyTorch Kimi K3 Delta Attention.

    The module owns one shared parameter set. ``recurrent``, ``chunkwise``, and
    ``decode`` select only the core execution strategy.
    """

    def __init__(self, config: KDAConfig):
        super().__init__()
        self.config = config
        self.projections = KDAProjections(config)
        self.decay = LowerBoundedDecay(config.num_heads, config.g_min)
        self.output_norm = HeadwiseRMSNorm(
            config.num_heads,
            config.value_head_dim,
            eps=config.eps,
            elementwise_affine=True,
            per_head_affine=True,
        )
        self.output_gate = FullRankOutputGate(
            config.d_model,
            bias=config.output_gate_bias,
            output_bias=config.output_bias,
            init_std=config.init_std,
        )

    def _validate_state(
        self, hidden_states: torch.Tensor, state: KDAState | None
    ) -> None:
        if state is None:
            return
        batch = hidden_states.shape[0]
        expected_recurrent = (
            batch,
            self.config.num_heads,
            self.config.key_head_dim,
            self.config.value_head_dim,
        )
        if state.recurrent_state.shape != expected_recurrent:
            raise ValueError(
                f"state recurrent tensor must have shape {expected_recurrent}"
            )
        history = self.config.short_conv_kernel_size - 1
        expected_buffers = (
            (batch, history, self.config.key_width),
            (batch, history, self.config.key_width),
            (batch, history, self.config.value_width),
        )
        actual_buffers = (
            state.q_conv_state.buffer.shape,
            state.k_conv_state.buffer.shape,
            state.v_conv_state.buffer.shape,
        )
        if actual_buffers != expected_buffers:
            raise ValueError(
                f"state ShortConv buffers must have shapes {expected_buffers}"
            )
        if state.recurrent_state.device != hidden_states.device:
            raise ValueError("state and hidden_states must be on the same device")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        state: KDAState | None = None,
        mode: Literal["recurrent", "chunkwise", "decode"] = "chunkwise",
        output_final_state: bool = False,
        output_diagnostics: bool = False,
    ) -> KDAOutput:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.config.d_model:
            raise ValueError(
                f"hidden_states must have shape [B,T,{self.config.d_model}], "
                f"got {tuple(hidden_states.shape)}"
            )
        batch, tokens, _ = hidden_states.shape
        if tokens == 0:
            raise ValueError("KDA does not support empty sequences")
        if mode not in ("recurrent", "chunkwise", "decode"):
            raise ValueError(f"unsupported KDA mode {mode!r}")
        if mode == "decode" and tokens != 1:
            raise ValueError("decode mode requires exactly one token")
        validate_attention_mask(attention_mask, batch, tokens)
        self._validate_state(hidden_states, state)

        return_state = output_final_state or mode == "decode"
        projected = self.projections(
            hidden_states,
            q_conv_state=state.q_conv_state if state is not None else None,
            k_conv_state=state.k_conv_state if state is not None else None,
            v_conv_state=state.v_conv_state if state is not None else None,
            attention_mask=attention_mask,
            output_final_state=return_state,
        )
        g, alpha = self.decay(projected.decay_logits)
        initial_recurrent = state.recurrent_state if state is not None else None
        core_kwargs = dict(
            q=projected.q,
            k=projected.k,
            v=projected.v,
            g=g,
            beta=projected.beta,
            initial_state=initial_recurrent,
            output_final_state=return_state or output_diagnostics,
            attention_mask=attention_mask,
            accumulate_state_in_fp32=self.config.accumulate_state_in_fp32,
        )
        if mode in ("recurrent", "decode"):
            core_output = recurrent_kda(**core_kwargs)
        else:
            core_output = chunkwise_kda(
                **core_kwargs,
                chunk_size=self.config.chunk_size,
                secondary_tile_size=self.config.secondary_tile_size,
            )

        normalized = self.output_norm(core_output.hidden_states)
        combined = combine_heads(normalized)
        final_output, gate = self.output_gate(
            combined, hidden_states, return_gate=True
        )
        if attention_mask is not None:
            final_output = final_output * attention_mask[..., None].to(
                final_output.dtype
            )

        next_state = None
        if return_state:
            lengths = (
                torch.full(
                    (batch,), tokens, dtype=torch.long, device=hidden_states.device
                )
                if attention_mask is None
                else attention_mask.sum(dim=1).to(torch.long)
            )
            offset = (
                lengths
                if state is None
                else state.sequence_offset.to(hidden_states.device) + lengths
            )
            next_state = KDAState(
                recurrent_state=core_output.final_state,
                q_conv_state=projected.q_conv_state,
                k_conv_state=projected.k_conv_state,
                v_conv_state=projected.v_conv_state,
                sequence_offset=offset,
            )

        diagnostics = None
        if output_diagnostics:
            diagnostics_state = core_output.final_state
            diagnostics = build_kda_diagnostics(
                projected.q,
                projected.k,
                projected.beta,
                g,
                alpha,
                diagnostics_state,
                core_output.hidden_states,
                final_output,
                gate,
                attention_mask,
                self.config.chunk_size,
                math.exp(self.config.g_min),
            )
        return KDAOutput(
            hidden_states=final_output,
            state=next_state,
            diagnostics=diagnostics,
        )
