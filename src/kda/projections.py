"""Kimi Delta Attention operators, projections, states, and diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.kimi_primitives import CausalShortConv1D, ShortConvState

from .config import KDAConfig
from .utils import validate_attention_mask


@dataclass
class KDAProjectionResult:
    """Projected KDA queries, keys, values, gates, and decay parameters."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    beta: torch.Tensor
    decay_logits: torch.Tensor
    q_conv_state: ShortConvState | None = None
    k_conv_state: ShortConvState | None = None
    v_conv_state: ShortConvState | None = None


class KDAProjections(nn.Module):
    """Neural projections around the core KDA operator."""

    def __init__(self, config: KDAConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(
            config.d_model, config.key_width, bias=config.projection_bias
        )
        self.k_proj = nn.Linear(
            config.d_model, config.key_width, bias=config.projection_bias
        )
        self.v_proj = nn.Linear(
            config.d_model, config.value_width, bias=config.projection_bias
        )
        self.q_conv = CausalShortConv1D(
            config.key_width,
            config.short_conv_kernel_size,
            bias=config.short_conv_bias,
        )
        self.k_conv = CausalShortConv1D(
            config.key_width,
            config.short_conv_kernel_size,
            bias=config.short_conv_bias,
        )
        self.v_conv = CausalShortConv1D(
            config.value_width,
            config.short_conv_kernel_size,
            bias=config.short_conv_bias,
        )
        self.beta_proj = nn.Linear(
            config.d_model, config.num_heads, bias=config.beta_bias
        )
        self.alpha_down = nn.Linear(
            config.d_model, config.resolved_decay_rank, bias=False
        )
        self.alpha_up = nn.Linear(
            config.resolved_decay_rank, config.key_width, bias=False
        )
        self.b_alpha = nn.Parameter(
            torch.empty(config.num_heads, config.key_head_dim)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.beta_proj,
            self.alpha_down,
            self.alpha_up,
        ):
            nn.init.normal_(
                projection.weight, mean=0.0, std=self.config.init_std
            )
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)
        if self.config.decay_initializer == "official_fla":
            # FLA KDA layer initialization: log-uniform dt in [1e-3, 1e-1],
            # transformed through inverse softplus. K3 retains this bias recipe
            # while setting A_log=0 for its lower-bounded gate.
            dt = torch.exp(
                torch.rand_like(self.b_alpha)
                * (math.log(0.1) - math.log(0.001))
                + math.log(0.001)
            ).clamp_min(1e-4)
            inverse_softplus = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                self.b_alpha.copy_(inverse_softplus)
        else:
            nn.init.zeros_(self.b_alpha)

    def _run_conv(
        self,
        module: CausalShortConv1D,
        x: torch.Tensor,
        state: ShortConvState | None,
        attention_mask: torch.Tensor | None,
        return_state: bool,
    ) -> tuple[torch.Tensor, ShortConvState | None]:
        if attention_mask is None or bool(attention_mask.all()):
            if return_state:
                output, next_state = module(x, state, return_state=True)
                return output, next_state
            return module(x, state), None

        # Right-padded examples have different effective chunk lengths.
        # Process each valid prefix so padding never enters the conv cache.
        outputs = []
        buffers = []
        for batch_index in range(x.shape[0]):
            length = int(attention_mask[batch_index].sum().item())
            batch_state = (
                ShortConvState(state.buffer[batch_index : batch_index + 1])
                if state is not None
                else None
            )
            valid_output, next_state = module(
                x[batch_index : batch_index + 1, :length],
                batch_state,
                return_state=True,
            )
            padded = F.pad(valid_output, (0, 0, 0, x.shape[1] - length))
            outputs.append(padded)
            buffers.append(next_state.buffer)
        output = torch.cat(outputs, dim=0)
        next_state = (
            ShortConvState(torch.cat(buffers, dim=0)) if return_state else None
        )
        return output, next_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        q_conv_state: ShortConvState | None = None,
        k_conv_state: ShortConvState | None = None,
        v_conv_state: ShortConvState | None = None,
        attention_mask: torch.Tensor | None = None,
        output_final_state: bool = False,
    ) -> KDAProjectionResult:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.config.d_model:
            raise ValueError(
                f"hidden_states must have shape [B,T,{self.config.d_model}], "
                f"got {tuple(hidden_states.shape)}"
            )
        batch, tokens, _ = hidden_states.shape
        if tokens == 0:
            raise ValueError("KDA projections do not support empty sequences")
        validate_attention_mask(attention_mask, batch, tokens)

        q_linear = self.q_proj(hidden_states)
        k_linear = self.k_proj(hidden_states)
        v_linear = self.v_proj(hidden_states)
        q, next_q = self._run_conv(
            self.q_conv,
            q_linear,
            q_conv_state,
            attention_mask,
            output_final_state,
        )
        k, next_k = self._run_conv(
            self.k_conv,
            k_linear,
            k_conv_state,
            attention_mask,
            output_final_state,
        )
        v, next_v = self._run_conv(
            self.v_conv,
            v_linear,
            v_conv_state,
            attention_mask,
            output_final_state,
        )
        q = F.normalize(
            F.silu(q).reshape(
                batch, tokens, self.config.num_heads, self.config.key_head_dim
            ),
            p=2,
            dim=-1,
            eps=self.config.eps,
        )
        k = F.normalize(
            F.silu(k).reshape(
                batch, tokens, self.config.num_heads, self.config.key_head_dim
            ),
            p=2,
            dim=-1,
            eps=self.config.eps,
        )
        v = F.silu(v).reshape(
            batch, tokens, self.config.num_heads, self.config.value_head_dim
        )
        beta = torch.sigmoid(self.beta_proj(hidden_states))
        beta_epsilon = torch.finfo(beta.dtype).eps
        beta = beta.clamp(beta_epsilon, 1.0 - beta_epsilon)
        decay_logits = self.alpha_up(
            self.alpha_down(hidden_states)
        ).reshape(
            batch, tokens, self.config.num_heads, self.config.key_head_dim
        )
        decay_logits = decay_logits + self.b_alpha[None, None]
        return KDAProjectionResult(
            q=q,
            k=k,
            v=v,
            beta=beta,
            decay_logits=decay_logits,
            q_conv_state=next_q,
            k_conv_state=next_k,
            v_conv_state=next_v,
        )
