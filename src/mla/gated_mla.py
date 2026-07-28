"""Gated Multi-head Latent Attention components and cache utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.kimi_primitives import FullRankOutputGate, combine_heads

from .attention import mla_attention
from .cache import MLACache
from .config import GatedMLAConfig
from .diagnostics import build_mla_diagnostics
from .projections import MLAProjections
from .utils import validate_hidden_states, validate_right_padding_mask


@dataclass
class GatedMLAOutput:
    """Container returned by Gated MLA with cache and optional diagnostics."""

    hidden_states: torch.Tensor
    cache: MLACache | None = None
    attentions: torch.Tensor | None = None
    diagnostics: dict[str, torch.Tensor] | None = None


class GatedMLA(nn.Module):
    """Independent Kimi K3 global attention: NoPE, latent KV, full-rank gate."""

    def __init__(self, config: GatedMLAConfig):
        super().__init__()
        self.config = config
        self.projections = MLAProjections(config)
        self.output_gate = FullRankOutputGate(
            config.d_model,
            bias=config.output_gate_bias,
            output_bias=config.output_bias,
            init_std=config.init_std,
        )
        self.output_dropout = nn.Dropout(config.output_dropout)

    def _validate_cache(
        self, hidden_states: torch.Tensor, cache: MLACache | None
    ) -> None:
        if cache is None:
            return
        if cache.latent_kv.shape[0] != hidden_states.shape[0]:
            raise ValueError("cache batch size must match hidden_states")
        if cache.latent_kv.shape[-1] != self.config.kv_latent_dim:
            raise ValueError("cache latent dimension does not match config")
        if cache.latent_kv.device != hidden_states.device:
            raise ValueError("cache and hidden_states must share device")
        if cache.latent_kv.dtype != hidden_states.dtype:
            raise TypeError("cache and hidden_states must share dtype")

    def _postprocess(
        self,
        attention_output: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = combine_heads(attention_output)
        gate = self.output_gate.gate_values(hidden_states)
        compute_dtype = combined.dtype
        gated = combined * gate.to(compute_dtype)
        output = F.linear(
            gated,
            self.output_gate.output_proj.weight.to(compute_dtype),
            None
            if self.output_gate.output_proj.bias is None
            else self.output_gate.output_proj.bias.to(compute_dtype),
        )
        output = self.output_dropout(output)
        return output.to(hidden_states.dtype), gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache: MLACache | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_diagnostics: bool = False,
        *,
        position_ids: torch.Tensor | None = None,
    ) -> GatedMLAOutput:
        if position_ids is not None:
            raise ValueError("position_ids are unsupported: Kimi Gated MLA is NoPE")
        batch, tokens = validate_hidden_states(
            hidden_states, self.config.d_model
        )
        validate_right_padding_mask(attention_mask, batch, tokens)
        self._validate_cache(hidden_states, cache)
        current_mask = (
            torch.ones(
                batch, tokens, dtype=torch.bool, device=hidden_states.device
            )
            if attention_mask is None
            else attention_mask
        )
        previous_lengths = (
            torch.zeros(batch, dtype=torch.long, device=hidden_states.device)
            if cache is None
            else cache.sequence_offset
        )

        query = self.projections.project_queries(hidden_states)
        current_latent = self.projections.compress_kv(hidden_states)
        base_cache = (
            MLACache.empty(
                batch,
                self.config.kv_latent_dim,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            if cache is None
            else cache
        )
        updated_cache = base_cache.append(current_latent, current_mask)
        key, value = self.projections.reconstruct_kv(updated_cache.latent_kv)
        # Lightweight QK-Clip control proxy. For approximately independent
        # Q/K features, dot-product RMS scales as
        # rms(Q) * rms(K) * sqrt(head_dim). Only one detached scalar survives.
        with torch.no_grad():
            self._last_qk_scale = (
                query.detach().float().square().mean().sqrt()
                * key.detach().float().square().mean().sqrt()
                * math.sqrt(self.config.q_head_dim)
            )
        query_positions = (
            previous_lengths[:, None]
            + torch.arange(tokens, device=hidden_states.device)[None, :]
        )
        need_probabilities = output_attentions or output_diagnostics
        attention_result = mla_attention(
            query,
            key,
            value,
            updated_cache.attention_mask,
            is_causal=True,
            dropout_p=self.config.attention_dropout,
            keep_output_fp32=self.config.keep_attention_output_fp32,
            backend=self.config.resolved_backend,
            query_mask=current_mask,
            query_positions=query_positions,
            training=self.training,
            return_attentions=need_probabilities,
        )
        if need_probabilities:
            raw_output, probabilities = attention_result
        else:
            raw_output, probabilities = attention_result, None
            
        final_output, gate = self._postprocess(raw_output, hidden_states)
        final_output = final_output * current_mask[..., None].to(final_output.dtype)

        diagnostics = None
        if output_diagnostics:
            diagnostics = build_mla_diagnostics(
                query,
                key,
                value,
                updated_cache.latent_kv,
                gate,
                probabilities,
                updated_cache.attention_mask,
                current_mask,
                updated_cache.cache_elements,
                self.config.full_kv_width,
                raw_output,
                final_output,
                self._last_qk_scale,
            )
        return GatedMLAOutput(
            hidden_states=final_output,
            cache=updated_cache if use_cache else None,
            attentions=probabilities if output_attentions else None,
            diagnostics=diagnostics,
        )
