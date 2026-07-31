"""In-place complete-head tensor sharding for Kimi KDA and Gated MLA."""

from __future__ import annotations

import copy
from dataclasses import replace
import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from src.hybrid_backbone.attention_layer import HybridAttentionLayer
from src.kda import KDAState
from src.kda.chunkwise import chunkwise_kda
from src.kda.decay import LowerBoundedDecay
from src.kda.diagnostics import build_kda_diagnostics
from src.kda.kda import KDAOutput, KimiDeltaAttention
from src.kda.recurrent import recurrent_kda
from src.kimi_primitives import (
    CausalShortConv1D,
    HeadwiseRMSNorm,
    ShortConvState,
)
from src.mla import MLACache
from src.mla.attention import mla_attention
from src.mla.diagnostics import build_mla_diagnostics
from src.mla.gated_mla import GatedMLA, GatedMLAOutput
from src.mla.utils import validate_hidden_states, validate_right_padding_mask

from .collectives import group_rank, group_size
from .tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    TensorParallelMetadata,
    copy_to_tensor_parallel_region,
)


def _head_slice(num_heads: int, group) -> tuple[int, int]:
    size = group_size(group)
    if num_heads % size:
        raise ValueError(
            f"attention heads={num_heads} must be divisible by TP size={size}"
        )
    local = num_heads // size
    start = group_rank(group) * local
    return start, start + local


def _replicated_parameter_hook(group):
    def synchronize(gradient: torch.Tensor) -> torch.Tensor:
        if group_size(group) > 1:
            dist.all_reduce(gradient, group=group)
        return gradient

    return synchronize


def _replicate_linear(linear: nn.Linear, group) -> nn.Linear:
    result = copy.deepcopy(linear)
    for parameter in result.parameters():
        parameter.register_hook(_replicated_parameter_hook(group))
    return result


def _shard_conv(
    convolution: CausalShortConv1D,
    *,
    start: int,
    end: int,
) -> CausalShortConv1D:
    result = CausalShortConv1D(
        end - start,
        convolution.kernel_size,
        bias=convolution.bias is not None,
        device=convolution.weight.device,
        dtype=convolution.weight.dtype,
    )
    result.weight = nn.Parameter(
        convolution.weight.detach()[start:end].clone(),
        requires_grad=convolution.weight.requires_grad,
    )
    if convolution.bias is not None:
        result.bias = nn.Parameter(
            convolution.bias.detach()[start:end].clone(),
            requires_grad=convolution.bias.requires_grad,
        )
    return result


def _run_conv(
    module: CausalShortConv1D,
    inputs: torch.Tensor,
    state: ShortConvState | None,
    mask: torch.Tensor | None,
    return_state: bool,
) -> tuple[torch.Tensor, ShortConvState | None]:
    if mask is None or bool(mask.all()):
        if return_state:
            return module(inputs, state, return_state=True)
        return module(inputs, state), None
    outputs, buffers = [], []
    for batch_index in range(inputs.shape[0]):
        length = int(mask[batch_index].sum().item())
        local_state = (
            None
            if state is None
            else ShortConvState(state.buffer[batch_index : batch_index + 1])
        )
        valid, next_state = module(
            inputs[batch_index : batch_index + 1, :length],
            local_state,
            return_state=True,
        )
        outputs.append(
            F.pad(valid, (0, 0, 0, inputs.shape[1] - length))
        )
        buffers.append(next_state.buffer)
    return (
        torch.cat(outputs),
        ShortConvState(torch.cat(buffers)) if return_state else None,
    )


class TensorParallelKDA(nn.Module):
    """KDA with complete recurrent heads owned by each TP rank."""

    def __init__(self, module: KimiDeltaAttention, *, group=None):
        super().__init__()
        self.config = module.config
        self.group = group
        start, end = _head_slice(module.config.num_heads, group)
        self.head_start = start
        self.head_end = end
        self.num_heads = end - start
        key_start = start * module.config.key_head_dim
        key_end = end * module.config.key_head_dim
        value_start = start * module.config.value_head_dim
        value_end = end * module.config.value_head_dim

        projections = module.projections
        self.q_proj = ColumnParallelLinear(projections.q_proj, group=group)
        self.k_proj = ColumnParallelLinear(projections.k_proj, group=group)
        self.v_proj = ColumnParallelLinear(projections.v_proj, group=group)
        self.q_proj._kimi_role = "attention_q"
        self.k_proj._kimi_role = "attention_k"
        self.v_proj._kimi_role = "attention_v"
        self.q_proj._kimi_head_spec = (
            self.num_heads,
            self.config.key_head_dim,
        )
        self.k_proj._kimi_head_spec = (
            self.num_heads,
            self.config.key_head_dim,
        )
        self.v_proj._kimi_head_spec = (
            self.num_heads,
            self.config.value_head_dim,
        )
        self.beta_proj = ColumnParallelLinear(
            projections.beta_proj, group=group
        )
        self.alpha_down = _replicate_linear(projections.alpha_down, group)
        self.alpha_up = ColumnParallelLinear(
            projections.alpha_up, group=group
        )
        self.b_alpha = nn.Parameter(
            projections.b_alpha.detach()[start:end].clone(),
            requires_grad=projections.b_alpha.requires_grad,
        )
        self.q_conv = _shard_conv(
            projections.q_conv, start=key_start, end=key_end
        )
        self.k_conv = _shard_conv(
            projections.k_conv, start=key_start, end=key_end
        )
        self.v_conv = _shard_conv(
            projections.v_conv, start=value_start, end=value_end
        )
        self.decay = LowerBoundedDecay(
            self.num_heads, module.config.g_min
        )
        self.decay.A_log = nn.Parameter(
            module.decay.A_log.detach()[start:end].clone(),
            requires_grad=module.decay.A_log.requires_grad,
        )
        self.output_norm = HeadwiseRMSNorm(
            self.num_heads,
            module.config.value_head_dim,
            eps=module.config.eps,
            elementwise_affine=module.output_norm.elementwise_affine,
            per_head_affine=module.output_norm.per_head_affine,
        )
        if module.output_norm.weight is not None:
            weight = module.output_norm.weight.detach()
            if module.output_norm.per_head_affine:
                weight = weight[start:end]
            self.output_norm.weight = nn.Parameter(
                weight.clone(),
                requires_grad=module.output_norm.weight.requires_grad,
            )
        self.gate_proj = ColumnParallelLinear(
            module.output_gate.gate_proj, group=group
        )
        self.output_proj = RowParallelLinear(
            module.output_gate.output_proj,
            group=group,
            input_is_parallel=True,
        )

    def _validate_state(
        self, hidden_states: torch.Tensor, state: KDAState | None
    ) -> None:
        if state is None:
            return
        batch = hidden_states.shape[0]
        expected = (
            batch,
            self.num_heads,
            self.config.key_head_dim,
            self.config.value_head_dim,
        )
        if state.recurrent_state.shape != expected:
            raise ValueError(
                f"local TP KDA state must have shape {expected}, got "
                f"{tuple(state.recurrent_state.shape)}"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        state: KDAState | None = None,
        mode: str = "chunkwise",
        output_final_state: bool = False,
        output_diagnostics: bool = False,
    ) -> KDAOutput:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.config.d_model:
            raise ValueError(
                f"hidden_states must have shape [B,T,{self.config.d_model}]"
            )
        if mode == "decode" and hidden_states.shape[1] != 1:
            raise ValueError("decode mode requires exactly one token")
        self._validate_state(hidden_states, state)
        batch, tokens, _ = hidden_states.shape
        return_state = output_final_state or mode == "decode"
        q_linear = self.q_proj(hidden_states)
        k_linear = self.k_proj(hidden_states)
        v_linear = self.v_proj(hidden_states)
        q, q_state = _run_conv(
            self.q_conv,
            q_linear,
            None if state is None else state.q_conv_state,
            attention_mask,
            return_state,
        )
        k, k_state = _run_conv(
            self.k_conv,
            k_linear,
            None if state is None else state.k_conv_state,
            attention_mask,
            return_state,
        )
        v, v_state = _run_conv(
            self.v_conv,
            v_linear,
            None if state is None else state.v_conv_state,
            attention_mask,
            return_state,
        )
        q = F.normalize(
            F.silu(q).reshape(
                batch,
                tokens,
                self.num_heads,
                self.config.key_head_dim,
            ),
            dim=-1,
            eps=self.config.eps,
        )
        k = F.normalize(
            F.silu(k).reshape(
                batch,
                tokens,
                self.num_heads,
                self.config.key_head_dim,
            ),
            dim=-1,
            eps=self.config.eps,
        )
        v = F.silu(v).reshape(
            batch,
            tokens,
            self.num_heads,
            self.config.value_head_dim,
        )
        beta = torch.sigmoid(self.beta_proj(hidden_states))
        epsilon = torch.finfo(beta.dtype).eps
        beta = beta.clamp(epsilon, 1.0 - epsilon)
        decay_logits = self.alpha_up(
            self.alpha_down(
                copy_to_tensor_parallel_region(hidden_states, self.group)
            )
        ).reshape(
            batch,
            tokens,
            self.num_heads,
            self.config.key_head_dim,
        )
        decay_logits = decay_logits + self.b_alpha[None, None]
        log_decay, alpha = self.decay(decay_logits)
        core_kwargs = {
            "q": q,
            "k": k,
            "v": v,
            "g": log_decay,
            "beta": beta,
            "initial_state": (
                None if state is None else state.recurrent_state
            ),
            "output_final_state": return_state or output_diagnostics,
            "attention_mask": attention_mask,
            "accumulate_state_in_fp32": (
                self.config.accumulate_state_in_fp32
            ),
        }
        core = (
            recurrent_kda(**core_kwargs)
            if mode in {"recurrent", "decode"}
            else chunkwise_kda(
                **core_kwargs,
                chunk_size=self.config.chunk_size,
                secondary_tile_size=self.config.secondary_tile_size,
            )
        )
        local_output = self.output_norm(core.hidden_states).reshape(
            batch, tokens, -1
        )
        gate = torch.sigmoid(self.gate_proj(hidden_states))
        final_output = self.output_proj(gate * local_output)
        if attention_mask is not None:
            final_output = final_output * attention_mask[..., None].to(
                final_output.dtype
            )
        next_state = None
        if return_state:
            lengths = (
                torch.full(
                    (batch,),
                    tokens,
                    dtype=torch.long,
                    device=hidden_states.device,
                )
                if attention_mask is None
                else attention_mask.sum(1).long()
            )
            next_state = KDAState(
                core.final_state,
                q_state,
                k_state,
                v_state,
                lengths
                if state is None
                else state.sequence_offset + lengths,
            )
        diagnostics = (
            build_kda_diagnostics(
                q,
                k,
                beta,
                log_decay,
                alpha,
                core.final_state,
                core.hidden_states,
                final_output,
                gate,
                attention_mask,
                self.config.chunk_size,
                math.exp(self.config.g_min),
            )
            if output_diagnostics
            else None
        )
        return KDAOutput(final_output, next_state, diagnostics)


class TensorParallelMLA(nn.Module):
    """Gated MLA with complete Q/K/V heads and a replicated latent cache."""

    def __init__(self, module: GatedMLA, *, group=None):
        super().__init__()
        self.config = module.config
        self.group = group
        start, end = _head_slice(module.config.num_heads, group)
        self.head_start = start
        self.head_end = end
        self.num_heads = end - start
        projections = module.projections
        self.query = ColumnParallelLinear(projections.query, group=group)
        self.query._kimi_role = "attention_q"
        self.query._kimi_head_spec = (
            self.num_heads,
            self.config.q_head_dim,
        )
        latent = projections.latent_kv
        self.compression = _replicate_linear(latent.compression, group)
        self.key_up = ColumnParallelLinear(latent.key_up, group=group)
        self.value_up = ColumnParallelLinear(latent.value_up, group=group)
        self.key_up._kimi_role = "attention_k"
        self.key_up._kimi_head_spec = (
            self.num_heads,
            self.config.q_head_dim,
        )
        self.value_up._kimi_role = "attention_v"
        self.value_up._kimi_head_spec = (
            self.num_heads,
            self.config.v_head_dim,
        )
        self.gate_proj = ColumnParallelLinear(
            module.output_gate.gate_proj, group=group
        )
        self.output_proj = RowParallelLinear(
            module.output_gate.output_proj,
            group=group,
            input_is_parallel=True,
        )
        self.output_dropout = copy.deepcopy(module.output_dropout)
        self._kimi_qk_weights = (self.query.weight, self.key_up.weight)
        self._kimi_qk_group = group

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
            raise ValueError("Kimi Gated MLA uses NoPE")
        batch, tokens = validate_hidden_states(
            hidden_states, self.config.d_model
        )
        validate_right_padding_mask(attention_mask, batch, tokens)
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
        query = self.query(hidden_states).reshape(
            batch, tokens, self.num_heads, self.config.q_head_dim
        )
        latent = self.compression(
            copy_to_tensor_parallel_region(hidden_states, self.group)
        )
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
        updated_cache = base_cache.append(latent, current_mask)
        cached = updated_cache.latent_kv
        key = self.key_up(cached).reshape(
            batch,
            cached.shape[1],
            self.num_heads,
            self.config.q_head_dim,
        )
        value = self.value_up(cached).reshape(
            batch,
            cached.shape[1],
            self.num_heads,
            self.config.v_head_dim,
        )
        with torch.no_grad():
            qk_scale = (
                query.detach().float().square().mean().sqrt()
                * key.detach().float().square().mean().sqrt()
                * math.sqrt(self.config.q_head_dim)
            )
            self._last_qk_scale = qk_scale
        query_positions = (
            previous_lengths[:, None]
            + torch.arange(tokens, device=hidden_states.device)[None]
        )
        need_probabilities = output_attentions or output_diagnostics
        attention_result = mla_attention(
            query,
            key,
            value,
            attention_mask=updated_cache.attention_mask,
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
        local_output = raw_output.reshape(batch, tokens, -1)
        gate = torch.sigmoid(self.gate_proj(hidden_states))
        compute_dtype = local_output.dtype
        final_output = self.output_proj(
            gate.to(compute_dtype) * local_output
        )
        final_output = self.output_dropout(final_output).to(hidden_states.dtype)
        final_output = final_output * current_mask[..., None].to(
            final_output.dtype
        )
        diagnostics = (
            build_mla_diagnostics(
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
                qk_scale,
            )
            if output_diagnostics
            else None
        )
        return GatedMLAOutput(
            hidden_states=final_output,
            cache=updated_cache if use_cache else None,
            attentions=probabilities if output_attentions else None,
            diagnostics=diagnostics,
        )


def shard_kimi_attention(
    model: nn.Module, *, group=None
) -> TensorParallelMetadata:
    """Replace existing Kimi attention owners in place; no second model."""
    transformed = 0
    for module in model.modules():
        if not isinstance(module, HybridAttentionLayer):
            continue
        if isinstance(module.attention, KimiDeltaAttention):
            module.attention = TensorParallelKDA(
                module.attention, group=group
            )
            transformed += 1
        elif isinstance(module.attention, GatedMLA):
            module.attention = TensorParallelMLA(
                module.attention, group=group
            )
            transformed += 1
    if transformed == 0:
        raise ValueError("no Kimi KDA/MLA modules were found for TP sharding")
    return TensorParallelMetadata(
        size=group_size(group),
        rank=group_rank(group),
        vocab_start=None,
        vocab_end=None,
        cache_layout="replicated",
        transformed_attention_layers=transformed,
    )


__all__ = [
    "TensorParallelKDA",
    "TensorParallelMLA",
    "shard_kimi_attention",
]
