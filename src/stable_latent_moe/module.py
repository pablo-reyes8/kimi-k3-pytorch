"""Stable LatentMoE routing, expert dispatch, and load-balancing components."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.transformer_modules.rms_norm import RMSNorm

from .config import StableLatentMoEConfig
from .diagnostics import build_moe_diagnostics
from .dispatch import (
    reference_sparse_dispatch,
    vectorized_sparse_dispatch,
)
from .experts import RoutedExpert, SharedExpert
from .histogram import HistogramQuantileBalancer
from .outputs import RouterOutput, StableLatentMoEOutput
from .quantile_balancing import ExactQuantileBalancer
from .router import TopKRouter
from .utils import policy_dtype


class StableLatentMoE(nn.Module):
    """Kimi K3 Stable LatentMoE reference implementation."""

    def __init__(self, config: StableLatentMoEConfig):
        super().__init__()
        self.config = config
        self.shared_experts = nn.ModuleList(
            SharedExpert(
                config.d_model,
                config.shared_expert_hidden_dim,
                config.beta_gate,
                config.beta_up,
                config.expert_bias,
                config.init_std,
            )
            for _ in range(config.num_shared_experts)
        )
        self.down_projection = nn.Linear(
            config.d_model,
            config.latent_dim,
            bias=config.projection_bias,
        )
        self.router = TopKRouter(config)
        self.routed_experts = nn.ModuleList(
            RoutedExpert(
                config.latent_dim,
                config.routed_expert_hidden_dim,
                config.beta_gate,
                config.beta_up,
                config.expert_bias,
                config.init_std,
            )
            for _ in range(config.num_routed_experts)
        )
        self.routed_aggregate_norm = RMSNorm(
            config.latent_dim, eps=config.norm_eps
        )
        self.up_projection = nn.Linear(
            config.latent_dim,
            config.d_model,
            bias=config.projection_bias,
        )
        self.exact_balancer = (
            ExactQuantileBalancer(
                config.num_routed_experts, config.top_k
            )
            if config.enable_quantile_balancing
            else None
        )
        self.histogram_balancer = (
            HistogramQuantileBalancer(
                config.num_routed_experts,
                config.top_k,
                config.histogram_num_bins,
                config.histogram_min_margin,
                config.histogram_max_margin,
            )
            if config.enable_quantile_balancing
            and config.quantile_backend == "histogram"
            else None
        )
        self._balance_accumulating = False
        self._balance_old_bias: torch.Tensor | None = None
        self._balance_exact_scores: list[torch.Tensor] = []
        self._balance_exact_cutoffs: list[torch.Tensor] = []
        self.reset_projection_parameters()

    def reset_projection_parameters(self) -> None:
        for projection in (self.down_projection, self.up_projection):
            nn.init.normal_(
                projection.weight,
                mean=0.0,
                std=self.config.init_std,
            )
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)

    @property
    def routing_bias(self) -> torch.Tensor:
        return self.router.routing_bias

    @torch.no_grad()
    def begin_balance_accumulation(self) -> None:
        if not self.training:
            raise RuntimeError("routing-bias accumulation is training-only")
        if not self.config.enable_quantile_balancing:
            raise RuntimeError("Quantile Balancing is disabled")
        if self._balance_accumulating:
            raise RuntimeError("balance accumulation is already active")
        if self.histogram_balancer is not None:
            self.histogram_balancer.reset()
        self._balance_old_bias = self.routing_bias.detach().clone()
        self._balance_exact_scores.clear()
        self._balance_exact_cutoffs.clear()
        self._balance_accumulating = True

    @torch.no_grad()
    def finalize_and_commit_balance(self):
        if not self._balance_accumulating:
            raise RuntimeError("balance accumulation is not active")
        if self.config.quantile_backend == "exact":
            if not self._balance_exact_scores:
                raise RuntimeError("cannot finalize an empty QB window")
            update = self.exact_balancer.compute_next_bias(
                torch.cat(self._balance_exact_scores, dim=0),
                torch.cat(self._balance_exact_cutoffs, dim=0),
                self._balance_old_bias,
            )
        else:
            update = self.histogram_balancer.compute_next_bias(
                self._balance_old_bias
            )
        self.router.commit_next_bias(update.next_bias)
        self.discard_balance_accumulation()
        return update

    @torch.no_grad()
    def discard_balance_accumulation(self) -> None:
        if self.histogram_balancer is not None:
            self.histogram_balancer.reset()
        self._balance_exact_scores.clear()
        self._balance_exact_cutoffs.clear()
        self._balance_old_bias = None
        self._balance_accumulating = False

    def _update_balance(
        self,
        router_output: RouterOutput,
        *,
        commit: bool,
    ):
        if not self.config.enable_quantile_balancing:
            raise RuntimeError("Quantile Balancing is disabled")
        raw_scores = router_output.raw_scores
        cutoff = router_output.cutoff_k_plus_one
        old_bias = router_output.routing_bias_before
        if self.config.quantile_backend == "exact":
            if self._balance_accumulating:
                self._balance_exact_scores.append(raw_scores.detach())
                self._balance_exact_cutoffs.append(cutoff.detach())
                return None
            update = self.exact_balancer.compute_next_bias(
                raw_scores, cutoff, old_bias
            )
            if commit:
                self.router.commit_next_bias(update.next_bias)
            return update
        self.histogram_balancer.accumulate(raw_scores, cutoff)
        if self._balance_accumulating:
            return None
        update = self.histogram_balancer.compute_next_bias(old_bias)
        if commit:
            self.router.commit_next_bias(update.next_bias)
        self.histogram_balancer.reset()
        return update

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        update_routing_bias: bool = False,
        return_router_diagnostics: bool = False,
        return_branch_outputs: bool = False,
    ) -> torch.Tensor | StableLatentMoEOutput:
        if inputs.ndim < 1 or inputs.shape[-1] != self.config.d_model:
            raise ValueError(
                f"inputs must have shape [...,{self.config.d_model}]"
            )
        if update_routing_bias and not self.training:
            raise RuntimeError("routing bias cannot update during eval")
        original_shape = inputs.shape
        flat_inputs = inputs.reshape(-1, self.config.d_model)
        shared_output = torch.zeros_like(flat_inputs)

        for expert in self.shared_experts:
            shared_output = shared_output + expert(flat_inputs)

        latent = self.down_projection(flat_inputs)
        need_balance = update_routing_bias or self._balance_accumulating

        router_output = self.router(
            flat_inputs,
            need_qb_cutoff=need_balance,
            return_full_scores=need_balance,
        )

        accumulation_dtype = policy_dtype(
            latent.dtype, self.config.routed_accumulation_dtype

        )
        dispatch = (
            reference_sparse_dispatch
            if self.config.routing_backend == "reference"
            else vectorized_sparse_dispatch
        )

        aggregate = dispatch(
            latent,
            self.routed_experts,
            router_output.selected_experts,
            router_output.selected_weights,
            accumulation_dtype=accumulation_dtype,
        )

        normalized = self.routed_aggregate_norm(aggregate).to(latent.dtype)
        routed_output = self.up_projection(normalized)
        hidden_states = (shared_output + routed_output).reshape(original_shape)

        if need_balance:
            update = self._update_balance(
                router_output,
                commit=update_routing_bias and not self._balance_accumulating,
            )
            if update is not None and update_routing_bias:
                router_output.routing_bias_after = (
                    self.routing_bias.detach().clone()
                )

        want_router = (
            return_router_diagnostics
            or self.config.return_router_diagnostics
        )
        want_enriched = want_router or return_branch_outputs
        if not want_enriched:
            return hidden_states
        diagnostics = (
            build_moe_diagnostics(
                router_output,
                self.config.num_shared_experts,
                shared_output,
                routed_output,
                aggregate,
                normalized,
            )
            if want_router
            else None
        )
        router_output.raw_logits = None
        router_output.raw_scores = None
        router_output.biased_scores = None
        
        return StableLatentMoEOutput(
            hidden_states=hidden_states,
            shared_output=(
                shared_output.reshape(original_shape)
                if return_branch_outputs
                else None
            ),
            routed_output=(
                routed_output.reshape(original_shape)
                if return_branch_outputs
                else None
            ),
            router_output=router_output if want_router else None,
            diagnostics=diagnostics,
        )
