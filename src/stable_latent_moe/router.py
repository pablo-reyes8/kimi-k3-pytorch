"""Stable LatentMoE routing, expert dispatch, and load-balancing components."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import StableLatentMoEConfig
from .outputs import RouterOutput
from .utils import policy_dtype


class TopKRouter(nn.Module):
    """Sigmoid router with biased dispatch and unbiased mixture weights."""

    tie_policy = "torch.topk deterministic behavior on the current device"

    def __init__(self, config: StableLatentMoEConfig):
        super().__init__()
        self.config = config
        self.projection = nn.Linear(
            config.d_model,
            config.num_routed_experts,
            bias=config.router_bias,
        )
        self.register_buffer(
            "routing_bias",
            torch.zeros(config.num_routed_experts),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.projection.weight,
            mean=0.0,
            std=self.config.init_std,
        )
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    @torch.no_grad()
    def commit_next_bias(self, next_bias: torch.Tensor) -> None:
        if next_bias.shape != self.routing_bias.shape:
            raise ValueError("next routing bias has an invalid shape")
        if not torch.isfinite(next_bias).all():
            raise ValueError("next routing bias must be finite")
        self.routing_bias.copy_(
            next_bias.to(
                device=self.routing_bias.device,
                dtype=self.routing_bias.dtype,
            )
        )

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        need_qb_cutoff: bool = False,
        return_full_scores: bool = False,
    ) -> RouterOutput:
        if inputs.ndim != 2 or inputs.shape[-1] != self.config.d_model:
            raise ValueError(
                f"router inputs must have shape [M,{self.config.d_model}]"
            )
        if need_qb_cutoff and self.config.top_k >= self.config.num_routed_experts:
            raise ValueError("Top-(k+1) cutoff requires k < num_experts")
        logits = self.projection(inputs)
        compute_dtype = policy_dtype(
            inputs.dtype, self.config.router_logits_dtype
        )
        raw_scores = torch.sigmoid(logits.to(compute_dtype))
        biased_scores = raw_scores + self.routing_bias.to(compute_dtype)
        requested = self.config.top_k + int(need_qb_cutoff)
        top_values, top_indices = torch.topk(
            biased_scores,
            k=requested,
            dim=-1,
        )
        selected_experts = top_indices[:, : self.config.top_k]
        selected_raw_scores = raw_scores.gather(
            dim=-1, index=selected_experts
        )
        weight_dtype = policy_dtype(
            inputs.dtype, self.config.routing_weights_dtype
        )
        selected_raw_for_weights = selected_raw_scores.to(weight_dtype)
        denominator = selected_raw_for_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(self.config.router_eps)
        selected_weights = selected_raw_for_weights / denominator
        expert_load = torch.bincount(
            selected_experts.reshape(-1),
            minlength=self.config.num_routed_experts,
        )
        return RouterOutput(
            selected_experts=selected_experts,
            selected_raw_scores=selected_raw_scores,
            selected_weights=selected_weights,
            expert_load=expert_load,
            cutoff_k_plus_one=(
                top_values[:, self.config.top_k]
                if need_qb_cutoff
                else None
            ),
            routing_bias_before=self.routing_bias.detach().clone(),
            raw_logits=logits if return_full_scores else None,
            raw_scores=raw_scores if return_full_scores else None,
            biased_scores=biased_scores if return_full_scores else None,
        )
