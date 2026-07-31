"""No-drop expert parallelism for Kimi Stable LatentMoE."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.nn import functional as dist_nn
import torch.nn as nn

from src.stable_latent_moe import StableLatentMoE
from src.stable_latent_moe.diagnostics import build_moe_diagnostics
from src.stable_latent_moe.outputs import StableLatentMoEOutput
from src.stable_latent_moe.utils import policy_dtype

from .collectives import all_gather_variable, group_rank, group_size


def _exchange_counts(counts: torch.Tensor, group) -> list[int]:
    received = torch.empty_like(counts)
    dist.all_to_all_single(received, counts, group=group)
    return [int(value) for value in received.cpu().tolist()]


def all_to_all_expert_dispatch(
    latent: torch.Tensor,
    experts: nn.ModuleList,
    selected_experts: torch.Tensor,
    selected_weights: torch.Tensor,
    *,
    first_expert: int,
    experts_per_rank: int,
    accumulation_dtype: torch.dtype,
    group=None,
) -> torch.Tensor:
    """Dispatch variable token assignments to owners and return in order."""
    size = group_size(group)
    if size == 1:
        from src.stable_latent_moe.dispatch import vectorized_sparse_dispatch

        return vectorized_sparse_dispatch(
            latent,
            experts,
            selected_experts - first_expert,
            selected_weights,
            accumulation_dtype=accumulation_dtype,
        )
    tokens, latent_dim = latent.shape
    top_k = selected_experts.shape[1]
    flat_experts = selected_experts.reshape(-1)
    flat_tokens = (
        torch.arange(tokens, device=latent.device)[:, None]
        .expand(tokens, top_k)
        .reshape(-1)
    )
    flat_weights = selected_weights.reshape(-1)
    destinations = torch.div(
        flat_experts, experts_per_rank, rounding_mode="floor"
    )
    order = torch.argsort(destinations, stable=True)
    destinations = destinations[order]
    send_tokens = flat_tokens[order]
    send_experts = flat_experts[order]
    send_latent = latent.index_select(0, send_tokens)
    send_weights = flat_weights[order]
    send_counts_tensor = torch.bincount(
        destinations, minlength=size
    ).to(torch.int64)
    send_counts = [int(value) for value in send_counts_tensor.cpu().tolist()]
    receive_counts = _exchange_counts(send_counts_tensor, group)
    received_total = sum(receive_counts)

    received_latent = latent.new_empty(received_total, latent_dim)
    received_weights = selected_weights.new_empty(received_total)
    received_experts = torch.empty(
        received_total, dtype=torch.long, device=latent.device
    )
    dist_nn.all_to_all_single(
        received_latent,
        send_latent,
        output_split_sizes=receive_counts,
        input_split_sizes=send_counts,
        group=group,
    )
    dist_nn.all_to_all_single(
        received_weights,
        send_weights,
        output_split_sizes=receive_counts,
        input_split_sizes=send_counts,
        group=group,
    )
    dist.all_to_all_single(
        received_experts,
        send_experts,
        output_split_sizes=receive_counts,
        input_split_sizes=send_counts,
        group=group,
    )

    computed = latent.new_zeros(
        received_total, latent_dim, dtype=accumulation_dtype
    )
    local_ids = received_experts - first_expert
    for local_id, expert in enumerate(experts):
        positions = torch.nonzero(
            local_ids == local_id, as_tuple=False
        ).flatten()
        if positions.numel() == 0:
            continue
        expert_output = expert(received_latent.index_select(0, positions))
        weighted = expert_output.to(accumulation_dtype) * (
            received_weights.index_select(0, positions)
            .to(accumulation_dtype)
            .unsqueeze(-1)
        )
        computed = computed.index_copy(0, positions, weighted)

    returned = computed.new_empty(len(send_tokens), latent_dim)
    dist_nn.all_to_all_single(
        returned,
        computed,
        output_split_sizes=send_counts,
        input_split_sizes=receive_counts,
        group=group,
    )
    aggregate = latent.new_zeros(
        tokens, latent_dim, dtype=accumulation_dtype
    )
    return aggregate.index_add(0, send_tokens, returned)


class ExpertParallelMoE(StableLatentMoE):
    """Stable LatentMoE retaining only this rank's contiguous experts."""

    def __init__(self, module: StableLatentMoE, *, group=None):
        nn.Module.__init__(self)
        self.config = module.config
        self.group = group
        size = group_size(group)
        if module.config.num_routed_experts % size:
            raise ValueError(
                "num_routed_experts must be divisible by expert parallel size"
            )
        self.experts_per_rank = module.config.num_routed_experts // size
        self.first_expert = group_rank(group) * self.experts_per_rank
        self.last_expert = self.first_expert + self.experts_per_rank
        self.shared_experts = module.shared_experts
        self.down_projection = module.down_projection
        self.router = module.router
        self.routed_experts = nn.ModuleList(
            list(module.routed_experts)[self.first_expert : self.last_expert]
        )
        self.routed_aggregate_norm = module.routed_aggregate_norm
        self.up_projection = module.up_projection
        self.exact_balancer = module.exact_balancer
        self.histogram_balancer = module.histogram_balancer
        self._balance_accumulating = False
        self._balance_old_bias = None
        self._balance_exact_scores = []
        self._balance_exact_cutoffs = []

    @torch.no_grad()
    def finalize_and_commit_balance(self):
        if not self._balance_accumulating:
            raise RuntimeError("balance accumulation is not active")
        if self.config.quantile_backend == "exact":
            if not self._balance_exact_scores:
                raise RuntimeError("cannot finalize an empty QB window")
            scores, _ = all_gather_variable(
                torch.cat(self._balance_exact_scores), group=self.group
            )
            cutoffs, _ = all_gather_variable(
                torch.cat(self._balance_exact_cutoffs), group=self.group
            )
            update = self.exact_balancer.compute_next_bias(
                scores, cutoffs, self._balance_old_bias
            )
        else:
            for tensor in (
                self.histogram_balancer.counts,
                self.histogram_balancer.underflow,
                self.histogram_balancer.overflow,
            ):
                if group_size(self.group) > 1:
                    dist.all_reduce(tensor, group=self.group)
            update = self.histogram_balancer.compute_next_bias(
                self._balance_old_bias
            )
        self.router.commit_next_bias(update.next_bias)
        self.discard_balance_accumulation()
        return update

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        update_routing_bias: bool = False,
        return_router_diagnostics: bool = False,
        return_branch_outputs: bool = False,
    ):
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
        aggregate = all_to_all_expert_dispatch(
            latent,
            self.routed_experts,
            router_output.selected_experts,
            router_output.selected_weights,
            first_expert=self.first_expert,
            experts_per_rank=self.experts_per_rank,
            accumulation_dtype=policy_dtype(
                latent.dtype, self.config.routed_accumulation_dtype
            ),
            group=self.group,
        )
        normalized = self.routed_aggregate_norm(aggregate).to(latent.dtype)
        routed_output = self.up_projection(normalized)
        hidden_states = (shared_output + routed_output).reshape(original_shape)
        if need_balance:
            if not self._balance_accumulating:
                self.begin_balance_accumulation()
            self._update_balance(router_output, commit=False)
            if update_routing_bias:
                update = self.finalize_and_commit_balance()
                router_output.routing_bias_after = update.next_bias.detach()
        want_router = (
            return_router_diagnostics
            or self.config.return_router_diagnostics
        )
        if want_router and group_size(self.group) > 1:
            dist.all_reduce(router_output.expert_load, group=self.group)
        if not (want_router or return_branch_outputs):
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
            hidden_states,
            shared_output.reshape(original_shape)
            if return_branch_outputs
            else None,
            routed_output.reshape(original_shape)
            if return_branch_outputs
            else None,
            router_output if want_router else None,
            diagnostics,
        )


def shard_kimi_experts(model: nn.Module, *, group=None) -> int:
    """Replace every Stable LatentMoE owner without changing layer order."""
    replacements: list[tuple[nn.Module, str, StableLatentMoE]] = []
    for parent in model.modules():
        for name, child in parent.named_children():
            if (
                isinstance(child, StableLatentMoE)
                and not isinstance(child, ExpertParallelMoE)
            ):
                replacements.append((parent, name, child))
    for parent, name, child in replacements:
        setattr(parent, name, ExpertParallelMoE(child, group=group))
    if not replacements:
        raise ValueError("no Stable LatentMoE modules found for EP sharding")
    return len(replacements)


def local_expert_parameter_ids(model: nn.Module) -> set[int]:
    result: set[int] = set()
    for module in model.modules():
        if isinstance(module, ExpertParallelMoE):
            result.update(id(parameter) for parameter in module.routed_experts.parameters())
    return result


def scale_local_expert_gradients(model: nn.Module, *, group=None) -> None:
    """Average expert gradients already accumulated from every EP source."""
    scale = float(group_size(group))
    if scale == 1:
        return
    for module in model.modules():
        if not isinstance(module, ExpertParallelMoE):
            continue
        for parameter in module.routed_experts.parameters():
            parameter.register_hook(
                lambda gradient, denominator=scale: gradient / denominator
            )


__all__ = [
    "ExpertParallelMoE",
    "all_to_all_expert_dispatch",
    "local_expert_parameter_ids",
    "scale_local_expert_gradients",
    "shard_kimi_experts",
]
