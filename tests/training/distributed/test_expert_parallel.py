from __future__ import annotations

import copy

import torch

from src.kimi_k3.config import kimi_k3_cpu_tiny_config
from src.stable_latent_moe import StableLatentMoE
from training.distributed import (
    DataParallelConfig,
    DistributedConfig,
    ExpertParallelConfig,
    all_gather_variable,
    build_device_mesh,
    initialize_distributed,
)
from training.distributed.expert_parallel import ExpertParallelMoE

from tests.training.distributed.helpers import configure_rank, launch


def _ep_worker(rank, world_size, port, output, _unused):
    configure_rank(rank, world_size, port)
    distributed = DistributedConfig(
        enabled=True,
        backend="gloo",
        data_parallel=DataParallelConfig(),
        expert_parallel=ExpertParallelConfig(
            enabled=True, size=world_size
        ),
    )
    context = build_device_mesh(
        initialize_distributed(distributed), distributed
    )
    try:
        torch.manual_seed(29)
        moe_config = kimi_k3_cpu_tiny_config(
            enable_vision=False, enable_mtp=False
        ).backbone.stable_latent_moe_config
        full = StableLatentMoE(moe_config).train()
        reference = copy.deepcopy(full).train()
        parallel = ExpertParallelMoE(full, group=context.ep_group).train()
        inputs = (
            torch.randn(2 + rank, moe_config.d_model) + rank * 0.3
        )
        reference_inputs = inputs.detach().clone().requires_grad_(True)
        parallel_inputs = inputs.detach().clone().requires_grad_(True)
        expected = reference(reference_inputs)
        actual = parallel(parallel_inputs)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

        expected.sum().backward()
        (actual.sum() / world_size).backward()
        torch.testing.assert_close(
            parallel_inputs.grad * world_size,
            reference_inputs.grad,
            atol=3e-4,
            rtol=1e-3,
        )
        first = parallel.first_expert
        local_parameter = next(
            parallel.routed_experts[0].parameters()
        )
        local_gradient = (
            torch.zeros_like(local_parameter)
            if local_parameter.grad is None
            else local_parameter.grad
        )
        reference_gradients = torch.stack(
            [
                torch.zeros_like(next(expert.parameters()))
                if next(expert.parameters()).grad is None
                else next(expert.parameters()).grad
                for expert in reference.routed_experts
            ]
        )
        torch.distributed.all_reduce(
            reference_gradients, group=context.ep_group
        )
        expected_gradient = reference_gradients[first] / world_size
        torch.testing.assert_close(
            local_gradient,
            expected_gradient,
            atol=2e-3,
            rtol=2e-3,
        )

        parallel.zero_grad(set_to_none=True)
        parallel.begin_balance_accumulation()
        parallel(parallel_inputs.detach())
        update = parallel.finalize_and_commit_balance()
        gathered_inputs, _ = all_gather_variable(
            parallel_inputs.detach(), group=context.ep_group
        )
        routed = reference.router(
            gathered_inputs,
            need_qb_cutoff=True,
            return_full_scores=True,
        )
        expected_update = reference.exact_balancer.compute_next_bias(
            routed.raw_scores,
            routed.cutoff_k_plus_one,
            reference.routing_bias,
        )
        torch.testing.assert_close(
            update.next_bias,
            expected_update.next_bias,
            atol=1e-7,
            rtol=1e-7,
        )
        assert parallel.routing_bias.shape == (moe_config.num_routed_experts,)
        if rank == 0:
            torch.save(
                {
                    "local_experts": len(parallel.routed_experts),
                    "tokens": update.num_tokens,
                },
                output,
            )
    finally:
        context.close()


def test_two_rank_ep_matches_outputs_gradients_and_global_qb(tmp_path):
    output = tmp_path / "ep.pt"
    launch(_ep_worker, output, None)
    result = torch.load(output, weights_only=True)
    assert result == {"local_experts": 2, "tokens": 5}
