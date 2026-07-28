import copy

import pytest
import torch
import torch.nn as nn

from src.stable_latent_moe import (
    reference_sparse_dispatch,
    vectorized_sparse_dispatch,
)


def experts(count=4, width=3):
    modules = nn.ModuleList(
        nn.Linear(width, width, bias=False) for _ in range(count)
    ).double()
    with torch.no_grad():
        for index, expert in enumerate(modules, 1):
            expert.weight.copy_(torch.eye(width) * index)
    return modules


def routing():
    indices = torch.tensor([[0, 2], [1, 2], [0, 1], [2, 0]])
    weights = torch.tensor(
        [[0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.5, 0.5]],
        dtype=torch.float64,
    )
    return indices, weights


def test_reference_dispatch_matches_hand_computed_token_assignment_loop():
    modules = experts()
    z = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    indices, weights = routing()
    output = reference_sparse_dispatch(
        z, modules, indices, weights, accumulation_dtype=torch.float64
    )
    expected = torch.stack(
        [
            sum(
                weights[token, slot]
                * modules[int(indices[token, slot])](z[token])
                for slot in range(2)
            )
            for token in range(4)
        ]
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tokens,top_k", [(1, 1), (3, 2), (7, 3)])
def test_vectorized_matches_reference_across_shapes(tokens, top_k):
    torch.manual_seed(227)
    modules = experts(count=5, width=4)
    z = torch.randn(tokens, 4, dtype=torch.float64)
    indices = torch.stack(
        [torch.randperm(5)[:top_k] for _ in range(tokens)]
    )
    weights = torch.rand(tokens, top_k, dtype=torch.float64)
    weights /= weights.sum(-1, keepdim=True)
    reference = reference_sparse_dispatch(
        z, modules, indices, weights, accumulation_dtype=torch.float64
    )
    actual = vectorized_sparse_dispatch(
        z, modules, indices, weights, accumulation_dtype=torch.float64
    )
    torch.testing.assert_close(actual, reference, rtol=2e-15, atol=2e-15)


def test_vectorized_gradient_parity_and_zero_load_expert():
    left_experts = experts()
    right_experts = copy.deepcopy(left_experts)
    indices, weights = routing()
    left = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    right = left.detach().clone().requires_grad_()
    left_output = reference_sparse_dispatch(
        left,
        left_experts,
        indices,
        weights,
        accumulation_dtype=torch.float64,
    )
    right_output = vectorized_sparse_dispatch(
        right,
        right_experts,
        indices,
        weights,
        accumulation_dtype=torch.float64,
    )
    left_output.square().sum().backward()
    right_output.square().sum().backward()
    torch.testing.assert_close(left.grad, right.grad, rtol=1e-14, atol=1e-14)
    for left_expert, right_expert in zip(left_experts, right_experts):
        if left_expert.weight.grad is None:
            assert right_expert.weight.grad is None
        else:
            torch.testing.assert_close(
                left_expert.weight.grad,
                right_expert.weight.grad,
                rtol=1e-14,
                atol=1e-14,
            )
    assert left_experts[3].weight.grad is None


def test_noncontiguous_latent_and_token_order_are_preserved():
    modules = experts(width=3)
    base = torch.randn(4, 2, 3, dtype=torch.float64)
    latent = base.transpose(0, 1).reshape(8, 3)
    indices = torch.tensor([[index % 4] for index in range(8)])
    weights = torch.ones(8, 1, dtype=torch.float64)
    actual = vectorized_sparse_dispatch(
        latent,
        modules,
        indices,
        weights,
        accumulation_dtype=torch.float64,
    )
    expected = torch.stack(
        [modules[index % 4](latent[index]) for index in range(8)]
    )
    torch.testing.assert_close(actual, expected)
