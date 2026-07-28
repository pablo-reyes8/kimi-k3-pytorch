import copy

import torch

from src.kimi_primitives import situ_glu_activation
from src.stable_latent_moe import TopKRouter
from tests.stable_latent_moe.conftest import tiny_moe, tiny_moe_config


def test_shared_and_routed_experts_have_canonical_widths_and_independent_storage():
    model = tiny_moe()
    assert all(expert.width == 8 for expert in model.shared_experts)
    assert all(expert.width == 4 for expert in model.routed_experts)
    shared_ptrs = [
        expert.transform.gate_proj.weight.data_ptr()
        for expert in model.shared_experts
    ]
    routed_ptrs = [
        expert.transform.gate_proj.weight.data_ptr()
        for expert in model.routed_experts
    ]
    assert len(set(shared_ptrs)) == len(shared_ptrs)
    assert len(set(routed_ptrs)) == len(routed_ptrs)


def test_expert_matches_exact_situ_glu_equation():
    expert = tiny_moe().shared_experts[0].double()
    x = torch.randn(2, 3, 8, dtype=torch.float64)
    transform = expert.transform
    expected = transform.down_proj(
        situ_glu_activation(
            transform.gate_proj(x),
            transform.up_proj(x),
            beta_gate=4.0,
            beta_up=25.0,
        )
    )
    torch.testing.assert_close(expert(x), expected, rtol=0, atol=0)


def test_router_matches_manual_sigmoid_biased_topk_and_raw_normalization():
    router = TopKRouter(tiny_moe_config()).double()
    with torch.no_grad():
        router.projection.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1.0, 0, 0, 0, 0, 0, 0],
                    [-1.0, 0, 0, 0, 0, 0, 0, 0],
                    [0, -1.0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=torch.float64,
            )
        )
        router.routing_bias.copy_(
            torch.tensor([0.0, -0.4, 0.7, 0.0], dtype=torch.float64)
        )
    x = torch.tensor([[2.0, 1.0, 0, 0, 0, 0, 0, 0]], dtype=torch.float64)
    output = router(x, need_qb_cutoff=True, return_full_scores=True)
    logits = x @ router.projection.weight.T
    scores = torch.sigmoid(logits)
    expected_indices = torch.topk(
        scores + router.routing_bias, 3, dim=-1
    ).indices
    selected = expected_indices[:, :2]
    raw = scores.gather(-1, selected)
    torch.testing.assert_close(output.raw_scores, scores, rtol=0, atol=0)
    torch.testing.assert_close(output.selected_experts, selected)
    torch.testing.assert_close(output.selected_raw_scores, raw)
    torch.testing.assert_close(
        output.selected_weights, raw / raw.sum(-1, keepdim=True)
    )
    torch.testing.assert_close(
        output.cutoff_k_plus_one,
        (scores + router.routing_bias).gather(
            -1, expected_indices[:, 2:]
        ).squeeze(-1),
    )
    assert not torch.allclose(scores.sum(-1), torch.ones(1, dtype=torch.float64))


def test_bias_that_preserves_selected_set_cannot_change_mixture_weights():
    router = TopKRouter(tiny_moe_config()).double()
    x = torch.randn(5, 8, dtype=torch.float64)
    first = router(x)
    with torch.no_grad():
        router.routing_bias.add_(2.0)
    second = router(x)
    torch.testing.assert_close(first.selected_experts, second.selected_experts)
    torch.testing.assert_close(first.selected_weights, second.selected_weights)


def test_router_contract_load_indices_ties_and_no_bias_gradient():
    router = TopKRouter(tiny_moe_config()).double()
    x = torch.zeros(7, 8, dtype=torch.float64, requires_grad=True)
    output = router(x)
    assert output.selected_experts.dtype == torch.int64
    assert output.selected_experts.shape == (7, 2)
    assert output.expert_load.sum() == 14
    torch.testing.assert_close(
        output.selected_weights.sum(-1),
        torch.ones(7, dtype=torch.float64),
    )
    assert "torch.topk" in router.tie_policy
    output.selected_weights.square().sum().backward()
    assert router.projection.weight.grad is not None
    assert router.routing_bias.grad is None


def test_shared_experts_are_always_summed_and_absent_from_router():
    model = tiny_moe().eval()
    calls = [0, 0]
    handles = []
    for index, expert in enumerate(model.shared_experts):
        handles.append(
            expert.register_forward_hook(
                lambda _m, _i, _o, index=index: calls.__setitem__(
                    index, calls[index] + 1
                )
            )
        )
    x = torch.randn(2, 3, 8)
    output = model(x, return_branch_outputs=True)
    expected = sum(expert(x) for expert in model.shared_experts)
    for handle in handles:
        handle.remove()
    assert calls == [2, 2]
    torch.testing.assert_close(output.shared_output, expected)
    assert model.router.projection.out_features == len(model.routed_experts)
