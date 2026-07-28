import copy

import pytest
import torch

from src.stable_latent_moe import StableLatentMoEOutput
from tests.stable_latent_moe.conftest import tiny_moe


def explicit_forward(model, inputs):
    shape = inputs.shape
    flat = inputs.reshape(-1, model.config.d_model)
    shared = sum(expert(flat) for expert in model.shared_experts)
    latent = model.down_projection(flat)
    routing = model.router(flat)
    aggregate = torch.zeros(
        flat.shape[0],
        model.config.latent_dim,
        dtype=flat.dtype,
        device=flat.device,
    )
    for token in range(flat.shape[0]):
        for slot in range(model.config.top_k):
            expert = model.routed_experts[
                int(routing.selected_experts[token, slot])
            ]
            aggregate[token] += (
                routing.selected_weights[token, slot]
                * expert(latent[token])
            )
    normalized = model.routed_aggregate_norm(aggregate)
    routed = model.up_projection(normalized)
    return (shared + routed).reshape(shape), shared, routed, aggregate


@pytest.mark.parametrize("backend", ["reference", "vectorized"])
@pytest.mark.parametrize("shape", [(8,), (3, 8), (2, 4, 8)])
def test_forward_matches_explicit_manual_pipeline_and_preserves_shape(
    backend, shape
):
    model = tiny_moe(
        routing_backend=backend,
        router_logits_dtype="input",
        routing_weights_dtype="input",
        routed_accumulation_dtype="input",
    ).double().eval()
    x = torch.randn(*shape, dtype=torch.float64)
    expected, _, _, _ = explicit_forward(model, x)
    actual = model(x)
    torch.testing.assert_close(actual, expected, rtol=3e-15, atol=3e-15)
    assert actual.shape == x.shape


def test_shared_sum_routed_sum_and_no_input_residual():
    model = tiny_moe().double().eval()
    x = torch.randn(2, 3, 8, dtype=torch.float64)
    output = model(x, return_branch_outputs=True)
    assert isinstance(output, StableLatentMoEOutput)
    torch.testing.assert_close(
        output.hidden_states,
        output.shared_output + output.routed_output,
        rtol=0,
        atol=0,
    )
    assert not torch.equal(output.hidden_states, x + output.shared_output)
    shared_sum = sum(expert(x) for expert in model.shared_experts)
    torch.testing.assert_close(output.shared_output, shared_sum)
    assert not torch.equal(
        output.shared_output, shared_sum / len(model.shared_experts)
    )


def test_zero_up_projection_recovers_shared_and_zero_shared_recovers_routed():
    model = tiny_moe().double().eval()
    x = torch.randn(2, 3, 8, dtype=torch.float64)
    with torch.no_grad():
        model.up_projection.weight.zero_()
    shared_only = model(x, return_branch_outputs=True)
    torch.testing.assert_close(
        shared_only.hidden_states, shared_only.shared_output
    )
    routed_model = tiny_moe().double().eval()
    with torch.no_grad():
        for expert in routed_model.shared_experts:
            for parameter in expert.parameters():
                parameter.zero_()
    routed_only = routed_model(x, return_branch_outputs=True)
    torch.testing.assert_close(
        routed_only.hidden_states, routed_only.routed_output
    )


def test_rmsnorm_is_after_aggregate_before_up_and_has_latent_width():
    model = tiny_moe().double().eval()
    assert model.routed_aggregate_norm.dim == model.config.latent_dim
    x = torch.randn(2, 3, 8, dtype=torch.float64)
    output = model(
        x,
        return_branch_outputs=True,
        return_router_diagnostics=True,
    )
    expected, _, routed, aggregate = explicit_forward(model, x)
    torch.testing.assert_close(output.hidden_states, expected)
    torch.testing.assert_close(
        output.routed_output.reshape(-1, 8), routed
    )
    wrong = model.up_projection(aggregate)
    assert not torch.allclose(output.routed_output.reshape(-1, 8), wrong)


def test_routed_norm_blocks_positive_expert_output_scale_propagation():
    baseline = tiny_moe(norm_eps=1e-20).double().eval()
    scaled = copy.deepcopy(baseline)
    with torch.no_grad():
        for expert in scaled.routed_experts:
            expert.transform.down_proj.weight.mul_(10.0)
    x = torch.randn(2, 4, 8, dtype=torch.float64) * 10
    expected = baseline(x, return_branch_outputs=True).routed_output
    actual = scaled(x, return_branch_outputs=True).routed_output
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-8)


def test_diagnostics_are_complete_finite_and_do_not_change_output():
    model = tiny_moe().eval()
    x = torch.randn(2, 5, 8)
    baseline = model(x)
    enriched = model(
        x,
        return_router_diagnostics=True,
        return_branch_outputs=True,
    )
    torch.testing.assert_close(enriched.hidden_states, baseline, rtol=0, atol=0)
    diagnostics = enriched.diagnostics
    assert diagnostics.num_tokens == 10
    assert diagnostics.num_assignments == 20
    assert diagnostics.shared_token_evaluations == 20
    assert diagnostics.expert_load.sum() == 20
    for value in vars(diagnostics).values():
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            assert torch.isfinite(value).all()
    router = enriched.router_output
    assert router.raw_scores is None
    assert router.biased_scores is None
    assert router.raw_logits is None


def test_module_is_tokenwise_batchwise_and_accepts_noncontiguous_input():
    model = tiny_moe().double().eval()
    x = torch.randn(2, 3, 8, dtype=torch.float64)
    baseline = model(x)
    changed = x.clone()
    changed[1, 2] += 100
    actual = model(changed)
    torch.testing.assert_close(actual[0], baseline[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1, :2], baseline[1, :2], rtol=0, atol=0)
    noncontiguous = torch.randn(2, 8, 3, dtype=torch.float64).transpose(1, 2)
    assert not noncontiguous.is_contiguous()
    torch.testing.assert_close(
        model(noncontiguous),
        model(noncontiguous.contiguous()),
        rtol=0,
        atol=0,
    )


def test_reference_and_vectorized_modules_share_state_dict_and_output():
    reference = tiny_moe(routing_backend="reference").double().eval()
    vectorized = tiny_moe(routing_backend="vectorized").double().eval()
    vectorized.load_state_dict(reference.state_dict())
    x = torch.randn(2, 4, 8, dtype=torch.float64)
    torch.testing.assert_close(
        reference(x), vectorized(x), rtol=3e-15, atol=3e-15
    )
