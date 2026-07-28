import copy
import io

import pytest
import torch

from tests.stable_latent_moe.conftest import tiny_moe


def gradient_snapshot(model, inputs):
    output = model(inputs)
    output.square().sum().backward()
    return inputs.grad.clone(), {
        name: (
            None if parameter.grad is None else parameter.grad.clone()
        )
        for name, parameter in model.named_parameters()
    }


def test_reference_and_vectorized_forward_and_gradient_parity():
    reference = tiny_moe(
        routing_backend="reference",
        router_logits_dtype="input",
        routing_weights_dtype="input",
        routed_accumulation_dtype="input",
    ).double()
    vectorized = tiny_moe(
        routing_backend="vectorized",
        router_logits_dtype="input",
        routing_weights_dtype="input",
        routed_accumulation_dtype="input",
    ).double()
    vectorized.load_state_dict(reference.state_dict())
    left = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    right = left.detach().clone().requires_grad_()
    left_grad = gradient_snapshot(reference, left)
    right_grad = gradient_snapshot(vectorized, right)
    torch.testing.assert_close(left_grad[0], right_grad[0], rtol=1e-11, atol=1e-13)
    for name in left_grad[1]:
        if left_grad[1][name] is None:
            assert right_grad[1][name] is None
        else:
            torch.testing.assert_close(
                left_grad[1][name],
                right_grad[1][name],
                rtol=2e-11,
                atol=2e-13,
            )


def test_gradcheck_with_fixed_well_separated_routing():
    model = tiny_moe(
        routing_backend="reference",
        router_logits_dtype="input",
        routing_weights_dtype="input",
        routed_accumulation_dtype="input",
    ).double()
    with torch.no_grad():
        model.router.projection.weight.copy_(
            torch.tensor(
                [
                    [2.0] * 8,
                    [1.0] * 8,
                    [-1.0] * 8,
                    [-2.0] * 8,
                ],
                dtype=torch.float64,
            )
        )
    x = (
        torch.rand(1, 2, 8, dtype=torch.float64) + 0.5
    ).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda value: model(value),
        (x,),
        fast_mode=True,
    )


def test_gradients_reach_active_parameters_and_zero_load_is_valid():
    model = tiny_moe().double()
    with torch.no_grad():
        model.router.routing_bias.copy_(
            torch.tensor([5.0, 4.0, -5.0, -6.0], dtype=torch.float64)
        )
    x = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    model(x).square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert model.router.projection.weight.grad is not None
    assert model.down_projection.weight.grad is not None
    assert model.up_projection.weight.grad is not None
    assert model.routed_aggregate_norm.weight.grad is not None
    assert all(
        expert.transform.gate_proj.weight.grad is not None
        for expert in model.shared_experts
    )
    assert model.routed_experts[2].transform.gate_proj.weight.grad is None
    assert model.routing_bias.grad is None
    assert torch.count_nonzero(
        model.router.projection.weight.grad[2:]
    ) == 0
    assert torch.count_nonzero(
        model.router.projection.weight.grad[:2]
    ) > 0


@pytest.mark.parametrize("backend", ["reference", "vectorized"])
def test_large_inputs_and_extreme_router_logits_remain_finite(backend):
    model = tiny_moe(routing_backend=backend)
    with torch.no_grad():
        model.router.projection.weight.fill_(100)
        model.router.projection.weight[::2].mul_(-1)
    x = (torch.randn(2, 5, 8) * 1e3).requires_grad_()
    output = model(x, return_router_diagnostics=True)
    output.hidden_states.float().square().mean().backward()
    assert torch.isfinite(output.hidden_states).all()
    assert torch.isfinite(output.router_output.selected_weights).all()
    assert torch.isfinite(x.grad).all()


def test_cpu_bfloat16_forward_backward_uses_stable_routing():
    model = tiny_moe().bfloat16()
    x = torch.randn(2, 4, 8, dtype=torch.bfloat16, requires_grad=True)
    output = model(x, return_router_diagnostics=True)
    assert output.hidden_states.dtype == torch.bfloat16
    assert output.router_output.selected_weights.dtype == torch.float32
    output.hidden_states.float().square().mean().backward()
    assert torch.isfinite(output.hidden_states.float()).all()
    assert torch.isfinite(x.grad.float()).all()


def test_state_dict_roundtrip_preserves_output_bias_and_backend_semantics():
    source = tiny_moe(routing_backend="reference").double().train()
    source(torch.randn(2, 3, 8, dtype=torch.float64), update_routing_bias=True)
    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    buffer.seek(0)
    target = tiny_moe(routing_backend="vectorized").double().eval()
    target.load_state_dict(torch.load(buffer, weights_only=True))
    source.eval()
    x = torch.randn(2, 4, 8, dtype=torch.float64)
    torch.testing.assert_close(
        source(x), target(x), rtol=3e-15, atol=3e-15
    )
    torch.testing.assert_close(source.routing_bias, target.routing_bias)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_cuda_dtype_forward_backward(dtype):
    model = tiny_moe().cuda().to(dtype)
    x = torch.randn(2, 4, 8, device="cuda", dtype=dtype, requires_grad=True)
    output = model(x)
    output.float().square().mean().backward()
    assert torch.isfinite(output.float()).all()
    assert torch.isfinite(x.grad.float()).all()
