import copy

import pytest
import torch

from src.kimi_primitives import FullRankOutputGate


def test_output_gate_matches_manual_formula_exactly():
    torch.manual_seed(2)
    module = FullRankOutputGate(5, bias=True, output_bias=True).double()
    attention = torch.randn(2, 3, 5, dtype=torch.float64)
    residual = torch.randn(2, 3, 5, dtype=torch.float64)
    expected_gate = torch.sigmoid(module.gate_proj(residual))
    expected = module.output_proj(expected_gate * attention)
    output, gate = module(attention, residual, return_gate=True)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(gate, expected_gate, rtol=0, atol=0)


def test_gate_is_computed_from_residual_not_attention_output():
    module = FullRankOutputGate(4)
    attention = torch.ones(1, 2, 4)
    first_output, first_gate = module(
        attention, torch.zeros_like(attention), return_gate=True
    )
    second_output, second_gate = module(
        attention, torch.ones_like(attention) * 100, return_gate=True
    )
    assert not torch.equal(first_gate, second_gate)
    assert not torch.equal(first_output, second_output)

    changed_attention = attention * 7
    _, unchanged_gate = module(
        changed_attention, torch.zeros_like(attention), return_gate=True
    )
    torch.testing.assert_close(unchanged_gate, first_gate, rtol=0, atol=0)


def test_gate_saturates_near_zero_and_one():
    module = FullRankOutputGate(3, bias=True)
    with torch.no_grad():
        module.gate_proj.weight.zero_()
        module.output_proj.weight.copy_(torch.eye(3))
    attention = torch.randn(1, 2, 3)
    residual = torch.randn_like(attention)
    with torch.no_grad():
        module.gate_proj.bias.fill_(20)
    positive, high_gate = module(attention, residual, return_gate=True)
    torch.testing.assert_close(positive, attention, rtol=1e-5, atol=1e-6)
    assert high_gate.min() > 0.999
    with torch.no_grad():
        module.gate_proj.bias.fill_(-20)
    negative, low_gate = module(attention, residual, return_gate=True)
    assert negative.abs().max() < 1e-7
    assert low_gate.max() < 1e-7


def test_gate_modulates_each_channel_before_identity_output_projection():
    module = FullRankOutputGate(4, bias=True)
    with torch.no_grad():
        module.gate_proj.weight.zero_()
        module.gate_proj.bias.copy_(torch.tensor([-20.0, 0.0, 20.0, 0.0]))
        module.output_proj.weight.copy_(torch.eye(4))
    attention = torch.ones(1, 1, 4)
    output, gate = module(attention, torch.zeros_like(attention), return_gate=True)
    torch.testing.assert_close(output, gate, rtol=0, atol=0)
    assert gate[0, 0, 0] < 1e-7
    assert gate[0, 0, 1] == 0.5
    assert gate[0, 0, 2] > 0.999


def test_gate_projection_is_real_full_rank_shape_not_scalar_or_factorized():
    module = FullRankOutputGate(16)
    assert module.gate_proj.weight.shape == (16, 16)
    assert isinstance(module.gate_proj, torch.nn.Linear)
    assert not hasattr(module, "gate_up") and not hasattr(module, "gate_down")


def test_bias_policies_are_independent():
    module = FullRankOutputGate(8, bias=True, output_bias=False)
    assert module.gate_proj.bias is not None
    assert module.output_proj.bias is None


def test_batch_and_token_gate_independence():
    module = FullRankOutputGate(4)
    residual = torch.randn(2, 3, 4)
    changed = residual.clone()
    changed[1, 2] += 100
    first = module.gate_values(residual)
    second = module.gate_values(changed)
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    torch.testing.assert_close(first[1, :2], second[1, :2], rtol=0, atol=0)
    assert not torch.equal(first[1, 2], second[1, 2])


def test_zero_attention_produces_zero_in_bias_free_output():
    module = FullRankOutputGate(4, output_bias=False)
    output = module(torch.zeros(2, 3, 4), torch.randn(2, 3, 4))
    assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize("d_model", [0, -1])
def test_invalid_d_model_rejected(d_model):
    with pytest.raises(ValueError):
        FullRankOutputGate(d_model)


@pytest.mark.parametrize(
    "attention,residual",
    [
        (torch.randn(2, 4), torch.randn(2, 4)),
        (torch.randn(2, 3, 5), torch.randn(2, 3, 5)),
        (torch.randn(2, 3, 4), torch.randn(2, 2, 4)),
    ],
)
def test_incompatible_shapes_rejected(attention, residual):
    with pytest.raises(ValueError):
        FullRankOutputGate(4)(attention, residual)


def test_gradcheck_covers_both_inputs_and_parameters():
    module = FullRankOutputGate(3, bias=True, output_bias=True).double()
    attention = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
    residual = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        module, (attention, residual), fast_mode=True
    )


def test_complete_gradients_are_nonzero_and_finite():
    module = FullRankOutputGate(5, bias=True, output_bias=True)
    attention = torch.randn(2, 3, 5, requires_grad=True)
    residual = torch.randn(2, 3, 5, requires_grad=True)
    module(attention, residual).square().mean().backward()
    assert attention.grad is not None and attention.grad.abs().sum() > 0
    assert residual.grad is not None and residual.grad.abs().sum() > 0
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.abs().sum() > 0, name
        assert torch.isfinite(parameter.grad).all(), name


def test_bfloat16_forward_backward_preserves_dtype():
    module = FullRankOutputGate(8).to(torch.bfloat16)
    attention = torch.randn(
        2, 3, 8, dtype=torch.bfloat16, requires_grad=True
    )
    residual = torch.randn(
        2, 3, 8, dtype=torch.bfloat16, requires_grad=True
    )
    output = module(attention, residual)
    output.float().square().mean().backward()
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output.float()).all()
    assert torch.isfinite(attention.grad.float()).all()
    assert torch.isfinite(residual.grad.float()).all()


def test_state_dict_roundtrip_exact():
    module = FullRankOutputGate(5).eval()
    clone = copy.deepcopy(module).eval()
    clone.load_state_dict(module.state_dict())
    attention, residual = torch.randn(2, 3, 5), torch.randn(2, 3, 5)
    torch.testing.assert_close(
        module(attention, residual),
        clone(attention, residual),
        rtol=0,
        atol=0,
    )

