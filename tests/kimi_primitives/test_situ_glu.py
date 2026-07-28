import copy

import pytest
import torch
import torch.nn.functional as F

from src.kimi_primitives import (
    SiTUGLU,
    situ_glu_activation,
    softcap,
)


def test_softcap_matches_direct_formula_at_adversarial_values():
    values = torch.tensor(
        [-1e4, -100.0, -1e-6, 0.0, 1e-6, 100.0, 1e4],
        dtype=torch.float64,
    )
    for beta in (0.5, 4.0, 25.0):
        expected = beta * torch.tanh(values / beta)
        torch.testing.assert_close(softcap(values, beta), expected, rtol=0, atol=0)


@pytest.mark.parametrize("beta", [4.0, 25.0])
def test_softcap_is_locally_linear(beta):
    values = torch.linspace(-1e-4, 1e-4, 101, dtype=torch.float64)
    torch.testing.assert_close(
        softcap(values, beta), values, rtol=1e-8, atol=1e-12
    )


@pytest.mark.parametrize("beta", [0.1, 4.0, 25.0])
def test_softcap_is_smoothly_bounded(beta):
    values = torch.tensor([-1e6, -100, 0, 100, 1e6], dtype=torch.float64)
    output = softcap(values, beta)
    assert output.abs().max() <= beta
    assert torch.isfinite(output).all()


def test_situ_activation_matches_manual_formula():
    gate = torch.tensor(
        [[-100.0, -1e-5, 0.0, 2.0, 100.0]], dtype=torch.float64
    )
    up = torch.tensor(
        [[100.0, -25.0, 0.0, 1e-5, -100.0]], dtype=torch.float64
    )
    expected = (
        4 * torch.tanh(gate / 4) * torch.sigmoid(gate)
        * 25 * torch.tanh(up / 25)
    )
    torch.testing.assert_close(
        situ_glu_activation(gate, up), expected, rtol=1e-15, atol=1e-20
    )


def test_situ_activation_respects_product_bound_under_extremes():
    generator = torch.Generator().manual_seed(3)
    gate = torch.randn(64, 128, generator=generator) * 1e4
    up = torch.randn(64, 128, generator=generator) * 1e4
    output = situ_glu_activation(gate, up)
    assert output.abs().max() <= 4 * 25
    assert torch.isfinite(output).all()


def test_situ_controls_growth_that_swiglu_does_not():
    values = torch.tensor([10.0, 100.0, 1e4])
    situ = situ_glu_activation(values, values)
    swiglu = F.silu(values) * values
    assert situ[-1].abs() <= 100
    assert swiglu[-1] > situ[-1] * 1e5


def test_module_forward_matches_projection_composition_exactly():
    torch.manual_seed(2)
    module = SiTUGLU(5, 7, bias=True, output_bias=True).double()
    x = torch.randn(2, 3, 5, dtype=torch.float64)
    expected = module.down_proj(
        situ_glu_activation(
            module.gate_proj(x),
            module.up_proj(x),
            module.beta_gate,
            module.beta_up,
        )
    )
    torch.testing.assert_close(module(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(3, 8), (2, 4, 8), (2, 3, 4, 8)])
def test_module_preserves_arbitrary_leading_dimensions(shape):
    assert SiTUGLU(8, 13)(torch.randn(shape)).shape == shape


def test_zero_bias_free_input_produces_exact_zero():
    output = SiTUGLU(8, 12, bias=False)(torch.zeros(2, 3, 8))
    assert torch.count_nonzero(output) == 0


def test_gate_and_up_projections_are_independent_parameters_and_storage():
    module = SiTUGLU(8, 12)
    assert module.gate_proj.weight is not module.up_proj.weight
    assert module.gate_proj.weight.data_ptr() != module.up_proj.weight.data_ptr()
    with torch.no_grad():
        original_up = module.up_proj.weight.clone()
        module.gate_proj.weight.add_(1)
    torch.testing.assert_close(module.up_proj.weight, original_up, rtol=0, atol=0)


def test_bias_policy_allows_independent_output_bias():
    module = SiTUGLU(8, 12, bias=False, output_bias=True)
    assert module.gate_proj.bias is None and module.up_proj.bias is None
    assert module.down_proj.bias is not None
    inherited = SiTUGLU(8, 12, bias=True, output_bias=None)
    assert all(
        projection.bias is not None
        for projection in (
            inherited.gate_proj,
            inherited.up_proj,
            inherited.down_proj,
        )
    )


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: SiTUGLU(0, 4),
        lambda: SiTUGLU(4, 0),
        lambda: SiTUGLU(4, 8, beta_gate=0),
        lambda: SiTUGLU(4, 8, beta_up=-1),
        lambda: SiTUGLU(4, 8, init_std=0),
    ],
)
def test_invalid_configuration_is_rejected(constructor):
    with pytest.raises(ValueError):
        constructor()


@pytest.mark.parametrize("beta", [0, -1.0])
def test_softcap_rejects_invalid_beta(beta):
    with pytest.raises(ValueError):
        softcap(torch.ones(2), beta)


def test_situ_activation_rejects_shape_mismatch_without_broadcasting():
    with pytest.raises(ValueError, match="identical"):
        situ_glu_activation(torch.ones(2, 1), torch.ones(2, 3))


@pytest.mark.parametrize("shape", [(2, 3, 7), (2, 3, 8, 1)])
def test_module_rejects_wrong_input_contract(shape):
    with pytest.raises(ValueError):
        SiTUGLU(8, 12)(torch.randn(shape))


def test_activation_gradcheck():
    gate = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    up = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda g, u: situ_glu_activation(g, u, 4.0, 25.0),
        (gate, up),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


def test_module_gradcheck_includes_projection_parameters():
    module = SiTUGLU(3, 4, bias=True).double()
    x = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(module, (x,), fast_mode=True)


def test_all_input_and_parameter_gradients_are_nonzero_and_finite():
    module = SiTUGLU(8, 12, bias=True, output_bias=True)
    x = torch.randn(2, 3, 8, requires_grad=True)
    module(x).square().mean().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.abs().sum() > 0, name
        assert torch.isfinite(parameter.grad).all(), name


def test_bfloat16_extreme_forward_backward_is_finite():
    module = SiTUGLU(8, 12).to(torch.bfloat16)
    x = (torch.randn(2, 3, 8) * 1e3).to(torch.bfloat16).requires_grad_()
    output = module(x)
    output.float().square().mean().backward()
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output.float()).all()
    assert x.grad is not None and torch.isfinite(x.grad.float()).all()


def test_state_dict_roundtrip_is_exact():
    module = SiTUGLU(8, 12).eval()
    clone = copy.deepcopy(module).eval()
    clone.load_state_dict(module.state_dict())
    x = torch.randn(2, 3, 8)
    torch.testing.assert_close(module(x), clone(x), rtol=0, atol=0)
