import pytest
import torch
import torch.nn.functional as F

from src.transformer_modules import SwiGLUFeedForward, SwiGLUMLPConfig
from src.transformer_modules.swiglu import round_up_to_multiple


def make_config(**overrides):
    values = dict(
        d_model=32,
        hidden_dim=64,
        expansion_factor=4.0,
        multiple_of=1,
        dropout=0.0,
        use_bias=False,
        init_std=0.02,
    )
    values.update(overrides)
    return SwiGLUMLPConfig(**values)


def make_mlp(**overrides):
    return SwiGLUFeedForward(make_config(**overrides))


@pytest.mark.parametrize(
    "value,multiple,expected",
    [(1, 1, 1), (7, 4, 8), (64, 16, 64), (65, 16, 80)],
)
def test_round_up_to_multiple(value, multiple, expected):
    assert round_up_to_multiple(value, multiple) == expected


@pytest.mark.parametrize("multiple", [0, -1])
def test_round_up_rejects_invalid_multiple(multiple):
    with pytest.raises(ValueError):
        round_up_to_multiple(5, multiple)


def test_explicit_and_inferred_hidden_dimensions():
    assert make_mlp(hidden_dim=77).hidden_dim == 77
    assert make_mlp(d_model=30, hidden_dim=None, expansion_factor=2.1).hidden_dim == 63
    assert (
        make_mlp(
            d_model=30, hidden_dim=None, expansion_factor=2.1, multiple_of=16
        ).hidden_dim
        == 64
    )


@pytest.mark.parametrize(
    "override",
    [
        {"d_model": 0},
        {"d_model": -1},
        {"hidden_dim": 0},
        {"hidden_dim": -1},
        {"hidden_dim": None, "expansion_factor": 0},
        {"multiple_of": 0},
        {"dropout": -0.1},
        {"dropout": 1.0},
        {"init_std": 0},
    ],
)
def test_invalid_configurations_rejected(override):
    with pytest.raises(ValueError):
        make_mlp(**override)


def test_projection_shapes_and_bias_policy():
    without_bias = make_mlp(d_model=24, hidden_dim=55, use_bias=False)
    assert without_bias.gate_proj.weight.shape == (55, 24)
    assert without_bias.up_proj.weight.shape == (55, 24)
    assert without_bias.down_proj.weight.shape == (24, 55)
    assert all(
        layer.bias is None
        for layer in (without_bias.gate_proj, without_bias.up_proj, without_bias.down_proj)
    )
    with_bias = make_mlp(use_bias=True)
    assert all(
        layer.bias is not None
        for layer in (with_bias.gate_proj, with_bias.up_proj, with_bias.down_proj)
    )
    assert all(
        torch.count_nonzero(layer.bias) == 0
        for layer in (with_bias.gate_proj, with_bias.up_proj, with_bias.down_proj)
    )


def test_initialized_weight_distribution_is_finite_and_reasonable():
    torch.manual_seed(123)
    mlp = make_mlp(d_model=64, hidden_dim=256, init_std=0.03)
    for layer in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        assert torch.isfinite(layer.weight).all()
        assert abs(layer.weight.mean().item()) < 0.01
        assert abs(layer.weight.std(unbiased=False).item() - 0.03) < 0.005


def test_forward_matches_exact_swiglu_equation():
    torch.manual_seed(2)
    mlp = make_mlp(use_bias=True).eval()
    x = torch.randn(2, 7, 32)
    expected = mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
    torch.testing.assert_close(mlp(x), expected)


@pytest.mark.parametrize("shape", [(2, 7, 32), (0, 7, 32), (2, 0, 32)])
def test_forward_preserves_valid_btd_shapes(shape):
    output = make_mlp()(torch.randn(*shape))
    assert output.shape == shape


@pytest.mark.parametrize("shape", [(7, 32), (2, 7, 32, 1), (2, 7, 31)])
def test_invalid_input_contract_rejected(shape):
    with pytest.raises(ValueError):
        make_mlp()(torch.randn(*shape))


def test_zero_input_and_bias_free_network_produces_zero():
    output = make_mlp(use_bias=False)(torch.zeros(2, 5, 32))
    assert torch.count_nonzero(output) == 0


def test_dropout_train_eval_contract():
    x = torch.randn(4, 12, 32)
    mlp = make_mlp(dropout=0.5)
    mlp.train()
    assert not torch.equal(mlp(x), mlp(x))
    mlp.eval()
    assert torch.equal(mlp(x), mlp(x))
    deterministic = make_mlp(dropout=0.0).train()
    assert torch.equal(deterministic(x), deterministic(x))


def test_all_parameters_and_input_receive_nonzero_finite_gradients():
    mlp = make_mlp(use_bias=True)
    x = torch.randn(2, 7, 32, requires_grad=True)
    mlp(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, parameter in mlp.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_bfloat16_forward_backward_preserves_dtype():
    mlp = make_mlp().to(torch.bfloat16)
    x = torch.randn(2, 5, 32, dtype=torch.bfloat16, requires_grad=True)
    y = mlp(x)
    y.float().mean().backward()
    assert y.dtype == torch.bfloat16 and torch.isfinite(y.float()).all()
    assert x.grad is not None and torch.isfinite(x.grad.float()).all()


def test_state_dict_roundtrip_is_exact():
    torch.manual_seed(9)
    first = make_mlp().eval()
    second = make_mlp().eval()
    second.load_state_dict(first.state_dict())
    x = torch.randn(2, 4, 32)
    torch.testing.assert_close(first(x), second(x), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_forward_backward():
    mlp = make_mlp().cuda()
    x = torch.randn(2, 5, 32, device="cuda", requires_grad=True)
    mlp(x).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
