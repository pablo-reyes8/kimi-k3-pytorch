import pytest
import torch

from src.kda import LowerBoundedDecay


def test_decay_matches_scaled_sigmoid_formula_at_regular_logits():
    module = LowerBoundedDecay(2, -5).double()
    with torch.no_grad():
        module.A_log.copy_(torch.tensor([0.0, 0.7], dtype=torch.float32))
    logits = torch.tensor(
        [[[[ -2.0, 0.0, 2.0], [1.0, -1.0, 0.5]]]], dtype=torch.float64
    )
    g, alpha = module(logits)
    scale = module.A_log.double().exp()[None, None, :, None]
    expected = -5 * torch.sigmoid(scale * logits)
    torch.testing.assert_close(g, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(alpha, expected.exp(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_decay_open_bounds_hold_even_for_saturated_finite_logits(dtype):
    module = LowerBoundedDecay(2, -5).to(dtype)
    logits = torch.tensor(
        [-1e4, -100, 0, 100, 1e4], dtype=dtype
    ).reshape(1, 1, 1, 5).expand(1, 1, 2, 5)
    g, alpha = module(logits)
    assert torch.all(g > -5)
    assert torch.all(g < 0)
    assert torch.all(alpha > torch.tensor(-5, dtype=dtype).exp())
    assert torch.all(alpha < 1)
    assert torch.isfinite(g.float()).all() and torch.isfinite(alpha.float()).all()


def test_decay_limit_sign_is_not_inverted():
    module = LowerBoundedDecay(1, -5)
    logits = torch.tensor([[[[-20.0, 20.0]]]])
    g, _ = module(logits)
    assert g[..., 0].abs() < 1e-5
    assert (g[..., 1] + 5).abs() < 1e-5


def test_per_head_A_scale_changes_only_corresponding_head():
    module = LowerBoundedDecay(3, -5)
    logits = torch.ones(1, 2, 3, 4)
    baseline = module(logits)[0]
    with torch.no_grad():
        module.A_log[1] = 2
    changed = module(logits)[0]
    torch.testing.assert_close(baseline[:, :, 0], changed[:, :, 0])
    torch.testing.assert_close(baseline[:, :, 2], changed[:, :, 2])
    assert not torch.equal(baseline[:, :, 1], changed[:, :, 1])


def test_A_log_initialization_shape_and_value():
    module = LowerBoundedDecay(4)
    assert module.A_log.shape == (4,)
    torch.testing.assert_close(module.A_log, torch.zeros_like(module.A_log))


def test_decay_is_not_negative_softplus():
    module = LowerBoundedDecay(1, -5).double()
    logits = torch.tensor([[[[-10.0, 0.0, 10.0]]]], dtype=torch.float64)
    actual = module(logits)[0]
    legacy = -torch.nn.functional.softplus(logits)
    assert not torch.allclose(actual, legacy)
    assert actual.min() > -5 and legacy.min() < -5


def test_decay_gradcheck_and_all_parameter_gradients():
    module = LowerBoundedDecay(2).double()
    logits = torch.randn(1, 2, 2, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: module(x)[0], (logits,))
    module(logits)[0].sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert module.A_log.grad is not None and module.A_log.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "constructor", [lambda: LowerBoundedDecay(0), lambda: LowerBoundedDecay(2, 0)]
)
def test_decay_invalid_configuration(constructor):
    with pytest.raises(ValueError):
        constructor()

