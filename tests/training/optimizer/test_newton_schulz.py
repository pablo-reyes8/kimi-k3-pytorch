import math

import pytest
import torch

from training.optimizer import (
    match_update_rms,
    zeropower_via_newton_schulz,
)


def test_ns_validates_shape_numerics_and_configuration():
    with pytest.raises(ValueError, match="2D"):
        zeropower_via_newton_schulz(torch.ones(2))
    with pytest.raises(FloatingPointError):
        zeropower_via_newton_schulz(torch.tensor([[float("nan")]]))
    with pytest.raises(ValueError):
        zeropower_via_newton_schulz(torch.ones(2, 2), steps=0)


@pytest.mark.parametrize("shape", [(3, 5), (5, 3)])
def test_ns_preserves_shape_dtype_device_input_and_is_graph_free(shape):
    source = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    before = source.detach().clone()
    first = zeropower_via_newton_schulz(source)
    second = zeropower_via_newton_schulz(source)
    assert first.shape == source.shape
    assert first.dtype == source.dtype
    assert first.device == source.device
    assert not first.requires_grad
    torch.testing.assert_close(source, before)
    torch.testing.assert_close(first, second)


def test_ns_zero_is_exact_and_reduces_singular_value_dispersion():
    assert torch.equal(
        zeropower_via_newton_schulz(torch.zeros(3, 2)),
        torch.zeros(3, 2),
    )
    source = torch.diag(torch.tensor([8.0, 1.0, 0.25]))
    transformed = zeropower_via_newton_schulz(source, steps=5)
    before = torch.linalg.svdvals(source)
    after = torch.linalg.svdvals(transformed)
    assert after.std(unbiased=False) / after.mean() < (
        before.std(unbiased=False) / before.mean()
    )


def test_update_rms_scaling_matches_documented_formulas():
    semi_orthogonal = torch.eye(2, 4)
    shaped = match_update_rms(semi_orthogonal, mode="shape")
    assert shaped.square().mean().sqrt() == pytest.approx(1.0)
    reference = torch.full((2, 4), 3.0)
    matched = match_update_rms(
        semi_orthogonal, reference, mode="reference_rms"
    )
    assert matched.square().mean().sqrt() == pytest.approx(3.0)
    with pytest.raises(ValueError):
        match_update_rms(semi_orthogonal, mode="reference_rms")
