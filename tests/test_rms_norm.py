import pytest
import torch

from src.transformer_modules import RMSNorm


@pytest.mark.parametrize("shape", [(4, 16, 32), (2, 3, 4, 32), (1, 32)])
def test_shape_dtype_and_last_dimension_contract(shape):
    x = torch.randn(*shape)
    y = RMSNorm(32)(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.device == x.device


@pytest.mark.parametrize("dim", [0, -1, -32])
def test_invalid_dimension_rejected(dim):
    with pytest.raises(ValueError, match="dim must be"):
        RMSNorm(dim)


@pytest.mark.parametrize("eps", [0.0, -1e-6])
def test_invalid_epsilon_rejected(eps):
    with pytest.raises(ValueError, match="eps must be"):
        RMSNorm(8, eps=eps)


def test_parameter_shape_and_initialization():
    norm = RMSNorm(32)
    assert norm.weight.shape == (32,)
    torch.testing.assert_close(norm.weight, torch.ones_like(norm.weight))


def test_forward_matches_fp32_reference_equation():
    torch.manual_seed(1)
    norm = RMSNorm(17, eps=3e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, 17))
    x = torch.randn(3, 7, 17)
    expected = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + norm.eps)
    expected = expected.to(x.dtype) * norm.weight.to(x.dtype)
    torch.testing.assert_close(norm(x), expected, atol=1e-6, rtol=1e-5)


def test_only_last_dimension_is_normalized():
    norm = RMSNorm(5)
    x = torch.randn(2, 3, 4, 5)
    y = norm(x)
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm.eps)
    torch.testing.assert_close(y, expected)


@pytest.mark.parametrize("value", [0.0, 1e-8, 1e6, -1e6])
def test_extreme_constant_inputs_are_finite(value):
    y = RMSNorm(16)(torch.full((2, 4, 16), value))
    assert torch.isfinite(y).all()
    if value == 0:
        assert torch.count_nonzero(y) == 0


def test_wrong_feature_dimension_rejected():
    with pytest.raises(ValueError, match="Expected last dimension"):
        RMSNorm(8)(torch.randn(2, 4, 7))


def test_backward_matches_autograd_and_all_gradients_are_finite():
    norm = RMSNorm(13)
    x = torch.randn(2, 5, 13, requires_grad=True)
    norm(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert norm.weight.grad is not None and torch.isfinite(norm.weight.grad).all()
    assert norm.weight.grad.abs().sum() > 0


def test_scale_equivariance_when_epsilon_is_negligible():
    norm = RMSNorm(16, eps=1e-12)
    x = torch.randn(2, 3, 16)
    torch.testing.assert_close(norm(10 * x), norm(x), atol=2e-5, rtol=2e-5)


def test_bfloat16_uses_stable_accumulation_and_preserves_dtype():
    norm = RMSNorm(32).to(torch.bfloat16)
    x = (torch.randn(2, 8, 32) * 1000).to(torch.bfloat16)
    y = norm(x)
    reference = (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + norm.eps)
    ).to(torch.bfloat16) * norm.weight
    assert y.dtype == torch.bfloat16
    assert torch.isfinite(y.float()).all()
    torch.testing.assert_close(y, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_float16_forward_backward():
    norm = RMSNorm(32).cuda().half()
    x = torch.randn(2, 8, 32, device="cuda", dtype=torch.float16, requires_grad=True)
    y = norm(x)
    y.float().mean().backward()
    assert y.dtype == torch.float16 and y.device.type == "cuda"
    assert torch.isfinite(y.float()).all() and torch.isfinite(x.grad.float()).all()
