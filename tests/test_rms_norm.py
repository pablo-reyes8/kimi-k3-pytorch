import torch

from src.transformer_modules import RMSNorm


def test_rms_norm_is_finite_and_differentiable():
    x = torch.randn(2, 5, 16, requires_grad=True)
    output = RMSNorm(16)(x)
    output.square().mean().backward()
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
