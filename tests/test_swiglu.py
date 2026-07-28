import torch

from src.transformer_modules import SwiGLUFeedForward, SwiGLUMLPConfig


def test_swiglu_shape_and_gradient():
    module = SwiGLUFeedForward(SwiGLUMLPConfig(d_model=12, hidden_dim=24))
    x = torch.randn(2, 4, 12, requires_grad=True)
    output = module(x)
    output.sum().backward()
    assert output.shape == x.shape
    assert torch.isfinite(x.grad).all()
