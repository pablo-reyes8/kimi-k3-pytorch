import torch

from src.transformer_modules import BaselineTransformerBlock, TransformerBlockConfig


def test_baseline_block_shape_and_gradient():
    block = BaselineTransformerBlock(
        TransformerBlockConfig(
            d_model=16, n_heads=4, mlp_hidden_dim=32, max_seq_len=8
        )
    )
    x = torch.randn(2, 8, 16, requires_grad=True)
    output = block(x)
    output.mean().backward()
    assert output.shape == x.shape
    assert torch.isfinite(x.grad).all()
