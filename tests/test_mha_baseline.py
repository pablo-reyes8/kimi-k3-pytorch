import torch

from src.transformer_modules import CausalMHAConfig, MultiHeadSelfAttention


def test_mha_shape_and_causality_with_optional_rope():
    torch.manual_seed(1)
    attention = MultiHeadSelfAttention(
        CausalMHAConfig(d_model=16, n_heads=4, max_seq_len=8, use_rope=True)
    ).eval()
    original = torch.randn(1, 6, 16)
    changed = original.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:])
    first = attention(original)
    second = attention(changed)
    assert first.shape == original.shape
    torch.testing.assert_close(first[:, :4], second[:, :4], atol=1e-6, rtol=1e-5)

    _, weights = attention(
        original,
        attention_mask=torch.zeros(1, 6, dtype=torch.bool),
        need_weights=True,
    )
    assert torch.isfinite(weights).all()
    assert torch.count_nonzero(weights) == 0
