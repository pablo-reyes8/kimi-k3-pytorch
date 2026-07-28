import pytest
import torch

from src.vision import VisionMLP, VisionSelfAttention, VisionTransformerBlock


def _manual_attention(module, x, mask=None):
    batch, count, dim = x.shape
    qkv = module.qkv(x).reshape(
        batch, count, 3, module.num_heads, module.head_dim
    )
    q, k, value = qkv.permute(2, 0, 3, 1, 4)
    scores = (q @ k.transpose(-2, -1)) * module.scale
    if mask is not None:
        scores = scores.masked_fill(
            ~mask[:, None, None], torch.finfo(scores.dtype).min
        )
    probs = scores.softmax(-1)
    if mask is not None:
        probs = probs * mask[:, None, None]
        probs = probs / probs.sum(-1, keepdim=True).clamp_min(
            torch.finfo(probs.dtype).tiny
        )
    result = (probs @ value).transpose(1, 2).reshape(batch, count, dim)
    return module.projection(result), probs


@pytest.mark.parametrize("mask", [None, torch.tensor([[True, True, False]])])
def test_attention_matches_reference_equation(mask):
    torch.manual_seed(1)
    module = VisionSelfAttention(8, 2, qkv_bias=True, proj_bias=True)
    x = torch.randn(1, 3, 8)
    actual, weights = module(x, mask, output_attentions=True)
    expected, expected_weights = _manual_attention(module, x, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(weights, expected_weights, rtol=1e-5, atol=1e-6)


def test_attention_masked_keys_have_exactly_zero_probability():
    module = VisionSelfAttention(12, 3)
    _, weights = module(
        torch.randn(2, 4, 12),
        torch.tensor([[True, False, True, False], [False, True, True, True]]),
        output_attentions=True,
    )
    assert torch.count_nonzero(weights[0, :, :, [1, 3]]) == 0
    assert torch.count_nonzero(weights[1, :, :, 0]) == 0
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))


def test_attention_fully_masked_row_is_zero_not_nan():
    module = VisionSelfAttention(8, 2)
    output, weights = module(
        torch.randn(1, 3, 8),
        torch.zeros(1, 3, dtype=torch.bool),
        output_attentions=True,
    )
    assert torch.count_nonzero(weights) == 0
    assert torch.count_nonzero(output) == 0
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: VisionSelfAttention(7, 2),
        lambda: VisionSelfAttention(0, 1),
        lambda: VisionSelfAttention(8, 0),
        lambda: VisionSelfAttention(8, 2, attention_dropout=1.0),
        lambda: VisionSelfAttention(8, 2, projection_dropout=-0.1),
    ],
)
def test_attention_rejects_invalid_configuration(constructor):
    with pytest.raises(ValueError):
        constructor()


def test_attention_validates_input_and_mask_contract():
    module = VisionSelfAttention(8, 2)
    with pytest.raises(ValueError):
        module(torch.randn(2, 8))
    with pytest.raises(ValueError, match="padding_mask"):
        module(torch.randn(2, 3, 8), torch.ones(2, 4, dtype=torch.bool))
    with pytest.raises(TypeError, match="boolean"):
        module(torch.randn(2, 3, 8), torch.ones(2, 3))


def test_attention_bias_policy_is_complete():
    no_bias = VisionSelfAttention(8, 2)
    assert no_bias.qkv.bias is None and no_bias.projection.bias is None
    biased = VisionSelfAttention(8, 2, qkv_bias=True, proj_bias=True)
    assert biased.qkv.bias is not None and biased.projection.bias is not None


def test_attention_backward_reaches_every_parameter_and_input():
    module = VisionSelfAttention(8, 2, qkv_bias=True, proj_bias=True)
    x = torch.randn(2, 3, 8, requires_grad=True)
    module(x)[0].square().mean().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in module.parameters())


def test_mlp_matches_explicit_equation():
    module = VisionMLP(4, 7, bias=True)
    x = torch.randn(2, 3, 4)
    expected = module.fc2(module.activation(module.fc1(x)))
    torch.testing.assert_close(module(x), expected, rtol=0, atol=0)


def test_mlp_bias_and_shape_validation():
    module = VisionMLP(4, 8, bias=False)
    assert module.fc1.bias is None and module.fc2.bias is None
    with pytest.raises(ValueError):
        module(torch.randn(2, 3, 5))
    with pytest.raises(ValueError):
        VisionMLP(0, 8)
    with pytest.raises(ValueError):
        VisionMLP(4, 8, dropout=1)


def test_block_matches_pre_norm_residual_equation():
    block = VisionTransformerBlock(
        8, 2, mlp_ratio=2, norm_type="layernorm", qkv_bias=True,
        proj_bias=True, mlp_bias=True
    ).eval()
    x = torch.randn(2, 3, 8)
    attention, _ = block.attention(block.norm1(x))
    after_attention = x + attention
    expected = after_attention + block.mlp(block.norm2(after_attention))
    actual, _ = block(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_block_becomes_identity_when_residual_branches_are_zero():
    block = VisionTransformerBlock(8, 2, mlp_ratio=2)
    with torch.no_grad():
        for parameter in block.attention.parameters():
            parameter.zero_()
        for parameter in block.mlp.parameters():
            parameter.zero_()
    x = torch.randn(2, 3, 8)
    torch.testing.assert_close(block(x)[0], x, rtol=0, atol=0)


def test_block_bfloat16_forward_backward():
    block = VisionTransformerBlock(8, 2, mlp_ratio=2).to(torch.bfloat16)
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16, requires_grad=True)
    output, _ = block(x)
    assert output.dtype == torch.bfloat16
    output.float().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

