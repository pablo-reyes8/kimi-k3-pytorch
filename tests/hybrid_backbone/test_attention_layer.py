import torch
import torch.nn as nn

from src.hybrid_backbone import DenseKimiFFN, HybridAttentionLayer
from src.kda import KimiDeltaAttention
from tests.hybrid_backbone.conftest import tiny_kda_config


def make_layer(ffn=True):
    return HybridAttentionLayer(
        "kda",
        KimiDeltaAttention(tiny_kda_config()),
        DenseKimiFFN(8, 12) if ffn else None,
        8,
        1e-6,
        layer_index=0,
        group_index=0,
        position_in_group=0,
    )


def test_layer_matches_two_explicit_pre_norm_residual_equations():
    layer = make_layer().double().eval()
    x = torch.randn(2, 5, 8, dtype=torch.float64)
    mask = torch.ones(2, 5, dtype=torch.bool)
    attention = layer.attention(
        layer.attention_norm(x), mask, mode="chunkwise"
    ).hidden_states
    post_attention = x + attention
    expected = post_attention + layer.ffn(layer.ffn_norm(post_attention))
    actual = layer(x, mask).hidden_states
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_layer_is_identity_when_attention_and_ffn_are_zero():
    layer = make_layer().eval()
    with torch.no_grad():
        for parameter in layer.attention.parameters():
            parameter.zero_()
        for parameter in layer.ffn.parameters():
            parameter.zero_()
    x = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)
    torch.testing.assert_close(layer(x, mask).hidden_states, x, rtol=0, atol=0)


def test_attention_only_residual_has_no_double_addition():
    layer = make_layer().double().eval()
    with torch.no_grad():
        for parameter in layer.ffn.parameters():
            parameter.zero_()
    x = torch.randn(1, 4, 8, dtype=torch.float64)
    mask = torch.ones(1, 4, dtype=torch.bool)
    attention = layer.attention(
        layer.attention_norm(x), mask, mode="chunkwise"
    ).hidden_states
    torch.testing.assert_close(
        layer(x, mask).hidden_states, x + attention, rtol=0, atol=0
    )


def test_ffn_only_residual_matches_explicit_equation():
    layer = make_layer().double().eval()
    with torch.no_grad():
        for parameter in layer.attention.parameters():
            parameter.zero_()
    x = torch.randn(1, 4, 8, dtype=torch.float64)
    mask = torch.ones(1, 4, dtype=torch.bool)
    expected = x + layer.ffn(layer.ffn_norm(x))
    torch.testing.assert_close(
        layer(x, mask).hidden_states, expected, rtol=0, atol=0
    )


def test_layer_accepts_a_generic_future_ffn_interface():
    class FutureChannelMixer(nn.Module):
        def forward(self, hidden_states):
            return torch.ones_like(hidden_states)

    layer = HybridAttentionLayer(
        "kda",
        KimiDeltaAttention(tiny_kda_config()),
        FutureChannelMixer(),
        8,
        1e-6,
        layer_index=0,
        group_index=0,
        position_in_group=0,
    ).eval()
    with torch.no_grad():
        for parameter in layer.attention.parameters():
            parameter.zero_()
    x = torch.randn(1, 3, 8)
    mask = torch.ones(1, 3, dtype=torch.bool)
    torch.testing.assert_close(
        layer(x, mask).hidden_states, x + 1, rtol=0, atol=0
    )
