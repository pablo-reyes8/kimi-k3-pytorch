import math

import pytest
import torch
import torch.nn.functional as F

from src.transformer_modules import CausalMHAConfig, MultiHeadSelfAttention


def make_config(**overrides):
    values = dict(
        d_model=32,
        n_heads=4,
        head_dim=8,
        attention_dropout=0.0,
        residual_dropout=0.0,
        use_bias=False,
        use_rope=True,
        rope_theta=10_000.0,
        rotary_dim=8,
        max_seq_len=32,
        init_std=0.02,
    )
    values.update(overrides)
    return CausalMHAConfig(**values)


def make_attention(**overrides):
    return MultiHeadSelfAttention(make_config(**overrides))


@pytest.mark.parametrize(
    "override",
    [
        {"d_model": 0},
        {"n_heads": 0},
        {"head_dim": 0},
        {"head_dim": 7},
        {"attention_dropout": -0.1},
        {"attention_dropout": 1.0},
        {"residual_dropout": -0.1},
        {"residual_dropout": 1.0},
        {"max_seq_len": 0},
        {"init_std": 0},
        {"rope_theta": 0},
        {"rotary_dim": 0},
        {"rotary_dim": 9},
        {"rotary_dim": 3},
    ],
)
def test_invalid_configuration_rejected(override):
    with pytest.raises(ValueError):
        make_attention(**override)


def test_head_dimension_is_inferred_and_divisibility_enforced():
    attention = MultiHeadSelfAttention(
        CausalMHAConfig(d_model=24, n_heads=3, head_dim=None)
    )
    assert attention.head_dim == 8 and attention.inner_dim == 24
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadSelfAttention(CausalMHAConfig(d_model=25, n_heads=3))


def test_projection_shapes_bias_policy_and_initialization():
    torch.manual_seed(1)
    attention = make_attention(use_bias=True, init_std=0.03)
    for layer in (
        attention.q_proj,
        attention.k_proj,
        attention.v_proj,
        attention.out_proj,
    ):
        assert layer.weight.shape == (32, 32)
        assert layer.bias is not None and torch.count_nonzero(layer.bias) == 0
        assert torch.isfinite(layer.weight).all()
        assert abs(layer.weight.std(unbiased=False).item() - 0.03) < 0.005
    without_bias = make_attention(use_bias=False)
    assert all(
        layer.bias is None
        for layer in (
            without_bias.q_proj,
            without_bias.k_proj,
            without_bias.v_proj,
            without_bias.out_proj,
        )
    )


def test_rope_is_optional_and_not_constructed_when_disabled():
    assert make_attention(use_rope=True).rope is not None
    assert make_attention(use_rope=False).rope is None


@pytest.mark.parametrize("shape", [(2, 8, 32), (0, 8, 32), (2, 0, 32)])
def test_output_shape_for_valid_btd_inputs(shape):
    output = make_attention()(torch.randn(*shape))
    assert output.shape == shape


@pytest.mark.parametrize("shape", [(8, 32), (2, 8, 32, 1), (2, 8, 31)])
def test_invalid_input_contract_rejected(shape):
    with pytest.raises(ValueError):
        make_attention()(torch.randn(*shape))


def test_sequence_longer_than_config_rejected():
    with pytest.raises(ValueError, match="max_seq_len"):
        make_attention(max_seq_len=4)(torch.randn(1, 5, 32))


def test_forward_matches_manual_scaled_dot_product_equation_without_rope():
    torch.manual_seed(4)
    attention = make_attention(use_rope=False).eval()
    x = torch.randn(2, 6, 32)
    q = attention._shape_projection(attention.q_proj(x)).transpose(1, 2)
    k = attention._shape_projection(attention.k_proj(x)).transpose(1, 2)
    v = attention._shape_projection(attention.v_proj(x)).transpose(1, 2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(attention.head_dim)
    causal = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
    weights = F.softmax(scores.float().masked_fill(causal, float("-inf")), dim=-1)
    context = (weights.to(v.dtype) @ v).transpose(1, 2).reshape(2, 6, 32)
    expected = attention.out_proj(context)
    torch.testing.assert_close(attention(x), expected, atol=1e-6, rtol=1e-5)


def test_attention_weights_shape_normalization_and_exact_causality():
    attention = make_attention().eval()
    _, weights = attention(torch.randn(2, 9, 32), need_weights=True)
    assert weights.shape == (2, 4, 9, 9)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 9))
    future = torch.triu(torch.ones(9, 9, dtype=torch.bool), diagonal=1)
    assert torch.count_nonzero(weights[:, :, future]) == 0


def test_changing_future_tokens_cannot_change_past_outputs():
    torch.manual_seed(5)
    attention = make_attention().eval()
    first_input = torch.randn(2, 10, 32)
    second_input = first_input.clone()
    second_input[:, 5:] = torch.randn_like(second_input[:, 5:])
    first, second = attention(first_input), attention(second_input)
    torch.testing.assert_close(first[:, :5], second[:, :5], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "mask",
    [torch.ones(8), torch.ones(2, 8, 1), torch.ones(2, 9)],
)
def test_invalid_attention_mask_shape_rejected(mask):
    with pytest.raises(ValueError):
        make_attention()(torch.randn(2, 8, 32), attention_mask=mask)


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.long, torch.float32])
def test_padding_keys_receive_exactly_zero_probability(mask_dtype):
    attention = make_attention().eval()
    mask = torch.ones(2, 8, dtype=mask_dtype)
    mask[0, 3] = 0
    mask[1, 5] = 0
    _, weights = attention(
        torch.randn(2, 8, 32), attention_mask=mask, need_weights=True
    )
    assert torch.count_nonzero(weights[0, :, :, 3]) == 0
    assert torch.count_nonzero(weights[1, :, :, 5]) == 0


def test_all_masked_rows_are_zero_not_uniform_or_nan():
    attention = make_attention().eval()
    output, weights = attention(
        torch.randn(2, 8, 32),
        attention_mask=torch.zeros(2, 8),
        need_weights=True,
    )
    assert torch.isfinite(output).all() and torch.isfinite(weights).all()
    assert torch.count_nonzero(weights) == 0
    assert torch.count_nonzero(output) == 0


def test_safe_masked_softmax_normalizes_only_rows_with_allowed_keys():
    attention = make_attention()
    scores = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    allowed = torch.tensor([[[[True, False], [False, False]]]])
    weights = attention._safe_masked_softmax(scores, allowed)
    torch.testing.assert_close(weights[0, 0, 0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(weights[0, 0, 1], torch.tensor([0.0, 0.0]))


def test_nope_ignores_positions_while_rope_responds_to_relative_positions():
    x = torch.randn(2, 8, 32)
    regular = torch.arange(8)
    stretched = torch.arange(8) * 3
    nope = make_attention(use_rope=False).eval()
    torch.testing.assert_close(
        nope(x, position_ids=regular), nope(x, position_ids=stretched)
    )
    rope = make_attention(use_rope=True).eval()
    assert not torch.allclose(
        rope(x, position_ids=regular),
        rope(x, position_ids=stretched),
        atol=1e-7,
        rtol=1e-7,
    )


def test_start_position_matches_explicit_position_ids():
    attention = make_attention().eval()
    x = torch.randn(2, 8, 32)
    torch.testing.assert_close(
        attention(x, start_pos=10),
        attention(x, position_ids=torch.arange(10, 18)),
    )


def test_batched_positions_supported():
    attention = make_attention().eval()
    x = torch.randn(2, 8, 32)
    positions = torch.stack((torch.arange(8), torch.arange(10, 18)))
    assert attention(x, position_ids=positions).shape == x.shape


def test_dropout_train_eval_contract():
    x = torch.randn(4, 12, 32)
    stochastic = make_attention(attention_dropout=0.5, residual_dropout=0.5)
    stochastic.train()
    assert not torch.equal(stochastic(x), stochastic(x))
    stochastic.eval()
    assert torch.equal(stochastic(x), stochastic(x))
    deterministic = make_attention().train()
    assert torch.equal(deterministic(x), deterministic(x))


def test_all_parameters_and_input_receive_finite_gradients():
    attention = make_attention(use_bias=True)
    x = torch.randn(2, 7, 32, requires_grad=True)
    attention(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, parameter in attention.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_bfloat16_forward_backward():
    attention = make_attention().to(torch.bfloat16)
    x = torch.randn(2, 7, 32, dtype=torch.bfloat16, requires_grad=True)
    output = attention(x)
    output.float().mean().backward()
    assert output.dtype == torch.bfloat16 and torch.isfinite(output.float()).all()
    assert x.grad is not None and torch.isfinite(x.grad.float()).all()


class TinyKVCache:
    def __init__(self):
        self.keys = []
        self.values = []

    def append(self, key, value, position_ids):
        self.keys.append(key)
        self.values.append(value)

    def get_kv(self):
        return torch.cat(self.keys, dim=2), torch.cat(self.values, dim=2)


def test_token_by_token_decode_matches_full_attention():
    torch.manual_seed(7)
    attention = make_attention().eval()
    x = torch.randn(2, 9, 32)
    full = attention(x)
    cache = TinyKVCache()
    decoded = []
    for position in range(x.shape[1]):
        output, returned_cache, aux = attention.forward_decode(
            x[:, position : position + 1],
            cache,
            position_ids=torch.tensor([position]),
            need_weights=True,
        )
        assert returned_cache is cache
        assert aux["attn_weights"].shape[-1] == position + 1
        decoded.append(output)
    torch.testing.assert_close(torch.cat(decoded, dim=1), full, atol=1e-5, rtol=1e-5)


def test_decode_rejects_non_single_token_input_and_bad_mask():
    attention = make_attention()
    with pytest.raises(ValueError, match="\\[B,1,D\\]"):
        attention.forward_decode(torch.randn(2, 2, 32), TinyKVCache())
    with pytest.raises(ValueError, match="decode attention_mask"):
        attention.forward_decode(
            torch.randn(2, 1, 32),
            TinyKVCache(),
            position_ids=torch.tensor([0]),
            attention_mask=torch.ones(2, 1, 1),
        )


def test_state_dict_roundtrip_is_exact():
    first = make_attention().eval()
    second = make_attention().eval()
    second.load_state_dict(first.state_dict())
    x = torch.randn(2, 8, 32)
    torch.testing.assert_close(first(x), second(x), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_forward_backward():
    attention = make_attention().cuda()
    x = torch.randn(2, 8, 32, device="cuda", requires_grad=True)
    attention(x).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
