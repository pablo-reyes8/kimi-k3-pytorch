import math

import pytest
import torch

from src.mla import manual_causal_attention, mla_attention
from tests.mla.conftest import random_qkv


@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("query_tokens,key_tokens", [(1, 7), (4, 4), (3, 7)])
def test_manual_and_sdpa_forward_match(is_causal, query_tokens, key_tokens):
    q, k, v = random_qkv(query_tokens=query_tokens, key_tokens=key_tokens)
    manual = mla_attention(q, k, v, is_causal=is_causal, backend="manual")
    sdpa = mla_attention(q, k, v, is_causal=is_causal, backend="sdpa")
    torch.testing.assert_close(manual, sdpa, rtol=2e-12, atol=2e-12)


def test_manual_attention_matches_explicit_scalar_reference():
    q, k, v = random_qkv(batch=1, query_tokens=4, heads=1)
    actual, weights = manual_causal_attention(
        q, k, v, return_attentions=True
    )
    outputs = []
    expected_weights = torch.zeros(1, 1, 4, 4, dtype=torch.float64)
    for token in range(4):
        scores = torch.tensor(
            [
                torch.dot(q[0, token, 0], k[0, source, 0])
                / math.sqrt(q.shape[-1])
                for source in range(token + 1)
            ]
        )
        probabilities = torch.softmax(scores, dim=0)
        expected_weights[0, 0, token, : token + 1] = probabilities
        outputs.append(
            sum(
                probabilities[source] * v[0, source, 0]
                for source in range(token + 1)
            )
        )
    expected = torch.stack(outputs)[None, :, None, :]
    torch.testing.assert_close(actual, expected, rtol=2e-15, atol=2e-15)
    torch.testing.assert_close(weights, expected_weights, rtol=2e-15, atol=2e-15)


def test_causal_diagonal_is_included_and_future_is_exactly_zero():
    q, k, v = random_qkv(query_tokens=6)
    _, weights = manual_causal_attention(q, k, v, return_attentions=True)
    assert torch.all(weights.diagonal(dim1=-2, dim2=-1) > 0)
    assert torch.count_nonzero(torch.triu(weights, diagonal=1)) == 0
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights[..., 0]))


def test_zero_queries_and_keys_produce_uniform_causal_average():
    q = torch.zeros(1, 4, 1, 2)
    k = torch.zeros_like(q)
    v = torch.arange(4.0).reshape(1, 4, 1, 1)
    output = manual_causal_attention(q, k, v)
    expected = torch.tensor([0.0, 0.5, 1.0, 1.5]).reshape(1, 4, 1, 1)
    torch.testing.assert_close(output, expected)


def test_dominant_key_approaches_one_hot_attention():
    q = torch.ones(1, 1, 1, 1, dtype=torch.float64)
    k = torch.tensor([[[[-100.0]], [[100.0]]]], dtype=torch.float64)
    v = torch.tensor([[[[3.0]], [[9.0]]]], dtype=torch.float64)
    output = manual_causal_attention(q, k, v, is_causal=False)
    torch.testing.assert_close(output, torch.tensor([[[[9.0]]]], dtype=torch.float64))


def test_all_masked_rows_are_zero_not_nan_for_both_backends():
    q, k, v = random_qkv(batch=2, query_tokens=3)
    mask = torch.tensor([[False, False, False], [True, True, True]])
    for backend in ("manual", "sdpa"):
        output = mla_attention(q, k, v, mask, backend=backend)
        assert torch.count_nonzero(output[0]) == 0
        assert torch.isfinite(output).all()


def test_output_attention_shape_and_normalization():
    q, k, v = random_qkv(query_tokens=3, key_tokens=5)
    output, weights = mla_attention(
        q, k, v, backend="sdpa", return_attentions=True
    )
    assert output.shape == (2, 3, 3, 4)
    assert weights.shape == (2, 3, 3, 5)
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[..., 0]))


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
def test_large_scores_remain_finite(backend):
    q, k, v = random_qkv(dtype=torch.float32)
    output = mla_attention(q * 1e10, k * 1e10, v, backend=backend)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("backend", ["invalid", "flash"])
def test_invalid_backend_is_rejected(backend):
    q, k, v = random_qkv()
    with pytest.raises(ValueError):
        mla_attention(q, k, v, backend=backend)
