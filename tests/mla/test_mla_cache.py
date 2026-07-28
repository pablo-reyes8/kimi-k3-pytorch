from dataclasses import fields

import pytest
import torch

from src.mla import MLACache
from tests.mla.conftest import tiny_mla


def test_cache_contains_only_latent_mask_and_offset():
    assert [field.name for field in fields(MLACache)] == [
        "latent_kv", "attention_mask", "sequence_offset"
    ]


def test_prefill_cache_is_exactly_compression_projection():
    model = tiny_mla().double().eval()
    x = torch.randn(2, 6, 12, dtype=torch.float64)
    output = model(x, use_cache=True)
    expected = model.projections.compress_kv(x)
    torch.testing.assert_close(output.cache.latent_kv, expected, rtol=0, atol=0)


def test_cache_element_count_and_compression_ratio():
    model = tiny_mla().eval()
    cache = model(torch.randn(2, 7, 12), use_cache=True).cache
    assert cache.cache_elements == 2 * 7 * 5
    full_kv_elements = 2 * 7 * 3 * (2 + 4)
    assert cache.cache_elements < full_kv_elements
    assert full_kv_elements / cache.cache_elements == 18 / 5


def test_each_valid_token_grows_cache_by_one_latent_vector():
    model = tiny_mla().eval()
    cache = None
    for expected_length in range(1, 8):
        cache = model(
            torch.randn(2, 1, 12), cache=cache, use_cache=True
        ).cache
        assert cache.cache_length == expected_length
        assert cache.cache_elements == 2 * expected_length * 5


def test_cache_clone_has_no_aliasing():
    model = tiny_mla().eval()
    cache = model(torch.randn(2, 4, 12), use_cache=True).cache
    cloned = cache.clone()
    cloned.latent_kv.add_(1)
    cloned.attention_mask.logical_not_()
    cloned.sequence_offset.add_(1)
    assert not torch.equal(cache.latent_kv, cloned.latent_kv)
    assert not torch.equal(cache.attention_mask, cloned.attention_mask)
    assert not torch.equal(cache.sequence_offset, cloned.sequence_offset)


def test_cache_batch_reorder():
    model = tiny_mla().eval()
    cache = model(torch.randn(3, 4, 12), use_cache=True).cache
    indices = torch.tensor([2, 0, 2], dtype=torch.long)
    reordered = cache.reorder(indices)
    torch.testing.assert_close(reordered.latent_kv, cache.latent_kv[indices])
    torch.testing.assert_close(reordered.attention_mask, cache.attention_mask[indices])
    torch.testing.assert_close(reordered.sequence_offset, cache.sequence_offset[indices])


def test_empty_cache_resets_prefix_dependency():
    model = tiny_mla().eval()
    token = torch.randn(2, 1, 12)
    old = model(torch.randn(2, 5, 12), use_cache=True).cache
    with_old = model(token, cache=old).hidden_states
    empty = MLACache.empty(2, 5, dtype=token.dtype)
    reset = model(token, cache=empty).hidden_states
    fresh = model(token).hidden_states
    torch.testing.assert_close(reset, fresh, rtol=0, atol=0)
    assert not torch.equal(with_old, fresh)


@pytest.mark.parametrize("bad_shape", [(1, 2, 5), (2, 2, 4)])
def test_invalid_cache_shape_is_rejected_by_module(bad_shape):
    model = tiny_mla()
    latent = torch.randn(*bad_shape)
    mask = torch.ones(bad_shape[:2], dtype=torch.bool)
    offset = mask.sum(1)
    cache = MLACache(latent, mask, offset)
    with pytest.raises(ValueError):
        model(torch.randn(2, 1, 12), cache=cache)
