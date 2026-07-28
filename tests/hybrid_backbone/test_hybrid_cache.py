import io

import pytest
import torch

from src.hybrid_backbone import (
    HybridBackboneCache,
    HybridLayerCache,
)
from src.mla import MLACache
from tests.hybrid_backbone.conftest import tiny_backbone


def test_cache_layer_count_order_types_offsets_and_final_mla():
    model = tiny_backbone(num_hybrid_groups=2).eval()
    cache = model(
        torch.randn(3, 6, 8), mode="prefill", use_cache=True
    ).cache
    assert len(cache.layer_caches) == 9
    expected = tuple(
        "gated_mla" if kind == "gated_mla_final" else kind
        for kind in model.attention_types
    )
    assert tuple(item.attention_type for item in cache.layer_caches) == expected
    assert all(item.sequence_offsets.tolist() == [6, 6, 6] for item in cache.layer_caches)
    assert cache.layer_caches[-1].attention_type == "gated_mla"
    assert isinstance(cache.layer_caches[-1].state, MLACache)


def test_kda_cache_is_fixed_size_while_mla_cache_grows_linearly():
    model = tiny_backbone().eval()
    short = model(
        torch.randn(2, 3, 8), mode="prefill", use_cache=True
    ).cache
    long = model(
        torch.randn(2, 7, 8), mode="prefill", use_cache=True
    ).cache
    assert short.kda_elements == long.kda_elements
    assert long.mla_elements / short.mla_elements == 7 / 3


def test_memory_accounting_matches_every_actual_tensor():
    model = tiny_backbone().eval()
    cache = model(
        torch.randn(2, 5, 8), mode="prefill", use_cache=True
    ).cache
    expected_kda = sum(
        item.num_elements for item in cache.layer_caches
        if item.attention_type == "kda"
    )
    expected_mla = sum(
        item.state.latent_kv.numel() for item in cache.layer_caches
        if item.attention_type == "gated_mla"
    )
    assert cache.kda_elements == expected_kda
    assert cache.mla_elements == expected_mla
    assert cache.total_elements == expected_kda + expected_mla


def test_clone_has_no_aliasing_and_original_cache_is_functional():
    model = tiny_backbone().eval()
    cache = model(
        torch.randn(2, 4, 8), mode="prefill", use_cache=True
    ).cache
    snapshot = cache.clone()
    _ = model(
        torch.randn(2, 1, 8),
        cache=cache,
        mode="decode",
        use_cache=True,
    )
    for original, cloned in zip(cache.layer_caches, snapshot.layer_caches):
        assert original.state is not cloned.state
        torch.testing.assert_close(
            original.sequence_offsets, cloned.sequence_offsets
        )
        if original.attention_type == "kda":
            assert (
                original.state.recurrent_state.data_ptr()
                != cloned.state.recurrent_state.data_ptr()
            )
        else:
            assert (
                original.state.latent_kv.data_ptr()
                != cloned.state.latent_kv.data_ptr()
            )


def test_batch_reorder_applies_to_every_layer():
    model = tiny_backbone().eval()
    cache = model(
        torch.randn(3, 5, 8), mode="prefill", use_cache=True
    ).cache
    indices = torch.tensor([2, 0, 2], dtype=torch.long)
    reordered = cache.reorder(indices)
    for actual, expected in zip(reordered.layer_caches, cache.layer_caches):
        torch.testing.assert_close(
            actual.sequence_offsets, expected.sequence_offsets[indices]
        )
        if actual.attention_type == "kda":
            torch.testing.assert_close(
                actual.state.recurrent_state,
                expected.state.recurrent_state[indices],
            )
        else:
            torch.testing.assert_close(
                actual.state.latent_kv, expected.state.latent_kv[indices]
            )


def test_cache_serialization_roundtrip():
    cache = tiny_backbone().eval()(
        torch.randn(2, 4, 8), mode="prefill", use_cache=True
    ).cache
    buffer = io.BytesIO()
    torch.save(cache, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)
    assert isinstance(loaded, HybridBackboneCache)
    assert loaded.sequence_length == cache.sequence_length
    assert loaded.total_elements == cache.total_elements


def test_wrong_cache_count_and_type_are_rejected():
    model = tiny_backbone().eval()
    cache = model(
        torch.randn(2, 3, 8), mode="prefill", use_cache=True
    ).cache
    incomplete = HybridBackboneCache(cache.layer_caches[:-1], 3)
    with pytest.raises(ValueError):
        model(
            torch.randn(2, 1, 8),
            cache=incomplete,
            mode="decode",
        )
    wrong = list(cache.layer_caches)
    wrong[0] = HybridLayerCache("gated_mla", cache.layer_caches[3].state)
    invalid = HybridBackboneCache(tuple(wrong), 3)
    with pytest.raises(ValueError):
        model(
            torch.randn(2, 1, 8),
            cache=invalid,
            mode="decode",
        )


def test_full_mode_rejects_external_cache_reset_is_cache_none():
    model = tiny_backbone().eval()
    prefix = model(
        torch.randn(2, 3, 8), mode="prefill", use_cache=True
    ).cache
    token = torch.randn(2, 1, 8)
    with pytest.raises(ValueError):
        model(token, cache=prefix, mode="full")
    fresh = model(token).last_hidden_state
    reset = model(token, cache=None).last_hidden_state
    torch.testing.assert_close(fresh, reset, rtol=0, atol=0)


def test_cache_batch_mismatch_is_rejected():
    model = tiny_backbone().eval()
    cache = model(
        torch.randn(2, 3, 8), mode="prefill", use_cache=True
    ).cache
    with pytest.raises(ValueError):
        model(
            torch.randn(3, 1, 8),
            cache=cache,
            mode="decode",
        )
