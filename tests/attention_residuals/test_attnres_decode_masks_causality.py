import pytest
import torch

from tests.attention_residuals.conftest import (
    activate_depth_queries,
    assert_hybrid_caches_close,
    attnres_backbone,
)


@pytest.mark.parametrize(
    "depth_mode,backend",
    [("full", "eager"), ("block", "eager"), ("block", "two_phase")],
)
def test_full_equals_prefill_and_token_decode(depth_mode, backend):
    model = attnres_backbone(
        depth_mode=depth_mode, backend=backend, block_size=4
    ).double().eval()
    activate_depth_queries(model)
    x = torch.randn(2, 7, 8, dtype=torch.float64)
    full = model(x)
    prefill = model(x, mode="prefill", use_cache=True)
    torch.testing.assert_close(
        full.last_hidden_state, prefill.last_hidden_state,
        rtol=3e-10, atol=3e-11,
    )
    cache = None
    pieces = []
    for token in range(7):
        output = model(
            x[:, token : token + 1],
            cache=cache,
            use_cache=True,
            mode="prefill" if cache is None else "decode",
        )
        pieces.append(output.last_hidden_state)
        cache = output.cache
    torch.testing.assert_close(
        torch.cat(pieces, 1), full.last_hidden_state,
        rtol=4e-9, atol=4e-10,
    )
    assert_hybrid_caches_close(
        prefill.cache, cache, rtol=4e-8, atol=4e-10
    )


@pytest.mark.parametrize("depth_mode", ["full", "block"])
@pytest.mark.parametrize("split", range(1, 7))
def test_every_prefill_decode_split_matches_reference(depth_mode, split):
    model = attnres_backbone(depth_mode=depth_mode).double().eval()
    x = torch.randn(2, 7, 8, dtype=torch.float64)
    reference = model(x, mode="prefill", use_cache=True)
    prefix = model(x[:, :split], mode="prefill", use_cache=True)
    pieces, cache = [prefix.last_hidden_state], prefix.cache
    for token in range(split, 7):
        output = model(
            x[:, token : token + 1],
            cache=cache,
            mode="decode",
            use_cache=True,
        )
        pieces.append(output.last_hidden_state)
        cache = output.cache
    torch.testing.assert_close(
        torch.cat(pieces, 1), reference.last_hidden_state,
        rtol=4e-9, atol=4e-10,
    )
    assert_hybrid_caches_close(
        reference.cache, cache, rtol=4e-8, atol=4e-10
    )


@pytest.mark.parametrize("depth_mode", ["full", "block"])
def test_irregular_temporal_streaming_is_invariant(depth_mode):
    model = attnres_backbone(depth_mode=depth_mode).double().eval()
    activate_depth_queries(model)
    x = torch.randn(2, 13, 8, dtype=torch.float64)
    reference = model(x, mode="prefill", use_cache=True)
    boundaries = (0, 2, 3, 7, 10, 13)
    cache, pieces = None, []
    for start, end in zip(boundaries, boundaries[1:]):
        output = model(
            x[:, start:end],
            cache=cache,
            mode="prefill",
            use_cache=True,
        )
        pieces.append(output.last_hidden_state)
        cache = output.cache
    torch.testing.assert_close(
        torch.cat(pieces, 1), reference.last_hidden_state,
        rtol=5e-9, atol=5e-10,
    )
    assert_hybrid_caches_close(
        reference.cache, cache, rtol=5e-8, atol=5e-10
    )


@pytest.mark.parametrize(
    "depth_mode,backend",
    [("full", "eager"), ("block", "eager"), ("block", "two_phase")],
)
@pytest.mark.parametrize("prefix", range(1, 7))
def test_end_to_end_prefix_causality(depth_mode, backend, prefix):
    model = attnres_backbone(
        depth_mode=depth_mode, backend=backend
    ).double().eval()
    activate_depth_queries(model)
    x = torch.randn(2, 7, 8, dtype=torch.float64)
    changed = x.clone()
    changed[:, prefix:] = torch.randn_like(changed[:, prefix:]) * 100
    torch.testing.assert_close(
        model(x).last_hidden_state[:, :prefix],
        model(changed).last_hidden_state[:, :prefix],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("depth_mode", ["full", "block"])
def test_padded_batch_matches_individual_valid_outputs_and_decode(depth_mode):
    model = attnres_backbone(depth_mode=depth_mode).double().eval()
    x = torch.randn(2, 5, 8, dtype=torch.float64)
    lengths = (5, 3)
    mask = torch.arange(5)[None, :] < torch.tensor(lengths)[:, None]
    batched = model(x, mask, mode="prefill", use_cache=True)
    next_token = torch.randn(2, 1, 8, dtype=torch.float64)
    batched_decode = model(
        next_token,
        cache=batched.cache,
        mode="decode",
        use_cache=True,
    )
    for batch, length in enumerate(lengths):
        individual = model(
            x[batch : batch + 1, :length],
            mode="prefill",
            use_cache=True,
        )
        torch.testing.assert_close(
            batched.last_hidden_state[batch : batch + 1, :length],
            individual.last_hidden_state,
            rtol=5e-9,
            atol=5e-10,
        )
        individual_decode = model(
            next_token[batch : batch + 1],
            cache=individual.cache,
            mode="decode",
            use_cache=True,
        )
        torch.testing.assert_close(
            batched_decode.last_hidden_state[batch : batch + 1],
            individual_decode.last_hidden_state,
            rtol=5e-9,
            atol=5e-10,
        )


def test_attnres_adds_no_persistent_cache_entry_or_sequence_growth():
    model = attnres_backbone(depth_mode="block").eval()
    output = model(
        torch.randn(2, 4, 8), mode="prefill", use_cache=True
    )
    assert len(output.cache.layer_caches) == len(model.layers)
    assert not hasattr(output.cache, "attention_residual_state")
    assert not any("depth" in field for field in vars(output.cache))
    original_count = len(output.cache.layer_caches)
    cache = output.cache
    for _ in range(5):
        cache = model(
            torch.randn(2, 1, 8),
            cache=cache,
            mode="decode",
            use_cache=True,
        ).cache
        assert len(cache.layer_caches) == original_count
