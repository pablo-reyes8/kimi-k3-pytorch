import pytest
import torch

from tests.hybrid_backbone.conftest import (
    assert_hybrid_caches_close,
    tiny_backbone,
)


@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("tokens", [1, 3, 8])
def test_full_prefill_and_token_decode_are_equivalent(groups, tokens):
    model = tiny_backbone(num_hybrid_groups=groups).double().eval()
    x = torch.randn(2, tokens, 8, dtype=torch.float64)
    full = model(x).last_hidden_state
    prefill = model(x, mode="prefill", use_cache=True)
    torch.testing.assert_close(full, prefill.last_hidden_state, rtol=3e-11, atol=3e-12)

    cache = None
    pieces = []
    for token in range(tokens):
        output = model(
            x[:, token : token + 1],
            cache=cache,
            use_cache=True,
            mode="prefill" if cache is None else "decode",
        )
        pieces.append(output.last_hidden_state)
        cache = output.cache
    decoded = torch.cat(pieces, dim=1)
    torch.testing.assert_close(full, decoded, rtol=3e-10, atol=3e-11)
    assert_hybrid_caches_close(
        prefill.cache, cache, rtol=3e-9, atol=3e-11
    )


@pytest.mark.parametrize("split", range(1, 7))
def test_split_prefill_then_decode_matches_full_and_final_cache(split):
    model = tiny_backbone().double().eval()
    x = torch.randn(2, 7, 8, dtype=torch.float64)
    reference = model(x, mode="prefill", use_cache=True)
    prefix = model(x[:, :split], mode="prefill", use_cache=True)
    pieces = [prefix.last_hidden_state]
    cache = prefix.cache
    for token in range(split, 7):
        output = model(
            x[:, token : token + 1],
            cache=cache,
            use_cache=True,
            mode="decode",
        )
        pieces.append(output.last_hidden_state)
        cache = output.cache
    torch.testing.assert_close(
        torch.cat(pieces, dim=1),
        reference.last_hidden_state,
        rtol=3e-10,
        atol=3e-11,
    )
    assert_hybrid_caches_close(
        reference.cache, cache, rtol=3e-9, atol=3e-11
    )


def test_irregular_chunk_streaming_matches_full_and_cache():
    model = tiny_backbone().double().eval()
    x = torch.randn(2, 13, 8, dtype=torch.float64)
    reference = model(x, mode="prefill", use_cache=True)
    boundaries = (0, 2, 3, 7, 10, 13)
    cache = None
    pieces = []
    for start, end in zip(boundaries, boundaries[1:]):
        output = model(
            x[:, start:end],
            cache=cache,
            use_cache=True,
            mode="prefill",
        )
        pieces.append(output.last_hidden_state)
        cache = output.cache
    torch.testing.assert_close(
        torch.cat(pieces, 1),
        reference.last_hidden_state,
        rtol=3e-10,
        atol=3e-11,
    )
    assert_hybrid_caches_close(
        reference.cache, cache, rtol=3e-9, atol=3e-11
    )


def test_decode_rejects_missing_cache_and_multiple_tokens():
    model = tiny_backbone()
    with pytest.raises(ValueError):
        model(torch.randn(2, 1, 8), mode="decode")
    prefill = model(
        torch.randn(2, 2, 8), mode="prefill", use_cache=True
    )
    with pytest.raises(ValueError):
        model(
            torch.randn(2, 2, 8),
            cache=prefill.cache,
            mode="decode",
        )
