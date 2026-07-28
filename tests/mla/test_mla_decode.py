import pytest
import torch

from tests.mla.conftest import tiny_mla


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
@pytest.mark.parametrize("tokens", [1, 2, 7, 11])
def test_full_forward_equals_token_by_token_decode(backend, tokens):
    model = tiny_mla(attention_backend=backend).double().eval()
    x = torch.randn(2, tokens, 12, dtype=torch.float64)
    full = model(x).hidden_states
    cache = None
    pieces = []
    for token in range(tokens):
        output = model(x[:, token : token + 1], cache=cache, use_cache=True)
        pieces.append(output.hidden_states)
        cache = output.cache
    decoded = torch.cat(pieces, dim=1)
    torch.testing.assert_close(full, decoded, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("split", range(1, 8))
def test_full_forward_equals_all_two_chunk_split_points(split):
    model = tiny_mla().double().eval()
    x = torch.randn(2, 8, 12, dtype=torch.float64)
    full = model(x).hidden_states
    prefix = model(x[:, :split], use_cache=True)
    suffix = model(x[:, split:], cache=prefix.cache, use_cache=True)
    streamed = torch.cat((prefix.hidden_states, suffix.hidden_states), dim=1)
    torch.testing.assert_close(full, streamed, rtol=3e-12, atol=3e-12)


def test_irregular_streaming_chunks_match_full():
    model = tiny_mla().double().eval()
    x = torch.randn(2, 13, 12, dtype=torch.float64)
    boundaries = (0, 2, 3, 8, 10, 13)
    cache = None
    pieces = []
    for start, end in zip(boundaries, boundaries[1:]):
        output = model(x[:, start:end], cache=cache, use_cache=True)
        cache = output.cache
        pieces.append(output.hidden_states)
    torch.testing.assert_close(
        torch.cat(pieces, dim=1), model(x).hidden_states,
        rtol=3e-12, atol=3e-12,
    )


def test_cache_is_functional_and_not_mutated_during_decode():
    model = tiny_mla().eval()
    cache = model(torch.randn(2, 4, 12), use_cache=True).cache
    snapshot = cache.clone()
    _ = model(torch.randn(2, 1, 12), cache=cache, use_cache=True)
    torch.testing.assert_close(cache.latent_kv, snapshot.latent_kv)
    torch.testing.assert_close(cache.attention_mask, snapshot.attention_mask)
    torch.testing.assert_close(cache.sequence_offset, snapshot.sequence_offset)
