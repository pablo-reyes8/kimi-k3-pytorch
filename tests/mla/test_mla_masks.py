import pytest
import torch

from tests.mla.conftest import tiny_mla


def test_batched_right_padding_matches_individual_unpadded_execution():
    model = tiny_mla().double().eval()
    x = torch.randn(3, 6, 12, dtype=torch.float64)
    lengths = (6, 4, 1)
    mask = torch.arange(6)[None, :] < torch.tensor(lengths)[:, None]
    batched = model(x, attention_mask=mask).hidden_states
    for batch, length in enumerate(lengths):
        individual = model(x[batch : batch + 1, :length]).hidden_states
        torch.testing.assert_close(
            batched[batch : batch + 1, :length],
            individual,
            rtol=3e-12,
            atol=3e-12,
        )
        assert torch.count_nonzero(batched[batch, length:]) == 0


def test_padded_prefill_then_decode_compacts_cache_and_matches_individual():
    model = tiny_mla().double().eval()
    prefix = torch.randn(2, 5, 12, dtype=torch.float64)
    prefix_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )
    next_tokens = torch.randn(2, 2, 12, dtype=torch.float64)
    prefill = model(prefix, prefix_mask, use_cache=True)
    decoded = model(next_tokens, cache=prefill.cache, use_cache=True)
    assert decoded.cache.sequence_offset.tolist() == [7, 5]
    assert decoded.cache.attention_mask.tolist() == [
        [True] * 7,
        [True] * 5 + [False] * 2,
    ]
    for batch, prefix_length in enumerate((5, 3)):
        sequence = torch.cat(
            (prefix[batch : batch + 1, :prefix_length], next_tokens[batch : batch + 1]),
            dim=1,
        )
        expected = model(sequence).hidden_states[:, -2:]
        torch.testing.assert_close(
            decoded.hidden_states[batch : batch + 1],
            expected,
            rtol=3e-12,
            atol=3e-12,
        )


@pytest.mark.parametrize(
    "mask,error",
    [
        (torch.ones(2, 3, 1, dtype=torch.bool), ValueError),
        (torch.ones(2, 4, dtype=torch.bool), ValueError),
        (torch.ones(2, 3), TypeError),
        (torch.tensor([[True, False, True], [True, True, True]]), ValueError),
        (torch.tensor([[False, False, False], [True, True, True]]), ValueError),
    ],
)
def test_invalid_masks_are_rejected(mask, error):
    with pytest.raises(error):
        tiny_mla()(torch.randn(2, 3, 12), attention_mask=mask)


@pytest.mark.parametrize("prefix", range(1, 8))
def test_future_tokens_cannot_change_prefix(prefix):
    model = tiny_mla().double().eval()
    x = torch.randn(2, 8, 12, dtype=torch.float64)
    changed = x.clone()
    changed[:, prefix:] = torch.randn_like(changed[:, prefix:]) * 100
    torch.testing.assert_close(
        model(x).hidden_states[:, :prefix],
        model(changed).hidden_states[:, :prefix],
        rtol=0,
        atol=0,
    )
