import pytest
import torch

from tests.hybrid_backbone.conftest import (
    assert_kda_states_close,
    tiny_backbone,
)


@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("prefix", range(1, 7))
def test_end_to_end_prefix_causality(groups, prefix):
    model = tiny_backbone(num_hybrid_groups=groups).double().eval()
    x = torch.randn(2, 7, 8, dtype=torch.float64)
    changed = x.clone()
    changed[:, prefix:] = torch.randn_like(changed[:, prefix:]) * 100
    torch.testing.assert_close(
        model(x).last_hidden_state[:, :prefix],
        model(changed).last_hidden_state[:, :prefix],
        rtol=0,
        atol=0,
    )


def test_batched_right_padding_matches_individual_outputs_and_caches():
    model = tiny_backbone().double().eval()
    x = torch.randn(3, 6, 8, dtype=torch.float64)
    lengths = (6, 4, 1)
    mask = torch.arange(6)[None, :] < torch.tensor(lengths)[:, None]
    batched = model(
        x, mask, mode="prefill", use_cache=True
    )
    assert batched.cache.sequence_lengths.tolist() == list(lengths)
    for batch, length in enumerate(lengths):
        individual = model(
            x[batch : batch + 1, :length],
            mode="prefill",
            use_cache=True,
        )
        torch.testing.assert_close(
            batched.last_hidden_state[batch : batch + 1, :length],
            individual.last_hidden_state,
            rtol=4e-10,
            atol=4e-11,
        )
        for batched_layer, individual_layer in zip(
            batched.cache.layer_caches, individual.cache.layer_caches
        ):
            if batched_layer.attention_type == "kda":
                sliced = batched_layer.state
                target = individual_layer.state
                torch.testing.assert_close(
                    sliced.recurrent_state[batch : batch + 1],
                    target.recurrent_state,
                    rtol=4e-9,
                    atol=4e-11,
                )
                torch.testing.assert_close(
                    sliced.q_conv_state.buffer[batch : batch + 1],
                    target.q_conv_state.buffer,
                    rtol=4e-9,
                    atol=4e-11,
                )
            else:
                torch.testing.assert_close(
                    batched_layer.state.latent_kv[
                        batch : batch + 1, :length
                    ],
                    individual_layer.state.latent_kv,
                    rtol=4e-10,
                    atol=4e-11,
                )


def test_unequal_padded_prefill_then_decode_matches_individual_streams():
    model = tiny_backbone().double().eval()
    prefix = torch.randn(2, 5, 8, dtype=torch.float64)
    lengths = (5, 3)
    mask = torch.arange(5)[None, :] < torch.tensor(lengths)[:, None]
    next_token = torch.randn(2, 1, 8, dtype=torch.float64)
    prefill = model(prefix, mask, mode="prefill", use_cache=True)
    decoded = model(
        next_token,
        cache=prefill.cache,
        mode="decode",
        use_cache=True,
    )
    assert decoded.cache.sequence_lengths.tolist() == [6, 4]
    for batch, length in enumerate(lengths):
        individual_prefix = model(
            prefix[batch : batch + 1, :length],
            mode="prefill",
            use_cache=True,
        )
        individual_decode = model(
            next_token[batch : batch + 1],
            cache=individual_prefix.cache,
            mode="decode",
            use_cache=True,
        )
        torch.testing.assert_close(
            decoded.last_hidden_state[batch : batch + 1],
            individual_decode.last_hidden_state,
            rtol=5e-10,
            atol=5e-11,
        )


@pytest.mark.parametrize(
    "mask,error",
    [
        (torch.ones(2, 4, dtype=torch.bool), ValueError),
        (torch.ones(2, 3), TypeError),
        (torch.tensor([[True, False, True], [True, True, True]]), ValueError),
        (torch.tensor([[False, False, False], [True, True, True]]), ValueError),
    ],
)
def test_invalid_masks_are_rejected_once_at_backbone_boundary(mask, error):
    with pytest.raises(error):
        tiny_backbone()(torch.randn(2, 3, 8), attention_mask=mask)
