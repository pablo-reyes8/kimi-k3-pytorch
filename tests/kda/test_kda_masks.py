import pytest
import torch

from tests.kda.conftest import tiny_kda


@pytest.mark.parametrize("mode", ["recurrent", "chunkwise"])
def test_padded_batch_matches_individual_unpadded_executions(mode):
    model = tiny_kda().double().eval()
    x = torch.randn(2, 7, 12, dtype=torch.float64)
    lengths = [7, 4]
    mask = torch.arange(7)[None, :] < torch.tensor(lengths)[:, None]
    batched = model(
        x, mask, mode=mode, output_final_state=True
    )
    individuals = [
        model(
            x[index : index + 1, :length],
            mode=mode,
            output_final_state=True,
        )
        for index, length in enumerate(lengths)
    ]
    for index, length in enumerate(lengths):
        torch.testing.assert_close(
            batched.hidden_states[index, :length],
            individuals[index].hidden_states[0],
            rtol=2e-11,
            atol=2e-12,
        )
        assert torch.count_nonzero(batched.hidden_states[index, length:]) == 0
        torch.testing.assert_close(
            batched.state.recurrent_state[index],
            individuals[index].state.recurrent_state[0],
            rtol=2e-11,
            atol=2e-12,
        )
        for name in ("q_conv_state", "k_conv_state", "v_conv_state"):
                torch.testing.assert_close(
                    getattr(batched.state, name).buffer[index],
                    getattr(individuals[index].state, name).buffer[0],
                    rtol=2e-14,
                    atol=2e-15,
                )
    torch.testing.assert_close(
        batched.state.sequence_offset, torch.tensor(lengths)
    )


def test_right_padding_does_not_change_later_streamed_valid_tokens():
    model = tiny_kda().double().eval()
    prefix = torch.randn(2, 6, 12, dtype=torch.float64)
    mask = torch.tensor(
        [[True] * 6, [True, True, True, True, False, False]]
    )
    padded = model(
        prefix, mask, mode="chunkwise", output_final_state=True
    )
    first = model(prefix[:1], mode="chunkwise", output_final_state=True)
    second = model(
        prefix[1:2, :4], mode="chunkwise", output_final_state=True
    )
    continuation = torch.randn(2, 2, 12, dtype=torch.float64)
    batched_next = model(
        continuation, state=padded.state, mode="recurrent"
    )
    first_next = model(
        continuation[:1], state=first.state, mode="recurrent"
    )
    second_next = model(
        continuation[1:], state=second.state, mode="recurrent"
    )
    torch.testing.assert_close(batched_next.hidden_states[:1], first_next.hidden_states)
    torch.testing.assert_close(batched_next.hidden_states[1:], second_next.hidden_states)


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones(2, 7),
        torch.ones(2, 6, dtype=torch.bool),
        torch.tensor([[True, False, True], [True, True, False]]),
        torch.zeros(2, 7, dtype=torch.bool),
    ],
)
def test_invalid_masks_rejected(mask):
    with pytest.raises((TypeError, ValueError)):
        tiny_kda()(torch.randn(2, 7, 12), mask)


def test_all_valid_mask_equals_no_mask_exactly():
    model = tiny_kda().eval()
    x = torch.randn(2, 7, 12)
    without = model(x, mode="chunkwise", output_final_state=True)
    with_mask = model(
        x,
        torch.ones(2, 7, dtype=torch.bool),
        mode="chunkwise",
        output_final_state=True,
    )
    torch.testing.assert_close(without.hidden_states, with_mask.hidden_states, rtol=0, atol=0)
    torch.testing.assert_close(
        without.state.recurrent_state, with_mask.state.recurrent_state, rtol=0, atol=0
    )
