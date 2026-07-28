import copy

import pytest
import torch

from tests.kda.conftest import tiny_kda


def decode_tokens(model, x, state=None):
    outputs = []
    for index in range(x.shape[1]):
        result = model(x[:, index : index + 1], state=state, mode="decode")
        outputs.append(result.hidden_states)
        state = result.state
    return torch.cat(outputs, dim=1), state


def assert_states_close(first, second, *, tolerance=2e-11):
    torch.testing.assert_close(
        first.recurrent_state, second.recurrent_state,
        rtol=tolerance, atol=tolerance * 0.1
    )
    for name in ("q_conv_state", "k_conv_state", "v_conv_state"):
        torch.testing.assert_close(
            getattr(first, name).buffer,
            getattr(second, name).buffer,
            rtol=2e-14,
            atol=2e-15,
        )
    torch.testing.assert_close(first.sequence_offset, second.sequence_offset)


def test_full_recurrent_chunkwise_and_token_decode_equivalence():
    model = tiny_kda().double().eval()
    x = torch.randn(2, 11, 12, dtype=torch.float64)
    recurrent = model(x, mode="recurrent", output_final_state=True)
    chunkwise = model(x, mode="chunkwise", output_final_state=True)
    decoded_output, decoded_state = decode_tokens(model, x)
    torch.testing.assert_close(
        recurrent.hidden_states, decoded_output, rtol=2e-11, atol=2e-12
    )
    torch.testing.assert_close(
        chunkwise.hidden_states, decoded_output, rtol=2e-11, atol=2e-12
    )
    assert_states_close(recurrent.state, decoded_state)
    assert_states_close(chunkwise.state, decoded_state)


@pytest.mark.parametrize("split", [1, 2, 5, 8, 10])
@pytest.mark.parametrize("prefill_mode", ["recurrent", "chunkwise"])
def test_split_prefill_then_decode_matches_one_shot(split, prefill_mode):
    model = tiny_kda().double().eval()
    x = torch.randn(2, 11, 12, dtype=torch.float64)
    full = model(x, mode="recurrent", output_final_state=True)
    prefix = model(
        x[:, :split], mode=prefill_mode, output_final_state=True
    )
    suffix, final_state = decode_tokens(model, x[:, split:], prefix.state)
    combined = torch.cat((prefix.hidden_states, suffix), dim=1)
    torch.testing.assert_close(combined, full.hidden_states, rtol=2e-11, atol=2e-12)
    assert_states_close(final_state, full.state)


def test_irregular_streaming_chunks_match_full():
    model = tiny_kda().double().eval()
    x = torch.randn(2, 19, 12, dtype=torch.float64)
    full = model(x, mode="recurrent", output_final_state=True)
    sizes = [3, 5, 1, 7, 3]
    state, offset, outputs = None, 0, []
    for size in sizes:
        result = model(
            x[:, offset : offset + size],
            state=state,
            mode="decode" if size == 1 else "chunkwise",
            output_final_state=True,
        )
        outputs.append(result.hidden_states)
        state = result.state
        offset += size
    torch.testing.assert_close(
        torch.cat(outputs, 1), full.hidden_states, rtol=2e-11, atol=2e-12
    )
    assert_states_close(state, full.state)


def test_cache_size_is_constant_with_context_length():
    model = tiny_kda().eval()
    short = model(
        torch.randn(2, 2, 12), mode="chunkwise", output_final_state=True
    ).state
    long = model(
        torch.randn(2, 50, 12), mode="chunkwise", output_final_state=True
    ).state
    short_elements = sum(
        tensor.numel()
        for tensor in (
            short.recurrent_state, short.q_conv_state.buffer,
            short.k_conv_state.buffer, short.v_conv_state.buffer,
            short.sequence_offset,
        )
    )
    long_elements = sum(
        tensor.numel()
        for tensor in (
            long.recurrent_state, long.q_conv_state.buffer,
            long.k_conv_state.buffer, long.v_conv_state.buffer,
            long.sequence_offset,
        )
    )
    assert short_elements == long_elements


def test_new_state_reset_removes_previous_sequence_dependency():
    model = tiny_kda().eval()
    prefix = model(
        torch.randn(1, 5, 12), mode="recurrent", output_final_state=True
    )
    suffix = torch.randn(1, 3, 12)
    continued = model(
        suffix, state=prefix.state, mode="recurrent"
    ).hidden_states
    reset = model(suffix, state=None, mode="recurrent").hidden_states
    fresh = model(suffix, mode="recurrent").hidden_states
    assert not torch.allclose(continued, reset)
    torch.testing.assert_close(reset, fresh, rtol=0, atol=0)


def test_state_reorder_matches_reordered_independent_execution():
    model = tiny_kda().eval()
    x = torch.randn(3, 5, 12)
    state = model(x, mode="chunkwise", output_final_state=True).state
    indices = torch.tensor([2, 0, 2], dtype=torch.long)
    reordered = state.reorder(indices)
    continuation = torch.randn(3, 2, 12)
    actual = model(
        continuation, state=reordered, mode="recurrent", output_final_state=True
    )
    expected_outputs = []
    for target, source in enumerate(indices.tolist()):
        individual_state = state.reorder(torch.tensor([source]))
        result = model(
            continuation[target : target + 1],
            state=individual_state,
            mode="recurrent",
            output_final_state=True,
        )
        expected_outputs.append(result.hidden_states)
    torch.testing.assert_close(
        actual.hidden_states, torch.cat(expected_outputs), rtol=1e-5, atol=1e-6
    )


def test_decode_returns_state_even_without_explicit_flag():
    result = tiny_kda()(torch.randn(2, 1, 12), mode="decode")
    assert result.state is not None
    torch.testing.assert_close(
        result.state.sequence_offset, torch.ones(2, dtype=torch.long)
    )
