import itertools

import pytest
import torch

from src.kda import chunkwise_kda, recurrent_kda
from tests.kda.conftest import random_core


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1, 1, 1),
        (1, 2, 1, 2, 3),
        (2, 7, 3, 2, 5),
        (1, 16, 2, 4, 3),
        (1, 17, 1, 3, 5),
        (2, 31, 2, 1, 3),
        (1, 65, 1, 2, 1),
    ],
)
@pytest.mark.parametrize("chunk_size", [1, 2, 4, 8, 16])
def test_chunkwise_outputs_and_final_state_match_recurrent(shape, chunk_size):
    batch, tokens, heads, key_dim, value_dim = shape
    q, k, v, g, beta, state = random_core(
        batch, tokens, heads, key_dim, value_dim
    )
    recurrent = recurrent_kda(q, k, v, g, beta, state, True)
    chunkwise = chunkwise_kda(
        q, k, v, g, beta, state, True,
        chunk_size=chunk_size,
        secondary_tile_size=min(chunk_size, 4),
    )
    torch.testing.assert_close(
        chunkwise.hidden_states, recurrent.hidden_states, rtol=2e-11, atol=2e-12
    )
    torch.testing.assert_close(
        chunkwise.final_state, recurrent.final_state, rtol=2e-11, atol=2e-12
    )


def test_chunk_size_invariance_including_partial_final_chunks():
    inputs = random_core(tokens=19, key_dim=3, value_dim=5)
    baseline = recurrent_kda(*inputs, output_final_state=True)
    for chunk_size in (1, 2, 4, 8, 16, 64):
        actual = chunkwise_kda(
            *inputs,
            output_final_state=True,
            chunk_size=chunk_size,
            secondary_tile_size=min(chunk_size, 4),
        )
        torch.testing.assert_close(
            actual.hidden_states, baseline.hidden_states, rtol=2e-11, atol=2e-12
        )
        torch.testing.assert_close(
            actual.final_state, baseline.final_state, rtol=2e-11, atol=2e-12
        )


@pytest.mark.parametrize("prefix", range(1, 9))
def test_chunkwise_prefix_causality_across_chunk_boundaries(prefix):
    q, k, v, g, beta, state = random_core(tokens=9)
    baseline = chunkwise_kda(
        q, k, v, g, beta, state, chunk_size=4, secondary_tile_size=2
    ).hidden_states
    changed = [tensor.clone() for tensor in (q, k, v, g, beta)]
    for tensor in changed:
        tensor[:, prefix:] = torch.randn_like(tensor[:, prefix:]) * 50
    actual = chunkwise_kda(
        *changed, state, chunk_size=4, secondary_tile_size=2
    ).hidden_states
    torch.testing.assert_close(baseline[:, :prefix], actual[:, :prefix], rtol=0, atol=0)


def test_chunkwise_mask_matches_unpadded_recurrent_and_preserves_state():
    q, k, v, g, beta, state = random_core(
        batch=1, tokens=9, heads=2, key_dim=3, value_dim=4
    )
    mask = torch.tensor([[True] * 6 + [False] * 3])
    chunked = chunkwise_kda(
        q, k, v, g, beta, state, True, mask,
        chunk_size=4, secondary_tile_size=2
    )
    valid = recurrent_kda(
        q[:, :6], k[:, :6], v[:, :6], g[:, :6], beta[:, :6], state, True
    )
    torch.testing.assert_close(chunked.hidden_states[:, :6], valid.hidden_states)
    assert torch.count_nonzero(chunked.hidden_states[:, 6:]) == 0
    torch.testing.assert_close(chunked.final_state, valid.final_state)


def test_recurrent_and_chunkwise_input_initial_state_gradients_match():
    inputs = random_core(
        batch=1, tokens=5, heads=2, key_dim=2, value_dim=3
    )
    recurrent_inputs = tuple(t.clone().requires_grad_() for t in inputs)
    chunk_inputs = tuple(t.clone().requires_grad_() for t in inputs)
    recurrent = recurrent_kda(
        *recurrent_inputs, output_final_state=True
    )
    chunked = chunkwise_kda(
        *chunk_inputs,
        output_final_state=True,
        chunk_size=3,
        secondary_tile_size=2,
    )
    (recurrent.hidden_states.square().sum() + recurrent.final_state.square().sum()).backward()
    (chunked.hidden_states.square().sum() + chunked.final_state.square().sum()).backward()
    for recurrent_input, chunk_input in zip(recurrent_inputs, chunk_inputs):
        torch.testing.assert_close(
            recurrent_input.grad, chunk_input.grad, rtol=2e-10, atol=2e-11
        )


def test_chunkwise_gradcheck():
    inputs = random_core(
        batch=1, tokens=3, heads=1, key_dim=2, value_dim=2
    )
    inputs = tuple(t.requires_grad_() for t in inputs)
    function = lambda *xs: (
        chunkwise_kda(
            *xs,
            output_final_state=True,
            chunk_size=2,
            secondary_tile_size=1,
        ).hidden_states,
        chunkwise_kda(
            *xs,
            output_final_state=True,
            chunk_size=2,
            secondary_tile_size=1,
        ).final_state,
    )
    assert torch.autograd.gradcheck(function, inputs, fast_mode=True)


def test_long_negative_cumulative_decay_remains_finite():
    q, k, v, _, beta, state = random_core(
        batch=1, tokens=96, heads=1, key_dim=4, value_dim=3,
        dtype=torch.float32
    )
    g = torch.full_like(q, -4.999)
    output = chunkwise_kda(
        q, k, v, g, beta, state, True,
        chunk_size=64, secondary_tile_size=16
    )
    assert torch.isfinite(output.hidden_states).all()
    assert torch.isfinite(output.final_state).all()

