import pytest
import torch

from src.kda import recurrent_kda
from tests.kda.conftest import random_core


def manual_recurrent(q, k, v, g, beta, initial_state=None, mask=None):
    batch, tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    state = (
        torch.zeros(batch, heads, key_dim, value_dim, dtype=q.dtype)
        if initial_state is None
        else initial_state.clone()
    )
    outputs, states = [], []
    for t in range(tokens):
        if mask is not None and not mask[:, t].all():
            # This helper's masked cases use batch size one.
            assert batch == 1
            if not mask[0, t]:
                outputs.append(torch.zeros(batch, heads, value_dim, dtype=q.dtype))
                states.append(state.clone())
                continue
        decayed = state * torch.exp(g[:, t])[..., None]
        error = v[:, t] - torch.einsum("bhkv,bhk->bhv", decayed, k[:, t])
        state = decayed + torch.einsum(
            "bhk,bhv->bhkv", beta[:, t, :, None] * k[:, t], error
        )
        outputs.append(torch.einsum("bhkv,bhk->bhv", state, q[:, t]))
        states.append(state)
    return torch.stack(outputs, 1), state, states


def test_recurrent_matches_explicit_loop_with_nonzero_state_k_not_v():
    inputs = random_core(
        batch=1, tokens=5, heads=2, key_dim=2, value_dim=3
    )
    q, k, v, g, beta, state = inputs
    expected_o, expected_s, _ = manual_recurrent(*inputs)
    actual = recurrent_kda(*inputs, output_final_state=True)
    torch.testing.assert_close(
        actual.hidden_states, expected_o, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(actual.final_state, expected_s, rtol=1e-12, atol=1e-12)


def test_one_step_matches_matrix_equation_and_error_correction_form():
    q, k, v, g, beta, state = random_core(
        batch=1, tokens=1, heads=1, key_dim=2, value_dim=3
    )
    alpha = torch.exp(g[:, 0])
    decayed = alpha[..., None] * state
    identity = torch.eye(2, dtype=torch.float64).reshape(1, 1, 2, 2)
    transition = identity - beta[:, 0, :, None, None] * torch.einsum(
        "bhk,bhj->bhkj", k[:, 0], k[:, 0]
    )
    matrix_state = torch.matmul(transition, decayed) + torch.einsum(
        "bhk,bhv->bhkv", beta[:, 0, :, None] * k[:, 0], v[:, 0]
    )
    error = v[:, 0] - torch.einsum("bhkv,bhk->bhv", decayed, k[:, 0])
    correction_state = decayed + torch.einsum(
        "bhk,bhv->bhkv", beta[:, 0, :, None] * k[:, 0], error
    )
    actual = recurrent_kda(*((q, k, v, g, beta, state)), output_final_state=True)
    torch.testing.assert_close(actual.final_state, matrix_state, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual.final_state, correction_state, rtol=1e-12, atol=1e-12)


def test_read_after_write_uses_current_state():
    q = k = torch.tensor([[[[1.0]]]], dtype=torch.float64)
    v = torch.tensor([[[[3.0]]]], dtype=torch.float64)
    g = torch.zeros_like(q)
    beta = torch.ones(1, 1, 1, dtype=torch.float64)
    output = recurrent_kda(q, k, v, g, beta).hidden_states
    torch.testing.assert_close(output, v, rtol=0, atol=0)


def test_decay_happens_before_error_and_update():
    q = k = torch.tensor([[[[1.0]]]], dtype=torch.float64)
    v = torch.tensor([[[[0.0]]]], dtype=torch.float64)
    g = torch.tensor([[[[torch.log(torch.tensor(0.5)).item()]]]], dtype=torch.float64)
    beta = torch.tensor([[[0.5]]], dtype=torch.float64)
    state = torch.tensor([[[[2.0]]]], dtype=torch.float64)
    actual = recurrent_kda(q, k, v, g, beta, state, True)
    # decay -> 1; error=-1; update=-0.5 => state/output=0.5
    torch.testing.assert_close(
        actual.final_state, torch.tensor([[[[0.5]]]], dtype=torch.float64)
    )


def test_beta_zero_only_decays_state():
    q, k, v, g, beta, state = random_core(tokens=4)
    beta.zero_()
    actual = recurrent_kda(q, k, v, g, beta, state, True)
    expected = state * torch.exp(g.sum(dim=1))[..., None]
    torch.testing.assert_close(actual.final_state, expected, rtol=1e-12, atol=1e-12)


def test_alpha_one_reduces_to_classic_delta_rule():
    q, k, v, g, beta, state = random_core(tokens=5)
    g.zero_()
    actual = recurrent_kda(q, k, v, g, beta, state, True)
    expected = manual_recurrent(q, k, v, g, beta, state)[1]
    torch.testing.assert_close(actual.final_state, expected)


def test_zero_state_first_update_is_beta_k_outer_v():
    q, k, v, g, beta, _ = random_core(
        batch=1, tokens=1, heads=2, key_dim=3, value_dim=4
    )
    actual = recurrent_kda(q, k, v, g, beta, output_final_state=True)
    expected = torch.einsum("bhk,bhv->bhkv", beta[:, 0, :, None] * k[:, 0], v[:, 0])
    torch.testing.assert_close(actual.final_state, expected, rtol=1e-12, atol=1e-12)


def test_zero_reconstruction_error_means_no_delta_update():
    batch, heads, key_dim, value_dim = 1, 1, 2, 3
    k = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)
    q = k.clone()
    g = torch.log(torch.tensor([[[[0.5, 0.8]]]], dtype=torch.float64))
    state = torch.randn(batch, heads, key_dim, value_dim, dtype=torch.float64)
    decayed = state * torch.exp(g[:, 0])[..., None]
    v = torch.einsum("bhkv,bhk->bhv", decayed, k[:, 0])[:, None]
    beta = torch.ones(batch, 1, heads, dtype=torch.float64)
    actual = recurrent_kda(q, k, v, g, beta, state, True)
    torch.testing.assert_close(actual.final_state, decayed, rtol=1e-12, atol=1e-12)


def test_unit_key_beta_one_writes_target_in_key_direction():
    q, k, v, g, beta, state = random_core(
        batch=1, tokens=1, heads=1, key_dim=3, value_dim=2
    )
    k = torch.nn.functional.normalize(k, dim=-1)
    g.zero_()
    beta.fill_(1)
    actual = recurrent_kda(q, k, v, g, beta, state, True)
    recalled = torch.einsum("bhkv,bhk->bhv", actual.final_state, k[:, 0])
    torch.testing.assert_close(recalled, v[:, 0], rtol=1e-12, atol=1e-12)


def test_batch_and_head_independence():
    q, k, v, g, beta, state = random_core(batch=2, heads=3)
    baseline = recurrent_kda(q, k, v, g, beta, state, True)
    v_changed = v.clone()
    v_changed[1, :, 2] += 100
    changed = recurrent_kda(q, k, v_changed, g, beta, state, True)
    torch.testing.assert_close(baseline.hidden_states[0], changed.hidden_states[0])
    torch.testing.assert_close(
        baseline.hidden_states[1, :, :2], changed.hidden_states[1, :, :2]
    )


@pytest.mark.parametrize("prefix", range(1, 8))
def test_recurrent_core_prefix_causality(prefix):
    q, k, v, g, beta, state = random_core(tokens=8)
    baseline = recurrent_kda(q, k, v, g, beta, state).hidden_states
    changed_inputs = [tensor.clone() for tensor in (q, k, v, g, beta)]
    for tensor in changed_inputs:
        tensor[:, prefix:] = torch.randn_like(tensor[:, prefix:]) * 10
    changed = recurrent_kda(*changed_inputs, state).hidden_states
    torch.testing.assert_close(
        baseline[:, :prefix], changed[:, :prefix], rtol=0, atol=0
    )


def test_padding_token_neither_decays_writes_nor_reads():
    q, k, v, g, beta, state = random_core(
        batch=1, tokens=5, heads=2, key_dim=2, value_dim=3
    )
    mask = torch.tensor([[True, True, True, False, False]])
    actual = recurrent_kda(
        q, k, v, g, beta, state, True, mask, return_states_per_token=True
    )
    valid = recurrent_kda(
        q[:, :3], k[:, :3], v[:, :3], g[:, :3], beta[:, :3], state, True
    )
    torch.testing.assert_close(actual.final_state, valid.final_state, rtol=0, atol=0)
    assert torch.count_nonzero(actual.hidden_states[:, 3:]) == 0
    torch.testing.assert_close(actual.states_per_token[2], actual.states_per_token[3])
    torch.testing.assert_close(actual.states_per_token[3], actual.states_per_token[4])


def test_recurrent_gradcheck_and_gradgradcheck():
    q, k, v, g, beta, state = random_core(
        batch=1, tokens=2, heads=1, key_dim=2, value_dim=2
    )
    inputs = tuple(tensor.requires_grad_() for tensor in (q, k, v, g, beta, state))
    function = lambda *xs: (
        recurrent_kda(*xs, output_final_state=True).hidden_states,
        recurrent_kda(*xs, output_final_state=True).final_state,
    )
    assert torch.autograd.gradcheck(function, inputs, fast_mode=True)
    assert torch.autograd.gradgradcheck(
        lambda *xs: recurrent_kda(*xs).hidden_states,
        inputs,
        fast_mode=True,
    )

