import inspect

import pytest
import torch

from src.kda import ut_transform
from src.kda.chunkwise import chunkwise_kda
from tests.kda.conftest import random_core


def explicit_ut(k, v, g, beta):
    # Canonical B,C,H -> internal B,H,C.
    ki = k.permute(0, 2, 1, 3)
    vi = v.permute(0, 2, 1, 3)
    gi = g.permute(0, 2, 1, 3)
    bi = beta.permute(0, 2, 1)
    log_gamma = gi.cumsum(2)
    count = k.shape[1]
    lower = torch.eye(count, dtype=k.dtype).expand(
        k.shape[0], k.shape[2], count, count
    ).clone()
    for i in range(count):
        for j in range(i):
            relative = torch.exp(log_gamma[:, :, i] - log_gamma[:, :, j])
            lower[:, :, i, j] = bi[:, :, i] * (
                ki[:, :, i] * relative * ki[:, :, j]
            ).sum(-1)
    diagonal_beta = torch.diag_embed(bi)
    m_inverse = torch.linalg.inv(lower) @ diagonal_beta
    kgamma = torch.exp(log_gamma) * ki
    return lower, m_inverse, m_inverse @ kgamma, m_inverse @ vi, log_gamma


def test_ut_lower_triangular_solve_matches_explicit_inverse_reference():
    _, k, v, g, beta, _ = random_core(
        batch=2, tokens=5, heads=2, key_dim=3, value_dim=4
    )
    output = ut_transform(k, v, g, beta, secondary_tile_size=2)
    lower, expected_m, expected_w, expected_u, log_gamma = explicit_ut(
        k, v, g, beta
    )
    assert torch.count_nonzero(torch.triu(lower, diagonal=1)) == 0
    diagonal = lower.diagonal(dim1=-2, dim2=-1)
    torch.testing.assert_close(diagonal, torch.ones_like(diagonal))
    torch.testing.assert_close(output.M, expected_m, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(
        output.W.permute(0, 2, 1, 3), expected_w, rtol=1e-11, atol=1e-12
    )
    torch.testing.assert_close(
        output.U.permute(0, 2, 1, 3), expected_u, rtol=1e-11, atol=1e-12
    )


def test_ut_W_U_match_forward_substitution_auxiliary_recurrence():
    _, k, v, g, beta, _ = random_core(
        batch=1, tokens=6, heads=2, key_dim=3, value_dim=4
    )
    output = ut_transform(k, v, g, beta, secondary_tile_size=3)
    lower, _, _, _, log_gamma = explicit_ut(k, v, g, beta)
    kgamma = torch.exp(log_gamma) * k.permute(0, 2, 1, 3)
    bi = beta.permute(0, 2, 1)
    w_rows, u_rows = [], []
    for i in range(k.shape[1]):
        w_i = bi[:, :, i, None] * kgamma[:, :, i]
        u_i = bi[:, :, i, None] * v.permute(0, 2, 1, 3)[:, :, i]
        for j in range(i):
            w_i = w_i - lower[:, :, i, j, None] * w_rows[j]
            u_i = u_i - lower[:, :, i, j, None] * u_rows[j]
        w_rows.append(w_i)
        u_rows.append(u_i)
    expected_w = torch.stack(w_rows, dim=2).permute(0, 2, 1, 3)
    expected_u = torch.stack(u_rows, dim=2).permute(0, 2, 1, 3)
    torch.testing.assert_close(output.W, expected_w, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(output.U, expected_u, rtol=1e-11, atol=1e-12)


def test_pseudo_values_match_manual_matrix_expression():
    _, k, v, g, beta, state = random_core(
        batch=1, tokens=4, heads=2, key_dim=3, value_dim=5
    )
    output = ut_transform(k, v, g, beta, secondary_tile_size=2)
    pseudo = output.U.permute(0, 2, 1, 3) - torch.matmul(
        output.W.permute(0, 2, 1, 3), state
    )
    manual = torch.empty_like(pseudo)
    for b in range(1):
        for h in range(2):
            manual[b, h] = (
                output.U[b, :, h] - output.W[b, :, h] @ state[b, h]
            )
    torch.testing.assert_close(pseudo, manual, rtol=0, atol=0)


@pytest.mark.parametrize("tile", [1, 2, 3, 8])
def test_secondary_tile_size_does_not_change_ut(tile):
    _, k, v, g, beta, _ = random_core(
        batch=1, tokens=7, heads=2, key_dim=3, value_dim=4
    )
    baseline = ut_transform(k, v, g, beta, secondary_tile_size=1)
    actual = ut_transform(k, v, g, beta, secondary_tile_size=tile)
    torch.testing.assert_close(actual.M, baseline.M, rtol=0, atol=0)
    torch.testing.assert_close(actual.W, baseline.W, rtol=0, atol=0)
    torch.testing.assert_close(actual.U, baseline.U, rtol=0, atol=0)


def test_ut_is_finite_for_extreme_lower_bounded_decays():
    _, k, v, _, beta, _ = random_core(
        batch=1, tokens=64, heads=1, key_dim=3, value_dim=4,
        dtype=torch.float32
    )
    g = torch.full_like(k, -5 + 1e-5)
    output = ut_transform(k, v, g, beta, secondary_tile_size=16)
    for tensor in (output.M, output.U, output.W, output.log_gamma):
        assert torch.isfinite(tensor).all()


def test_ut_gradcheck_and_gradgradcheck():
    _, k, v, g, beta, _ = random_core(
        batch=1, tokens=3, heads=1, key_dim=2, value_dim=2
    )
    inputs = tuple(tensor.requires_grad_() for tensor in (k, v, g, beta))
    fn = lambda *xs: (
        ut_transform(*xs, secondary_tile_size=2).W,
        ut_transform(*xs, secondary_tile_size=2).U,
    )
    assert torch.autograd.gradcheck(fn, inputs, fast_mode=True)
    assert torch.autograd.gradgradcheck(fn, inputs, fast_mode=True)


def test_productive_chunkwise_source_has_no_token_recurrent_loop_or_inverse():
    source = inspect.getsource(chunkwise_kda)
    assert "linalg.inv" not in source
    assert "for index in range(tokens)" not in source
    assert "for token" not in source
    assert "solve_triangular" not in source  # delegated to the UT transform
