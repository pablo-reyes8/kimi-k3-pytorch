import copy
import math

import pytest
import torch
import torch.nn.functional as F

from src.kda import KDAProjections
from tests.kda.conftest import tiny_config


def test_projection_shapes_include_channelwise_decay_and_scalar_beta():
    module = KDAProjections(tiny_config())
    output = module(torch.randn(2, 7, 12))
    assert output.q.shape == output.k.shape == (2, 7, 3, 2)
    assert output.v.shape == (2, 7, 3, 4)
    assert output.beta.shape == (2, 7, 3)
    assert output.decay_logits.shape == (2, 7, 3, 2)


def test_q_k_v_manual_projection_conv_activation_pipeline():
    module = KDAProjections(tiny_config()).double()
    x = torch.randn(2, 5, 12, dtype=torch.float64)
    output = module(x)
    q_linear = module.q_proj(x)
    k_linear = module.k_proj(x)
    v_linear = module.v_proj(x)
    q_expected = F.normalize(
        F.silu(module.q_conv(q_linear)).reshape(2, 5, 3, 2),
        dim=-1,
        eps=module.config.eps,
    )
    k_expected = F.normalize(
        F.silu(module.k_conv(k_linear)).reshape(2, 5, 3, 2),
        dim=-1,
        eps=module.config.eps,
    )
    v_expected = F.silu(module.v_conv(v_linear)).reshape(2, 5, 3, 4)
    torch.testing.assert_close(output.q, q_expected, rtol=0, atol=0)
    torch.testing.assert_close(output.k, k_expected, rtol=0, atol=0)
    torch.testing.assert_close(output.v, v_expected, rtol=0, atol=0)


def test_beta_and_decay_logits_match_manual_projections():
    module = KDAProjections(tiny_config()).double()
    x = torch.randn(2, 5, 12, dtype=torch.float64)
    output = module(x)
    expected_beta = torch.sigmoid(module.beta_proj(x))
    expected_logits = module.alpha_up(module.alpha_down(x)).reshape(2, 5, 3, 2)
    expected_logits = expected_logits + module.b_alpha[None, None]
    torch.testing.assert_close(output.beta, expected_beta, rtol=0, atol=0)
    torch.testing.assert_close(output.decay_logits, expected_logits, rtol=0, atol=0)


def test_q_and_k_are_unit_norm_while_v_is_not_forced_to_unit_norm():
    module = KDAProjections(tiny_config())
    output = module(torch.randn(2, 7, 12) * 100)
    torch.testing.assert_close(
        output.q.norm(dim=-1), torch.ones(2, 7, 3), rtol=2e-5, atol=2e-5
    )
    torch.testing.assert_close(
        output.k.norm(dim=-1), torch.ones(2, 7, 3), rtol=2e-5, atol=2e-5
    )
    assert not torch.allclose(output.v.norm(dim=-1), torch.ones(2, 7, 3))


def test_zero_preactivation_qk_is_finite_not_nan():
    module = KDAProjections(tiny_config(decay_initializer="zeros"))
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
    output = module(torch.zeros(2, 3, 12))
    assert torch.count_nonzero(output.q) == 0
    assert torch.count_nonzero(output.k) == 0
    assert torch.isfinite(output.q).all() and torch.isfinite(output.k).all()


def test_projection_and_conv_parameters_are_independent():
    module = KDAProjections(tiny_config())
    assert len(
        {
            module.q_proj.weight.data_ptr(),
            module.k_proj.weight.data_ptr(),
            module.v_proj.weight.data_ptr(),
        }
    ) == 3
    assert len(
        {
            module.q_conv.weight.data_ptr(),
            module.k_conv.weight.data_ptr(),
            module.v_conv.weight.data_ptr(),
        }
    ) == 3


def test_official_fla_bias_initializer_range_and_inverse_softplus_relation():
    torch.manual_seed(3)
    module = KDAProjections(tiny_config(decay_initializer="official_fla"))
    recovered_dt = F.softplus(module.b_alpha)
    assert recovered_dt.min() >= 0.001 - 1e-6
    assert recovered_dt.max() <= 0.1 + 1e-6
    assert module.b_alpha.shape == (3, 2)


def test_zero_initializer_is_explicit_experimental_fallback():
    module = KDAProjections(tiny_config(decay_initializer="zeros"))
    torch.testing.assert_close(module.b_alpha, torch.zeros_like(module.b_alpha))


def test_masked_projection_cache_matches_individual_unpadded_prefixes():
    module = KDAProjections(tiny_config()).eval()
    x = torch.randn(2, 6, 12)
    mask = torch.tensor(
        [[True] * 6, [True, True, True, True, False, False]]
    )
    batched = module(x, attention_mask=mask, output_final_state=True)
    first = module(x[:1], output_final_state=True)
    second = module(x[1:2, :4], output_final_state=True)
    torch.testing.assert_close(batched.q[0], first.q[0])
    torch.testing.assert_close(batched.q[1, :4], second.q[0])
    torch.testing.assert_close(
        batched.q_conv_state.buffer[1], second.q_conv_state.buffer[0]
    )
    torch.testing.assert_close(
        batched.k_conv_state.buffer[1], second.k_conv_state.buffer[0]
    )
    torch.testing.assert_close(
        batched.v_conv_state.buffer[1], second.v_conv_state.buffer[0]
    )


def test_projection_cached_chunks_match_full_sequence():
    module = KDAProjections(tiny_config()).eval()
    x = torch.randn(2, 9, 12)
    full = module(x, output_final_state=True)
    first = module(x[:, :4], output_final_state=True)
    second = module(
        x[:, 4:],
        q_conv_state=first.q_conv_state,
        k_conv_state=first.k_conv_state,
        v_conv_state=first.v_conv_state,
        output_final_state=True,
    )
    for name in ("q", "k", "v"):
        combined = torch.cat((getattr(first, name), getattr(second, name)), dim=1)
        torch.testing.assert_close(combined, getattr(full, name), rtol=1e-5, atol=1e-6)


def test_projection_complete_gradients_and_roundtrip():
    module = KDAProjections(tiny_config())
    x = torch.randn(2, 5, 12, requires_grad=True)
    output = module(x)
    loss = sum(
        tensor.square().mean()
        for tensor in (
            output.q, output.k, output.v, output.beta, output.decay_logits
        )
    )
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in module.parameters())
    clone = copy.deepcopy(module).eval()
    module.eval()
    reference = torch.randn(1, 4, 12)
    for actual, expected in zip(
        module(reference).__dict__.values(), clone(reference).__dict__.values()
    ):
        if isinstance(actual, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(2, 12), (2, 3, 11), (2, 0, 12)])
def test_projection_invalid_hidden_contract(shape):
    with pytest.raises(ValueError):
        KDAProjections(tiny_config())(torch.randn(shape))

