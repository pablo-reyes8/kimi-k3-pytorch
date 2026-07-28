import copy

import pytest
import torch

from src.kda import KimiDeltaAttention
from tests.kda.conftest import tiny_config, tiny_kda


def test_full_module_recurrent_and_chunkwise_outputs_states_diagnostics():
    model = tiny_kda().double().eval()
    x = torch.randn(2, 9, 12, dtype=torch.float64)
    recurrent = model(
        x, mode="recurrent", output_final_state=True, output_diagnostics=True
    )
    chunkwise = model(
        x, mode="chunkwise", output_final_state=True, output_diagnostics=True
    )
    torch.testing.assert_close(
        recurrent.hidden_states, chunkwise.hidden_states, rtol=2e-11, atol=2e-12
    )
    torch.testing.assert_close(
        recurrent.state.recurrent_state,
        chunkwise.state.recurrent_state,
        rtol=2e-11,
        atol=2e-12,
    )
    for left, right in (
        (recurrent.state.q_conv_state, chunkwise.state.q_conv_state),
        (recurrent.state.k_conv_state, chunkwise.state.k_conv_state),
        (recurrent.state.v_conv_state, chunkwise.state.v_conv_state),
    ):
        torch.testing.assert_close(left.buffer, right.buffer, rtol=0, atol=0)
    assert recurrent.diagnostics.keys() == chunkwise.diagnostics.keys()


def test_full_output_matches_manual_postprocessing_from_same_projections():
    model = tiny_kda().double().eval()
    x = torch.randn(1, 5, 12, dtype=torch.float64)
    projected = model.projections(x)
    g, _ = model.decay(projected.decay_logits)
    from src.kda import recurrent_kda
    from src.kimi_primitives import combine_heads

    core = recurrent_kda(
        projected.q, projected.k, projected.v, g, projected.beta
    ).hidden_states
    expected = model.output_gate(
        combine_heads(model.output_norm(core)), x
    )
    torch.testing.assert_close(
        model(x, mode="recurrent").hidden_states, expected, rtol=0, atol=0
    )


def test_output_gate_uses_residual_hidden_states():
    model = tiny_kda().eval()
    x = torch.randn(1, 4, 12)
    residual_gate = model.output_gate.gate_values(x)
    changed = x.clone()
    changed[:, 2] += 100
    changed_gate = model.output_gate.gate_values(changed)
    torch.testing.assert_close(residual_gate[:, :2], changed_gate[:, :2])
    assert not torch.equal(residual_gate[:, 2], changed_gate[:, 2])


def test_no_residual_connection_is_hidden_inside_kda():
    model = tiny_kda(decay_initializer="zeros")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    x = torch.randn(2, 3, 12)
    assert torch.count_nonzero(model(x, mode="recurrent").hidden_states) == 0


def test_diagnostics_are_scalar_finite_and_complete():
    model = tiny_kda().eval()
    output = model(
        torch.randn(2, 7, 12),
        mode="chunkwise",
        output_final_state=True,
        output_diagnostics=True,
    )
    expected = {
        "alpha_min", "alpha_max", "alpha_mean", "log_decay_min",
        "log_decay_max", "beta_mean", "beta_saturation_low",
        "beta_saturation_high", "state_norm_mean", "state_norm_max",
        "q_norm_error", "k_norm_error", "output_gate_mean",
        "output_gate_saturation", "chunk_count",
    }
    assert output.diagnostics.keys() == expected
    assert all(value.ndim == 0 for value in output.diagnostics.values())
    assert all(torch.isfinite(value.float()) for value in output.diagnostics.values())
    assert output.diagnostics["chunk_count"] == 2
    assert output.diagnostics["alpha_min"] > torch.exp(torch.tensor(-5.0))
    assert output.diagnostics["alpha_max"] < 1


@pytest.mark.parametrize("mode", ["recurrent", "chunkwise"])
@pytest.mark.parametrize("prefix", range(1, 8))
def test_full_module_prefix_causality_including_shortconv(mode, prefix):
    model = tiny_kda().double().eval()
    x = torch.randn(2, 8, 12, dtype=torch.float64)
    changed = x.clone()
    changed[:, prefix:] = torch.randn_like(changed[:, prefix:]) * 100
    baseline = model(x, mode=mode).hidden_states
    actual = model(changed, mode=mode).hidden_states
    torch.testing.assert_close(
        baseline[:, :prefix], actual[:, :prefix], rtol=0, atol=0
    )


def test_k_not_equal_v_full_module():
    model = KimiDeltaAttention(
        tiny_config(
            d_model=15,
            num_heads=3,
            key_head_dim=2,
            value_head_dim=5,
        )
    )
    output = model(torch.randn(2, 7, 15), output_final_state=True)
    assert output.hidden_states.shape == (2, 7, 15)
    assert output.state.recurrent_state.shape == (2, 3, 2, 5)


def test_full_module_parameter_gradients_recurrent_vs_chunkwise():
    recurrent_model = tiny_kda().double()
    chunk_model = copy.deepcopy(recurrent_model)
    x_recurrent = torch.randn(
        1, 5, 12, dtype=torch.float64, requires_grad=True
    )
    x_chunk = x_recurrent.detach().clone().requires_grad_()
    recurrent_state = recurrent_model(
        x_recurrent, mode="recurrent", output_final_state=True
    )
    chunk_state = chunk_model(
        x_chunk, mode="chunkwise", output_final_state=True
    )
    recurrent_loss = (
        recurrent_state.hidden_states.square().sum()
        + recurrent_state.state.recurrent_state.square().sum()
    )
    chunk_loss = (
        chunk_state.hidden_states.square().sum()
        + chunk_state.state.recurrent_state.square().sum()
    )
    recurrent_loss.backward()
    chunk_loss.backward()
    torch.testing.assert_close(
        x_recurrent.grad, x_chunk.grad, rtol=2e-9, atol=2e-11
    )
    recurrent_parameters = dict(recurrent_model.named_parameters())
    chunk_parameters = dict(chunk_model.named_parameters())
    assert recurrent_parameters.keys() == chunk_parameters.keys()
    for name in recurrent_parameters:
        left, right = recurrent_parameters[name].grad, chunk_parameters[name].grad
        assert left is not None and right is not None, name
        torch.testing.assert_close(left, right, rtol=2e-8, atol=2e-10)


def test_full_module_gradcheck_small_dimensions():
    model = KimiDeltaAttention(
        tiny_config(
            d_model=2,
            num_heads=1,
            key_head_dim=1,
            value_head_dim=2,
            short_conv_kernel_size=1,
            decay_rank=1,
            chunk_size=2,
            secondary_tile_size=1,
            decay_initializer="zeros",
        )
    ).double()
    x = torch.randn(1, 2, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value: model(value, mode="chunkwise").hidden_states,
        (x,),
        fast_mode=True,
    )


@pytest.mark.parametrize(
    "mode,shape",
    [
        ("bad", (1, 1, 12)),
        ("decode", (1, 2, 12)),
        ("recurrent", (1, 0, 12)),
        ("chunkwise", (1, 2, 11)),
    ],
)
def test_full_module_rejects_invalid_modes_and_shapes(mode, shape):
    with pytest.raises(ValueError):
        tiny_kda()(torch.randn(shape), mode=mode)


def test_noncontiguous_hidden_states_supported():
    model = tiny_kda().eval()
    source = torch.randn(2, 7, 24)
    x = source[..., ::2]
    assert not x.is_contiguous()
    output = model(x, mode="chunkwise")
    assert output.hidden_states.shape == x.shape
    assert torch.isfinite(output.hidden_states).all()

