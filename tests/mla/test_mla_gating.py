import torch
import torch.nn.functional as F

from src.kimi_primitives import combine_heads
from src.mla import mla_attention
from tests.mla.conftest import tiny_mla


def test_full_module_matches_gate_equation_exactly():
    model = tiny_mla(attention_backend="manual").double().eval()
    x = torch.randn(2, 5, 12, dtype=torch.float64)
    projected = model.projections(x)
    raw = mla_attention(
        projected.query, projected.key, projected.value, backend="manual"
    )
    gate = torch.sigmoid(model.output_gate.gate_proj(x))
    expected = F.linear(
        gate * combine_heads(raw),
        model.output_gate.output_proj.weight,
        model.output_gate.output_proj.bias,
    )
    torch.testing.assert_close(model(x).hidden_states, expected, rtol=0, atol=0)


def test_gate_is_full_rank_d_to_d_and_uses_residual_input():
    model = tiny_mla().eval()
    assert model.output_gate.gate_proj.weight.shape == (12, 12)
    x = torch.randn(1, 4, 12)
    baseline = model.output_gate.gate_values(x)
    changed = x.clone()
    changed[:, 2] += 100
    actual = model.output_gate.gate_values(changed)
    torch.testing.assert_close(baseline[:, :2], actual[:, :2])
    assert not torch.equal(baseline[:, 2], actual[:, 2])


def test_neutral_gate_is_one_half():
    model = tiny_mla()
    with torch.no_grad():
        model.output_gate.gate_proj.weight.zero_()
        if model.output_gate.gate_proj.bias is not None:
            model.output_gate.gate_proj.bias.zero_()
    gate = model.output_gate.gate_values(torch.randn(2, 3, 12))
    torch.testing.assert_close(gate, torch.full_like(gate, 0.5))


def test_closed_gate_suppresses_output():
    model = tiny_mla(output_gate_bias=True).eval()
    with torch.no_grad():
        model.output_gate.gate_proj.weight.zero_()
        model.output_gate.gate_proj.bias.fill_(-100)
    output = model(torch.randn(2, 4, 12)).hidden_states
    assert output.abs().max() < 1e-30


def test_diagnostics_are_scalar_finite_and_complete():
    model = tiny_mla().eval()
    output = model(torch.randn(2, 6, 12), output_diagnostics=True)
    expected = {
        "attention_entropy", "attention_max_probability", "gate_mean",
        "gate_min", "gate_max", "gate_saturation_low",
        "gate_saturation_high", "latent_norm_mean", "latent_norm_max",
        "query_norm_mean", "key_norm_mean", "cache_length",
        "cache_elements", "compression_ratio",
    }
    assert output.diagnostics.keys() == expected
    assert all(value.ndim == 0 for value in output.diagnostics.values())
    assert all(torch.isfinite(value.float()) for value in output.diagnostics.values())
    assert output.diagnostics["cache_length"] == 6
    assert output.diagnostics["cache_elements"] == 60
    torch.testing.assert_close(
        output.diagnostics["compression_ratio"], torch.tensor(3.6)
    )


def test_diagnostics_support_padded_prefill_and_cached_decode():
    model = tiny_mla().eval()
    mask = torch.tensor(
        [[True, True, True, True], [True, True, False, False]]
    )
    prefill = model(
        torch.randn(2, 4, 12),
        attention_mask=mask,
        use_cache=True,
        output_diagnostics=True,
    )
    decoded = model(
        torch.randn(2, 1, 12),
        cache=prefill.cache,
        use_cache=True,
        output_diagnostics=True,
    )
    assert all(
        torch.isfinite(value.float())
        for value in decoded.diagnostics.values()
    )
    assert decoded.diagnostics["cache_length"] == 5
