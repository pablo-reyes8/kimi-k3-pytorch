import copy

import pytest
import torch

from src.hybrid_backbone import HybridAttentionBackbone
from tests.hybrid_backbone.conftest import tiny_backbone, tiny_config


def test_diagnostics_are_complete_per_layer_and_globally():
    model = tiny_backbone().eval()
    output = model(
        torch.randn(2, 6, 8),
        mode="prefill",
        use_cache=True,
        output_diagnostics=True,
    )
    diagnostics = output.diagnostics
    assert len(diagnostics["layers"]) == len(model.layers)
    assert diagnostics["num_kda_layers"] == 3
    assert diagnostics["num_mla_layers"] == 2
    for index, layer in enumerate(diagnostics["layers"]):
        assert layer["layer_index"] == index
        for name in (
            "input_norm",
            "attention_output_norm",
            "post_attention_residual_norm",
            "ffn_output_norm",
            "post_ffn_residual_norm",
        ):
            assert layer[name].ndim == 0
            assert torch.isfinite(layer[name])
        assert isinstance(layer["mechanism"], dict)
    counts = diagnostics["num_parameters_by_component"]
    assert counts.keys() == {"kda", "gated_mla", "ffn", "norms", "total"}
    assert counts["total"] == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert diagnostics["kda_cache_elements"] == output.cache.kda_elements
    assert diagnostics["mla_cache_elements"] == output.cache.mla_elements
    assert diagnostics["total_cache_elements"] == output.cache.total_elements


def test_state_dict_roundtrip_is_exact_and_eval_is_deterministic():
    source = tiny_backbone().eval()
    target = HybridAttentionBackbone(tiny_config()).eval()
    target.load_state_dict(source.state_dict())
    x = torch.randn(2, 6, 8)
    expected = source(x).last_hidden_state
    torch.testing.assert_close(
        target(x).last_hidden_state, expected, rtol=0, atol=0
    )
    torch.testing.assert_close(
        source(x).last_hidden_state, expected, rtol=0, atol=0
    )


def test_dropout_is_disabled_in_eval_and_active_in_training():
    model = tiny_backbone(residual_dropout=0.5, ffn_dropout=0.5)
    x = torch.randn(2, 8, 8)
    model.eval()
    torch.testing.assert_close(
        model(x).last_hidden_state,
        model(x).last_hidden_state,
        rtol=0,
        atol=0,
    )
    model.train()
    assert not torch.equal(
        model(x).last_hidden_state, model(x).last_hidden_state
    )


def test_to_float64_moves_every_floating_parameter_and_buffer():
    model = tiny_backbone().to(torch.float64)
    assert all(parameter.dtype == torch.float64 for parameter in model.parameters())
    assert all(
        not buffer.dtype.is_floating_point or buffer.dtype == torch.float64
        for buffer in model.buffers()
    )


@pytest.mark.parametrize("groups", [1, 3])
def test_fp32_stress_outputs_caches_and_gradients_are_finite(groups):
    model = tiny_backbone(num_hybrid_groups=groups)
    x = (torch.randn(2, 9, 8) * 1e3).requires_grad_()
    output = model(x, mode="prefill", use_cache=True)
    output.last_hidden_state.square().mean().backward()
    assert torch.isfinite(output.last_hidden_state).all()
    assert torch.isfinite(x.grad).all()
    for layer_cache in output.cache.layer_caches:
        state = layer_cache.state
        tensor = (
            state.recurrent_state
            if layer_cache.attention_type == "kda"
            else state.latent_kv
        )
        assert torch.isfinite(tensor).all()


def test_cpu_bfloat16_full_prefill_and_decode_are_finite():
    model = tiny_backbone().bfloat16().eval()
    x = torch.randn(2, 5, 8, dtype=torch.bfloat16)
    prefill = model(x[:, :4], mode="prefill", use_cache=True)
    decode = model(
        x[:, 4:],
        cache=prefill.cache,
        mode="decode",
        use_cache=True,
    )
    assert prefill.last_hidden_state.dtype == torch.bfloat16
    assert decode.last_hidden_state.dtype == torch.bfloat16
    assert torch.isfinite(prefill.last_hidden_state).all()
    assert torch.isfinite(decode.last_hidden_state).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bfloat16_full_prefill_decode():
    fp32 = tiny_backbone().cuda().eval()
    bf16 = copy.deepcopy(fp32).bfloat16()
    x = torch.randn(2, 6, 8, device="cuda")
    reference = fp32(x).last_hidden_state
    prefill = bf16(
        x[:, :5].bfloat16(), mode="prefill", use_cache=True
    )
    decode = bf16(
        x[:, 5:].bfloat16(),
        cache=prefill.cache,
        mode="decode",
        use_cache=True,
    )
    torch.testing.assert_close(
        decode.last_hidden_state.float(),
        reference[:, 5:],
        rtol=0.08,
        atol=0.02,
    )
