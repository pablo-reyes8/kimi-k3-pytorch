import copy

import pytest
import torch

from src.attention_residuals import AttentionResidualConfig
from src.kimi_block import KimiBlock, KimiBlockConfig
from src.stable_latent_moe import StableLatentMoE
from tests.stable_latent_moe.conftest import (
    tiny_kimi,
    tiny_kimi_config,
)


def test_canonical_kimi_block_is_3kda_1mla_final_mla_and_five_moe():
    model = tiny_kimi()
    assert model.attention_types == (
        "kda",
        "kda",
        "kda",
        "gated_mla",
        "gated_mla_final",
    )
    assert len(model.moe_layers) == 5
    assert all(isinstance(moe, StableLatentMoE) for moe in model.moe_layers)
    assert all(layer.ffn_norm.dim == 8 for layer in model.layers)
    assert all(
        layer.ffn.routed_aggregate_norm.dim == 4 for layer in model.layers
    )


def test_attention_pattern_is_explicitly_parametrizable_in_kimi_block():
    config = tiny_kimi_config(
        num_pattern_repeats=2,
        attention_pattern=("kda", "gated_mla"),
        attention_residual_config=AttentionResidualConfig(
            8,
            mode="block",
            sublayers_per_depth_block=3,
        ),
    )
    model = KimiBlock(config)
    assert model.attention_types == (
        "kda",
        "gated_mla",
        "kda",
        "gated_mla",
        "gated_mla_final",
    )
    assert config.num_moe_layers == 5


def test_every_layer_owns_independent_moe_router_experts_bias_and_histogram():
    model = tiny_kimi(
        stable_latent_moe_config=tiny_kimi_config()
        .stable_latent_moe_config
    )
    router_ptrs = [moe.router.projection.weight.data_ptr() for moe in model.moe_layers]
    expert_ptrs = [
        moe.routed_experts[0].transform.gate_proj.weight.data_ptr()
        for moe in model.moe_layers
    ]
    bias_ptrs = [moe.routing_bias.data_ptr() for moe in model.moe_layers]
    assert len(set(router_ptrs)) == len(router_ptrs)
    assert len(set(expert_ptrs)) == len(expert_ptrs)
    assert len(set(bias_ptrs)) == len(bias_ptrs)


@pytest.mark.parametrize("depth_mode", ["full", "block"])
def test_moe_is_one_attnres_site_and_history_records_only_total_output(depth_mode):
    config = tiny_kimi_config(
        attention_residual_config=AttentionResidualConfig(
            8,
            mode=depth_mode,
            sublayers_per_depth_block=4 if depth_mode == "block" else None,
        )
    )
    model = KimiBlock(config).double().eval()
    x = torch.randn(1, 4, 8, dtype=torch.float64)
    output = model(x, output_hidden_states=True, output_depth_weights=True)
    trace = output.hidden_state_trace
    assert len(trace.ffn_outputs) == len(model.layers)
    assert len(output.depth_outputs.site_stats) == 2 * len(model.layers)
    first = model.layers[0]
    expected = first.ffn(first.ffn_norm(trace.pre_ffn[0]))
    torch.testing.assert_close(trace.ffn_outputs[0], expected, rtol=0, atol=0)
    assert output.depth_outputs.site_stats[1].metadata.site_kind == "pre_ffn"


@pytest.mark.parametrize("backend", ["eager", "two_phase"])
def test_kimi_block_full_prefill_decode_and_cache_equivalence(backend):
    config = tiny_kimi_config(
        attention_residual_config=AttentionResidualConfig(
            8,
            mode="block",
            sublayers_per_depth_block=4,
            backend=backend,
        )
    )
    model = KimiBlock(config).double().eval()
    x = torch.randn(2, 7, 8, dtype=torch.float64)
    full = model(x)
    prefill = model(x, mode="prefill", use_cache=True)
    torch.testing.assert_close(
        full.last_hidden_state,
        prefill.last_hidden_state,
        rtol=4e-10,
        atol=4e-11,
    )
    cache = None
    pieces = []
    for token in range(x.shape[1]):
        output = model(
            x[:, token : token + 1],
            cache=cache,
            use_cache=True,
            mode="prefill" if cache is None else "decode",
        )
        pieces.append(output.last_hidden_state)
        cache = output.cache
    torch.testing.assert_close(
        torch.cat(pieces, dim=1),
        full.last_hidden_state,
        rtol=5e-9,
        atol=5e-10,
    )
    assert len(cache.layer_caches) == len(model.layers)
    assert not hasattr(cache, "moe")
    assert not hasattr(cache, "routing_bias")


def test_irregular_prefill_and_frozen_routing_bias_are_invariant():
    model = tiny_kimi().double().eval()
    x = torch.randn(1, 9, 8, dtype=torch.float64)
    expected = model(x, mode="prefill", use_cache=True)
    cache = None
    pieces = []
    boundaries = (0, 2, 5, 6, 9)
    biases = [moe.routing_bias.clone() for moe in model.moe_layers]
    for start, end in zip(boundaries, boundaries[1:]):
        output = model(
            x[:, start:end],
            cache=cache,
            mode="prefill",
            use_cache=True,
        )
        pieces.append(output.last_hidden_state)
        cache = output.cache
    torch.testing.assert_close(
        torch.cat(pieces, 1),
        expected.last_hidden_state,
        rtol=5e-9,
        atol=5e-10,
    )
    for before, moe in zip(biases, model.moe_layers):
        torch.testing.assert_close(before, moe.routing_bias)


def test_routing_diagnostics_exist_for_every_kda_mla_and_final_mla():
    model = tiny_kimi().eval()
    output = model(torch.randn(2, 5, 8), output_diagnostics=True)
    assert len(output.diagnostics["layers"]) == 5
    for layer in output.diagnostics["layers"]:
        diagnostics = layer["channel_mixer"]
        assert diagnostics.num_tokens == 10
        assert diagnostics.num_assignments == 20
        assert diagnostics.expert_load.sum() == 20


def test_backbone_training_can_causally_update_every_independent_router():
    model = tiny_kimi().train()
    before = [moe.routing_bias.clone() for moe in model.moe_layers]
    model(torch.randn(2, 5, 8), update_routing_bias=True)
    assert all(
        not torch.equal(old, moe.routing_bias)
        for old, moe in zip(before, model.moe_layers)
    )
    model.eval()
    with pytest.raises(RuntimeError):
        model(torch.randn(2, 2, 8), update_routing_bias=True)


def test_kimi_block_state_dict_and_config_roundtrip_are_exact():
    source = tiny_kimi().double().eval()
    target = KimiBlock(
        KimiBlockConfig.from_dict(source.config.to_dict())
    ).double().eval()
    target.load_state_dict(source.state_dict())
    x = torch.randn(2, 4, 8, dtype=torch.float64)
    torch.testing.assert_close(
        source(x).last_hidden_state,
        target(x).last_hidden_state,
        rtol=0,
        atol=0,
    )


def test_no_standard_residual_surrounds_moe_in_attnres_path():
    model = tiny_kimi().eval()
    with torch.no_grad():
        for layer in model.layers:
            for parameter in layer.attention.parameters():
                parameter.zero_()
            for parameter in layer.ffn.parameters():
                parameter.zero_()
    x = torch.randn(1, 3, 8)
    output = model(x, output_hidden_states=True)
    assert all(
        torch.count_nonzero(value) == 0
        for value in output.hidden_state_trace.ffn_outputs
    )
    assert not torch.equal(output.last_hidden_state, x)
