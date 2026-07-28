import copy

import pytest
import torch

from src.hybrid_backbone import HybridAttentionBackbone
from tests.attention_residuals.conftest import (
    activate_depth_queries,
    attnres_backbone,
    backbone_config,
)


@pytest.mark.parametrize("depth_mode", ["full", "block"])
def test_pattern_final_mla_and_two_sites_per_layer_are_preserved(depth_mode):
    model = attnres_backbone(depth_mode=depth_mode)
    assert model.attention_types == (
        "kda", "kda", "kda", "gated_mla", "gated_mla_final"
    )
    assert model.final_global_layer.ffn is not None
    for layer in model.layers:
        assert layer.pre_attention_attnres is not None
        assert layer.pre_ffn_attnres is not None
    assert model.final_output_attnres is not None


def test_final_output_mixer_occurs_after_final_mla_ffn_and_is_not_last_output():
    model = attnres_backbone(depth_mode="full").double().eval()
    x = torch.randn(1, 4, 8, dtype=torch.float64)
    output = model(x, output_hidden_states=True, output_depth_weights=True)
    trace = output.hidden_state_trace
    assert len(trace.ffn_outputs) == len(model.layers)
    sources = [trace.embedding]
    for attention, ffn in zip(trace.attention_outputs, trace.ffn_outputs):
        sources.extend((attention, ffn))
    expected = model.final_output_attnres(
        torch.stack(sources, dim=2)
    ).mixed_state
    torch.testing.assert_close(trace.final_mixed, expected, rtol=0, atol=0)
    assert not torch.equal(trace.final_mixed, trace.ffn_outputs[-1])
    torch.testing.assert_close(
        output.last_hidden_state,
        model.final_norm(trace.final_mixed),
        rtol=0,
        atol=0,
    )


def test_attnres_path_has_no_standard_residual_leakage():
    model = attnres_backbone(depth_mode="full").eval()
    with torch.no_grad():
        for layer in model.layers:
            for parameter in layer.attention.parameters():
                parameter.zero_()
            for parameter in layer.ffn.parameters():
                parameter.zero_()
    x = torch.randn(2, 4, 8)
    output = model(x, output_hidden_states=True)
    assert all(
        torch.count_nonzero(value) == 0
        for value in output.hidden_state_trace.attention_outputs
    )
    assert all(
        torch.count_nonzero(value) == 0
        for value in output.hidden_state_trace.ffn_outputs
    )
    expected_mixed = x / (1 + 2 * len(model.layers))
    torch.testing.assert_close(
        output.hidden_state_trace.final_mixed, expected_mixed
    )
    assert not torch.equal(output.hidden_state_trace.final_mixed, x)


def test_full_and_block_size_one_are_exactly_equivalent_at_zero_queries():
    full = attnres_backbone(depth_mode="full").double().eval()
    block = attnres_backbone(depth_mode="block", block_size=1).double().eval()
    block.load_state_dict(full.state_dict())
    x = torch.randn(2, 6, 8, dtype=torch.float64)
    full_output = full(
        x, output_hidden_states=True, output_depth_weights=True
    )
    block_output = block(
        x, output_hidden_states=True, output_depth_weights=True
    )
    torch.testing.assert_close(
        full_output.last_hidden_state,
        block_output.last_hidden_state,
        rtol=0,
        atol=0,
    )
    for left, right in zip(
        full_output.hidden_states, block_output.hidden_states
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(
        full_output.depth_outputs.averaged_weight_matrix,
        block_output.depth_outputs.averaged_weight_matrix,
        rtol=0,
        atol=0,
    )


def test_full_and_block_size_one_match_with_active_queries():
    full = attnres_backbone(depth_mode="full").double().eval()
    activate_depth_queries(full)
    block = attnres_backbone(depth_mode="block", block_size=1).double().eval()
    block.load_state_dict(full.state_dict())
    x = torch.randn(2, 6, 8, dtype=torch.float64)
    torch.testing.assert_close(
        full(x).last_hidden_state,
        block(x).last_hidden_state,
        rtol=2e-13,
        atol=2e-13,
    )


def test_block_eager_and_two_phase_backbones_are_equivalent():
    eager = attnres_backbone(
        depth_mode="block", backend="eager", block_size=4
    ).double().eval()
    activate_depth_queries(eager)
    two_phase = attnres_backbone(
        depth_mode="block", backend="two_phase", block_size=4
    ).double().eval()
    two_phase.load_state_dict(eager.state_dict())
    x = torch.randn(2, 7, 8, dtype=torch.float64)
    left = eager(x, output_depth_weights=True)
    right = two_phase(x, output_depth_weights=True)
    torch.testing.assert_close(
        left.last_hidden_state, right.last_hidden_state,
        rtol=3e-13, atol=3e-13,
    )
    torch.testing.assert_close(
        left.depth_outputs.averaged_weight_matrix,
        right.depth_outputs.averaged_weight_matrix,
        rtol=3e-13, atol=3e-13,
    )
    assert right.depth_outputs.inter_block_scan_count == 3


def explicit_full_traversal(model, hidden_states, mask):
    controller = model.depth_controller
    state = controller.initialize(hidden_states)
    attention_outputs, ffn_outputs = [], []
    for layer in model.layers:
        attention_mix = controller.mix_for_site(
            layer.pre_attention_attnres, state
        )
        attention, _, _ = layer._attention_transform(
            attention_mix.mixed_state,
            mask,
            None,
            use_cache=False,
            mode="full",
            output_diagnostics=False,
        )
        attention = layer.residual_dropout(attention)
        controller.append_output(state, attention)
        ffn_mix = controller.mix_for_site(layer.pre_ffn_attnres, state)
        ffn = layer.residual_dropout(
            layer.ffn(layer.ffn_norm(ffn_mix.mixed_state))
        )
        controller.append_output(state, ffn)
        attention_outputs.append(attention)
        ffn_outputs.append(ffn)
    final = controller.finalize(state, model.final_output_attnres).mixed_state
    return model.final_norm(final), attention_outputs, ffn_outputs


def test_backbone_matches_explicit_controller_layer_traversal():
    model = attnres_backbone(depth_mode="full").double().eval()
    activate_depth_queries(model)
    x = torch.randn(2, 5, 8, dtype=torch.float64)
    mask = torch.ones(2, 5, dtype=torch.bool)
    expected, _, _ = explicit_full_traversal(model, x, mask)
    torch.testing.assert_close(
        model(x).last_hidden_state, expected, rtol=0, atol=0
    )


def test_standard_mode_has_no_attnres_parameters_and_remains_deterministic():
    config = backbone_config(
        attention_residual_config=None,
    )
    first = HybridAttentionBackbone(config).double().eval()
    second = copy.deepcopy(first)
    assert first.config.depth_mixing == "standard"
    assert first.depth_site_metadata == ()
    assert first.final_output_attnres is None
    assert not any("attnres" in name for name, _ in first.named_parameters())
    x = torch.randn(2, 5, 8, dtype=torch.float64)
    torch.testing.assert_close(
        first(x).last_hidden_state,
        second(x).last_hidden_state,
        rtol=0,
        atol=0,
    )


def test_attnres_parameter_count_matches_two_d_per_site():
    model = attnres_backbone(depth_mode="block")
    sites = 2 * len(model.layers) + 1
    attnres_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "attnres" in name
    )
    assert sites == 11
    assert attnres_parameters == 2 * model.config.d_model * sites == 176


def test_full_hidden_trace_has_unambiguous_fields():
    model = attnres_backbone(depth_mode="block").eval()
    output = model(torch.randn(2, 4, 8), output_hidden_states=True)
    trace = output.hidden_state_trace
    assert trace.embedding.shape == (2, 4, 8)
    assert len(trace.pre_attention) == len(model.layers)
    assert len(trace.attention_outputs) == len(model.layers)
    assert len(trace.pre_ffn) == len(model.layers)
    assert len(trace.ffn_outputs) == len(model.layers)
    assert trace.final_mixed.shape == (2, 4, 8)


def test_single_token_zero_input_and_oversized_depth_block():
    model = attnres_backbone(depth_mode="block", block_size=64).eval()
    output = model(
        torch.zeros(1, 1, 8),
        output_hidden_states=True,
        output_depth_weights=True,
    )
    assert output.last_hidden_state.shape == (1, 1, 8)
    assert torch.isfinite(output.last_hidden_state).all()
    assert output.depth_outputs.num_depth_blocks == 1
    assert output.depth_outputs.partial_final_block_size == 10
