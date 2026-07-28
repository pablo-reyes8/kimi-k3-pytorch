from collections import Counter

import torch

from tests.hybrid_backbone.conftest import tiny_backbone


def test_pattern_repeats_without_shift_and_final_global_mla_is_present():
    model = tiny_backbone(num_hybrid_groups=3)
    expected = (
        "kda", "kda", "kda", "gated_mla",
        "kda", "kda", "kda", "gated_mla",
        "kda", "kda", "kda", "gated_mla",
        "gated_mla_final",
    )
    assert model.attention_types == expected
    assert tuple(group.attention_types for group in model.groups) == (
        ("kda", "kda", "kda", "gated_mla"),
    ) * 3


def test_expected_attention_layer_counts_for_multiple_group_counts():
    for groups in (1, 2, 4):
        model = tiny_backbone(num_hybrid_groups=groups)
        counts = Counter(
            "gated_mla" if kind == "gated_mla_final" else kind
            for kind in model.attention_types
        )
        assert counts == {"kda": 3 * groups, "gated_mla": groups + 1}


def test_layer_metadata_is_complete_and_stable():
    model = tiny_backbone(num_hybrid_groups=2)
    for index, layer in enumerate(model.layers):
        assert layer.layer_index == index
        if index < 8:
            assert layer.group_index == index // 4
            assert layer.position_in_group == index % 4
            assert not layer.is_final_global
        else:
            assert layer.group_index is None
            assert layer.position_in_group is None
            assert layer.is_final_global


def test_attention_ffn_and_norm_parameters_are_never_shared():
    model = tiny_backbone(num_hybrid_groups=2)
    for attribute in ("attention", "ffn", "attention_norm", "ffn_norm"):
        modules = [
            getattr(layer, attribute)
            for layer in model.layers
            if getattr(layer, attribute) is not None
        ]
        parameter_pointers = [
            next(module.parameters()).data_ptr() for module in modules
        ]
        assert len(parameter_pointers) == len(set(parameter_pointers))


def test_final_global_layer_has_no_ffn_by_default():
    model = tiny_backbone()
    assert model.final_global_layer.attention_type == "gated_mla"
    assert model.final_global_layer.ffn is None
    assert model.final_global_layer.ffn_norm is None


def test_final_global_ffn_can_be_enabled_explicitly():
    model = tiny_backbone(add_ffn_after_final_global=True)
    assert model.final_global_layer.ffn is not None
    assert model.final_global_layer.ffn_norm is not None


def test_parameter_count_matches_closed_form_for_one_group():
    model = tiny_backbone()
    config = model.config
    d = config.d_model
    h = config.kda_config.num_heads
    k = config.kda_config.key_head_dim
    v = config.kda_config.value_head_dim
    key_width, value_width = h * k, h * v
    rank = config.kda_config.resolved_decay_rank
    kernel = config.kda_config.short_conv_kernel_size
    kda = (
        d * (2 * key_width + value_width + h + rank)
        + rank * key_width
        + kernel * (2 * key_width + value_width)
        + key_width
        + h
        + value_width
        + 2 * d * d
    )
    mla_config = config.mla_config
    mla = (
        d * mla_config.query_width
        + d * mla_config.kv_latent_dim
        + mla_config.kv_latent_dim * mla_config.query_width
        + mla_config.kv_latent_dim * mla_config.value_width
        + 2 * d * d
    )
    ffn = 3 * d * config.resolved_mlp_hidden_dim
    norms = 10 * d
    expected = 3 * kda + 2 * mla + 4 * ffn + norms
    observed = sum(parameter.numel() for parameter in model.parameters())
    assert observed == expected
