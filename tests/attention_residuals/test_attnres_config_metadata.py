import pytest

from src.attention_residuals import AttentionResidualConfig
from src.hybrid_backbone import HybridBackboneConfig
from tests.attention_residuals.conftest import (
    attnres_backbone,
    attnres_config,
    backbone_config,
)


def test_full_block_and_standard_config_contracts():
    full = attnres_config("full")
    block = attnres_config("block", block_size=6)
    standard = AttentionResidualConfig(
        8, mode="standard", sublayers_per_depth_block=None
    )
    assert full.mode == "full"
    assert block.resolved_sublayers_per_depth_block == 6
    assert standard.mode == "standard"


def test_transformer_layer_block_size_derives_sublayer_units():
    config = AttentionResidualConfig(
        8,
        mode="block",
        transformer_layers_per_depth_block=3,
        sublayers_per_depth_block=None,
    )
    assert config.resolved_sublayers_per_depth_block == 6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_model": 0},
        {"mode": "unknown"},
        {"rms_norm_eps": 0},
        {"query_init": "normal"},
        {"include_embedding_source": False},
        {"add_final_output_mixer": False},
        {"mode": "block", "sublayers_per_depth_block": 0},
        {"mode": "block", "sublayers_per_depth_block": None},
        {
            "mode": "block",
            "transformer_layers_per_depth_block": 3,
            "sublayers_per_depth_block": 5,
        },
        {"mode": "full", "sublayers_per_depth_block": 4},
        {"mode": "standard", "sublayers_per_depth_block": 4},
        {"backend": "unknown"},
        {
            "mode": "full",
            "backend": "two_phase",
            "sublayers_per_depth_block": None,
        },
    ],
)
def test_invalid_attnres_configs_are_rejected(kwargs):
    values = dict(d_model=8)
    values.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        AttentionResidualConfig(**values)


def test_config_roundtrip_is_lossless():
    config = attnres_config(
        "block",
        backend="two_phase",
        block_size=8,
        target_num_depth_blocks=3,
        return_depth_weights=True,
    )
    assert AttentionResidualConfig.from_dict(config.to_dict()) == config
    backbone = backbone_config(
        attention_residual_config=config,
        groups=2,
    )
    assert HybridBackboneConfig.from_dict(backbone.to_dict()) == backbone


def test_k3_topology_represents_93_layers_and_eight_depth_blocks():
    attnres = AttentionResidualConfig(
        8,
        mode="block",
        transformer_layers_per_depth_block=12,
        sublayers_per_depth_block=24,
        target_num_depth_blocks=8,
    )
    config = backbone_config(groups=23, attention_residual_config=attnres)
    assert config.num_attention_layers == 93
    assert config.num_kda_layers == 69
    assert config.num_mla_layers == 24
    assert attnres.num_depth_blocks(93) == 8


def test_attnres_rejects_final_global_layer_without_ffn():
    with pytest.raises(ValueError):
        backbone_config(
            add_ffn_after_final_global=False,
            attention_residual_config=attnres_config("full"),
        )


@pytest.mark.parametrize("depth_mode,block_size", [("full", 4), ("block", 4)])
def test_depth_site_metadata_is_contiguous_alternating_and_complete(
    depth_mode, block_size
):
    model = attnres_backbone(depth_mode=depth_mode, block_size=block_size)
    metadata = model.depth_site_metadata
    assert len(metadata) == 2 * len(model.layers) + 1
    assert [item.site_index for item in metadata] == list(range(len(metadata)))
    for layer_index, attention_type in enumerate(model.attention_types):
        expected_type = (
            "gated_mla"
            if attention_type == "gated_mla_final"
            else attention_type
        )
        attention_site, ffn_site = metadata[2 * layer_index : 2 * layer_index + 2]
        assert attention_site.site_kind == "pre_attention"
        assert ffn_site.site_kind == "pre_ffn"
        assert attention_site.transformer_layer_index == layer_index
        assert ffn_site.transformer_layer_index == layer_index
        assert attention_site.attention_type == expected_type
        assert ffn_site.attention_type == expected_type
    assert metadata[-1].site_kind == "final_output"
    assert metadata[-1].transformer_layer_index is None


def test_block_metadata_boundaries_use_sublayer_units():
    model = attnres_backbone(depth_mode="block", block_size=3)
    transformation_sites = model.depth_site_metadata[:-1]
    assert [item.depth_block_index for item in transformation_sites] == [
        index // 3 for index in range(10)
    ]
    assert [item.position_in_depth_block for item in transformation_sites] == [
        index % 3 for index in range(10)
    ]
