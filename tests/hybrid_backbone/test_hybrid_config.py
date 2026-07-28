import pytest

from src.hybrid_backbone import (
    CANONICAL_ATTENTION_PATTERN,
    HybridBackboneConfig,
)
from tests.hybrid_backbone.conftest import (
    tiny_config,
    tiny_kda_config,
    tiny_mla_config,
)


def test_canonical_config_counts_and_defaults():
    config = tiny_config(num_hybrid_groups=2)
    assert config.attention_pattern == CANONICAL_ATTENTION_PATTERN
    assert config.resolved_mlp_hidden_dim == 12
    assert config.num_group_layers == 8
    assert config.num_attention_layers == 9
    assert config.num_kda_layers == 6
    assert config.num_mla_layers == 3
    assert config.add_final_gated_mla
    assert config.add_ffn_after_final_global


@pytest.mark.parametrize(
    "overrides",
    [
        {"d_model": 0},
        {"num_hybrid_groups": 0},
        {"attention_pattern": ()},
        {"attention_pattern": ("kda", "unknown")},
        {"attention_pattern": ("kda", "kda", "gated_mla")},
        {"add_final_gated_mla": False},
        {"kda_config": None},
        {"mla_config": None},
        {"kda_config": tiny_kda_config(d_model=4, num_heads=1)},
        {
            "mla_config": tiny_mla_config(
                d_model=4, num_heads=1, v_head_dim=4
            )
        },
        {"rms_norm_eps": 0},
        {"residual_dropout": -0.1},
        {"residual_dropout": 1.0},
        {"ffn_dropout": -0.1},
        {"ffn_dropout": 1.0},
        {"mlp_hidden_dim": 0},
        {"use_dense_ffn": False},
        {"activation": "swiglu"},
        {"init_std": 0},
    ],
)
def test_invalid_configurations_are_rejected(overrides):
    with pytest.raises((ValueError, TypeError)):
        tiny_config(**overrides)


def test_config_roundtrip_restores_nested_configs_and_tuple():
    config = tiny_config(num_hybrid_groups=3, ffn_bias=True)
    restored = HybridBackboneConfig.from_dict(config.to_dict())
    assert restored == config
    assert isinstance(restored.attention_pattern, tuple)
    assert restored.kda_config == config.kda_config
    assert restored.mla_config == config.mla_config
