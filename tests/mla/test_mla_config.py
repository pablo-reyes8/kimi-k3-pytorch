import pytest

from src.mla import GatedMLAConfig
from tests.mla.conftest import tiny_config


def test_config_derived_dimensions_and_compression():
    config = tiny_config()
    assert config.query_width == 6
    assert config.value_width == 12
    assert config.full_kv_width == 18
    assert config.cache_compression_ratio == 18 / 5
    assert config.resolved_backend == "sdpa"


@pytest.mark.parametrize(
    "override",
    [
        {"d_model": 0},
        {"num_heads": 0},
        {"q_head_dim": 0},
        {"v_head_dim": 0},
        {"kv_latent_dim": 0},
        {"d_model": 11},
        {"kv_latent_dim": 19},
        {"attention_dropout": -0.1},
        {"attention_dropout": 1.0},
        {"output_dropout": -0.1},
        {"output_dropout": 1.0},
        {"use_nope": False},
        {"init_std": 0.0},
        {"attention_backend": "flash"},
        {"attention_backend": "sdpa", "use_sdpa": False},
    ],
)
def test_invalid_configs_are_rejected(override):
    with pytest.raises((ValueError, TypeError)):
        tiny_config(**override)


def test_manual_backend_can_be_selected_explicitly():
    config = tiny_config(attention_backend="manual", use_sdpa=False)
    assert config.resolved_backend == "manual"


def test_config_roundtrip_is_lossless():
    config = tiny_config(projection_bias=True, attention_backend="manual")
    assert GatedMLAConfig.from_dict(config.to_dict()) == config
