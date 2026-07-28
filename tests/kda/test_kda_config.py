import pytest

from src.kda import KDAConfig
from tests.kda.conftest import tiny_config


def test_config_resolved_dimensions_and_roundtrip():
    config = tiny_config(decay_rank=None)
    assert config.key_width == 6
    assert config.value_width == config.d_model == 12
    assert config.resolved_decay_rank == config.value_head_dim
    assert KDAConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_model": 0},
        {"num_heads": 0},
        {"key_head_dim": 0},
        {"value_head_dim": 0},
        {"d_model": 11},
        {"short_conv_kernel_size": 0},
        {"decay_rank": 0},
        {"g_min": 0},
        {"chunk_size": 0},
        {"secondary_tile_size": 0},
        {"chunk_size": 4, "secondary_tile_size": 5},
        {"eps": 0},
        {"init_std": 0},
        {"decay_initializer": "claimed_official"},
    ],
)
def test_invalid_configuration_rejected(kwargs):
    values = tiny_config().to_dict()
    values.update(kwargs)
    with pytest.raises(ValueError):
        KDAConfig(**values)


def test_key_and_value_dimensions_need_not_match():
    config = KDAConfig(
        d_model=15, num_heads=3, key_head_dim=2, value_head_dim=5
    )
    assert config.key_head_dim != config.value_head_dim


def test_default_k3_decay_constants():
    config = KDAConfig(
        d_model=16, num_heads=4, key_head_dim=3, value_head_dim=4
    )
    assert config.g_min == -5
    assert config.secondary_tile_size == 16
    assert config.accumulate_state_in_fp32 is True
    assert config.decay_initializer == "official_fla"

