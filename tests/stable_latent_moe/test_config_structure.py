import pytest

from src.attention_residuals import AttentionResidualConfig
from src.hybrid_backbone import HybridAttentionBackbone, HybridBackboneConfig
from src.stable_latent_moe import StableLatentMoEConfig
from tests.hybrid_backbone.conftest import tiny_kda_config, tiny_mla_config
from tests.stable_latent_moe.conftest import (
    tiny_kimi_config,
    tiny_moe,
    tiny_moe_config,
)


@pytest.mark.parametrize(
    "overrides",
    [
        {"d_model": 0},
        {"latent_dim": 0},
        {"num_shared_experts": 0},
        {"num_routed_experts": 0},
        {"routed_experts_per_token": 0},
        {"routed_experts_per_token": 5},
        {
            "routed_experts_per_token": 4,
            "enable_quantile_balancing": True,
        },
        {"shared_expert_hidden_dim": 0},
        {"routed_expert_hidden_dim": 0},
        {"beta_gate": 0},
        {"beta_up": 0},
        {"norm_eps": 0},
        {"router_eps": 0},
        {"routing_backend": "dense"},
        {"quantile_backend": "approx"},
        {"router_logits_dtype": "float16"},
        {"routing_weights_dtype": "float16"},
        {"routed_accumulation_dtype": "float16"},
        {"histogram_num_bins": 1},
        {"histogram_min_margin": 2.0, "histogram_max_margin": 2.0},
        {"init_std": 0},
    ],
)
def test_invalid_configs_are_rejected(overrides):
    values = tiny_moe_config().to_dict()
    values.update(overrides)
    with pytest.raises((ValueError, TypeError)):
        StableLatentMoEConfig(**values)


def test_dense_routed_k_equals_n_is_allowed_only_without_balancing():
    config = tiny_moe_config(
        routed_experts_per_token=4,
        enable_quantile_balancing=False,
    )
    assert config.top_k == config.num_routed_experts


def test_config_roundtrips_are_lossless():
    moe = tiny_moe_config(quantile_backend="histogram")
    assert StableLatentMoEConfig.from_dict(moe.to_dict()) == moe
    kimi = tiny_kimi_config()
    assert type(kimi).from_dict(kimi.to_dict()) == kimi
    backbone = kimi.to_backbone_config()
    assert HybridBackboneConfig.from_dict(backbone.to_dict()) == backbone


def test_full_k3_metadata_is_representable_without_allocating_experts():
    config = StableLatentMoEConfig.kimi_k3()
    assert config.d_model == 7168
    assert config.latent_dim == 3584
    assert config.num_shared_experts == 2
    assert config.num_routed_experts == 896
    assert config.top_k == 16
    kimi = tiny_kimi_config(
        d_model=7168,
        num_pattern_repeats=23,
        kda_config=tiny_kda_config(
            d_model=7168,
            value_head_dim=3584,
        ),
        mla_config=tiny_mla_config(
            d_model=7168,
            v_head_dim=3584,
        ),
        stable_latent_moe_config=config,
        attention_residual_config=AttentionResidualConfig(
            7168,
            mode="block",
            sublayers_per_depth_block=24,
            target_num_depth_blocks=8,
        ),
    )
    assert (kimi.num_kda_layers, kimi.num_mla_layers, kimi.num_moe_layers) == (
        69,
        24,
        93,
    )


def test_parameter_count_matches_closed_form_and_excludes_routing_bias():
    model = tiny_moe()
    config = model.config
    expected = (
        config.num_shared_experts
        * 3
        * config.d_model
        * config.shared_expert_hidden_dim
        + config.num_routed_experts
        * 3
        * config.latent_dim
        * config.routed_expert_hidden_dim
        + 2 * config.d_model * config.latent_dim
        + config.num_routed_experts * config.d_model
        + config.latent_dim
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == expected
    assert "router.routing_bias" in model.state_dict()
    assert all(
        name != "router.routing_bias"
        for name, _ in model.named_parameters()
    )


def test_hybrid_backbone_selects_exactly_one_channel_mixer_type():
    moe = tiny_moe_config()
    config = HybridBackboneConfig(
        d_model=8,
        num_hybrid_groups=1,
        kda_config=tiny_kda_config(),
        mla_config=tiny_mla_config(),
        use_dense_ffn=False,
        channel_mixer_type="stable_latent_moe",
        stable_latent_moe_config=moe,
    )
    model = HybridAttentionBackbone(config)
    assert all(type(layer.ffn).__name__ == "StableLatentMoE" for layer in model.layers)
    with pytest.raises(ValueError):
        HybridBackboneConfig(
            d_model=8,
            num_hybrid_groups=1,
            kda_config=tiny_kda_config(),
            mla_config=tiny_mla_config(),
            channel_mixer_type="stable_latent_moe",
            stable_latent_moe_config=moe,
        )
