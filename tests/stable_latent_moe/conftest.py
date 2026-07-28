import torch

from src.attention_residuals import AttentionResidualConfig
from src.kimi_block import KimiBlock, KimiBlockConfig
from src.stable_latent_moe import StableLatentMoE, StableLatentMoEConfig
from tests.hybrid_backbone.conftest import tiny_kda_config, tiny_mla_config


def tiny_moe_config(**overrides):
    values = dict(
        d_model=8,
        latent_dim=4,
        num_shared_experts=2,
        num_routed_experts=4,
        routed_experts_per_token=2,
        shared_expert_hidden_dim=6,
        routed_expert_hidden_dim=5,
        routing_backend="vectorized",
        quantile_backend="exact",
        enable_quantile_balancing=True,
    )
    values.update(overrides)
    return StableLatentMoEConfig(**values)


def tiny_moe(**overrides):
    torch.manual_seed(211)
    return StableLatentMoE(tiny_moe_config(**overrides))


def tiny_kimi_config(**overrides):
    values = dict(
        d_model=8,
        num_pattern_repeats=1,
        kda_config=tiny_kda_config(),
        mla_config=tiny_mla_config(),
        stable_latent_moe_config=tiny_moe_config(),
        attention_residual_config=AttentionResidualConfig(
            8,
            mode="block",
            sublayers_per_depth_block=4,
        ),
    )
    values.update(overrides)
    return KimiBlockConfig(**values)


def tiny_kimi(**overrides):
    torch.manual_seed(223)
    return KimiBlock(tiny_kimi_config(**overrides))
