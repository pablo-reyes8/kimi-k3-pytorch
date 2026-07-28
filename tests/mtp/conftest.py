import torch
import torch.nn as nn

from src.attention_residuals import AttentionResidualConfig
from src.mtp import KimiMTPConfig, KimiMTPHead
from tests.hybrid_backbone.conftest import tiny_kda_config, tiny_mla_config
from tests.stable_latent_moe.conftest import tiny_moe_config


def tiny_mtp_config(**overrides):
    values = dict(
        d_model=8,
        vocab_size=23,
        kda_config=tiny_kda_config(),
        mla_config=tiny_mla_config(),
        stable_latent_moe_config=tiny_moe_config(
            num_shared_experts=1,
        ),
        attention_residual_config=AttentionResidualConfig(
            8,
            mode="block",
            sublayers_per_depth_block=4,
        ),
        init_std=0.02,
    )
    values.update(overrides)
    return KimiMTPConfig(**values)


def tiny_mtp_head(*, tied=False, seed=701, **overrides):
    torch.manual_seed(seed)
    config = tiny_mtp_config(**overrides)
    embedding = nn.Embedding(config.vocab_size, config.d_model)
    lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    if tied:
        lm_head.weight = embedding.weight
    return KimiMTPHead(config, embedding, lm_head)
