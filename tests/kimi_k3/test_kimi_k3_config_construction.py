from dataclasses import FrozenInstanceError, replace

import pytest
import torch

import src
from src import (
    KimiK3,
    KimiK3Config,
    kimi_k3_canonical_config,
    kimi_k3_cpu_tiny_config,
)
from src.kda import KimiDeltaAttention
from src.mla import GatedMLA
from src.stable_latent_moe import StableLatentMoE


def test_canonical_config_has_reported_topology_without_instantiation():
    config = kimi_k3_canonical_config()
    assert config.vocab_size == 160000
    assert config.d_model == 7168
    assert config.backbone.num_kda_layers == 69
    assert config.backbone.num_mla_layers == 24
    assert config.backbone.num_moe_layers == 93
    assert config.backbone.attention_residual_config.num_depth_blocks(93) == 8
    assert config.backbone.stable_latent_moe_config.num_routed_experts == 896
    assert config.backbone.stable_latent_moe_config.top_k == 16
    assert config.vision.depth == 27
    assert config.mtp.num_mtp_layers == 1


def test_cpu_tiny_config_is_frozen_and_roundtrips_exactly():
    config = kimi_k3_cpu_tiny_config()
    assert KimiK3Config.from_dict(config.to_dict()) == config
    with pytest.raises(FrozenInstanceError):
        config.vocab_size = 10


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("vocab_size", 0, "vocab_size"),
        ("d_model", 0, "d_model"),
        ("initializer_range", 0, "initializer_range"),
        ("image_token_id", 128, "vocabulary"),
        ("image_token_id", 2, "distinct"),
    ],
)
def test_unified_config_rejects_invalid_values(field, value, match):
    config = kimi_k3_cpu_tiny_config()
    with pytest.raises(ValueError, match=match):
        replace(config, **{field: value})


def test_public_entrypoint_is_the_real_kimi_class():
    from src.kimi_k3_mini import KimiK3 as stable_entrypoint

    assert src.KimiK3 is KimiK3
    assert stable_entrypoint is KimiK3


def test_disabled_capabilities_instantiate_no_hidden_modules(
    config_no_vision,
    config_no_mtp,
):
    no_vision = KimiK3(config_no_vision)
    assert no_vision.vision_encoder is None
    assert no_vision.vision_projector is None
    assert no_vision.multimodal_composer is None
    no_mtp = KimiK3(config_no_mtp)
    assert no_mtp.mtp is None
    assert not any(name.startswith("mtp.") for name, _ in no_mtp.named_parameters())


def test_main_backbone_has_exact_attention_and_moe_counts(tiny_kimi_model):
    layers = tiny_kimi_model.backbone.layers
    assert len(layers) == 5
    assert sum(isinstance(layer.attention, KimiDeltaAttention) for layer in layers) == 3
    assert sum(isinstance(layer.attention, GatedMLA) for layer in layers) == 2
    assert sum(isinstance(layer.ffn, StableLatentMoE) for layer in layers) == 5
    assert all(isinstance(layer.ffn, StableLatentMoE) for layer in layers)


def test_mtp_group_is_separate_and_has_no_shared_module_registration(
    tiny_kimi_model,
):
    assert tiny_kimi_model.mtp.block.attention_types == (
        "kda",
        "kda",
        "kda",
        "gated_mla",
    )
    assert "_registered_input_embeddings" not in tiny_kimi_model.mtp._modules
    assert "_registered_lm_head" not in tiny_kimi_model.mtp._modules
    assert not any(
        name.startswith("mtp.") and (
            "input_embeddings" in name or "lm_head" in name
        )
        for name, _ in tiny_kimi_model.named_parameters()
    )


def test_specialized_initialization_survives_top_level_construction(
    tiny_kimi_model,
):
    attnres_queries = [
        site.pseudo_query
        for layer in tiny_kimi_model.backbone.layers
        for site in (
            layer.pre_attention_attnres,
            layer.pre_ffn_attnres,
        )
    ]
    assert all(torch.count_nonzero(query) == 0 for query in attnres_queries)
    assert all(
        "routing_bias" not in dict(module.named_parameters())
        for module in tiny_kimi_model.modules()
        if isinstance(module, StableLatentMoE)
    )


def test_same_seed_constructs_identical_state_dicts():
    torch.manual_seed(77)
    first = KimiK3(kimi_k3_cpu_tiny_config())
    torch.manual_seed(77)
    second = KimiK3(kimi_k3_cpu_tiny_config())
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
