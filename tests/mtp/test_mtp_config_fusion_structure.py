import pytest
import torch

from src.kda import KimiDeltaAttention
from src.mla import GatedMLA
from src.mtp import KimiMTPConfig, KimiMTPFusion
from src.stable_latent_moe import StableLatentMoE

from .conftest import tiny_mtp_config, tiny_mtp_head


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"num_mtp_layers": 2}, "exactly one"),
        ({"future_offset": 3}, "x\\[t\\+2\\]"),
        ({"loss_weight": -0.1}, "loss_weight"),
        ({"fusion_kind": "add"}, "concat_project"),
        ({"fusion_bias": True}, "bias-free"),
        ({"share_main_lm_head": False}, "shares"),
        ({"detach_backbone_hidden": True}, None),
    ],
)
def test_config_enforces_canonical_choices(overrides, match):
    if match is None:
        assert tiny_mtp_config(**overrides).detach_backbone_hidden
    else:
        with pytest.raises(ValueError, match=match):
            tiny_mtp_config(**overrides)


def test_config_roundtrip_is_lossless_and_immutable():
    config = tiny_mtp_config()
    assert KimiMTPConfig.from_dict(config.to_dict()) == config
    with pytest.raises(Exception):
        config.loss_weight = 1.0


def test_disabled_config_needs_no_auxiliary_subconfigs():
    config = KimiMTPConfig(d_model=8, vocab_size=23, enabled=False)
    head = tiny_mtp_head(enabled=False)
    assert config.enabled is False
    assert head.block is None
    assert head.fusion is None


def test_fusion_matches_explicit_normalize_concat_projection():
    torch.manual_seed(1)
    fusion = KimiMTPFusion(4, eps=1e-5)
    hidden = torch.randn(2, 3, 4)
    future = torch.randn(2, 3, 4)
    expected = torch.nn.functional.linear(
        torch.cat(
            (fusion.hidden_norm(hidden), fusion.future_embedding_norm(future)),
            dim=-1,
        ),
        fusion.projection.weight,
    )
    torch.testing.assert_close(fusion(hidden, future), expected)
    assert fusion.projection.bias is None


def test_fusion_has_closed_form_parameter_count_and_gradcheck():
    fusion = KimiMTPFusion(3).double()
    assert sum(p.numel() for p in fusion.parameters()) == 2 * 3 + 2 * 3 * 3
    inputs = (
        torch.randn(1, 2, 3, dtype=torch.double, requires_grad=True),
        torch.randn(1, 2, 3, dtype=torch.double, requires_grad=True),
    )
    assert torch.autograd.gradcheck(fusion, inputs, atol=1e-5, rtol=1e-4)


def test_mtp_block_has_exact_canonical_group_and_no_extra_global_mla():
    head = tiny_mtp_head()
    block = head.block
    assert block.attention_types == ("kda", "kda", "kda", "gated_mla")
    assert len(block.layers) == 4
    assert sum(isinstance(layer.attention, KimiDeltaAttention) for layer in block.layers) == 3
    assert sum(isinstance(layer.attention, GatedMLA) for layer in block.layers) == 1
    assert sum(isinstance(layer.ffn, StableLatentMoE) for layer in block.layers) == 4
    assert block.backbone.final_global_layer is None


def test_mtp_block_has_eight_independent_sites_and_final_local_site():
    block = tiny_mtp_head().block.backbone
    sites = [
        site
        for layer in block.layers
        for site in (layer.pre_attention_attnres, layer.pre_ffn_attnres)
    ]
    assert len(sites) == 8
    assert all(site is not None for site in sites)
    assert len({site.pseudo_query.data_ptr() for site in sites}) == 8
    assert block.final_output_attnres is not None
    assert block.final_output_attnres.pseudo_query.data_ptr() not in {
        site.pseudo_query.data_ptr() for site in sites
    }


def test_attnres_zero_init_is_uniform_over_available_sources():
    backbone = tiny_mtp_head().block.backbone
    output = backbone(
        torch.randn(2, 5, 8),
        output_depth_weights=True,
    )
    weights = output.depth_outputs.averaged_weight_matrix
    source_mask = output.depth_outputs.source_mask
    for row, mask in zip(weights, source_mask):
        valid = row[mask]
        torch.testing.assert_close(
            valid,
            torch.full_like(valid, 1.0 / valid.numel()),
            atol=1e-6,
            rtol=0,
        )


def test_mtp_parameters_are_independent_across_instances():
    left = tiny_mtp_head(seed=10)
    right = tiny_mtp_head(seed=10)
    assert not {
        parameter.data_ptr() for parameter in left.block.parameters()
    } & {
        parameter.data_ptr() for parameter in right.block.parameters()
    }
