import torch

from src.attention_residuals import AttentionResidualConfig
from src.hybrid_backbone import HybridAttentionBackbone, HybridBackboneConfig
from tests.hybrid_backbone.conftest import (
    assert_hybrid_caches_close,
    tiny_kda_config,
    tiny_mla_config,
)


def attnres_config(mode="block", backend="eager", block_size=4, **overrides):
    values = dict(
        d_model=8,
        mode=mode,
        backend=backend,
        sublayers_per_depth_block=block_size if mode == "block" else None,
    )
    values.update(overrides)
    return AttentionResidualConfig(**values)


def backbone_config(
    depth_mode="block",
    backend="eager",
    block_size=4,
    groups=1,
    **overrides,
):
    values = dict(
        d_model=8,
        num_hybrid_groups=groups,
        mlp_hidden_dim=12,
        kda_config=tiny_kda_config(),
        mla_config=tiny_mla_config(),
        attention_residual_config=attnres_config(
            depth_mode, backend, block_size
        ),
    )
    values.update(overrides)
    return HybridBackboneConfig(**values)


def attnres_backbone(**kwargs):
    torch.manual_seed(53)
    return HybridAttentionBackbone(backbone_config(**kwargs))


def activate_depth_queries(model, seed=67):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in model.layers:
            for site in (
                layer.pre_attention_attnres,
                layer.pre_ffn_attnres,
            ):
                site.pseudo_query.copy_(
                    torch.randn(
                        site.d_model,
                        generator=generator,
                        dtype=site.pseudo_query.dtype,
                        device=site.pseudo_query.device,
                    )
                    * 0.2
                )
        model.final_output_attnres.pseudo_query.copy_(
            torch.randn(
                model.config.d_model,
                generator=generator,
                dtype=model.final_output_attnres.pseudo_query.dtype,
                device=model.final_output_attnres.pseudo_query.device,
            )
            * 0.2
        )


__all__ = [
    "activate_depth_queries",
    "assert_hybrid_caches_close",
    "attnres_backbone",
    "attnres_config",
    "backbone_config",
]
