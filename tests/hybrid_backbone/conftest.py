import torch

from src.hybrid_backbone import HybridAttentionBackbone, HybridBackboneConfig
from src.kda import KDAConfig
from src.mla import GatedMLAConfig


def tiny_kda_config(**overrides):
    values = dict(
        d_model=8,
        num_heads=2,
        key_head_dim=2,
        value_head_dim=4,
        short_conv_kernel_size=2,
        decay_rank=3,
        chunk_size=4,
        secondary_tile_size=2,
        decay_initializer="zeros",
    )
    values.update(overrides)
    return KDAConfig(**values)


def tiny_mla_config(**overrides):
    values = dict(
        d_model=8,
        num_heads=2,
        q_head_dim=2,
        v_head_dim=4,
        kv_latent_dim=3,
        attention_backend="manual",
    )
    values.update(overrides)
    return GatedMLAConfig(**values)


def tiny_config(**overrides):
    values = dict(
        d_model=8,
        num_hybrid_groups=1,
        mlp_hidden_dim=12,
        kda_config=tiny_kda_config(),
        mla_config=tiny_mla_config(),
    )
    values.update(overrides)
    return HybridBackboneConfig(**values)


def tiny_backbone(**overrides):
    torch.manual_seed(41)
    return HybridAttentionBackbone(tiny_config(**overrides))


def assert_kda_states_close(left, right, **kwargs):
    torch.testing.assert_close(
        left.recurrent_state, right.recurrent_state, **kwargs
    )
    torch.testing.assert_close(
        left.q_conv_state.buffer, right.q_conv_state.buffer, **kwargs
    )
    torch.testing.assert_close(
        left.k_conv_state.buffer, right.k_conv_state.buffer, **kwargs
    )
    torch.testing.assert_close(
        left.v_conv_state.buffer, right.v_conv_state.buffer, **kwargs
    )
    torch.testing.assert_close(
        left.sequence_offset, right.sequence_offset, rtol=0, atol=0
    )


def assert_hybrid_caches_close(left, right, **kwargs):
    assert len(left.layer_caches) == len(right.layer_caches)
    assert left.sequence_length == right.sequence_length
    for left_layer, right_layer in zip(left.layer_caches, right.layer_caches):
        assert left_layer.attention_type == right_layer.attention_type
        if left_layer.attention_type == "kda":
            assert_kda_states_close(
                left_layer.state, right_layer.state, **kwargs
            )
        else:
            torch.testing.assert_close(
                left_layer.state.latent_kv,
                right_layer.state.latent_kv,
                **kwargs,
            )
            torch.testing.assert_close(
                left_layer.state.attention_mask,
                right_layer.state.attention_mask,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                left_layer.state.sequence_offset,
                right_layer.state.sequence_offset,
                rtol=0,
                atol=0,
            )
