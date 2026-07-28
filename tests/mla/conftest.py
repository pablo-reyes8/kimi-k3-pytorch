import torch

from src.mla import GatedMLA, GatedMLAConfig


def tiny_config(**overrides):
    values = dict(
        d_model=12,
        num_heads=3,
        q_head_dim=2,
        v_head_dim=4,
        kv_latent_dim=5,
        attention_dropout=0.0,
        output_dropout=0.0,
    )
    values.update(overrides)
    return GatedMLAConfig(**values)


def tiny_mla(**overrides):
    torch.manual_seed(29)
    return GatedMLA(tiny_config(**overrides))


def random_qkv(
    batch=2,
    query_tokens=5,
    key_tokens=None,
    heads=3,
    query_dim=2,
    value_dim=4,
    dtype=torch.float64,
    seed=31,
):
    key_tokens = query_tokens if key_tokens is None else key_tokens
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(
        batch, query_tokens, heads, query_dim, generator=generator, dtype=dtype
    )
    k = torch.randn(
        batch, key_tokens, heads, query_dim, generator=generator, dtype=dtype
    )
    v = torch.randn(
        batch, key_tokens, heads, value_dim, generator=generator, dtype=dtype
    )
    return q, k, v
