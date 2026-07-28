import torch

from src.kda import KDAConfig, KimiDeltaAttention


def tiny_config(**overrides):
    values = dict(
        d_model=12,
        num_heads=3,
        key_head_dim=2,
        value_head_dim=4,
        short_conv_kernel_size=3,
        decay_rank=5,
        chunk_size=4,
        secondary_tile_size=2,
        eps=1e-6,
    )
    values.update(overrides)
    return KDAConfig(**values)


def tiny_kda(**overrides):
    torch.manual_seed(17)
    return KimiDeltaAttention(tiny_config(**overrides))


def random_core(
    batch=2,
    tokens=7,
    heads=3,
    key_dim=2,
    value_dim=4,
    dtype=torch.float64,
    seed=11,
):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(
        batch, tokens, heads, key_dim, generator=generator, dtype=dtype
    )
    k = torch.randn(
        batch, tokens, heads, key_dim, generator=generator, dtype=dtype
    )
    v = torch.randn(
        batch, tokens, heads, value_dim, generator=generator, dtype=dtype
    )
    g = -5 * torch.rand(
        batch, tokens, heads, key_dim, generator=generator, dtype=dtype
    )
    beta = torch.rand(
        batch, tokens, heads, generator=generator, dtype=dtype
    )
    state = torch.randn(
        batch, heads, key_dim, value_dim, generator=generator, dtype=dtype
    )
    return q, k, v, g, beta, state

