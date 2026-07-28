import torch

from src.vision import (
    HierarchicalMoonViTEncoder,
    HierarchicalVisionConfig,
    MoonViTEncoder,
    SwinMoonViTEncoder,
    SwinVisionConfig,
    VisionEncoderConfig,
)


def tiny_moonvit(**overrides):
    values = dict(
        image_size=(28, 42),
        patch_size=14,
        in_channels=3,
        embed_dim=24,
        depth=2,
        num_heads=6,
        mlp_ratio=2.0,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
    )
    values.update(overrides)
    torch.manual_seed(7)
    return MoonViTEncoder(VisionEncoderConfig(**values))


def tiny_hierarchical(**overrides):
    values = dict(
        image_size=(56, 84),
        patch_size=14,
        embed_dims=(24, 48),
        depths=(1, 1),
        num_heads=(6, 6),
        mlp_ratio=2.0,
    )
    values.update(overrides)
    torch.manual_seed(7)
    return HierarchicalMoonViTEncoder(HierarchicalVisionConfig(**values))


def tiny_swin(**overrides):
    values = dict(
        image_size=(56, 84),
        patch_size=14,
        embed_dims=(24, 48),
        depths=(2, 2),
        num_heads=(6, 6),
        window_size=2,
        mlp_ratio=2.0,
    )
    values.update(overrides)
    torch.manual_seed(7)
    return SwinMoonViTEncoder(SwinVisionConfig(**values))

