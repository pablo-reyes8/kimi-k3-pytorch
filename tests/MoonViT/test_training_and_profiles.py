from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src import (
    HierarchicalMoonViTEncoder,
    MoonViTEncoder,
    SwinMoonViTEncoder,
)
from src.vision import (
    HierarchicalVisionConfig,
    SwinVisionConfig,
    VisionEncoderConfig,
)


class ShapeClassifier(nn.Module):
    """Test-only head: production vision encoders remain headless."""

    def __init__(self, encoder: nn.Module, output_dim: int, classes: int = 2):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(output_dim, classes)

    def forward(self, images):
        tokens = self.encoder(images).last_hidden_state
        return self.head(tokens.mean(dim=1))


def synthetic_left_right_shapes(samples_per_class=8, size=16):
    generator = torch.Generator().manual_seed(123)
    images = 0.03 * torch.randn(
        2 * samples_per_class, 3, size, size, generator=generator
    )
    labels = torch.arange(2).repeat_interleave(samples_per_class)
    images[:samples_per_class, :, 4:12, 1:5] += 1
    images[samples_per_class:, :, 4:12, -5:-1] += 1
    return images, labels


def test_standard_moonvit_overfits_download_free_spatial_task():
    torch.manual_seed(4)
    encoder = MoonViTEncoder(
        VisionEncoderConfig(
            image_size=16,
            patch_size=4,
            embed_dim=16,
            depth=1,
            num_heads=4,
            mlp_ratio=2,
            initializer_std=0.05,
        )
    )
    model = ShapeClassifier(encoder, 16)
    images, labels = synthetic_left_right_shapes()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0)
    losses = []
    for _ in range(35):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    accuracy = (model(images).argmax(-1) == labels).float().mean()
    assert losses[-1] < losses[0] * 0.1
    assert losses[-1] < 0.05
    assert accuracy == 1


@pytest.mark.parametrize(
    "encoder,output_dim",
    [
        (
            HierarchicalMoonViTEncoder(
                HierarchicalVisionConfig(
                    image_size=16,
                    patch_size=4,
                    embed_dims=(8, 16),
                    depths=(1, 1),
                    num_heads=(2, 4),
                    mlp_ratio=2,
                    position_embedding_type="none",
                )
            ),
            16,
        ),
        (
            SwinMoonViTEncoder(
                SwinVisionConfig(
                    image_size=16,
                    patch_size=4,
                    embed_dims=(8, 16),
                    depths=(2, 1),
                    num_heads=(2, 4),
                    window_size=2,
                    mlp_ratio=2,
                )
            ),
            16,
        ),
    ],
)
def test_variant_training_step_updates_backbone_not_only_head(encoder, output_dim):
    torch.manual_seed(5)
    model = ShapeClassifier(encoder, output_dim)
    images, labels = synthetic_left_right_shapes(samples_per_class=2)
    before = encoder.patch_embedding.projection.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = nn.functional.cross_entropy(model(images), labels)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before, encoder.patch_embedding.projection.weight)
    assert encoder.patch_embedding.projection.weight.grad.abs().sum() > 0


def test_three_profiles_are_architecturally_distinct():
    standard = MoonViTEncoder(
        VisionEncoderConfig(
            image_size=28, embed_dim=24, depth=1, num_heads=6
        )
    )
    hierarchical = HierarchicalMoonViTEncoder(
        HierarchicalVisionConfig(
            image_size=28,
            embed_dims=(24, 48),
            depths=(1, 1),
            num_heads=(6, 6),
        )
    )
    swin = SwinMoonViTEncoder(
        SwinVisionConfig(
            image_size=28,
            embed_dims=(24, 48),
            depths=(2, 1),
            num_heads=(6, 6),
            window_size=2,
        )
    )
    names = {type(standard).__name__, type(hierarchical).__name__, type(swin).__name__}
    assert names == {
        "MoonViTEncoder",
        "HierarchicalMoonViTEncoder",
        "SwinMoonViTEncoder",
    }
    assert not hasattr(standard, "pools") and not hasattr(standard, "mergers")
    assert hasattr(hierarchical, "pools") and not hasattr(hierarchical, "mergers")
    assert hasattr(swin, "mergers") and not hasattr(swin, "pools")


def test_maxvit_and_volo_were_not_copied_to_production_source():
    vision_files = {
        path.name.lower()
        for path in (Path(__file__).parents[2] / "src" / "vision").glob("*.py")
    }
    assert not any("maxvit" in name or "volo" in name for name in vision_files)


def test_required_profile_files_exist_and_encode_patch14(tmp_path):
    yaml = pytest.importorskip("yaml")
    root = Path(__file__).parents[2]
    required = [
        "vit_baseline_tiny.yaml",
        "moonvit_proxy_tiny.yaml",
        "moonvit_proxy_mini.yaml",
        "moonvit_hierarchical_tiny.yaml",
        "moonvit_swin_tiny.yaml",
    ]
    for filename in required:
        path = root / "config" / "vision" / filename
        assert path.is_file()
        values = yaml.safe_load(path.read_text(encoding="utf-8"))["model"]
        if filename != "vit_baseline_tiny.yaml":
            assert values["patch_size"] == 14
            assert values["norm_type"] == "rmsnorm"


def test_public_api_exports_all_three_backbones():
    import src

    assert src.MoonViTEncoder is MoonViTEncoder
    assert src.HierarchicalMoonViTEncoder is HierarchicalMoonViTEncoder
    assert src.SwinMoonViTEncoder is SwinMoonViTEncoder

