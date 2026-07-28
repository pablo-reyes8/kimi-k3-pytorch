import copy

import pytest
import torch
import torch.nn as nn

from src.vision import (
    MoonViTEncoder,
    SpatialTokenPixelShuffle,
    VisionEncoderConfig,
    VisionProjector,
)
from src.transformer_modules.rms_norm import RMSNorm

from tests.MoonViT.helpers import tiny_moonvit


@pytest.mark.parametrize(
    "shape,expected_grid",
    [((2, 3, 28, 42), (2, 3)), ((1, 3, 56, 28), (4, 2))],
)
def test_moonvit_variable_rectangular_resolution(shape, expected_grid):
    model = tiny_moonvit()
    output = model(torch.randn(shape))
    assert output.grid_size == expected_grid
    assert output.last_hidden_state.shape == (
        shape[0], expected_grid[0] * expected_grid[1], 24
    )


def test_moonvit_output_contract_and_diagnostics():
    model = tiny_moonvit(depth=3)
    output = model(
        torch.randn(2, 3, 28, 42),
        output_hidden_states=True,
        output_attentions=True,
    )
    assert len(output.hidden_states) == 4
    assert len(output.attentions) == 3
    assert output.hidden_states[-1].data_ptr() == output.last_hidden_state.data_ptr()
    assert all(weights.shape == (2, 6, 6, 6) for weights in output.attentions)
    assert all(
        torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)))
        for weights in output.attentions
    )


def test_moonvit_defaults_match_reported_structural_policy():
    config = VisionEncoderConfig()
    assert config.patch_size == 14
    assert config.num_heads == 12
    assert config.norm_type == "rmsnorm"
    assert config.use_cls_token is False
    assert not any(
        (config.patch_bias, config.qkv_bias, config.proj_bias, config.mlp_bias)
    )


def test_moonvit_proxy_has_no_classifier_or_bias_parameters():
    model = tiny_moonvit()
    assert not any("head" in name or "classifier" in name for name, _ in model.named_modules())
    assert not any(name.endswith(".bias") for name, _ in model.named_parameters())
    assert all(
        isinstance(block.norm1, RMSNorm) and isinstance(block.norm2, RMSNorm)
        for block in model.blocks
    )


def test_moonvit_cls_is_optional_and_prepended_only_when_requested():
    without_cls = tiny_moonvit(use_cls_token=False)
    with_cls = tiny_moonvit(use_cls_token=True)
    image = torch.randn(2, 3, 28, 42)
    assert without_cls(image).last_hidden_state.shape[1] == 6
    assert with_cls(image).last_hidden_state.shape[1] == 7
    assert without_cls.cls_token is None
    assert with_cls.cls_token.shape == (1, 1, 24)


def test_moonvit_none_position_mode_has_no_position_parameters():
    model = tiny_moonvit(position_embedding_type="none")
    assert model.position_embedding is None
    assert not any("position" in name for name, _ in model.named_parameters())


def test_moonvit_padding_mask_blocks_masked_key_in_every_layer():
    model = tiny_moonvit(depth=2)
    mask = torch.tensor([[True, True, False, True, False, True]])
    output = model(
        torch.randn(1, 3, 28, 42),
        mask,
        output_attentions=True,
    )
    for attention in output.attentions:
        assert torch.count_nonzero(attention[..., [2, 4]]) == 0


def test_moonvit_batch_independence():
    model = tiny_moonvit().eval()
    first, second = torch.randn(1, 3, 28, 42), torch.randn(1, 3, 28, 42)
    together = model(torch.cat((first, second))).last_hidden_state
    torch.testing.assert_close(
        together[:1], model(first).last_hidden_state, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        together[1:], model(second).last_hidden_state, rtol=1e-5, atol=1e-6
    )


def test_moonvit_causality_is_not_accidentally_applied():
    model = tiny_moonvit(depth=1, position_embedding_type="none").eval()
    image = torch.randn(1, 3, 28, 42)
    changed = image.clone()
    changed[:, :, 14:28, 28:42] += 20
    first = model(image).last_hidden_state[:, 0]
    second = model(changed).last_hidden_state[:, 0]
    assert not torch.allclose(first, second)


def test_moonvit_backward_reaches_input_and_every_parameter():
    model = tiny_moonvit()
    image = torch.randn(2, 3, 28, 42, requires_grad=True)
    model(image).last_hidden_state.square().mean().backward()
    assert image.grad is not None and image.grad.abs().sum() > 0
    missing = [name for name, value in model.named_parameters() if value.grad is None]
    assert missing == []


def test_moonvit_state_dict_roundtrip_exact():
    model = tiny_moonvit().eval()
    clone = copy.deepcopy(model).eval()
    clone.load_state_dict(model.state_dict())
    image = torch.randn(2, 3, 28, 42)
    torch.testing.assert_close(
        model(image).last_hidden_state,
        clone(image).last_hidden_state,
        rtol=0,
        atol=0,
    )


def test_moonvit_complete_visual_path_contract():
    encoder = tiny_moonvit(image_size=56)
    packer = SpatialTokenPixelShuffle()
    projector = VisionProjector(24 * 4, 64, 32)
    encoded = encoder(torch.randn(2, 3, 56, 56))
    packed = packer(encoded.last_hidden_state, encoded.grid_size)
    projected = projector(packed.last_hidden_state)
    assert packed.grid_size == (2, 2)
    assert projected.shape == (2, 4, 32)
    projected.square().mean().backward()
    assert all(parameter.grad is not None for parameter in encoder.parameters())
    assert all(parameter.grad is not None for parameter in projector.parameters())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image_size": 30},
        {"embed_dim": 25},
        {"depth": 0},
        {"position_embedding_type": "rope"},
        {"drop_path_rate": 1},
        {"initializer_std": 0},
    ],
)
def test_moonvit_config_validation(kwargs):
    with pytest.raises(ValueError):
        VisionEncoderConfig(**kwargs)


def test_moonvit_validates_masks_and_image_divisibility():
    model = tiny_moonvit()
    with pytest.raises(ValueError, match="divisible"):
        model(torch.randn(1, 3, 29, 42))
    with pytest.raises(ValueError, match="padding_mask"):
        model(torch.randn(1, 3, 28, 42), torch.ones(1, 5, dtype=torch.bool))
    with pytest.raises(TypeError):
        model(torch.randn(1, 3, 28, 42), torch.ones(1, 6))


def test_moonvit_drop_path_schedule_is_monotonic():
    model = tiny_moonvit(depth=4, drop_path_rate=0.4)
    rates = [block.drop_path1.drop_prob for block in model.blocks]
    torch.testing.assert_close(torch.tensor(rates), torch.linspace(0, 0.4, 4))


def test_moonvit_parameter_initialization_is_non_degenerate():
    model = tiny_moonvit(depth=4)
    linear_weights = [
        module.weight.detach().flatten()
        for module in model.modules()
        if isinstance(module, nn.Linear)
    ]
    combined = torch.cat(linear_weights)
    assert abs(combined.mean().item()) < 0.01
    assert 0.01 < combined.std().item() < 0.03


def test_moonvit_bfloat16_forward_backward():
    model = tiny_moonvit().to(torch.bfloat16)
    image = torch.randn(2, 3, 28, 42, dtype=torch.bfloat16, requires_grad=True)
    output = model(image).last_hidden_state
    assert output.dtype == torch.bfloat16
    output.float().square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
