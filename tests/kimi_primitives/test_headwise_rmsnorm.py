import copy

import pytest
import torch

from src.kimi_primitives import HeadwiseRMSNorm


def manual_norm(module, x):
    accumulation = (
        x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    )
    normalized = accumulation * torch.rsqrt(
        accumulation.square().mean(-1, keepdim=True) + module.eps
    )
    normalized = normalized.to(x.dtype)
    if module.weight is not None:
        weight = module.weight.to(x.dtype)
        normalized = normalized * (
            weight[None, None]
            if module.per_head_affine
            else weight[None, None, None]
        )
    return normalized


@pytest.mark.parametrize(
    "affine,per_head", [(True, True), (True, False), (False, True)]
)
def test_headwise_norm_matches_manual_formula(affine, per_head):
    module = HeadwiseRMSNorm(
        3, 5, eps=3e-6, elementwise_affine=affine,
        per_head_affine=per_head
    )
    if module.weight is not None:
        with torch.no_grad():
            module.weight.copy_(
                torch.linspace(0.5, 1.5, module.weight.numel()).reshape(
                    module.weight.shape
                )
            )
    x = torch.randn(2, 4, 3, 5)
    torch.testing.assert_close(module(x), manual_norm(module, x))


def test_unit_affine_produces_unit_rms_per_batch_token_head():
    module = HeadwiseRMSNorm(4, 16, eps=1e-12)
    output = module(torch.randn(3, 7, 4, 16))
    rms = output.square().mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), rtol=2e-5, atol=2e-5)


def test_headwise_norm_does_not_mix_heads():
    module = HeadwiseRMSNorm(3, 8)
    x = torch.randn(2, 4, 3, 8)
    changed = x.clone()
    changed[:, :, 1] *= 1e5
    first, second = module(x), module(changed)
    torch.testing.assert_close(first[:, :, 0], second[:, :, 0], rtol=0, atol=0)
    torch.testing.assert_close(first[:, :, 2], second[:, :, 2], rtol=0, atol=0)


def test_headwise_norm_does_not_mix_tokens_or_batches():
    module = HeadwiseRMSNorm(2, 4)
    x = torch.randn(2, 3, 2, 4)
    changed = x.clone()
    changed[1, 2] += 100
    first, second = module(x), module(changed)
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    torch.testing.assert_close(first[1, :2], second[1, :2], rtol=0, atol=0)


@pytest.mark.parametrize("scale", [0.1, 10.0, 1e4])
def test_positive_scale_invariance(scale):
    module = HeadwiseRMSNorm(3, 8, eps=1e-12)
    x = torch.randn(2, 4, 3, 8)
    torch.testing.assert_close(
        module(scale * x), module(x), rtol=2e-4, atol=2e-4
    )


def test_zero_input_is_exactly_zero_and_finite():
    output = HeadwiseRMSNorm(3, 8)(torch.zeros(2, 4, 3, 8))
    assert torch.count_nonzero(output) == 0
    assert torch.isfinite(output).all()


def test_per_head_affine_scales_heads_and_channels_independently():
    module = HeadwiseRMSNorm(2, 3, eps=1e-12)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    x = torch.ones(1, 1, 2, 3)
    torch.testing.assert_close(
        module(x), module.weight.reshape(1, 1, 2, 3), rtol=1e-5, atol=1e-5
    )


def test_affine_parameter_shapes_are_explicit():
    assert HeadwiseRMSNorm(3, 5).weight.shape == (3, 5)
    assert HeadwiseRMSNorm(3, 5, per_head_affine=False).weight.shape == (5,)
    assert HeadwiseRMSNorm(3, 5, elementwise_affine=False).weight is None


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: HeadwiseRMSNorm(0, 4),
        lambda: HeadwiseRMSNorm(2, 0),
        lambda: HeadwiseRMSNorm(2, 4, eps=0),
    ],
)
def test_invalid_configuration_rejected(constructor):
    with pytest.raises(ValueError):
        constructor()


@pytest.mark.parametrize(
    "shape", [(2, 3, 8), (2, 3, 4, 7), (2, 3, 5, 8), (2, 3, 4, 8, 1)]
)
def test_ambiguous_or_wrong_shapes_rejected(shape):
    with pytest.raises(ValueError):
        HeadwiseRMSNorm(4, 8)(torch.randn(shape))


def test_gradcheck_input_and_per_head_affine():
    module = HeadwiseRMSNorm(2, 3, eps=1e-6).double()
    x = torch.randn(1, 2, 2, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(module, (x,), fast_mode=True)


def test_all_gradients_are_nonzero_and_finite():
    module = HeadwiseRMSNorm(3, 5)
    x = torch.randn(2, 4, 3, 5, requires_grad=True)
    target = torch.randn_like(x)
    (module(x) * target).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert module.weight.grad is not None and module.weight.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all() and torch.isfinite(module.weight.grad).all()


def test_bfloat16_uses_fp32_reduction_and_restores_dtype():
    module = HeadwiseRMSNorm(3, 16).to(torch.bfloat16)
    x = (torch.randn(2, 4, 3, 16) * 1e3).to(torch.bfloat16)
    output = module(x)
    expected = manual_norm(module, x)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output.float()).all()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_state_dict_roundtrip_exact():
    module = HeadwiseRMSNorm(3, 5)
    with torch.no_grad():
        module.weight.normal_()
    clone = copy.deepcopy(module)
    clone.load_state_dict(module.state_dict())
    x = torch.randn(2, 4, 3, 5)
    torch.testing.assert_close(module(x), clone(x), rtol=0, atol=0)
