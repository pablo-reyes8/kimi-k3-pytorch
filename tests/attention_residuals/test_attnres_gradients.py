import copy

import pytest
import torch

from src.attention_residuals import AttentionResidualSite
from tests.attention_residuals.conftest import (
    activate_depth_queries,
    attnres_backbone,
)


def test_site_gradcheck_and_key_value_gradient_paths():
    site = AttentionResidualSite(3).double()
    with torch.no_grad():
        site.pseudo_query.copy_(torch.tensor([0.5, -1.0, 2.0]))
    sources = torch.randn(
        1, 2, 3, 3, dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(
        lambda value: site(value).mixed_state,
        (sources,),
        fast_mode=True,
    )
    site(sources).mixed_state.square().sum().backward()
    assert torch.isfinite(sources.grad).all()
    assert torch.count_nonzero(sources.grad)
    assert torch.count_nonzero(site.pseudo_query.grad)
    assert torch.count_nonzero(site.key_norm.weight.grad)


@pytest.mark.parametrize(
    "depth_mode,backend",
    [("full", "eager"), ("block", "eager"), ("block", "two_phase")],
)
def test_all_backbone_parameters_have_valid_gradient_with_mathematical_exception(
    depth_mode, backend
):
    model = attnres_backbone(
        depth_mode=depth_mode, backend=backend
    ).double()
    activate_depth_queries(model)
    x = torch.randn(1, 5, 8, dtype=torch.float64, requires_grad=True)
    model(x).last_hidden_state.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        first_site_parameter = (
            "groups.0.layers.0.pre_attention_attnres" in name
        )
        if not first_site_parameter:
            assert torch.count_nonzero(parameter.grad), name
    first = model.layers[0].pre_attention_attnres
    assert torch.count_nonzero(first.pseudo_query.grad) == 0
    assert torch.count_nonzero(first.key_norm.weight.grad) == 0


def gradient_snapshot(model, x):
    output = model(x).last_hidden_state
    output.square().sum().backward()
    return x.grad.clone(), {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
    }


def test_full_and_block_size_one_input_and_parameter_gradients_match():
    full = attnres_backbone(depth_mode="full").double()
    activate_depth_queries(full)
    block = attnres_backbone(depth_mode="block", block_size=1).double()
    block.load_state_dict(full.state_dict())
    x_full = torch.randn(1, 4, 8, dtype=torch.float64, requires_grad=True)
    x_block = x_full.detach().clone().requires_grad_()
    full_grad = gradient_snapshot(full, x_full)
    block_grad = gradient_snapshot(block, x_block)
    torch.testing.assert_close(
        full_grad[0], block_grad[0], rtol=2e-11, atol=2e-12
    )
    assert full_grad[1].keys() == block_grad[1].keys()
    for name in full_grad[1]:
        torch.testing.assert_close(
            full_grad[1][name], block_grad[1][name],
            rtol=3e-10, atol=3e-12,
        )


def test_block_eager_and_two_phase_input_and_parameter_gradients_match():
    eager = attnres_backbone(
        depth_mode="block", backend="eager", block_size=4
    ).double()
    activate_depth_queries(eager)
    two = attnres_backbone(
        depth_mode="block", backend="two_phase", block_size=4
    ).double()
    two.load_state_dict(eager.state_dict())
    x_eager = torch.randn(1, 4, 8, dtype=torch.float64, requires_grad=True)
    x_two = x_eager.detach().clone().requires_grad_()
    eager_grad = gradient_snapshot(eager, x_eager)
    two_grad = gradient_snapshot(two, x_two)
    torch.testing.assert_close(
        eager_grad[0], two_grad[0], rtol=3e-10, atol=3e-12
    )
    for name in eager_grad[1]:
        torch.testing.assert_close(
            eager_grad[1][name], two_grad[1][name],
            rtol=5e-9, atol=5e-11,
        )


def test_small_full_backbone_gradcheck():
    model = attnres_backbone(depth_mode="full").double()
    activate_depth_queries(model)
    x = torch.randn(1, 2, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value: model(value).last_hidden_state,
        (x,),
        fast_mode=True,
    )
