import copy

import torch

from tests.hybrid_backbone.conftest import tiny_backbone


def test_every_active_parameter_and_input_receives_finite_nonzero_gradient():
    model = tiny_backbone(num_hybrid_groups=2).double()
    x = torch.randn(1, 5, 8, dtype=torch.float64, requires_grad=True)
    model(x).last_hidden_state.square().sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad), name


def test_backbone_and_explicit_layer_iteration_gradients_match():
    backbone = tiny_backbone().double()
    explicit = copy.deepcopy(backbone)
    x_backbone = torch.randn(
        1, 5, 8, dtype=torch.float64, requires_grad=True
    )
    x_explicit = x_backbone.detach().clone().requires_grad_()
    backbone(x_backbone).last_hidden_state.square().sum().backward()
    mask = torch.ones(1, 5, dtype=torch.bool)
    output = x_explicit
    for layer in explicit.layers:
        output = layer(output, mask).hidden_states
    explicit.final_norm(output).square().sum().backward()
    torch.testing.assert_close(
        x_backbone.grad, x_explicit.grad, rtol=2e-9, atol=2e-11
    )
    for (left_name, left), (right_name, right) in zip(
        backbone.named_parameters(), explicit.named_parameters()
    ):
        assert left_name == right_name
        torch.testing.assert_close(
            left.grad, right.grad, rtol=3e-8, atol=3e-10
        )


def test_small_backbone_gradcheck():
    model = tiny_backbone().double()
    x = torch.randn(1, 2, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value: model(value).last_hidden_state,
        (x,),
        fast_mode=True,
    )
