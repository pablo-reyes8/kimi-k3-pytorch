import copy

import torch

from src.mla import GatedMLA, manual_causal_attention
from tests.mla.conftest import random_qkv, tiny_config, tiny_mla


def test_all_architectural_parameters_receive_finite_nonzero_gradients():
    model = tiny_mla(attention_backend="manual").double()
    x = torch.randn(2, 5, 12, dtype=torch.float64, requires_grad=True)
    model(x).hidden_states.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad) > 0
    expected = {
        "projections.query.weight",
        "projections.latent_kv.compression.weight",
        "projections.latent_kv.key_up.weight",
        "projections.latent_kv.value_up.weight",
        "output_gate.gate_proj.weight",
        "output_gate.output_proj.weight",
    }
    parameters = dict(model.named_parameters())
    assert expected <= parameters.keys()
    for name in expected:
        gradient = parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient) > 0, name


def test_manual_and_sdpa_input_and_parameter_gradients_match():
    manual = tiny_mla(attention_backend="manual").double()
    sdpa = copy.deepcopy(manual)
    object.__setattr__(
        sdpa, "config",
        tiny_config(attention_backend="sdpa"),
    )
    x_manual = torch.randn(1, 5, 12, dtype=torch.float64, requires_grad=True)
    x_sdpa = x_manual.detach().clone().requires_grad_()
    manual(x_manual).hidden_states.square().sum().backward()
    sdpa(x_sdpa).hidden_states.square().sum().backward()
    torch.testing.assert_close(x_manual.grad, x_sdpa.grad, rtol=2e-10, atol=2e-12)
    for (left_name, left), (right_name, right) in zip(
        manual.named_parameters(), sdpa.named_parameters()
    ):
        assert left_name == right_name
        torch.testing.assert_close(
            left.grad, right.grad, rtol=3e-10, atol=3e-12
        )


def test_manual_attention_core_gradcheck():
    q, k, v = random_qkv(
        batch=1, query_tokens=2, heads=1, query_dim=1, value_dim=1
    )
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    assert torch.autograd.gradcheck(
        lambda a, b, c: manual_causal_attention(a, b, c),
        (q, k, v),
        fast_mode=True,
    )


def test_latent_reconstruction_gradcheck():
    model = tiny_mla().double()
    latent = torch.randn(1, 2, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value: torch.cat(model.projections.reconstruct_kv(value), dim=-1),
        (latent,),
        fast_mode=True,
    )


def test_small_full_gated_mla_gradcheck():
    model = GatedMLA(
        tiny_config(
            d_model=2,
            num_heads=1,
            q_head_dim=1,
            v_head_dim=2,
            kv_latent_dim=1,
            attention_backend="manual",
        )
    ).double()
    x = torch.randn(1, 2, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value: model(value).hidden_states,
        (x,),
        fast_mode=True,
    )
