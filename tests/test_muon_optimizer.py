import pytest
import torch

from src import BaselineCausalLM, BaselineCausalLMConfig
from training.muon_optimizer import (
    HybridMuonAdamW,
    Muon,
    build_muon_adamw_optimizer,
    build_muon_adamw_parameter_groups,
    should_use_adamw_no_decay,
    should_use_muon,
    zeropower_via_newtonschulz5,
)


@pytest.mark.parametrize("shape", [(4, 4), (3, 7), (7, 3)])
def test_newton_schulz_shape_dtype_finiteness_and_scale_invariance(shape):
    torch.manual_seed(2)
    matrix = torch.randn(*shape)
    first = zeropower_via_newtonschulz5(matrix)
    second = zeropower_via_newtonschulz5(matrix * 10)
    assert first.shape == matrix.shape and first.dtype == matrix.dtype
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second, atol=2e-5, rtol=2e-5)


def test_newton_schulz_zero_and_bfloat16():
    assert torch.count_nonzero(zeropower_via_newtonschulz5(torch.zeros(3, 5))) == 0
    output = zeropower_via_newtonschulz5(torch.randn(3, 5).bfloat16())
    assert output.dtype == torch.bfloat16 and torch.isfinite(output.float()).all()


@pytest.mark.parametrize(
    "matrix,kwargs,error",
    [
        (torch.randn(3), {}, ValueError),
        (torch.randn(2, 2), {"steps": 0}, ValueError),
        (torch.randn(2, 2), {"eps": 0}, ValueError),
        (torch.tensor([[float("nan")]]), {}, FloatingPointError),
    ],
)
def test_newton_schulz_invalid_inputs_rejected(matrix, kwargs, error):
    with pytest.raises(error):
        zeropower_via_newtonschulz5(matrix, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": 0},
        {"momentum": -0.1},
        {"momentum": 1.0},
        {"weight_decay": -0.1},
        {"ns_steps": 0},
        {"eps": 0},
    ],
)
def test_muon_invalid_hyperparameters_rejected(kwargs):
    with pytest.raises(ValueError):
        Muon([torch.nn.Parameter(torch.ones(2, 2))], **kwargs)


def test_muon_rejects_non_matrix_parameters_and_nonfinite_gradients():
    with pytest.raises(ValueError, match="2D"):
        Muon([torch.nn.Parameter(torch.ones(3))])
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = Muon([parameter])
    parameter.grad = torch.full_like(parameter, float("inf"))
    with pytest.raises(FloatingPointError):
        optimizer.step()


def test_muon_step_matches_explicit_first_step_without_momentum():
    parameter = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    gradient = torch.tensor([[0.3, -0.2], [0.1, 0.4]])
    parameter.grad = gradient.clone()
    expected = parameter.detach() - 0.1 * zeropower_via_newtonschulz5(gradient)
    optimizer = Muon([parameter], lr=0.1, momentum=0.0, nesterov=False)
    optimizer.step()
    torch.testing.assert_close(parameter, expected)
    torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], gradient)


def tiny_model():
    return BaselineCausalLM(
        BaselineCausalLMConfig(
            vocab_size=32,
            d_model=16,
            n_layers=1,
            n_heads=2,
            mlp_hidden_dim=32,
            max_seq_len=8,
            pad_token_id=0,
        )
    )


def test_parameter_policy_routes_expected_types():
    network = tiny_model()
    named = dict(network.named_parameters())
    assert should_use_muon(
        "backbone.layers.0.attention.q_proj.weight",
        named["backbone.layers.0.attention.q_proj.weight"],
        network.backbone.layers[0].attention.q_proj,
    )
    assert not should_use_muon(
        "embedding.token_embedding.weight",
        named["embedding.token_embedding.weight"],
        network.embedding.token_embedding,
    )
    assert should_use_adamw_no_decay(
        "final_norm.weight", named["final_norm.weight"], network.final_norm
    )


def test_hybrid_groups_cover_every_parameter_once_with_expected_metadata():
    network = tiny_model()
    muon, adamw, metadata = build_muon_adamw_parameter_groups(network)
    assigned = [p for group in muon + adamw for p in group["params"]]
    trainable = [p for p in network.parameters() if p.requires_grad]
    assert {id(p) for p in assigned} == {id(p) for p in trainable}
    assert len(assigned) == len({id(p) for p in assigned})
    assert all(p.ndim == 2 for p in muon[0]["params"])
    assert metadata["total_trainable_params"] == sum(p.numel() for p in trainable)
    assert "embedding.token_embedding.weight" in metadata["adamw_no_decay_names"]


def test_hybrid_optimizer_step_lr_control_and_state_roundtrip():
    network = tiny_model()
    optimizer, metadata = build_muon_adamw_optimizer(
        network, learning_rate=1e-3, muon_lr=1e-2
    )
    assert isinstance(optimizer, HybridMuonAdamW)
    assert metadata["num_muon_tensors"] > 0
    ids = torch.randint(1, 32, (2, 8))
    network(ids, labels=torch.randint(1, 32, (2, 8))).loss.backward()
    before = {name: p.detach().clone() for name, p in network.named_parameters()}
    optimizer.step()
    assert any(not torch.equal(before[name], p) for name, p in network.named_parameters())
    optimizer.set_lr(2e-3, muon_lr=2e-2)
    assert all(group["lr"] == 2e-2 for group in optimizer.muon.param_groups)
    assert all(group["lr"] == 2e-3 for group in optimizer.adamw.param_groups)
    state = optimizer.state_dict()
    restored, _ = build_muon_adamw_optimizer(network)
    restored.load_state_dict(state)
    assert restored.metadata == optimizer.metadata
