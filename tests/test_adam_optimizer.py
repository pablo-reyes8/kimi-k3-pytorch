import pytest
import torch

from src import BaselineCausalLM, BaselineCausalLMConfig
from training.adam_optimizer import (
    build_adamw_optimizer,
    build_adamw_parameter_groups,
)


def model():
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


def test_groups_cover_every_trainable_parameter_exactly_once():
    network = model()
    groups, info = build_adamw_parameter_groups(network, weight_decay=0.1)
    grouped = [parameter for group in groups for parameter in group["params"]]
    trainable = [parameter for parameter in network.parameters() if parameter.requires_grad]
    assert {id(p) for p in grouped} == {id(p) for p in trainable}
    assert len(grouped) == len({id(p) for p in grouped})
    assert info["num_decay_params"] + info["num_no_decay_params"] == sum(
        p.numel() for p in trainable
    )


def test_norm_bias_embedding_and_tied_head_are_no_decay():
    _, info = build_adamw_parameter_groups(model())
    no_decay = set(info["no_decay_names"])
    assert "embedding.token_embedding.weight" in no_decay
    assert all(name in no_decay for name in no_decay if name.endswith(".bias"))
    assert all("norm" not in name or name in no_decay for name in no_decay | set(info["decay_names"]))


def test_projection_matrices_receive_decay():
    _, info = build_adamw_parameter_groups(model())
    decay = set(info["decay_names"])
    assert "backbone.layers.0.attention.q_proj.weight" in decay
    assert "backbone.layers.0.mlp.gate_proj.weight" in decay


def test_frozen_parameter_is_excluded():
    network = model()
    network.backbone.layers[0].attention.q_proj.weight.requires_grad_(False)
    groups, _ = build_adamw_parameter_groups(network)
    assert id(network.backbone.layers[0].attention.q_proj.weight) not in {
        id(p) for group in groups for p in group["params"]
    }


@pytest.mark.parametrize("weight_decay", [-0.1, -1.0])
def test_negative_weight_decay_rejected(weight_decay):
    with pytest.raises(ValueError):
        build_adamw_parameter_groups(model(), weight_decay)


def test_optimizer_hyperparameters_and_actual_update():
    network = model()
    optimizer, info = build_adamw_optimizer(
        network,
        learning_rate=2e-3,
        weight_decay=0.1,
        betas=(0.8, 0.9),
        eps=1e-7,
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    assert all(group["lr"] == 2e-3 for group in optimizer.param_groups)
    before = network.backbone.layers[0].attention.q_proj.weight.detach().clone()
    ids = torch.randint(1, 32, (2, 8))
    network(ids, labels=torch.randint(1, 32, (2, 8))).loss.backward()
    optimizer.step()
    assert not torch.equal(
        before, network.backbone.layers[0].attention.q_proj.weight.detach()
    )
    assert info["num_decay_params"] > 0 and info["num_no_decay_params"] > 0


def test_nonpositive_learning_rate_rejected():
    with pytest.raises(ValueError):
        build_adamw_optimizer(model(), learning_rate=0)
