import pytest
import torch

from training.ema import EMA, ema_health, unwrap_model


def model():
    return torch.nn.Sequential(
        torch.nn.Linear(4, 3),
        torch.nn.LayerNorm(3),
        torch.nn.Linear(3, 2),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"decay": -0.1},
        {"decay": 1.0},
        {"update_after_step": -1},
        {"update_every": 0},
    ],
)
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        EMA(model(), **kwargs)


def test_initial_shadow_is_fp32_clone_of_trainable_parameters():
    network = model().to(torch.bfloat16)
    ema = EMA(network)
    named = dict(network.named_parameters())
    assert len(ema) == len(named)
    for name, shadow in ema.shadow.items():
        assert shadow.dtype == torch.float32
        torch.testing.assert_close(shadow, named[name].detach().float())
        assert shadow.data_ptr() != named[name].data_ptr()


def test_frozen_and_excluded_parameters_are_not_tracked():
    network = model()
    network[0].bias.requires_grad_(False)
    ema = EMA(network, exclude_names=("2.",))
    assert "0.bias" not in ema.shadow
    assert all(not name.startswith("2.") for name in ema.shadow)


def test_exact_update_equation_without_warmup_decay():
    network = model()
    ema = EMA(network, decay=0.75, use_num_updates=False)
    before = {name: value.clone() for name, value in ema.shadow.items()}
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.add_(2)
    assert ema.update(network)
    named = dict(network.named_parameters())
    for name, shadow in ema.shadow.items():
        expected = before[name] * 0.75 + named[name].float() * 0.25
        torch.testing.assert_close(shadow, expected)
    assert ema.num_updates == 1


def test_update_schedule_skips_expected_steps():
    network = model()
    ema = EMA(network, update_after_step=2, update_every=3)
    assert ema.update(network, step=1) is False
    assert ema.update(network, step=2) is True
    assert ema.update(network, step=3) is False
    assert ema.update(network, step=5) is True
    assert ema.num_updates == 2 and ema.total_steps_seen == 5


def test_store_copy_restore_and_context_manager_are_lossless():
    network = model()
    ema = EMA(network, use_num_updates=False)
    originals = {name: p.detach().clone() for name, p in network.named_parameters()}
    with torch.no_grad():
        for shadow in ema.shadow.values():
            shadow.add_(10)
    with ema.average_parameters(network):
        for name, parameter in network.named_parameters():
            torch.testing.assert_close(parameter, ema.shadow[name].to(parameter.dtype))
    for name, parameter in network.named_parameters():
        torch.testing.assert_close(parameter, originals[name])
    assert ema.backup == {}


def test_reinit_to_and_state_roundtrip():
    network = model()
    ema = EMA(network, decay=0.8)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.add_(1)
    ema.reinit_from_model(network)
    ema.to("cpu")
    for name, parameter in network.named_parameters():
        torch.testing.assert_close(ema.shadow[name], parameter.float())
    state = ema.state_dict()
    restored = EMA(network, decay=0.1)
    restored.load_state_dict(state, strict=True)
    assert restored.decay == 0.8
    for name in ema.shadow:
        torch.testing.assert_close(restored.shadow[name], ema.shadow[name])


def test_strict_state_mismatch_rejected():
    network = model()
    ema = EMA(network)
    state = ema.state_dict()
    state["shadow"].pop(next(iter(state["shadow"])))
    with pytest.raises(RuntimeError, match="mismatch"):
        ema.load_state_dict(state, strict=True)


def test_health_statuses_cover_ok_empty_nonfinite_and_zero_model():
    network = model()
    ema = EMA(network)
    ok, status, difference = ema_health(ema, network)
    assert ok and status == "ok" and difference == pytest.approx(0)
    empty = EMA(network, exclude_names=("",))
    assert ema_health(empty, network)[:2] == (False, "empty_ema")
    first = next(iter(ema.shadow))
    ema.shadow[first].view(-1)[0] = float("nan")
    assert ema_health(ema, network)[:2] == (False, "nan_or_inf_in_ema")
    zero_network = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        zero_network.weight.zero_()
    assert ema_health(EMA(zero_network), zero_network)[:2] == (
        False,
        "model_zero_norm",
    )


def test_unwrap_model_supports_module_wrappers():
    network = model()

    class Wrapper:
        module = network

    assert unwrap_model(network) is network
    assert unwrap_model(Wrapper()) is network
