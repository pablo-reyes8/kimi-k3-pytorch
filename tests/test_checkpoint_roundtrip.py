import random
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from conftest import tiny_model
from training import build_adamw_optimizer
from training.checkpoints import load_checkpoint, save_checkpoint
from training.scheduler import WarmupCosineLR


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    epochs: int = 2


def trained_state():
    model = tiny_model()
    optimizer, _ = build_adamw_optimizer(model, learning_rate=1e-3)
    scheduler = WarmupCosineLR(optimizer, total_steps=4, warmup_steps=1)
    ids = torch.randint(1, 48, (2, 8))
    labels = torch.randint(1, 48, (2, 8))
    model(ids, labels=labels).loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    return model.eval(), optimizer, scheduler, ids


def test_full_checkpoint_roundtrip_restores_model_optimizer_scheduler_and_metadata(tmp_path):
    model, optimizer, scheduler, ids = trained_state()
    expected = model(ids).logits.detach().clone()
    path = save_checkpoint(
        tmp_path / "nested" / "state.pt",
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=2,
        global_step=7,
        model_config=model.config,
        training_config=TrainingConfig(),
        history={"loss": [3.0, 2.0]},
    )
    assert path.exists() and not path.with_suffix(".pt.tmp").exists()

    restored = tiny_model().eval()
    restored_optimizer, _ = build_adamw_optimizer(restored)
    restored_scheduler = WarmupCosineLR(
        restored_optimizer, total_steps=99, warmup_steps=0
    )
    state = load_checkpoint(
        path,
        restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )
    torch.testing.assert_close(expected, restored(ids).logits)
    assert state["epoch"] == 2 and state["global_step"] == 7
    assert state["history"] == {"loss": [3.0, 2.0]}
    assert state["model_config"]["d_model"] == model.config.d_model
    assert state["training_config"] == {"learning_rate": 1e-3, "epochs": 2}
    assert state["missing_keys"] == state["unexpected_keys"] == []
    assert restored_scheduler.step_num == scheduler.step_num


def test_checkpoint_rng_restoration_is_exact(tmp_path):
    model = tiny_model()
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    path = save_checkpoint(tmp_path / "rng.pt", model)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    load_checkpoint(path, model, restore_rng=True)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])


def test_non_strict_load_reports_missing_and_unexpected_keys(tmp_path):
    model = tiny_model()
    path = save_checkpoint(tmp_path / "state.pt", model)
    payload = torch.load(path, weights_only=False)
    removed_name = next(iter(payload["model_state_dict"]))
    payload["model_state_dict"].pop(removed_name)
    payload["model_state_dict"]["unexpected"] = torch.ones(1)
    torch.save(payload, path)
    state = load_checkpoint(path, tiny_model(), strict=False)
    assert removed_name in state["missing_keys"]
    assert "unexpected" in state["unexpected_keys"]


def test_strict_load_rejects_incompatible_model(tmp_path):
    path = save_checkpoint(tmp_path / "state.pt", tiny_model())
    with pytest.raises(RuntimeError):
        load_checkpoint(path, tiny_model(vocab_size=49), strict=True)


def test_map_location_cpu_and_plain_config_serialization(tmp_path):
    model = tiny_model()
    path = save_checkpoint(
        tmp_path / "state.pt",
        model,
        model_config={"name": "baseline"},
        training_config=None,
    )
    state = load_checkpoint(path, tiny_model(), map_location="cpu")
    assert state["model_config"] == {"name": "baseline"}
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
