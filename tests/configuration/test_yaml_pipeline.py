from pathlib import Path
from types import SimpleNamespace
import importlib

import pytest

from configuration import ConfigError
from data import build_dataloaders_from_yaml, load_data_config
from src import kimi_k3_cpu_tiny_config, load_model_config
from training import (
    load_training_config,
    train_kimi_from_yaml,
    validate_pipeline_compatibility,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "relative",
    [
        "config/data/synthetic_cpu_smoke.yaml",
        "config/data/synthetic_retrieval.yaml",
        "config/data/tinystories.yaml",
        "config/data/tinystories_1024.yaml",
        "config/data/fineweb_edu_8192.yaml",
    ],
)
def test_public_data_profiles_parse_without_building_or_downloading(relative):
    config = load_data_config(ROOT / relative)
    assert config.max_seq_len > 0
    assert config.loader.batch_size > 0


@pytest.mark.parametrize(
    "relative",
    [
        "config/kimi_k3/cpu_tiny.yaml",
        "config/kimi_k3/t4_15gb.yaml",
        "config/kimi_k3/gpu_24gb.yaml",
        "config/kimi_k3/gpu_48gb.yaml",
        "config/kimi_k3/canonical.yaml",
    ],
)
def test_public_model_profiles_build_typed_configs_without_weights(relative):
    config = load_model_config(ROOT / relative)
    assert config.d_model == config.backbone.d_model
    assert config.backbone.num_transformer_layers > 0


def test_cpu_yaml_reproduces_the_programmatic_tiny_architecture_exactly():
    assert load_model_config(
        ROOT / "config/kimi_k3/cpu_tiny.yaml"
    ) == kimi_k3_cpu_tiny_config()


def test_synthetic_yaml_builds_cpu_loaders_without_training():
    bundle = build_dataloaders_from_yaml(
        ROOT / "config/data/synthetic_cpu_smoke.yaml"
    )
    batch = next(iter(bundle.train_loader))
    assert batch["input_ids"].shape[0] == 4
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    rebuilt = bundle.train_loader_factory(16)
    assert next(iter(rebuilt))["input_ids"].shape[1] <= 16


@pytest.mark.parametrize(
    "relative",
    [
        "config/training/cpu_yaml_smoke.yaml",
        "config/training/t4_15gb.yaml",
        "config/training/gpu_24gb.yaml",
        "config/training/gpu_48gb_pcc.yaml",
    ],
)
def test_public_training_profiles_parse_every_control_block(relative):
    config = load_training_config(ROOT / relative)
    assert config.training.epochs > 0
    assert config.loss.ignore_index < 0
    assert config.optimizer.adamw_lr > 0
    assert config.checkpoint.run_name


@pytest.mark.parametrize(
    "data_path,model_path,training_path",
    [
        (
            "config/data/synthetic_cpu_smoke.yaml",
            "config/kimi_k3/cpu_tiny.yaml",
            "config/training/cpu_yaml_smoke.yaml",
        ),
        (
            "config/data/tinystories.yaml",
            "config/kimi_k3/t4_15gb.yaml",
            "config/training/t4_15gb.yaml",
        ),
        (
            "config/data/tinystories_1024.yaml",
            "config/kimi_k3/gpu_24gb.yaml",
            "config/training/gpu_24gb.yaml",
        ),
        (
            "config/data/fineweb_edu_8192.yaml",
            "config/kimi_k3/gpu_48gb.yaml",
            "config/training/gpu_48gb_pcc.yaml",
        ),
    ],
)
def test_recommended_three_yaml_sets_are_cross_compatible(
    data_path, model_path, training_path
):
    validate_pipeline_compatibility(
        load_training_config(ROOT / training_path),
        model_config=load_model_config(ROOT / model_path),
        data_config=load_data_config(ROOT / data_path),
    )


def test_unknown_yaml_keys_fail_loudly(tmp_path):
    path = tmp_path / "bad_data.yaml"
    path.write_text(
        """
data:
  name: typo
  kind: synthetic_retrieval
  unexpected: true
  dataset:
    num_train_examples: 2
    num_val_examples: 1
    block_size: 8
  loader:
    batch_size: 1
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unexpected"):
        load_data_config(path)


def test_yaml_training_adapter_only_delegates_to_master(monkeypatch):
    captured = {}

    def fake_master(**kwargs):
        captured.update(kwargs)
        return {"delegated": True}

    trainer_module = importlib.import_module("training.train_kimi_k3")
    monkeypatch.setattr(trainer_module, "train_kimiK3", fake_master)
    model_config = load_model_config(
        ROOT / "config/kimi_k3/t4_15gb.yaml"
    )
    data_config = load_data_config(
        ROOT / "config/data/tinystories.yaml"
    )
    data = SimpleNamespace(
        config=data_config,
        train_loader="train-loader",
        val_loader="val-loader",
        train_loader_factory=lambda max_seq_len: max_seq_len,
        tokenizer="tokenizer",
    )
    model = SimpleNamespace(config=model_config)
    result = train_kimi_from_yaml(
        ROOT / "config/training/t4_15gb.yaml",
        model=model,
        data=data,
    )
    assert result == {"delegated": True}
    assert captured["model"] is model
    assert captured["train_loader"] == "train-loader"
    assert "train_loader_factory" not in captured
    assert captured["kimi_optimizer_config"].kind == "adamw"
    assert captured["loss_config"].mtp_loss_weight == pytest.approx(0.1)


def test_yaml_pcc_delegates_context_loader_factory(monkeypatch):
    captured = {}
    trainer_module = importlib.import_module("training.train_kimi_k3")
    monkeypatch.setattr(
        trainer_module,
        "train_kimiK3",
        lambda **kwargs: captured.update(kwargs) or {},
    )
    model = SimpleNamespace(
        config=load_model_config(
            ROOT / "config/kimi_k3/gpu_48gb.yaml"
        )
    )
    data = SimpleNamespace(
        config=load_data_config(
            ROOT / "config/data/fineweb_edu_8192.yaml"
        ),
        train_loader="static",
        val_loader=None,
        train_loader_factory="context-factory",
        tokenizer="tokenizer",
    )
    train_kimi_from_yaml(
        ROOT / "config/training/gpu_48gb_pcc.yaml",
        model=model,
        data=data,
    )
    assert captured["train_loader_factory"] == "context-factory"
    assert "train_loader" not in captured
