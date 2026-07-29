from pathlib import Path
from types import SimpleNamespace
import importlib

import pytest

from configuration import ConfigError, resolve_kimi_pipeline_profile
from data import build_dataloaders_from_yaml, load_data_config
from src import kimi_k3_cpu_tiny_config, load_model_config
from training import (
    load_training_config,
    train_kimi_from_yaml,
    validate_pipeline_compatibility,
)


ROOT = Path(__file__).resolve().parents[2]
PROFILE_ROOT = ROOT / "config/kimi_full_pipeline"
PROFILES = (
    "cpu_smoke",
    "low_gpu",
    "gpu_24gb",
    "gpu_48gb",
    "gpu_80gb",
    "canonical",
)


@pytest.mark.parametrize(
    "relative",
    [
        "config/data/synthetic_retrieval.yaml",
        "config/data/wikitext2.yaml",
        "config/data/fineweb_10bt_streaming.yaml",
        "config/data/fineweb_100bt_streaming.yaml",
        "config/data/fineweb_350bt_streaming.yaml",
        *[
            f"config/kimi_full_pipeline/{name}/data.yaml"
            for name in PROFILES
        ],
    ],
)
def test_public_data_profiles_parse_without_building_or_downloading(relative):
    config = load_data_config(ROOT / relative)
    assert config.max_seq_len > 0
    assert config.loader.batch_size > 0


@pytest.mark.parametrize(
    "relative,preset",
    [
        ("config/data/fineweb_10bt_streaming.yaml", "fineweb_10bt"),
        ("config/data/fineweb_100bt_streaming.yaml", "fineweb_100bt"),
        ("config/data/fineweb_350bt_streaming.yaml", "fineweb_350bt"),
    ],
)
def test_web_scale_data_yamls_require_streaming_and_document_caps(
    relative, preset
):
    config = load_data_config(ROOT / relative)
    assert config.dataset.preset_name == preset
    assert config.dataset.streaming
    assert config.dataset.max_tokenizer_documents is not None
    assert config.dataset.max_train_documents is not None


@pytest.mark.parametrize(
    "profile_name,preset",
    [
        ("low_gpu", "wikitext2"),
        ("gpu_24gb", "fineweb_10bt"),
        ("gpu_48gb", "fineweb_100bt"),
        ("gpu_80gb", "fineweb_350bt"),
        ("canonical", "fineweb_350bt"),
    ],
)
def test_profile_data_scale_increases_with_compute(profile_name, preset):
    profile = resolve_kimi_pipeline_profile(PROFILE_ROOT / profile_name)
    assert load_data_config(profile.data).dataset.preset_name == preset


@pytest.mark.parametrize("profile_name", PROFILES)
def test_public_model_profiles_build_typed_configs_without_weights(
    profile_name,
):
    profile = resolve_kimi_pipeline_profile(PROFILE_ROOT / profile_name)
    config = load_model_config(profile.model)
    assert config.d_model == config.backbone.d_model
    assert config.backbone.num_transformer_layers > 0


def test_cpu_yaml_reproduces_the_programmatic_tiny_architecture_exactly():
    assert load_model_config(
        PROFILE_ROOT / "cpu_smoke/model.yaml"
    ) == kimi_k3_cpu_tiny_config()


def test_synthetic_yaml_builds_cpu_loaders_without_training():
    bundle = build_dataloaders_from_yaml(
        PROFILE_ROOT / "cpu_smoke/data.yaml"
    )
    batch = next(iter(bundle.train_loader))
    assert batch["input_ids"].shape[0] == 4
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    rebuilt = bundle.train_loader_factory(16)
    assert next(iter(rebuilt))["input_ids"].shape[1] <= 16


@pytest.mark.parametrize("profile_name", PROFILES)
def test_public_training_profiles_parse_every_control_block(profile_name):
    profile = resolve_kimi_pipeline_profile(PROFILE_ROOT / profile_name)
    config = load_training_config(profile.training)
    assert config.training.epochs > 0
    assert config.loss.ignore_index < 0
    assert config.optimizer.adamw_lr > 0
    assert config.checkpoint.run_name


@pytest.mark.parametrize("profile_name", PROFILES)
def test_recommended_three_yaml_sets_are_cross_compatible(profile_name):
    profile = resolve_kimi_pipeline_profile(PROFILE_ROOT / profile_name)
    validate_pipeline_compatibility(
        load_training_config(profile.training),
        model_config=load_model_config(profile.model),
        data_config=load_data_config(profile.data),
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
        PROFILE_ROOT / "low_gpu/model.yaml"
    )
    data_config = load_data_config(
        PROFILE_ROOT / "low_gpu/data.yaml"
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
        PROFILE_ROOT / "low_gpu/training.yaml",
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
            PROFILE_ROOT / "gpu_48gb/model.yaml"
        )
    )
    data = SimpleNamespace(
        config=load_data_config(
            PROFILE_ROOT / "gpu_48gb/data.yaml"
        ),
        train_loader="static",
        val_loader=None,
        train_loader_factory="context-factory",
        tokenizer="tokenizer",
    )
    train_kimi_from_yaml(
        PROFILE_ROOT / "gpu_48gb/training.yaml",
        model=model,
        data=data,
    )
    assert captured["train_loader_factory"] == "context-factory"
    assert "train_loader" not in captured
