"""YAML control plane for the high-level Kimi K3 trainer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from configuration.yaml_utils import (
    ConfigError,
    dataclass_kwargs,
    expect_mapping,
    load_yaml_mapping,
    reject_unknown_keys,
)

from .config import (
    CheckpointConfig,
    PretrainingLossConfig,
    PredictionConfig,
    SchedulerConfig,
    TrainingConfig,
)
from .context_curriculum import (
    ContextCurriculumConfig,
    ContextStage,
)
from .diagnostics import DiagnosticsConfig
from .distributed import DistributedConfig, distributed_config_from_dict
from .distributed.parallelize import validate_distributed_model_config
from .optimizer import KimiOptimizerConfig


@dataclass(frozen=True)
class TrainingRuntimeConfig:
    device: str = "auto"
    total_steps: int | None = None
    verbose: bool = True
    use_ema: bool = False
    ema_decay: float = 0.999
    eval_use_ema: bool = False

    def __post_init__(self) -> None:
        if not self.device:
            raise ValueError("runtime.device must not be empty")
        if self.total_steps is not None and self.total_steps <= 0:
            raise ValueError("runtime.total_steps must be None or positive")
        if not 0 < self.ema_decay < 1:
            raise ValueError("runtime.ema_decay must be in (0, 1)")
        if self.eval_use_ema and not self.use_ema:
            raise ValueError("runtime.eval_use_ema requires use_ema=true")


@dataclass(frozen=True)
class TrainingYamlConfig:
    runtime: TrainingRuntimeConfig
    training: TrainingConfig
    loss: PretrainingLossConfig
    optimizer: KimiOptimizerConfig
    scheduler: SchedulerConfig
    diagnostics: DiagnosticsConfig
    context_curriculum: ContextCurriculumConfig
    checkpoint: CheckpointConfig
    prediction: PredictionConfig
    distributed: DistributedConfig
    source_path: Path


def _section(root: dict, name: str, cls, *, required: bool = True):
    values = expect_mapping(
        root, name, path="root", required=required
    )
    return cls(**dataclass_kwargs(values, cls, path=name))


def load_training_config(path: str | Path) -> TrainingYamlConfig:
    """Parse all trainer knobs without creating optimizer/model state."""
    source, root = load_yaml_mapping(path)
    runtime = _section(
        root, "runtime", TrainingRuntimeConfig, required=False
    )
    training = _section(root, "training", TrainingConfig)
    loss = _section(
        root, "loss", PretrainingLossConfig, required=False
    )
    optimizer = _section(root, "optimizer", KimiOptimizerConfig)
    scheduler = _section(root, "scheduler", SchedulerConfig)
    diagnostics = _section(
        root, "diagnostics", DiagnosticsConfig, required=False
    )
    curriculum_values = expect_mapping(
        root,
        "context_curriculum",
        path="root",
        required=False,
    )
    if "stages" in curriculum_values:
        stages = curriculum_values["stages"]
        if not isinstance(stages, list):
            raise ConfigError("context_curriculum.stages must be a list")
        curriculum_values["stages"] = tuple(
            ContextStage(**stage) if isinstance(stage, dict) else stage
            for stage in stages
        )
    context_curriculum = ContextCurriculumConfig(
        **dataclass_kwargs(
            curriculum_values,
            ContextCurriculumConfig,
            path="context_curriculum",
        )
    )
    checkpoint = _section(
        root, "checkpoint", CheckpointConfig, required=False
    )
    prediction = _section(
        root, "prediction", PredictionConfig, required=False
    )
    distributed_values = expect_mapping(
        root, "distributed", path="root", required=False
    )
    distributed = distributed_config_from_dict(distributed_values)
    reject_unknown_keys(root, path="root")
    return TrainingYamlConfig(
        runtime=runtime,
        training=training,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        diagnostics=diagnostics,
        context_curriculum=context_curriculum,
        checkpoint=checkpoint,
        prediction=prediction,
        distributed=distributed,
        source_path=source,
    )


def validate_pipeline_compatibility(
    config: TrainingYamlConfig,
    *,
    model_config,
    data_config,
) -> None:
    """Validate cross-YAML invariants before any optimizer is allocated."""
    if config.training.use_mtp and not model_config.enable_mtp:
        raise ConfigError(
            "training.use_mtp=true but model MTP is disabled"
        )
    if config.training.max_seq_len > data_config.max_seq_len:
        raise ConfigError(
            f"training.max_seq_len ({config.training.max_seq_len}) exceeds "
            f"data block_size ({data_config.max_seq_len})"
        )
    curriculum = config.context_curriculum
    if curriculum.enabled:
        final_length = curriculum.stages[-1].max_seq_len
        if final_length > config.training.max_seq_len:
            raise ConfigError(
                "final context stage exceeds training.max_seq_len"
            )
        if final_length > data_config.max_seq_len:
            raise ConfigError(
                "final context stage exceeds data block_size"
            )
    if (
        config.distributed.data_parallel.mode == "fsdp"
        and config.runtime.use_ema
    ):
        raise ConfigError(
            "runtime.use_ema=true is unsupported with FSDP; "
            "a sharded EMA is not implemented"
        )
    validate_distributed_model_config(config.distributed, model_config)


def train_kimi_from_yaml(
    path: str | Path,
    *,
    model,
    data,
    logger=None,
    distributed_context=None,
) -> dict[str, Any]:
    """Call only the master trainer using one validated training YAML."""
    config = load_training_config(path)
    validate_pipeline_compatibility(
        config,
        model_config=model.config,
        data_config=data.config,
    )
    from .train_kimi_k3 import train_kimiK3

    reset_loader = (
        config.context_curriculum.enabled
        and config.context_curriculum.reset_dataloader_on_transition
    )
    loader_kwargs = (
        {"train_loader_factory": data.train_loader_factory}
        if reset_loader
        else {"train_loader": data.train_loader}
    )
    return train_kimiK3(
        model=model,
        val_loader=data.val_loader,
        device=config.runtime.device,
        training_config=config.training,
        loss_config=config.loss,
        kimi_optimizer_config=config.optimizer,
        scheduler_config=config.scheduler,
        diagnostics_config=config.diagnostics,
        context_curriculum_config=config.context_curriculum,
        checkpoint_config=config.checkpoint,
        prediction_config=config.prediction,
        use_ema=config.runtime.use_ema,
        ema_decay=config.runtime.ema_decay,
        eval_use_ema=config.runtime.eval_use_ema,
        logger=logger,
        tokenizer=data.tokenizer,
        total_steps=config.runtime.total_steps,
        verbose=config.runtime.verbose,
        distributed_config=config.distributed,
        distributed_context=distributed_context,
        **loader_kwargs,
    )


__all__ = [
    "TrainingRuntimeConfig",
    "TrainingYamlConfig",
    "load_training_config",
    "train_kimi_from_yaml",
    "validate_pipeline_compatibility",
]
