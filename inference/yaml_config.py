"""Optional YAML profiles for generation-time sampling controls."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

from configuration.yaml_utils import (
    dataclass_kwargs,
    expect_mapping,
    load_yaml_mapping,
    reject_unknown_keys,
)

from .config import GenerationConfig
from training.distributed import (
    DistributedConfig,
    distributed_config_from_dict,
)


@dataclass(frozen=True)
class InferenceYamlConfig:
    generation: GenerationConfig
    distributed: DistributedConfig
    source_path: Path


def load_inference_yaml_config(
    path: str | Path,
) -> InferenceYamlConfig:
    source, root = load_yaml_mapping(path)
    values = expect_mapping(root, "inference", path="root")
    distributed_values = expect_mapping(
        root, "distributed", path="root", required=False
    )
    reject_unknown_keys(root, path="root")
    distributed = distributed_config_from_dict(distributed_values)
    if distributed.data_parallel.mode != "none":
        raise ValueError("distributed inference does not use DDP/FSDP")
    return InferenceYamlConfig(
        generation=GenerationConfig(
            **dataclass_kwargs(
                values, GenerationConfig, path="inference"
            )
        ),
        distributed=distributed,
        source_path=source,
    )


def load_generation_config(path: str | Path) -> GenerationConfig:
    return load_inference_yaml_config(path).generation


__all__ = [
    "InferenceYamlConfig",
    "load_generation_config",
    "load_inference_yaml_config",
]
