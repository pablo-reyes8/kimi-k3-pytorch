"""Optional YAML profiles for generation-time sampling controls."""

from __future__ import annotations

from pathlib import Path

from configuration.yaml_utils import (
    dataclass_kwargs,
    expect_mapping,
    load_yaml_mapping,
    reject_unknown_keys,
)

from .config import GenerationConfig


def load_generation_config(path: str | Path) -> GenerationConfig:
    _, root = load_yaml_mapping(path)
    values = expect_mapping(root, "inference", path="root")
    reject_unknown_keys(root, path="root")
    return GenerationConfig(
        **dataclass_kwargs(
            values, GenerationConfig, path="inference"
        )
    )


__all__ = ["load_generation_config"]
