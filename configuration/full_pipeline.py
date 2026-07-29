"""Resolve one self-contained Kimi data/model/training profile directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .yaml_utils import ConfigError


@dataclass(frozen=True)
class KimiPipelinePaths:
    root: Path
    data: Path
    model: Path
    training: Path


def resolve_kimi_pipeline_profile(
    profile: str | Path,
) -> KimiPipelinePaths:
    """Return the three canonical YAML paths and fail on incomplete profiles."""
    root = Path(profile).expanduser().resolve()
    if not root.is_dir():
        raise ConfigError(f"Kimi pipeline profile is not a directory: {root}")
    paths = KimiPipelinePaths(
        root=root,
        data=root / "data.yaml",
        model=root / "model.yaml",
        training=root / "training.yaml",
    )
    missing = [
        path.name
        for path in (paths.data, paths.model, paths.training)
        if not path.is_file()
    ]
    if missing:
        raise ConfigError(
            f"incomplete Kimi pipeline profile {root}: missing "
            + ", ".join(missing)
        )
    return paths


__all__ = ["KimiPipelinePaths", "resolve_kimi_pipeline_profile"]
