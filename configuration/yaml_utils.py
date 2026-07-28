"""Small, strict and dependency-light YAML parsing utilities."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as error:  # pragma: no cover - installation issue.
    yaml = None
    _YAML_IMPORT_ERROR = error
else:
    _YAML_IMPORT_ERROR = None


class ConfigError(ValueError):
    """Raised when a repository YAML violates its public schema."""


def load_yaml_mapping(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Load one YAML document and require a mapping at its root."""
    if yaml is None:
        raise ImportError(
            "YAML configuration requires PyYAML>=6"
        ) from _YAML_IMPORT_ERROR
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, Mapping):
        raise ConfigError(f"{resolved} must contain a YAML mapping")
    return resolved, dict(values)


def expect_mapping(
    values: Mapping[str, Any],
    key: str,
    *,
    path: str,
    required: bool = True,
) -> dict[str, Any]:
    """Pop and validate a nested mapping."""
    mutable = values if isinstance(values, dict) else dict(values)
    if key not in mutable:
        if required:
            raise ConfigError(f"{path}.{key} is required")
        return {}
    nested = mutable.pop(key)
    if not isinstance(nested, Mapping):
        raise ConfigError(f"{path}.{key} must be a mapping")
    return dict(nested)


def reject_unknown_keys(values: Mapping[str, Any], *, path: str) -> None:
    """Fail loudly on typos instead of silently ignoring configuration."""
    if values:
        keys = ", ".join(sorted(str(key) for key in values))
        raise ConfigError(f"unknown keys in {path}: {keys}")


def dataclass_kwargs(
    values: Mapping[str, Any],
    dataclass_type,
    *,
    path: str,
) -> dict[str, Any]:
    """Validate keys against a dataclass without constructing it."""
    fields = dataclass_type.__dataclass_fields__
    unknown = set(values) - set(fields)
    if unknown:
        reject_unknown_keys({key: values[key] for key in unknown}, path=path)
    return dict(values)


__all__ = [
    "ConfigError",
    "dataclass_kwargs",
    "expect_mapping",
    "load_yaml_mapping",
    "reject_unknown_keys",
]
