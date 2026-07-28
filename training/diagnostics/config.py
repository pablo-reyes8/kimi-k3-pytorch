"""Bounded diagnostic scheduling configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DiagnosticsConfig:
    enabled: bool = True
    cheap_every_steps: int = 1
    standard_every_steps: int = 10
    deep_every_steps: int = 500
    sample_layers_per_standard_step: int = 8
    rotate_sampled_layers: bool = True
    sample_tokens_per_layer: int = 256
    sample_parameters_per_group: int = 16
    sample_elements_per_parameter: int = 256
    max_diagnostic_time_fraction: float = 0.05
    max_persistent_gpu_bytes: int = 16 * 1024 * 1024
    alert_patience_steps: int = 5

    def __post_init__(self) -> None:
        for name in (
            "cheap_every_steps",
            "standard_every_steps",
            "deep_every_steps",
            "sample_layers_per_standard_step",
            "sample_tokens_per_layer",
            "sample_parameters_per_group",
            "sample_elements_per_parameter",
            "alert_patience_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.max_diagnostic_time_fraction <= 1:
            raise ValueError("max_diagnostic_time_fraction must be in (0, 1]")
        if self.max_persistent_gpu_bytes < 0:
            raise ValueError("max_persistent_gpu_bytes must be non-negative")

    def level_for_step(self, step: int) -> str | None:
        if not self.enabled or step <= 0:
            return None
        if step % self.deep_every_steps == 0:
            return "deep"
        if step % self.standard_every_steps == 0:
            return "standard"
        if step % self.cheap_every_steps == 0:
            return "cheap"
        return None

    def to_dict(self) -> dict:
        return asdict(self)
