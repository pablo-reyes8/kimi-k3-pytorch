"""Configuration for the project-defined Kimi training objectives."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class KimiLossConfig:
    """Configure unpublished engineering choices without calling them official."""

    ignore_index: int = -100
    label_smoothing: float = 0.0
    lambda_mtp: float = 0.0
    sft_reduction: str = "token_mean"
    rl_ratio_clip_min: float | None = None
    rl_ratio_clip_max: float | None = None
    rl_log_ratio_l2_coef: float | None = None
    rl_normalize_advantages: bool = False
    mopd_reward_clip_max: float | None = None
    mopd_mode: str = "sampled_token_pg"
    compute_loss_in_fp32: bool = True

    def __post_init__(self) -> None:
        if self.ignore_index >= 0:
            raise ValueError("ignore_index must be outside the non-negative vocabulary")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.lambda_mtp < 0:
            raise ValueError("lambda_mtp must be >= 0")
        if self.sft_reduction not in ("token_mean", "sequence_mean"):
            raise ValueError("unsupported SFT reduction")
        optional_nonnegative = (
            "rl_log_ratio_l2_coef",
            "mopd_reward_clip_max",
        )
        for name in optional_nonnegative:
            value = getattr(self, name)
            if value is not None and value <= 0 and name == "mopd_reward_clip_max":
                raise ValueError("mopd_reward_clip_max must be > 0")
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0")
        if (self.rl_ratio_clip_min is None) != (
            self.rl_ratio_clip_max is None
        ):
            raise ValueError("both RL ratio bounds must be provided together")
        if self.rl_ratio_clip_min is not None and not (
            0 < self.rl_ratio_clip_min <= self.rl_ratio_clip_max
        ):
            raise ValueError("RL ratio bounds must satisfy 0 < min <= max")
        if self.mopd_mode not in (
            "sampled_token_pg",
            "kimi_rl_regularized",
            "topk_reverse_kl",
        ):
            raise ValueError("unsupported MOPD mode")
        if not self.compute_loss_in_fp32:
            raise ValueError("phase 11 requires FP32 loss computation")

    def require_posttraining_values(self) -> None:
        """Reject construction of RL/MOPD objectives with unspecified values."""

        required = (
            self.rl_ratio_clip_min,
            self.rl_ratio_clip_max,
            self.rl_log_ratio_l2_coef,
            self.mopd_reward_clip_max,
        )
        if any(value is None for value in required):
            raise ValueError(
                "RL and MOPD values are unpublished; provide explicit project "
                "values before constructing post-training objectives"
            )

    def to_dict(self) -> dict:
        """Serialize the configuration to primitive values."""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "KimiLossConfig":
        """Restore a configuration serialized with :meth:`to_dict`."""

        return cls(**values)
