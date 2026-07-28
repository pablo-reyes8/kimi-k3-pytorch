"""Typed outputs and phase metadata shared by every Kimi loss."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

import torch


@dataclass
class LossOutput:
    """Scalar loss with unbiased distributed-reduction accounting."""

    loss: torch.Tensor
    loss_sum: torch.Tensor
    normalizer: torch.Tensor
    num_valid_tokens: torch.Tensor


@dataclass
class TokenCrossEntropyOutput(LossOutput):
    """Token cross-entropy result with optional unreduced diagnostics."""

    per_token_nll: torch.Tensor | None = None
    per_sample_loss_sum: torch.Tensor | None = None
    per_sample_num_tokens: torch.Tensor | None = None


@dataclass
class MTPLossOutput(TokenCrossEntropyOutput):
    """Multi-token loss with optional statistics for every future depth."""

    per_depth_loss_sum: torch.Tensor | None = None
    per_depth_normalizer: torch.Tensor | None = None
    future_offsets: tuple[int, ...] | None = None


@dataclass
class SFTLossOutput(TokenCrossEntropyOutput):
    """Assistant-only trajectory loss and optional component summaries."""

    per_component_loss_sum: dict[int, torch.Tensor] | None = None
    per_component_num_tokens: dict[int, torch.Tensor] | None = None
    reduction: str = "token_mean"


@dataclass
class PolicyOptimizationLossOutput(LossOutput):
    """Kimi policy objective and detached rollout diagnostics."""

    policy_objective_sum: torch.Tensor
    regularization_sum: torch.Tensor
    mean_advantage: torch.Tensor
    mean_log_ratio: torch.Tensor
    max_abs_log_ratio: torch.Tensor
    clipped_fraction: torch.Tensor
    mean_reward: torch.Tensor | None = None
    reward_std: torch.Tensor | None = None


@dataclass
class MOPDLossOutput(LossOutput):
    """Multi-teacher OPD loss and detached teacher/student diagnostics."""

    mean_token_reward: torch.Tensor
    reward_clip_fraction: torch.Tensor
    mean_teacher_logprob: torch.Tensor
    mean_student_logprob: torch.Tensor
    mean_teacher_student_gap: torch.Tensor
    teacher_token_count: torch.Tensor
    mode: str
    reward_reference: str
    policy_output: PolicyOptimizationLossOutput | None = None


@dataclass
class KimiPretrainingLossOutput:
    """Combined next-token and optional MTP pretraining objective."""

    loss: torch.Tensor
    ntp: TokenCrossEntropyOutput
    mtp: MTPLossOutput | None
    lambda_mtp: float
    moe_diagnostics: object | None = None
    moe_aux_loss: None = None


class SFTComponent(IntEnum):
    """Project-defined target categories for supervised trajectories."""

    IGNORE = 0
    REASONING = 1
    TOOL_CALL = 2
    TOOL_ARGUMENT = 3
    ASSISTANT_TEXT = 4
    FINAL_ANSWER = 5
    END_OF_TURN = 6


class KimiTrainingPhase(str, Enum):
    """Mutually exclusive optimization phases."""

    PRETRAIN = "pretrain"
    SFT = "sft"
    RL = "rl"
    MOPD = "mopd"


@dataclass(frozen=True)
class RewardWeights:
    """Explicit coefficients for available trajectory reward sources."""

    verifier: float | None = None
    generative: float | None = None
    task_specific: float | None = None


@dataclass
class TrajectoryRewards:
    """Detached rewards and grouping metadata consumed by policy training."""

    final_reward: torch.Tensor
    prompt_group_ids: torch.Tensor
    verifier_reward: torch.Tensor | None = None
    generative_reward: torch.Tensor | None = None
    task_specific_reward: torch.Tensor | None = None
    budget_override: torch.Tensor | None = None


@dataclass
class KimiLossBundle:
    """Strict dispatcher output carrying one phase-specific result."""

    phase: KimiTrainingPhase
    loss: torch.Tensor
    output: object

