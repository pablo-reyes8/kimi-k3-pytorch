"""Shared types, validation, reductions, and log-probability utilities."""

from .config import KimiLossConfig
from .logprobs import gather_token_logprobs
from .types import (
    KimiLossBundle,
    KimiPretrainingLossOutput,
    KimiTrainingPhase,
    LossOutput,
    MOPDLossOutput,
    MTPLossOutput,
    PolicyOptimizationLossOutput,
    RewardWeights,
    SFTComponent,
    SFTLossOutput,
    TokenCrossEntropyOutput,
    TrajectoryRewards,
)

__all__ = [
    "KimiLossBundle",
    "KimiLossConfig",
    "KimiPretrainingLossOutput",
    "KimiTrainingPhase",
    "LossOutput",
    "MOPDLossOutput",
    "MTPLossOutput",
    "PolicyOptimizationLossOutput",
    "RewardWeights",
    "SFTComponent",
    "SFTLossOutput",
    "TokenCrossEntropyOutput",
    "TrajectoryRewards",
    "gather_token_logprobs",
]
