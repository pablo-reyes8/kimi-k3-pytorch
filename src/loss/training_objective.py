"""Strict phase dispatcher for the five public Kimi loss modules."""

from __future__ import annotations

import torch.nn as nn

from .common.config import KimiLossConfig
from .common.types import KimiLossBundle, KimiTrainingPhase
from .mopd_loss import MultiTeacherOnPolicyDistillationLoss
from .policy_optimization_loss import KimiPolicyOptimizationLoss
from .pretraining_loss import KimiPretrainingLoss
from .sft_loss import SFTTrajectoryCrossEntropyLoss


_PHASE_KEYS = {
    KimiTrainingPhase.PRETRAIN: {
        "logits",
        "labels",
        "attention_mask",
        "loss_mask",
        "boundary_mask",
        "token_weights",
        "mtp_logits",
        "mtp_labels",
        "mtp_loss_mask",
        "future_offsets",
        "moe_diagnostics",
    },
    KimiTrainingPhase.SFT: {
        "logits",
        "labels",
        "assistant_mask",
        "attention_mask",
        "component_ids",
        "component_weights",
        "boundary_mask",
        "sample_weights",
        "return_per_component",
    },
    KimiTrainingPhase.RL: {
        "current_logprobs",
        "old_logprobs",
        "action_mask",
        "rewards",
        "prompt_group_ids",
        "advantages",
        "sample_weights",
        "expected_group_size",
    },
    KimiTrainingPhase.MOPD: {
        "current_student_logprobs",
        "teacher_sampled_token_logprobs",
        "action_mask",
        "teacher_ids",
        "student_logprobs_for_reward",
        "old_student_logprobs",
        "teacher_topk_token_ids",
        "teacher_topk_logprobs",
        "student_topk_logprobs",
        "sampled_token_ids",
        "teacher_sampled_token_ids",
    },
}


class KimiTrainingObjective(nn.Module):
    """Dispatch exactly one training phase without summing unrelated losses."""

    def __init__(self, config: KimiLossConfig):
        super().__init__()
        config.require_posttraining_values()
        self.config = config
        self.pretraining = KimiPretrainingLoss(
            lambda_mtp=config.lambda_mtp,
            ignore_index=config.ignore_index,
            label_smoothing=config.label_smoothing,
        )
        self.sft = SFTTrajectoryCrossEntropyLoss(
            ignore_index=config.ignore_index,
            reduction=config.sft_reduction,
        )
        self.policy = KimiPolicyOptimizationLoss(
            ratio_clip_min=config.rl_ratio_clip_min,
            ratio_clip_max=config.rl_ratio_clip_max,
            log_ratio_l2_coef=config.rl_log_ratio_l2_coef,
            normalize_advantages=config.rl_normalize_advantages,
        )
        self.mopd = MultiTeacherOnPolicyDistillationLoss(
            reward_clip_max=config.mopd_reward_clip_max,
            mode=config.mopd_mode,
            policy_loss=self.policy,
        )

    def forward(
        self,
        phase: KimiTrainingPhase | str,
        **batch,
    ) -> KimiLossBundle:
        """Validate phase-specific keys and call only the selected objective."""

        try:
            phase = KimiTrainingPhase(phase)
        except ValueError as exc:
            raise ValueError(f"unknown Kimi training phase: {phase!r}") from exc
        unexpected = set(batch) - _PHASE_KEYS[phase]
        if unexpected:
            raise ValueError(
                f"{phase.value} received inputs from another phase: "
                f"{sorted(unexpected)}"
            )
        module = {
            KimiTrainingPhase.PRETRAIN: self.pretraining,
            KimiTrainingPhase.SFT: self.sft,
            KimiTrainingPhase.RL: self.policy,
            KimiTrainingPhase.MOPD: self.mopd,
        }[phase]
        output = module(**batch)
        return KimiLossBundle(phase=phase, loss=output.loss, output=output)


__all__ = ["KimiTrainingObjective"]
