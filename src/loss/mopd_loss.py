"""High-level multi-teacher on-policy distillation objective."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .common.reduction import connected_zero
from .common.types import MOPDLossOutput
from .common.validation import boolean_mask, require_finite, validate_zero_policy
from .distillation.components import (
    clipped_teacher_student_reward,
    corrected_topk_reverse_kl,
    validate_teacher_ids,
)
from .policy_optimization_loss import KimiPolicyOptimizationLoss


class MultiTeacherOnPolicyDistillationLoss(nn.Module):
    """Consolidate one explicitly routed domain/effort teacher per trajectory.

    The default oracle is sampled-token policy gradient. A second backend
    injects the same detached dense reward into Kimi's stale-rollout objective;
    the optional Top-k backend includes MOPD's truncation-bias correction.
    Teacher models and full-vocabulary teacher logits are never owned here.
    """

    def __init__(
        self,
        *,
        reward_clip_max: float,
        mode: Literal[
            "sampled_token_pg",
            "kimi_rl_regularized",
            "topk_reverse_kl",
        ] = "sampled_token_pg",
        policy_loss: KimiPolicyOptimizationLoss | None = None,
        zero_valid_policy: Literal["raise", "connected_zero"] = "raise",
    ) -> None:
        super().__init__()
        if reward_clip_max <= 0:
            raise ValueError("reward_clip_max must be > 0")
        if mode not in (
            "sampled_token_pg",
            "kimi_rl_regularized",
            "topk_reverse_kl",
        ):
            raise ValueError("unsupported MOPD mode")
        if mode == "kimi_rl_regularized" and policy_loss is None:
            raise ValueError("regularized MOPD requires KimiPolicyOptimizationLoss")
        self.reward_clip_max = float(reward_clip_max)
        self.mode = mode
        self.policy_loss = policy_loss
        self.zero_valid_policy = validate_zero_policy(zero_valid_policy)

    def forward(
        self,
        *,
        current_student_logprobs: torch.Tensor,
        teacher_sampled_token_logprobs: torch.Tensor | None,
        action_mask: torch.Tensor,
        teacher_ids: torch.Tensor,
        student_logprobs_for_reward: torch.Tensor | None = None,
        old_student_logprobs: torch.Tensor | None = None,
        teacher_topk_token_ids: torch.Tensor | None = None,
        teacher_topk_logprobs: torch.Tensor | None = None,
        student_topk_logprobs: torch.Tensor | None = None,
        sampled_token_ids: torch.Tensor | None = None,
        teacher_sampled_token_ids: torch.Tensor | None = None,
    ) -> MOPDLossOutput:
        """Compute direct, Kimi-regularized, or corrected Top-k distillation."""

        if current_student_logprobs.ndim != 2:
            raise ValueError("current_student_logprobs must have shape [B,T]")
        batch, tokens = current_student_logprobs.shape
        valid = boolean_mask(
            "action_mask",
            action_mask,
            (batch, tokens),
            current_student_logprobs.device,
        )
        validate_teacher_ids(teacher_ids, batch, current_student_logprobs.device)
        if sampled_token_ids is not None or teacher_sampled_token_ids is not None:
            if sampled_token_ids is None or teacher_sampled_token_ids is None:
                raise ValueError("both sampled-token metadata tensors are required")
            if (
                sampled_token_ids.shape != (batch, tokens)
                or teacher_sampled_token_ids.shape != (batch, tokens)
                or torch.any(
                    sampled_token_ids[valid] != teacher_sampled_token_ids[valid]
                )
            ):
                raise ValueError("teacher/student tokenization metadata mismatch")
        normalizer = valid.sum().to(torch.float32)
        if normalizer.item() == 0 and self.zero_valid_policy == "raise":
            raise ValueError("MOPD batch contains no sampled action tokens")
        current = current_student_logprobs.float()
        require_finite("current_student_logprobs", current, valid)
        policy_output = None
        if self.mode == "topk_reverse_kl":
            if any(
                value is None
                for value in (
                    teacher_topk_token_ids,
                    teacher_topk_logprobs,
                    student_topk_logprobs,
                )
            ):
                raise ValueError("Top-k mode requires token IDs and both logprobs")
            expected_prefix = (batch, tokens)
            if (
                teacher_topk_token_ids.shape[:2] != expected_prefix
                or teacher_topk_logprobs.shape != teacher_topk_token_ids.shape
                or student_topk_logprobs.shape != teacher_topk_token_ids.shape
            ):
                raise ValueError("Top-k tensors must have shape [B,T,K]")
            if torch.any(
                teacher_topk_token_ids.sort(-1).values[..., 1:]
                == teacher_topk_token_ids.sort(-1).values[..., :-1]
            ):
                raise ValueError("teacher Top-k token IDs must be unique per token")
            per_token = corrected_topk_reverse_kl(
                teacher_topk_logprobs,
                student_topk_logprobs,
            )
            require_finite("topk_reverse_kl", per_token, valid)
            loss_sum = torch.where(
                valid, per_token, torch.zeros_like(per_token)
            ).sum()
            loss = (
                loss_sum / normalizer
                if normalizer.item()
                else connected_zero(student_topk_logprobs)
            )
            teacher_values = teacher_topk_logprobs.detach().float()
            student_values = student_topk_logprobs.detach().float()
            reward = torch.zeros_like(current)
            clipped = torch.zeros_like(valid)
        else:
            if (
                teacher_sampled_token_logprobs is None
                or teacher_sampled_token_logprobs.shape != (batch, tokens)
            ):
                raise ValueError("sampled MOPD requires teacher logprobs [B,T]")
            if teacher_sampled_token_logprobs.device != current.device:
                raise ValueError("teacher and student logprobs must share device")
            teacher_values = teacher_sampled_token_logprobs.detach().float()
            require_finite("teacher_logprobs", teacher_values, valid)
            if student_logprobs_for_reward is None:
                student_reference = current.detach()
                reward_reference = "current_detached"
            else:
                if student_logprobs_for_reward.shape != (batch, tokens):
                    raise ValueError("stored student logprobs must have shape [B,T]")
                student_reference = student_logprobs_for_reward.detach().float()
                reward_reference = "rollout_stored"
            require_finite("student_reward_reference", student_reference, valid)
            reward, clipped = clipped_teacher_student_reward(
                teacher_values, student_reference, self.reward_clip_max
            )
            if self.mode == "sampled_token_pg":
                loss_sum = -torch.where(
                    valid,
                    reward * current,
                    torch.zeros_like(current),
                ).sum()
                loss = (
                    loss_sum / normalizer
                    if normalizer.item()
                    else connected_zero(current_student_logprobs)
                )
            else:
                if old_student_logprobs is None:
                    raise ValueError("regularized MOPD requires old_student_logprobs")
                policy_output = self.policy_loss(
                    current_logprobs=current_student_logprobs,
                    old_logprobs=old_student_logprobs,
                    action_mask=valid,
                    advantages=reward,
                )
                loss, loss_sum = policy_output.loss, policy_output.loss_sum
            student_values = student_reference
        if self.mode == "topk_reverse_kl":
            reward_reference = "topk_distribution"
            mean_teacher = teacher_values[valid].mean() if torch.any(valid) else teacher_values.new_zeros(())
            mean_student = student_values[valid].mean() if torch.any(valid) else student_values.new_zeros(())
            mean_gap = (teacher_values - student_values)[valid].mean() if torch.any(valid) else teacher_values.new_zeros(())
        else:
            mean_teacher = teacher_values[valid].mean() if torch.any(valid) else teacher_values.new_zeros(())
            mean_student = student_values[valid].mean() if torch.any(valid) else student_values.new_zeros(())
            mean_gap = (teacher_values - student_values)[valid].mean() if torch.any(valid) else teacher_values.new_zeros(())
        zero = current.detach().new_zeros(())
        return MOPDLossOutput(
            loss=loss,
            loss_sum=loss_sum,
            normalizer=normalizer.detach(),
            num_valid_tokens=normalizer.detach(),
            mean_token_reward=(
                reward[valid].mean().detach() if torch.any(valid) else zero
            ),
            reward_clip_fraction=(
                clipped[valid].float().mean().detach() if torch.any(valid) else zero
            ),
            mean_teacher_logprob=mean_teacher.detach(),
            mean_student_logprob=mean_student.detach(),
            mean_teacher_student_gap=mean_gap.detach(),
            teacher_token_count=normalizer.detach(),
            mode=self.mode,
            reward_reference=reward_reference,
            policy_output=policy_output,
        )


__all__ = ["MultiTeacherOnPolicyDistillationLoss"]
