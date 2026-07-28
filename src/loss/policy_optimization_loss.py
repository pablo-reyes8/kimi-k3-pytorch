"""High-level Kimi K2.5 policy-optimization objective reused by Kimi K3."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

from .common.reduction import connected_zero
from .common.types import PolicyOptimizationLossOutput
from .common.validation import boolean_mask, require_finite, validate_zero_policy
from .rl.policy_components import group_mean_advantages


class KimiPolicyOptimizationLoss(nn.Module):
    """Implement clipped token ratios plus raw log-ratio L2 regularization.

    This is the published Kimi policy equation, not PPO's minimum surrogate:
    ``loss = -clip(exp(delta), alpha, beta) * advantage + tau * delta²``.
    No value head, GAE, entropy bonus, or critic objective is introduced.
    """

    def __init__(
        self,
        *,
        ratio_clip_min: float,
        ratio_clip_max: float,
        log_ratio_l2_coef: float,
        normalize_advantages: bool = False,
        require_complete_groups: bool = True,
        zero_valid_policy: Literal["raise", "connected_zero"] = "raise",
    ) -> None:
        super().__init__()
        if not 0 < ratio_clip_min <= ratio_clip_max:
            raise ValueError("ratio bounds must satisfy 0 < min <= max")
        if log_ratio_l2_coef < 0:
            raise ValueError("log_ratio_l2_coef must be >= 0")
        self.ratio_clip_min = float(ratio_clip_min)
        self.ratio_clip_max = float(ratio_clip_max)
        self.log_ratio_l2_coef = float(log_ratio_l2_coef)
        self.normalize_advantages = normalize_advantages
        self.require_complete_groups = require_complete_groups
        self.zero_valid_policy = validate_zero_policy(zero_valid_policy)

    def forward(
        self,
        *,
        current_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        action_mask: torch.Tensor,
        rewards: torch.Tensor | None = None,
        prompt_group_ids: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
        expected_group_size: int | None = None,
    ) -> PolicyOptimizationLossOutput:
        """Return the token-normalized negative Kimi policy objective."""

        if current_logprobs.ndim != 2 or old_logprobs.shape != current_logprobs.shape:
            raise ValueError("current_logprobs and old_logprobs must have shape [B,T]")
        if not current_logprobs.dtype.is_floating_point:
            raise TypeError("current_logprobs must be floating point")
        if old_logprobs.device != current_logprobs.device:
            raise ValueError("current and old logprobs must share device")
        batch, tokens = current_logprobs.shape
        valid = boolean_mask(
            "action_mask", action_mask, (batch, tokens), current_logprobs.device
        )
        reward_mode = rewards is not None or prompt_group_ids is not None
        if reward_mode == (advantages is not None):
            raise ValueError(
                "provide exactly rewards+prompt_group_ids or explicit advantages"
            )
        mean_reward = reward_std = None
        if reward_mode:
            if rewards is None or prompt_group_ids is None:
                raise ValueError("rewards and prompt_group_ids are required together")
            sample_advantages = group_mean_advantages(
                rewards,
                prompt_group_ids,
                expected_group_size=expected_group_size,
                require_complete_groups=self.require_complete_groups,
            )
            mean_reward = rewards.detach().float().mean()
            reward_std = rewards.detach().float().std(unbiased=False)
            token_advantages = sample_advantages[:, None].expand(batch, tokens)
        else:
            if advantages.shape not in ((batch,), (batch, tokens)):
                raise ValueError("advantages must have shape [B] or [B,T]")
            token_advantages = (
                advantages[:, None].expand(batch, tokens)
                if advantages.ndim == 1
                else advantages
            ).detach().float()
        if self.normalize_advantages:
            selected = token_advantages[valid]
            if selected.numel():
                token_advantages = (
                    token_advantages - selected.mean()
                ) / selected.std(unbiased=False).clamp_min(1e-6)
        if sample_weights is None:
            weights = torch.ones((batch, 1), device=current_logprobs.device)
        else:
            if sample_weights.shape != (batch,):
                raise ValueError("sample_weights must have shape [B]")
            weights = sample_weights.detach().float()[:, None]
            if not torch.isfinite(weights).all() or torch.any(weights < 0):
                raise ValueError("sample_weights must be finite and non-negative")
        token_weights = valid.float() * weights
        normalizer = token_weights.sum(dtype=torch.float32)
        num_valid = valid.sum().to(torch.float32)
        current = current_logprobs.float()
        old = old_logprobs.detach().float()
        require_finite("current_logprobs", current, valid)
        require_finite("old_logprobs", old, valid)
        require_finite("advantages", token_advantages, valid)
        log_ratio = current - old
        log_min, log_max = math.log(self.ratio_clip_min), math.log(self.ratio_clip_max)
        clipped_ratio = torch.exp(log_ratio.clamp(log_min, log_max))
        policy_terms = clipped_ratio * token_advantages
        regularization_terms = self.log_ratio_l2_coef * log_ratio.square()
        require_finite("policy_terms", policy_terms, valid)
        require_finite("regularization_terms", regularization_terms, valid)
        policy_sum = torch.where(
            valid,
            policy_terms * weights,
            torch.zeros_like(policy_terms),
        ).sum()
        regularization_sum = torch.where(
            valid,
            regularization_terms * weights,
            torch.zeros_like(regularization_terms),
        ).sum()
        loss_sum = -policy_sum + regularization_sum
        if normalizer.item() == 0:
            if self.zero_valid_policy == "raise":
                raise ValueError("RL batch contains no action tokens")
            loss = connected_zero(current_logprobs)
            loss_sum = loss
        else:
            loss = loss_sum / normalizer
        selected_ratio = log_ratio[valid]
        selected_advantage = token_advantages[valid]
        zero = current.detach().new_zeros(())
        clipped = valid & (
            (log_ratio < log_min) | (log_ratio > log_max)
        )
        return PolicyOptimizationLossOutput(
            loss=loss,
            loss_sum=loss_sum,
            normalizer=normalizer.detach(),
            num_valid_tokens=num_valid.detach(),
            policy_objective_sum=policy_sum.detach(),
            regularization_sum=regularization_sum.detach(),
            mean_advantage=(
                selected_advantage.mean().detach()
                if selected_advantage.numel() else zero
            ),
            mean_log_ratio=(
                selected_ratio.mean().detach() if selected_ratio.numel() else zero
            ),
            max_abs_log_ratio=(
                selected_ratio.abs().max().detach()
                if selected_ratio.numel() else zero
            ),
            clipped_fraction=(
                clipped[valid].float().mean().detach() if torch.any(valid) else zero
            ),
            mean_reward=None if mean_reward is None else mean_reward.detach(),
            reward_std=None if reward_std is None else reward_std.detach(),
        )


__all__ = ["KimiPolicyOptimizationLoss"]
