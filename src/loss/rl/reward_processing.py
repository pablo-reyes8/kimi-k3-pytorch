"""Pure upstream transformations for verifier, GRM, and budget rewards."""

from __future__ import annotations

import torch

from ..common.types import RewardWeights


class KimiRewardComposer:
    """Compose only explicitly weighted reward sources and detach the result."""

    def __call__(
        self,
        *,
        verifier_reward: torch.Tensor | None,
        generative_reward: torch.Tensor | None,
        task_specific_reward: torch.Tensor | None,
        weights: RewardWeights,
    ) -> torch.Tensor:
        """Return a shape-validated weighted reward sum without normalization."""

        sources = (
            ("verifier", verifier_reward, weights.verifier),
            ("generative", generative_reward, weights.generative),
            ("task_specific", task_specific_reward, weights.task_specific),
        )
        present = [(name, value, weight) for name, value, weight in sources if value is not None]
        if not present:
            raise ValueError("at least one reward source is required")
        batch = present[0][1].shape
        if len(batch) != 1:
            raise ValueError("reward sources must have shape [B]")
        result = torch.zeros_like(present[0][1], dtype=torch.float32)
        for name, value, weight in sources:
            if value is None:
                if weight is not None:
                    raise ValueError(f"{name} weight was supplied without its reward")
                continue
            if weight is None:
                raise ValueError(f"{name} reward requires an explicit weight")
            if value.shape != batch or value.device != result.device:
                raise ValueError("all reward sources must share shape and device")
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"{name} reward contains non-finite values")
            result = result + float(weight) * value.detach().float()
        return result.detach()


def apply_reasoning_budget_override(
    reward: torch.Tensor,
    used_tokens: torch.Tensor,
    base_budget: torch.Tensor,
    budget_multiplier: float,
    penalty_reward: float = -1.0,
) -> torch.Tensor:
    """Replace rewards whose model-output token usage exceeds the budget."""

    if reward.ndim != 1 or used_tokens.shape != reward.shape or base_budget.shape != reward.shape:
        raise ValueError("reward, used_tokens and base_budget must have shape [B]")
    if budget_multiplier <= 0:
        raise ValueError("budget_multiplier must be > 0")
    if torch.any(used_tokens < 0) or torch.any(base_budget < 0):
        raise ValueError("token counts and budgets must be non-negative")
    exceeded = used_tokens.float() > float(budget_multiplier) * base_budget.float()
    penalty = torch.full_like(reward.float(), float(penalty_reward))
    return torch.where(exceeded, penalty, reward.detach().float()).detach()


def apply_verbosity_binary_rule(
    wins: torch.Tensor,
    output_length: torch.Tensor,
    initial_verbosity: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Force a binary comparison loss when a candidate violates verbosity."""

    if wins.dtype != torch.bool:
        raise TypeError("wins must be boolean")
    if output_length.shape != wins.shape or initial_verbosity.shape != wins.shape:
        raise ValueError("verbosity inputs must share shape")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    violates = output_length.float() > sigma * initial_verbosity.float()
    return (wins & ~violates).detach()

