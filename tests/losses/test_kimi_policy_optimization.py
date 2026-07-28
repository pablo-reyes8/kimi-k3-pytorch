import math

import pytest
import torch

from src.loss import KimiPolicyOptimizationLoss


def _criterion(tau=0.2):
    return KimiPolicyOptimizationLoss(
        ratio_clip_min=0.8,
        ratio_clip_max=1.2,
        log_ratio_l2_coef=tau,
    )


def test_kimi_rl_matches_hand_computed_objective():
    current = torch.tensor([[0.1, -0.2], [0.3, -0.4]], requires_grad=True)
    old = torch.zeros_like(current)
    rewards = torch.tensor([1.0, 0.0])
    groups = torch.tensor([7, 7])
    mask = torch.ones_like(current, dtype=torch.bool)
    output = _criterion()(current_logprobs=current, old_logprobs=old, action_mask=mask,
                          rewards=rewards, prompt_group_ids=groups, expected_group_size=2)
    advantages = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
    ratio = torch.exp(current.clamp(math.log(0.8), math.log(1.2)))
    policy = (ratio * advantages).sum()
    regularization = 0.2 * current.square().sum()
    torch.testing.assert_close(output.policy_objective_sum, policy)
    torch.testing.assert_close(output.regularization_sum, regularization)
    torch.testing.assert_close(output.loss, (-policy + regularization) / 4)


def test_kimi_rl_reward_translation_and_batch_permutation_invariance():
    current = torch.tensor([[0.0, 0.1], [-0.1, 0.2], [0.3, 0.0], [-0.2, 0.1]])
    old = torch.zeros_like(current)
    rewards = torch.tensor([1.0, 3.0, -2.0, 4.0])
    groups = torch.tensor([0, 0, 1, 1])
    mask = torch.ones_like(current, dtype=torch.bool)
    base = _criterion()(current_logprobs=current, old_logprobs=old, action_mask=mask,
                        rewards=rewards, prompt_group_ids=groups, expected_group_size=2)
    shifted = _criterion()(current_logprobs=current, old_logprobs=old, action_mask=mask,
                           rewards=rewards + 99, prompt_group_ids=groups, expected_group_size=2)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = _criterion()(
        current_logprobs=current[permutation], old_logprobs=old[permutation],
        action_mask=mask[permutation], rewards=rewards[permutation],
        prompt_group_ids=groups[permutation], expected_group_size=2,
    )
    torch.testing.assert_close(base.loss, shifted.loss)
    torch.testing.assert_close(base.loss, permuted.loss)


def test_kimi_rl_uses_action_tokens_and_total_token_normalization():
    current = torch.tensor([[0.2, -0.1, 0.5]], requires_grad=True)
    old = torch.zeros_like(current)
    mask = torch.tensor([[False, True, True]])
    output = _criterion(tau=0)(
        current_logprobs=current, old_logprobs=old, action_mask=mask,
        advantages=torch.ones(1),
    )
    assert output.normalizer.item() == 2
    output.loss.backward()
    assert current.grad[0, 0].item() == 0
    assert current.grad[0, 1:].abs().sum() > 0


def test_kimi_rl_policy_gradient_zero_outside_clip_regularizer_remains():
    current = torch.tensor([[10.0]], requires_grad=True)
    old = torch.zeros_like(current)
    mask = torch.ones_like(current, dtype=torch.bool)
    policy_only = _criterion(tau=0)(
        current_logprobs=current, old_logprobs=old, action_mask=mask,
        advantages=torch.ones(1),
    )
    policy_grad, = torch.autograd.grad(policy_only.loss, current, retain_graph=True)
    assert policy_grad.item() == 0
    regularized = _criterion(tau=0.2)(
        current_logprobs=current, old_logprobs=old, action_mask=mask,
        advantages=torch.ones(1),
    )
    regularized_grad, = torch.autograd.grad(regularized.loss, current)
    assert regularized_grad.item() != 0
    assert torch.isfinite(regularized.loss)


def test_kimi_rl_old_logprobs_advantages_and_rewards_are_detached():
    current = torch.zeros(2, 1, requires_grad=True)
    old = torch.zeros(2, 1, requires_grad=True)
    rewards = torch.tensor([1.0, 0.0], requires_grad=True)
    output = _criterion()(current_logprobs=current, old_logprobs=old,
                          action_mask=torch.ones_like(current, dtype=torch.bool),
                          rewards=rewards, prompt_group_ids=torch.tensor([0, 0]))
    output.loss.backward()
    assert current.grad is not None
    assert old.grad is None
    assert rewards.grad is None


def test_kimi_rl_large_log_ratios_do_not_overflow():
    current = torch.tensor([[1e20, -1e20]], requires_grad=True)
    with pytest.raises(FloatingPointError, match="regularization"):
        _criterion()(current_logprobs=current, old_logprobs=torch.zeros_like(current),
                     action_mask=torch.ones_like(current, dtype=torch.bool),
                     advantages=torch.ones_like(current))


def test_kimi_rl_large_but_representable_stale_rollout_is_finite():
    current = torch.tensor([[1000.0, -1000.0]], requires_grad=True)
    output = _criterion()(current_logprobs=current, old_logprobs=torch.zeros_like(current),
                          action_mask=torch.ones_like(current, dtype=torch.bool),
                          advantages=torch.ones_like(current))
    assert torch.isfinite(output.loss)


def test_kimi_rl_incomplete_groups_and_empty_actions_raise():
    criterion = _criterion()
    values = torch.zeros(3, 1)
    with pytest.raises(ValueError, match="incomplete"):
        criterion(current_logprobs=values, old_logprobs=values,
                  action_mask=torch.ones_like(values, dtype=torch.bool),
                  rewards=torch.arange(3.0), prompt_group_ids=torch.tensor([0, 0, 1]),
                  expected_group_size=2)
    with pytest.raises(ValueError, match="no action"):
        criterion(current_logprobs=values, old_logprobs=values,
                  action_mask=torch.zeros_like(values, dtype=torch.bool),
                  advantages=torch.ones(3))


def test_kimi_rl_bf16_inputs_compute_fp32():
    current = torch.zeros(2, 3, dtype=torch.bfloat16, requires_grad=True)
    output = _criterion()(current_logprobs=current, old_logprobs=torch.zeros_like(current),
                          action_mask=torch.ones_like(current, dtype=torch.bool),
                          advantages=torch.tensor([1.0, -1.0]))
    assert output.loss.dtype == torch.float32
    output.loss.backward()
    assert torch.isfinite(current.grad).all()


def test_kimi_rl_nonfinite_masked_context_does_not_poison_loss():
    current = torch.tensor([[float("nan"), 0.1]])
    old = torch.tensor([[float("inf"), 0.0]])
    output = _criterion()(
        current_logprobs=current,
        old_logprobs=old,
        action_mask=torch.tensor([[False, True]]),
        advantages=torch.ones(1),
    )
    assert torch.isfinite(output.loss)


def test_kimi_rl_is_not_ppo_min_surrogate():
    current = torch.tensor([[math.log(1.1)]])
    output = _criterion(tau=0)(current_logprobs=current, old_logprobs=torch.zeros_like(current),
                               action_mask=torch.ones_like(current, dtype=torch.bool),
                               advantages=torch.tensor([-1.0]))
    # Kimi uses clipped_ratio*A directly: loss = -1.1*(-1) = +1.1.
    torch.testing.assert_close(output.loss, torch.tensor(1.1))
