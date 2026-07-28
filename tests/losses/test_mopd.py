import pytest
import torch

from src.loss import (
    KimiPolicyOptimizationLoss,
    MultiTeacherOnPolicyDistillationLoss,
)


def test_mopd_reward_and_direct_pg_match_manual_equation():
    student = torch.tensor([[-0.5, -1.0, -0.2]], requires_grad=True)
    teacher = torch.tensor([[-0.1, -2.0, 2.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    output = MultiTeacherOnPolicyDistillationLoss(reward_clip_max=0.6)(
        current_student_logprobs=student,
        teacher_sampled_token_logprobs=teacher,
        action_mask=mask,
        teacher_ids=torch.tensor([[0, 1]]),
    )
    reward = (teacher.detach() - student.detach()).clamp(-0.6, 0.6)
    expected = -(reward[mask] * student[mask]).mean()
    torch.testing.assert_close(output.loss, expected)
    torch.testing.assert_close(output.mean_token_reward, reward[mask].mean())
    assert output.reward_reference == "current_detached"


def test_mopd_teacher_and_reward_are_stop_gradient():
    student = torch.tensor([[-0.5, -1.0]], requires_grad=True)
    teacher = torch.tensor([[-0.1, -2.0]], requires_grad=True)
    output = MultiTeacherOnPolicyDistillationLoss(reward_clip_max=1.0)(
        current_student_logprobs=student,
        teacher_sampled_token_logprobs=teacher,
        action_mask=torch.ones_like(student, dtype=torch.bool),
        teacher_ids=torch.tensor([2]),
    )
    output.loss.backward()
    assert student.grad is not None
    assert teacher.grad is None
    assert not output.mean_token_reward.requires_grad


def test_mopd_clips_both_reward_sides():
    student = torch.zeros(1, 2, requires_grad=True)
    teacher = torch.tensor([[100.0, -100.0]])
    output = MultiTeacherOnPolicyDistillationLoss(reward_clip_max=2.0)(
        current_student_logprobs=student,
        teacher_sampled_token_logprobs=teacher,
        action_mask=torch.ones_like(student, dtype=torch.bool),
        teacher_ids=torch.tensor([0]),
    )
    assert output.mean_token_reward.item() == 0
    assert output.reward_clip_fraction.item() == 1


def test_mopd_action_mask_removes_context_gradient():
    student = torch.tensor([[-0.5, -1.0]], requires_grad=True)
    output = MultiTeacherOnPolicyDistillationLoss(reward_clip_max=2.0)(
        current_student_logprobs=student,
        teacher_sampled_token_logprobs=torch.zeros_like(student),
        action_mask=torch.tensor([[False, True]]),
        teacher_ids=torch.tensor([4]),
    )
    output.loss.backward()
    assert student.grad[0, 0].item() == 0
    assert student.grad[0, 1].item() != 0


def test_mopd_stored_reward_reference_is_explicit():
    student = torch.tensor([[-0.5]], requires_grad=True)
    stored = torch.tensor([[-1.5]], requires_grad=True)
    output = MultiTeacherOnPolicyDistillationLoss(reward_clip_max=3.0)(
        current_student_logprobs=student,
        teacher_sampled_token_logprobs=torch.tensor([[-1.0]]),
        action_mask=torch.ones_like(student, dtype=torch.bool),
        teacher_ids=torch.tensor([0]),
        student_logprobs_for_reward=stored,
    )
    assert output.reward_reference == "rollout_stored"
    output.loss.backward()
    assert stored.grad is None


def test_mopd_regularized_delegates_to_kimi_policy():
    policy = KimiPolicyOptimizationLoss(
        ratio_clip_min=0.8, ratio_clip_max=1.2, log_ratio_l2_coef=0.1
    )
    criterion = MultiTeacherOnPolicyDistillationLoss(
        reward_clip_max=2.0, mode="kimi_rl_regularized", policy_loss=policy
    )
    current = torch.tensor([[-0.5, -0.3]], requires_grad=True)
    output = criterion(
        current_student_logprobs=current,
        teacher_sampled_token_logprobs=torch.tensor([[-0.2, -0.7]]),
        action_mask=torch.ones_like(current, dtype=torch.bool),
        teacher_ids=torch.tensor([[1, 2]]),
        old_student_logprobs=torch.tensor([[-0.6, -0.3]]),
    )
    assert output.policy_output is not None
    torch.testing.assert_close(output.loss, output.policy_output.loss)


def test_mopd_direct_and_regularized_gradients_match_at_ratio_one_tau_zero():
    policy = KimiPolicyOptimizationLoss(
        ratio_clip_min=0.8, ratio_clip_max=1.2, log_ratio_l2_coef=0.0
    )
    current = torch.tensor([[-0.5, -0.3]], requires_grad=True)
    kwargs = dict(
        current_student_logprobs=current,
        teacher_sampled_token_logprobs=torch.tensor([[-0.2, -0.7]]),
        action_mask=torch.ones_like(current, dtype=torch.bool),
        teacher_ids=torch.tensor([0]),
    )
    direct = MultiTeacherOnPolicyDistillationLoss(reward_clip_max=2.0)(**kwargs)
    regularized = MultiTeacherOnPolicyDistillationLoss(
        reward_clip_max=2.0, mode="kimi_rl_regularized", policy_loss=policy
    )(**kwargs, old_student_logprobs=current.detach())
    direct_grad, = torch.autograd.grad(direct.loss, current, retain_graph=True)
    regularized_grad, = torch.autograd.grad(regularized.loss, current)
    # The objectives differ by a parameter-independent baseline at ratio one,
    # but induce the same policy gradient.
    torch.testing.assert_close(direct_grad, regularized_grad)


def test_mopd_corrected_topk_is_zero_at_teacher_student_match():
    logprobs = torch.log(torch.tensor([[[0.4, 0.3]]]))
    output = MultiTeacherOnPolicyDistillationLoss(
        reward_clip_max=1.0, mode="topk_reverse_kl"
    )(
        current_student_logprobs=torch.zeros(1, 1),
        teacher_sampled_token_logprobs=None,
        action_mask=torch.ones(1, 1, dtype=torch.bool),
        teacher_ids=torch.tensor([0]),
        teacher_topk_token_ids=torch.tensor([[[2, 5]]]),
        teacher_topk_logprobs=logprobs,
        student_topk_logprobs=logprobs.clone().requires_grad_(),
    )
    torch.testing.assert_close(output.loss, torch.tensor(0.0))


def test_mopd_topk_correction_matches_formula():
    teacher = torch.log(torch.tensor([[[0.5, 0.2]]]))
    student = torch.log(torch.tensor([[[0.4, 0.1]]])).requires_grad_()
    output = MultiTeacherOnPolicyDistillationLoss(
        reward_clip_max=1.0, mode="topk_reverse_kl"
    )(
        current_student_logprobs=torch.zeros(1, 1),
        teacher_sampled_token_logprobs=None,
        action_mask=torch.ones(1, 1, dtype=torch.bool),
        teacher_ids=torch.tensor([0]),
        teacher_topk_token_ids=torch.tensor([[[1, 3]]]),
        teacher_topk_logprobs=teacher,
        student_topk_logprobs=student,
    )
    ps, pt = student.exp(), teacher.exp()
    expected = (ps * (student - teacher) - ps + pt).sum()
    torch.testing.assert_close(output.loss, expected)


def test_mopd_teacher_routing_and_tokenization_are_validated():
    values = torch.zeros(1, 2)
    criterion = MultiTeacherOnPolicyDistillationLoss(reward_clip_max=1.0)
    with pytest.raises(ValueError, match="teacher IDs"):
        criterion(current_student_logprobs=values,
                  teacher_sampled_token_logprobs=values,
                  action_mask=torch.ones_like(values, dtype=torch.bool),
                  teacher_ids=torch.tensor([10]))
    with pytest.raises(ValueError, match="tokenization"):
        criterion(current_student_logprobs=values,
                  teacher_sampled_token_logprobs=values,
                  action_mask=torch.ones_like(values, dtype=torch.bool),
                  teacher_ids=torch.tensor([0]),
                  sampled_token_ids=torch.tensor([[1, 2]]),
                  teacher_sampled_token_ids=torch.tensor([[1, 3]]))


def test_mopd_bf16_computes_fp32_without_full_teacher_logits():
    current = torch.tensor([[-1.0, -2.0]], dtype=torch.bfloat16, requires_grad=True)
    output = MultiTeacherOnPolicyDistillationLoss(reward_clip_max=1.0)(
        current_student_logprobs=current,
        teacher_sampled_token_logprobs=torch.tensor([[-0.5, -3.0]], dtype=torch.bfloat16),
        action_mask=torch.ones_like(current, dtype=torch.bool),
        teacher_ids=torch.tensor([8]),
    )
    assert output.loss.dtype == torch.float32
    output.loss.backward()
    assert torch.isfinite(current.grad).all()
