"""Teacher routing, dense reward, and corrected Top-k OPD utilities."""

from __future__ import annotations

import torch


def validate_teacher_ids(
    teacher_ids: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> None:
    """Validate one explicit teacher or domain/effort pair per trajectory."""

    if teacher_ids.device != device:
        raise ValueError("teacher_ids must share the student device")
    if teacher_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("teacher_ids must use int32 or int64")
    if teacher_ids.shape == (batch_size,):
        if torch.any((teacher_ids < 0) | (teacher_ids >= 9)):
            raise ValueError("flat teacher IDs must be in [0, 9)")
        return
    if teacher_ids.shape == (batch_size, 2):
        if torch.any((teacher_ids < 0) | (teacher_ids >= 3)):
            raise ValueError("domain/effort IDs must each be in [0, 3)")
        return
    raise ValueError("teacher_ids must have shape [B] or [B,2]")


def clipped_teacher_student_reward(
    teacher_logprobs: torch.Tensor,
    student_reference_logprobs: torch.Tensor,
    reward_clip_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute detached bilateral ``teacher - student`` token rewards."""

    teacher = teacher_logprobs.detach().float()
    student = student_reference_logprobs.detach().float()
    raw = teacher - student
    reward = raw.clamp(-reward_clip_max, reward_clip_max).detach()
    clipped = raw.abs() > reward_clip_max
    return reward, clipped


def corrected_topk_reverse_kl(
    teacher_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
) -> torch.Tensor:
    """Evaluate corrected truncated reverse KL at every token position."""

    teacher = teacher_logprobs.detach().float()
    student = student_logprobs.float()
    student_p = student.exp()
    teacher_p = teacher.exp()
    return (
        student_p * (student - teacher) - student_p + teacher_p
    ).sum(dim=-1)

