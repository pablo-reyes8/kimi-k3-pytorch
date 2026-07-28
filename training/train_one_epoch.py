"""One epoch of token-exact causal-LM training."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, Optional

import torch

from data.batch import normalize_lm_batch

from .autocast import autocast_ctx, move_batch_to_device
from .loss_accounting import (
    LossContribution,
    combine_window_loss,
    extract_loss_contribution,
)
from .model_call import call_model
from .state import TrainerState


def _grad_norm(parameters) -> float | None:
    gradients = [
        parameter.grad.detach().float().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return None
    return float(torch.stack(gradients).norm(2).item())


def _batch_size(batch: dict[str, Any]) -> int:
    input_ids = batch.get("input_ids")
    return int(input_ids.shape[0]) if torch.is_tensor(input_ids) else 0


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    *,
    device: str | torch.device = "cpu",
    scheduler=None,
    scaler=None,
    ema=None,
    amp_enabled: bool = False,
    amp_dtype: str = "bf16",
    grad_accum_steps: int = 1,
    grad_clip: Optional[float] = None,
    max_batches: Optional[int] = None,
    use_mtp: bool | None = None,
    state: TrainerState | None = None,
    moe_controller=None,
    curriculum=None,
    log_every: int | None = None,
    logger=None,
    on_optimizer_step: Callable[[dict[str, float]], None] | None = None,
) -> Dict[str, float]:
    """Train a complete epoch while accumulating exact token loss sums.

    Kimi NTP and MTP sums are normalized independently at the end of each
    logical optimizer window. Generic scalar-loss models use valid-token
    weighting as a compatibility fallback.
    """

    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive")
    if grad_clip is not None and grad_clip <= 0:
        raise ValueError("grad_clip must be None or positive")
    if max_batches is not None and max_batches < 0:
        raise ValueError("max_batches must be None or non-negative")

    device = torch.device(device)
    model.to(device).train()
    state = TrainerState() if state is None else state
    optimizer.zero_grad(set_to_none=True)

    all_contributions: list[LossContribution] = []
    window: list[LossContribution] = []
    optimizer_steps = 0
    skipped_steps = 0
    grad_norm_pre_total = 0.0
    grad_norm_post_total = 0.0
    grad_norm_count = 0
    num_samples = 0
    started = time.perf_counter()

    def begin_window() -> None:
        if moe_controller is not None:
            moe_controller.begin()

    def discard_window() -> None:
        if moe_controller is not None:
            moe_controller.discard()
        optimizer.zero_grad(set_to_none=True)
        window.clear()

    def finish_window() -> None:
        nonlocal optimizer_steps, skipped_steps
        nonlocal grad_norm_pre_total, grad_norm_post_total, grad_norm_count

        objective, window_stats = combine_window_loss(window)
        if not math.isfinite(float(objective.detach().item())):
            discard_window()
            raise FloatingPointError("non-finite training loss")

        if scaler is None:
            objective.backward()
        else:
            scaler.scale(objective).backward()
            scaler.unscale_(optimizer)

        pre_clip = _grad_norm(model.parameters())
        if pre_clip is not None and not math.isfinite(pre_clip):
            discard_window()
            raise FloatingPointError("non-finite gradient norm")
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(grad_clip),
                error_if_nonfinite=True,
            )
        post_clip = _grad_norm(model.parameters())

        executed = True
        if scaler is None:
            optimizer.step()
        else:
            old_scale = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            executed = float(scaler.get_scale()) >= old_scale

        if executed:
            if scheduler is not None:
                scheduler.step()
            state.optimizer_step += 1
            optimizer_steps += 1
            if ema is not None:
                try:
                    ema.update(model, step=state.optimizer_step)
                except TypeError:
                    ema.update(model)
            if moe_controller is not None:
                moe_controller.commit()
            if curriculum is not None:
                curriculum.update(state.optimizer_step)
                state.curriculum_stage_index = curriculum.stage_index
        else:
            skipped_steps += 1
            state.skipped_optimizer_steps += 1
            if moe_controller is not None:
                moe_controller.discard()

        optimizer.zero_grad(set_to_none=True)
        if pre_clip is not None:
            grad_norm_pre_total += pre_clip
            grad_norm_post_total += post_clip if post_clip is not None else pre_clip
            grad_norm_count += 1

        step_metrics = {
            **window_stats,
            "optimizer_step": float(state.optimizer_step),
            "optimizer_step_executed": float(executed),
            "grad_norm_pre_clip": (
                float("nan") if pre_clip is None else pre_clip
            ),
            "grad_norm_post_clip": (
                float("nan") if post_clip is None else post_clip
            ),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "accumulation_size_effective": float(len(window)),
        }
        if (
            logger is not None
            and log_every is not None
            and executed
            and state.optimizer_step % log_every == 0
        ):
            logger.log(state.optimizer_step, step_metrics)
        if on_optimizer_step is not None:
            on_optimizer_step(step_metrics)
        window.clear()

    try:
        for batch_index, raw_batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch_to_device(normalize_lm_batch(raw_batch), device)
            if not window:
                begin_window()
            if curriculum is not None:
                curriculum.validate_sequence_length(
                    int(batch["input_ids"].shape[1])
                )

            with autocast_ctx(
                device, enabled=amp_enabled, amp_dtype=amp_dtype
            ):
                output = call_model(model, batch, use_mtp=use_mtp)
                contribution = extract_loss_contribution(output, batch)

            window.append(contribution)
            all_contributions.append(
                LossContribution(
                    ntp_loss_sum=contribution.ntp_loss_sum.detach(),
                    ntp_normalizer=contribution.ntp_normalizer,
                    mtp_loss_sum=(
                        None
                        if contribution.mtp_loss_sum is None
                        else contribution.mtp_loss_sum.detach()
                    ),
                    mtp_normalizer=contribution.mtp_normalizer,
                    lambda_mtp=contribution.lambda_mtp,
                    reported_loss=contribution.reported_loss,
                    batch_tokens=contribution.batch_tokens,
                )
            )
            num_samples += _batch_size(batch)
            state.micro_step += 1
            state.samples_seen += _batch_size(batch)
            state.tokens_seen += contribution.batch_tokens
            state.valid_ntp_tokens_seen += int(contribution.ntp_normalizer)
            state.valid_mtp_tokens_seen += int(contribution.mtp_normalizer)

            if len(window) == grad_accum_steps:
                finish_window()

        if window:
            finish_window()
    except BaseException:
        if window or (
            moe_controller is not None and moe_controller.window_open
        ):
            discard_window()
        raise

    if not all_contributions:
        epoch_stats = {
            "loss": 0.0,
            "ntp_loss": 0.0,
            "mtp_loss": float("nan"),
            "ntp_tokens": 0.0,
            "mtp_tokens": 0.0,
            "tokens": 0.0,
        }
    else:
        _, epoch_stats = combine_window_loss(all_contributions)
    elapsed = time.perf_counter() - started
    return {
        **epoch_stats,
        "num_batches": float(len(all_contributions)),
        "optimizer_steps": float(optimizer_steps),
        "skipped_optimizer_steps": float(skipped_steps),
        "num_samples": float(num_samples),
        "grad_norm_pre_clip": grad_norm_pre_total / max(grad_norm_count, 1),
        "grad_norm_post_clip": grad_norm_post_total / max(grad_norm_count, 1),
        "epoch_time_seconds": float(elapsed),
        "tokens_per_second": float(epoch_stats["tokens"] / max(elapsed, 1e-12)),
    }
