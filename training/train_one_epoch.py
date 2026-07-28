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
from .diagnostics.loss_metrics import compute_loss_metrics
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
    diagnostics=None,
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
    window_forward_time = 0.0
    window_data_time = 0.0
    window_samples = 0
    diagnostic_metrics_pending: dict[str, float] = {}
    diagnostic_alerts_pending = ()
    window_started = None
    started = time.perf_counter()
    previous_batch_completed = started

    def begin_window() -> None:
        nonlocal window_started
        window_started = time.perf_counter()
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
        nonlocal diagnostic_metrics_pending, diagnostic_alerts_pending
        nonlocal window_forward_time, window_samples
        nonlocal window_data_time

        objective, window_stats = combine_window_loss(window)
        if not math.isfinite(float(objective.detach().item())):
            discard_window()
            raise FloatingPointError("non-finite training loss")

        backward_started = time.perf_counter()
        if scaler is None:
            objective.backward()
        else:
            scaler.scale(objective).backward()
            scaler.unscale_(optimizer)
        backward_ms = (time.perf_counter() - backward_started) * 1000.0

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
        optimizer_metrics = (
            diagnostics.capture_before_optimizer_step()
            if diagnostics is not None
            else {}
        )

        executed = True
        optimizer_report = None
        optimizer_started = time.perf_counter()
        if scaler is None:
            optimizer_report = optimizer.step()
            executed = bool(
                getattr(optimizer_report, "executed", True)
            )
        else:
            old_scale = float(scaler.get_scale())
            optimizer_report = scaler.step(optimizer)
            scaler.update()
            executed = float(scaler.get_scale()) >= old_scale
            if optimizer_report is not None:
                executed = executed and bool(
                    getattr(optimizer_report, "executed", True)
                )
        optimizer_ms = (time.perf_counter() - optimizer_started) * 1000.0

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
        step_time_ms = (
            (time.perf_counter() - window_started) * 1000.0
            if window_started is not None
            else 0.0
        )
        after_optimizer_metrics = {}
        if diagnostics is not None:
            (
                after_optimizer_metrics,
                diagnostic_alerts_pending,
            ) = diagnostics.capture_after_optimizer_step(
                step=state.optimizer_step,
                step_time_ms=step_time_ms,
            )
        if pre_clip is not None:
            grad_norm_pre_total += pre_clip
            grad_norm_post_total += post_clip if post_clip is not None else pre_clip
            grad_norm_count += 1

        step_metrics = {
            **window_stats,
            **compute_loss_metrics(
                total_loss=window_stats["loss"],
                ntp_loss=window_stats["ntp_loss"],
                mtp_loss=(
                    None
                    if not math.isfinite(window_stats["mtp_loss"])
                    else window_stats["mtp_loss"]
                ),
                ntp_tokens=window_stats["ntp_tokens"],
                mtp_tokens=window_stats["mtp_tokens"],
                lambda_mtp=(
                    window[0].lambda_mtp if window else 0.0
                ),
            ),
            "optimizer_step": float(state.optimizer_step),
            "optimizer_step_executed": float(executed),
            "grad_norm_pre_clip": (
                float("nan") if pre_clip is None else pre_clip
            ),
            "grad_norm_post_clip": (
                float("nan") if post_clip is None else post_clip
            ),
            "train/grad_norm_global_preclip": (
                float("nan") if pre_clip is None else pre_clip
            ),
            "train/grad_norm_global_postclip": (
                float("nan") if post_clip is None else post_clip
            ),
            "train/grad_clip_coefficient": (
                1.0
                if pre_clip is None or grad_clip is None
                else min(1.0, float(grad_clip) / max(pre_clip, 1e-12))
            ),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "accumulation_size_effective": float(len(window)),
            "train/step_time_ms": step_time_ms,
            "train/forward_time_ms": window_forward_time * 1000.0,
            "train/data_time_ms": window_data_time * 1000.0,
            "train/backward_time_ms": backward_ms,
            "train/optimizer_time_ms": optimizer_ms,
            "train/samples_per_second": (
                window_samples / max(step_time_ms / 1000.0, 1e-12)
            ),
            "train/tokens_per_second": (
                window_stats["tokens"] / max(step_time_ms / 1000.0, 1e-12)
            ),
            "train/amp_scale": (
                1.0 if scaler is None else float(scaler.get_scale())
            ),
            "train/overflow_steps_total": float(
                state.skipped_optimizer_steps
            ),
            "train/skipped_steps_total": float(
                state.skipped_optimizer_steps
            ),
            "train/memory_allocated_mb": (
                torch.cuda.memory_allocated(device) / (1024 ** 2)
                if device.type == "cuda"
                else 0.0
            ),
            "train/memory_reserved_mb": (
                torch.cuda.memory_reserved(device) / (1024 ** 2)
                if device.type == "cuda"
                else 0.0
            ),
            "train/max_memory_allocated_mb": (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                if device.type == "cuda"
                else 0.0
            ),
            **diagnostic_metrics_pending,
            **optimizer_metrics,
            **after_optimizer_metrics,
        }
        if optimizer_report is not None:
            step_metrics.update(
                getattr(optimizer_report, "update_metrics", {})
            )
            if getattr(optimizer_report, "qk_clip_applied", False):
                for name in tuple(step_metrics):
                    if name.endswith("/qk_clip_active"):
                        step_metrics[name] = 1.0
            if hasattr(optimizer_report, "adamw_lr"):
                step_metrics["lr/adamw"] = float(
                    optimizer_report.adamw_lr
                )
                step_metrics["lr/muon"] = float(
                    optimizer_report.muon_lr
                )
        if (
            logger is not None
            and log_every is not None
            and executed
            and state.optimizer_step % log_every == 0
        ):
            logger.log(state.optimizer_step, step_metrics)
        if on_optimizer_step is not None:
            step_metrics["_alerts"] = diagnostic_alerts_pending
            on_optimizer_step(step_metrics)
        window.clear()
        diagnostic_metrics_pending = {}
        diagnostic_alerts_pending = ()
        window_forward_time = 0.0
        window_data_time = 0.0
        window_samples = 0

    try:
        for batch_index, raw_batch in enumerate(dataloader):
            window_data_time += time.perf_counter() - previous_batch_completed
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch_to_device(normalize_lm_batch(raw_batch), device)
            if not window:
                begin_window()
            if curriculum is not None:
                curriculum.validate_sequence_length(
                    int(batch["input_ids"].shape[1])
                )

            first_in_window = not window
            diagnostic_kwargs = (
                diagnostics.model_kwargs_for_step(state.optimizer_step + 1)
                if diagnostics is not None and first_in_window
                else {}
            )
            forward_started = time.perf_counter()
            with autocast_ctx(
                device, enabled=amp_enabled, amp_dtype=amp_dtype
            ):
                output = call_model(
                    model,
                    batch,
                    use_mtp=use_mtp,
                    extra_kwargs=diagnostic_kwargs,
                )
                contribution = extract_loss_contribution(output, batch)
            window_forward_time += time.perf_counter() - forward_started
            if diagnostics is not None and first_in_window:
                diagnostic_metrics_pending = diagnostics.collect_output(
                    output, step=state.optimizer_step + 1
                )

            window.append(contribution)
            window_samples += _batch_size(batch)
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
            previous_batch_completed = time.perf_counter()

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
