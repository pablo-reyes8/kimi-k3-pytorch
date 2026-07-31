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
    combine_distributed_window_loss,
    combine_window_loss,
    extract_loss_contribution,
)
from .model_call import call_model
from .diagnostics.loss_metrics import compute_loss_metrics
from .context_curriculum import truncate_batch_to_context
from .state import TrainerState
from .distributed import (
    all_ranks_true,
    clip_grad_norm,
    distributed_grad_norm,
    reduce_scalar_metrics,
)


def _distributed_counter(value: float, context) -> float:
    if context is None or not context.initialized:
        return float(value)
    tensor = torch.tensor(
        float(value), device=context.device, dtype=torch.float64
    )
    for group, size in (
        (context.dp_group, context.dp_size),
        (context.ep_group, context.ep_size),
    ):
        if size > 1:
            torch.distributed.all_reduce(tensor, group=group)
    return float(tensor.item())


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
    max_seq_len: int | None = None,
    context_ignore_index: int = -100,
    image_token_id: int | None = None,
    video_token_id: int | None = None,
    stop_on_context_transition: bool = True,
    distributed_context=None,
) -> Dict[str, Any]:
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
    window_token_capacity = 0
    window_padding_tokens = 0
    diagnostic_metrics_pending: dict[str, float] = {}
    diagnostic_alerts_pending = ()
    window_started = None
    started = time.perf_counter()
    previous_batch_completed = started
    transition_event: dict[str, float] | None = None
    transition_pending = False
    stop_requested = False

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
        nonlocal window_token_capacity, window_padding_tokens
        nonlocal transition_event
        nonlocal transition_pending, stop_requested

        local_ntp_tokens = sum(item.ntp_normalizer for item in window)
        local_mtp_tokens = sum(item.mtp_normalizer for item in window)
        local_tokens = sum(item.batch_tokens for item in window)
        objective, window_stats = combine_distributed_window_loss(
            window, distributed_context
        )
        finite_objective = math.isfinite(float(objective.detach().item()))
        if distributed_context is not None:
            finite_objective = all_ranks_true(
                finite_objective,
                device=device,
            )
        if not finite_objective:
            discard_window()
            raise FloatingPointError("non-finite training loss")
        global_samples = _distributed_counter(
            window_samples, distributed_context
        )
        state.samples_seen += int(global_samples - window_samples)
        state.tokens_seen += int(window_stats["tokens"] - local_tokens)
        state.valid_ntp_tokens_seen += int(
            window_stats["ntp_tokens"] - local_ntp_tokens
        )
        state.valid_mtp_tokens_seen += int(
            window_stats["mtp_tokens"] - local_mtp_tokens
        )

        backward_started = time.perf_counter()
        if scaler is None:
            objective.backward()
        else:
            scaler.scale(objective).backward()
            scaler.unscale_(optimizer)
        backward_ms = (time.perf_counter() - backward_started) * 1000.0

        pre_clip_tensor = distributed_grad_norm(
            model, distributed_context
        )
        pre_clip = (
            None
            if pre_clip_tensor is None
            else float(pre_clip_tensor.item())
        )
        gradients_finite = (
            pre_clip is None or math.isfinite(pre_clip)
        )
        if distributed_context is not None:
            gradients_finite = all_ranks_true(
                gradients_finite, device=device
            )
        if gradients_finite and grad_clip is not None:
            clip_grad_norm(
                model,
                float(grad_clip),
                context=distributed_context,
            )
        post_clip_tensor = distributed_grad_norm(
            model, distributed_context
        )
        post_clip = (
            None
            if post_clip_tensor is None
            else float(post_clip_tensor.item())
        )
        optimizer_metrics = (
            diagnostics.capture_before_optimizer_step()
            if diagnostics is not None
            else {}
        )

        executed = gradients_finite
        optimizer_report = None
        optimizer_started = time.perf_counter()
        if not gradients_finite:
            if scaler is not None:
                scaler.update()
        elif scaler is None:
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
        if distributed_context is not None:
            globally_executed = all_ranks_true(executed, device=device)
            if executed != globally_executed:
                raise RuntimeError(
                    "optimizer execution diverged across distributed ranks"
                )
            executed = globally_executed
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
                transitioned = curriculum.update(state.tokens_seen)
                state.curriculum_stage_index = curriculum.stage_index
                if transitioned:
                    transition = curriculum.last_transition
                    transition_event = {
                        **transition.to_dict(),
                        "loss_before_transition": float(window_stats["loss"]),
                    }
                    transition_pending = True
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
                global_samples / max(step_time_ms / 1000.0, 1e-12)
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
            "context/valid_tokens_per_step": float(window_stats["tokens"]),
            "context/padding_fraction": (
                window_padding_tokens / max(window_token_capacity, 1)
            ),
            "context/tokens_per_second": (
                window_stats["tokens"] / max(step_time_ms / 1000.0, 1e-12)
            ),
            "context/step_time_seconds": step_time_ms / 1000.0,
            "context/peak_memory_mb": (
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
        if curriculum is not None:
            step_metrics.update(curriculum.metrics())
        elif max_seq_len is not None:
            step_metrics.update(
                {
                    "context/stage_index": 0.0,
                    "context/max_seq_len": float(max_seq_len),
                    "context/tokens_seen": float(state.tokens_seen),
                    "context/transition_count": 0.0,
                }
            )
        if distributed_context is not None:
            step_metrics = reduce_scalar_metrics(
                step_metrics, context=distributed_context
            )
        if transition_pending:
            step_metrics.update(
                {
                    "context/old_max_seq_len": float(
                        transition_event["old_max_seq_len"]
                    ),
                    "context/new_max_seq_len": float(
                        transition_event["new_max_seq_len"]
                    ),
                    "context/tokens_seen_at_transition": float(
                        transition_event["tokens_seen_at_transition"]
                    ),
                    "context/loss_before_transition": float(
                        transition_event["loss_before_transition"]
                    ),
                }
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
        if transition_pending and stop_on_context_transition:
            stop_requested = True
        transition_pending = False
        window.clear()
        diagnostic_metrics_pending = {}
        diagnostic_alerts_pending = ()
        window_forward_time = 0.0
        window_data_time = 0.0
        window_samples = 0
        window_token_capacity = 0
        window_padding_tokens = 0

    try:
        for batch_index, raw_batch in enumerate(dataloader):
            window_data_time += time.perf_counter() - previous_batch_completed
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = normalize_lm_batch(raw_batch)
            active_max_seq_len = (
                curriculum.current_max_seq_len()
                if curriculum is not None
                else max_seq_len
            )
            if active_max_seq_len is not None:
                batch, context_batch_metrics = truncate_batch_to_context(
                    batch,
                    active_max_seq_len,
                    ignore_index=context_ignore_index,
                    image_token_id=image_token_id,
                    video_token_id=video_token_id,
                )
            else:
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask")
                capacity = int(input_ids.numel())
                valid = (
                    capacity
                    if attention_mask is None
                    else int(attention_mask.sum().item())
                )
                context_batch_metrics = {
                    "valid_tokens": float(valid),
                    "padding_fraction": 1.0 - valid / max(capacity, 1),
                    "sequence_length": float(input_ids.shape[1]),
                }
            batch = move_batch_to_device(batch, device)
            if not window:
                begin_window()
            if curriculum is not None:
                curriculum.validate_sequence_length(
                    int(batch["input_ids"].shape[1])
                )
                curriculum.validate_valid_tokens(
                    batch["attention_mask"].sum(dim=1)
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
            capacity = int(batch["input_ids"].numel())
            window_token_capacity += capacity
            window_padding_tokens += int(
                round(context_batch_metrics["padding_fraction"] * capacity)
            )
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
                if stop_requested:
                    break

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
        _, epoch_stats = combine_distributed_window_loss(
            all_contributions, distributed_context
        )
    elapsed = time.perf_counter() - started
    return {
        **epoch_stats,
        "num_batches": float(len(all_contributions)),
        "optimizer_steps": float(optimizer_steps),
        "skipped_optimizer_steps": float(skipped_steps),
        "num_samples": _distributed_counter(
            num_samples, distributed_context
        ),
        "grad_norm_pre_clip": grad_norm_pre_total / max(grad_norm_count, 1),
        "grad_norm_post_clip": grad_norm_post_total / max(grad_norm_count, 1),
        "epoch_time_seconds": float(elapsed),
        "tokens_per_second": float(epoch_stats["tokens"] / max(elapsed, 1e-12)),
        "context_transition": transition_event,
    }
