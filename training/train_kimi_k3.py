"""High-level Kimi K3 pretraining orchestration."""

from __future__ import annotations

import math
from pathlib import Path
import shutil
from typing import Any, Callable

import torch

from .autocast import resolve_device, setup_device_and_precision
from .checkpoints import load_checkpoint, save_checkpoint
from .context_curriculum import (
    ContextCurriculumConfig,
    ProgressiveContextCurriculum,
    build_context_loader,
)
from .config import (
    CheckpointConfig,
    OptimizerConfig,
    PretrainingLossConfig,
    PredictionConfig,
    SchedulerConfig,
    TrainingConfig,
)
from .ema import EMA
from .diagnostics import (
    DiagnosticsConfig,
    KimiDiagnosticCollector,
    KimiTrainingPrinter,
)
from .eval_one_epoch import eval_one_epoch
from .logger import JSONLLogger, MemoryLogger
from .moe_control import MoEController
from .optimizer import (
    KimiOptimizerConfig,
    build_kimi_optimizer,
    build_parameter_registry,
)
from .predictions import next_token_preview, print_next_token_preview
from .scheduler import build_warmup_cosine_scheduler
from .seed import set_seed
from .state import TrainerState
from .train_one_epoch import train_one_epoch
from .distributed import (
    DistributedConfig,
    broadcast_model_state,
    build_device_mesh,
    initialize_distributed,
    load_distributed_checkpoint,
    parallelize_kimi_model,
    print_topology,
    save_distributed_checkpoint,
    unwrap_model,
    wrap_data_parallel,
)


def _loader_batches(loader, maximum: int | None) -> int:
    try:
        batches = len(loader)
    except TypeError as error:
        raise ValueError(
            "total_steps is required for a dataloader without __len__"
        ) from error
    return min(batches, maximum) if maximum is not None else batches


def _prune_epoch_checkpoints(directory: Path, run_name: str, keep: int) -> None:
    paths = sorted(directory.glob(f"{run_name}_epoch_*"))
    for path in paths[:-keep]:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _merge_train_segments(segments: list[dict]) -> dict:
    if len(segments) == 1:
        return segments[0]
    additive = (
        "num_batches", "optimizer_steps", "skipped_optimizer_steps",
        "num_samples", "ntp_tokens", "mtp_tokens", "tokens",
        "epoch_time_seconds",
    )
    merged = {
        name: sum(float(segment.get(name, 0.0)) for segment in segments)
        for name in additive
    }
    ntp_tokens = max(merged["ntp_tokens"], 1.0)
    merged["ntp_loss"] = sum(
        float(segment["ntp_loss"]) * float(segment["ntp_tokens"])
        for segment in segments
    ) / ntp_tokens
    mtp_segments = [
        segment for segment in segments
        if math.isfinite(float(segment.get("mtp_loss", float("nan"))))
        and float(segment.get("mtp_tokens", 0.0)) > 0
    ]
    merged["mtp_loss"] = (
        float("nan")
        if not mtp_segments
        else sum(
            float(segment["mtp_loss"]) * float(segment["mtp_tokens"])
            for segment in mtp_segments
        ) / max(merged["mtp_tokens"], 1.0)
    )
    merged["loss"] = sum(
        float(segment["loss"]) * float(segment["tokens"])
        for segment in segments
    ) / max(merged["tokens"], 1.0)
    steps = max(merged["optimizer_steps"], 1.0)
    for name in ("grad_norm_pre_clip", "grad_norm_post_clip"):
        merged[name] = sum(
            float(segment[name]) * float(segment["optimizer_steps"])
            for segment in segments
        ) / steps
    merged["tokens_per_second"] = (
        merged["tokens"] / max(merged["epoch_time_seconds"], 1e-12)
    )
    merged["context_transition"] = next(
        (
            segment["context_transition"]
            for segment in reversed(segments)
            if segment.get("context_transition") is not None
        ),
        None,
    )
    return merged


def train_kimiK3(
    *,
    model,
    train_loader=None,
    train_loader_factory: Callable[[int], Any] | None = None,
    val_loader=None,
    device: str | torch.device = "auto",
    training_config: TrainingConfig | None = None,
    loss_config: PretrainingLossConfig | None = None,
    optimizer_config: OptimizerConfig | None = None,
    kimi_optimizer_config: KimiOptimizerConfig | None = None,
    scheduler_config: SchedulerConfig | None = None,
    diagnostics_config: DiagnosticsConfig | None = None,
    context_curriculum_config: ContextCurriculumConfig | None = None,
    checkpoint_config: CheckpointConfig | None = None,
    prediction_config: PredictionConfig | None = None,
    optimizer=None,
    scheduler=None,
    ema=None,
    use_ema: bool = False,
    ema_decay: float = 0.999,
    eval_use_ema: bool = False,
    curriculum=None,
    logger=None,
    tokenizer=None,
    id_to_text: Callable[[list[int]], str] | None = None,
    total_steps: int | None = None,
    verbose: bool = True,
    distributed_config: DistributedConfig | None = None,
    distributed_context=None,
) -> dict[str, Any]:
    """Train Kimi K3 through modular train/eval/checkpoint components.

    Optimizer and scheduler may be injected for experiments. When omitted,
    the canonical Kimi Per-Head Muon + Muon + AdamW optimizer and
    first-update-aware warmup-cosine schedule are constructed here.
    """

    training_config = training_config or TrainingConfig()
    distributed_config = distributed_config or DistributedConfig()
    if loss_config is not None:
        from src.loss import KimiPretrainingLoss

        if not hasattr(model, "pretraining_loss"):
            raise TypeError(
                "loss_config requires a Kimi model with pretraining_loss"
            )
        model_mtp = getattr(getattr(model, "config", None), "mtp", None)
        default_mtp_weight = (
            float(getattr(model_mtp, "loss_weight", 0.0))
            if training_config.use_mtp
            else 0.0
        )
        model.pretraining_loss = KimiPretrainingLoss(
            lambda_mtp=(
                default_mtp_weight
                if loss_config.mtp_loss_weight is None
                else loss_config.mtp_loss_weight
            ),
            ignore_index=loss_config.ignore_index,
            label_smoothing=loss_config.label_smoothing,
        )
    legacy_optimizer_config = optimizer_config
    optimizer_config = optimizer_config or OptimizerConfig()
    if kimi_optimizer_config is None:
        kimi_optimizer_config = (
            KimiOptimizerConfig(
                kind="adamw",
                adamw_lr=optimizer_config.learning_rate,
                adamw_betas=optimizer_config.betas,
                adamw_eps=optimizer_config.eps,
                weight_decay=optimizer_config.weight_decay,
                qk_clip_enabled=False,
            )
            if legacy_optimizer_config is not None
            else KimiOptimizerConfig()
        )
    scheduler_config = scheduler_config or SchedulerConfig()
    diagnostics_config = diagnostics_config or DiagnosticsConfig()
    context_curriculum_config = (
        context_curriculum_config or ContextCurriculumConfig()
    )
    checkpoint_config = checkpoint_config or CheckpointConfig()
    prediction_config = prediction_config or PredictionConfig()

    owns_distributed_context = distributed_context is None
    if distributed_context is None:
        distributed_context = initialize_distributed(distributed_config)
        distributed_context = build_device_mesh(
            distributed_context, distributed_config
        )
    if distributed_context.world_size != distributed_config.logical_world_size:
        raise RuntimeError("distributed context and configuration disagree")
    rank_seed = (
        training_config.seed
        + distributed_context.dp_rank * distributed_context.ep_size
        + distributed_context.ep_rank
    )
    set_seed(rank_seed, deterministic=training_config.deterministic)
    device_obj = (
        distributed_context.device
        if distributed_config.enabled
        else resolve_device(device)
    )
    if training_config.precision == "fp16" and device_obj.type != "cuda":
        raise RuntimeError("FP16 training requires CUDA")
    amp_enabled = training_config.precision != "fp32"
    precision = setup_device_and_precision(
        device=device_obj,
        amp_enabled=amp_enabled,
        amp_dtype=training_config.precision,
    )
    model.to(device_obj)
    distributed_report = None
    if distributed_config.enabled:
        broadcast_model_state(model, distributed_context)
        distributed_report = parallelize_kimi_model(
            model, distributed_config, distributed_context
        )
        model.to(device_obj)
        if (
            distributed_config.data_parallel.mode == "fsdp"
            and (use_ema or ema is not None)
        ):
            raise ValueError(
                "EMA with FSDP requires a sharded EMA and is unsupported"
            )
        model = wrap_data_parallel(
            model,
            distributed_config,
            distributed_context,
            training_precision=training_config.precision,
        )
    raw_model_config = getattr(unwrap_model(model), "config", None)

    if train_loader is not None and train_loader_factory is not None:
        raise ValueError(
            "pass train_loader or train_loader_factory, not both"
        )
    if curriculum is not None and context_curriculum_config.enabled:
        raise ValueError(
            "pass curriculum or context_curriculum_config, not both"
        )
    if curriculum is None and context_curriculum_config.enabled:
        model_config = raw_model_config
        configured_model_limit = getattr(
            model_config, "max_seq_len", training_config.max_seq_len
        )
        mtp_config = getattr(model_config, "mtp", None)
        mtp_min_seq_len = (
            int(getattr(mtp_config, "future_offset", 2)) + 1
            if training_config.use_mtp
            else 1
        )
        curriculum = ProgressiveContextCurriculum(
            context_curriculum_config,
            training_max_seq_len=training_config.max_seq_len,
            model_max_seq_len=configured_model_limit,
            mtp_min_seq_len=mtp_min_seq_len,
        )
    if (
        curriculum is not None
        and curriculum.enabled
        and curriculum.config.reset_dataloader_on_transition
        and train_loader_factory is None
    ):
        raise ValueError(
            "enabled PCC with loader reset requires train_loader_factory"
        )
    initial_context = (
        curriculum.current_max_seq_len()
        if curriculum is not None
        else training_config.max_seq_len
    )
    if train_loader_factory is not None:
        train_loader = build_context_loader(
            train_loader_factory, initial_context
        )
    if train_loader is None:
        raise ValueError("train_loader or train_loader_factory is required")

    optimizer_info = None
    parameter_registry = None
    if optimizer is None:
        optimizer, parameter_registry = build_kimi_optimizer(
            model, kimi_optimizer_config
        )
        optimizer_info = (
            parameter_registry.to_dict()
            if hasattr(parameter_registry, "to_dict")
            else parameter_registry
        )
    else:
        parameter_registry = (
            optimizer.parameter_assignment_report()
            if hasattr(optimizer, "parameter_assignment_report")
            else build_parameter_registry(model, kind="adamw", strict=False)
        )

    if total_steps is None:
        batches = _loader_batches(
            train_loader, training_config.max_batches_per_epoch
        )
        steps_per_epoch = math.ceil(
            batches / training_config.gradient_accumulation_steps
        )
        total_steps = max(steps_per_epoch * training_config.epochs, 1)
    warmup_steps = scheduler_config.resolve_warmup_steps(total_steps)
    if scheduler is None:
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr=(
                kimi_optimizer_config.adamw_lr
                * scheduler_config.min_lr_ratio
            ),
            min_muon_lr=(
                kimi_optimizer_config.resolved_muon_lr
                * scheduler_config.min_lr_ratio
            ),
            prepare_first_update=True,
        )

    if ema is None and use_ema:
        ema = EMA(model, decay=ema_decay, device="cpu")
    if eval_use_ema and ema is None:
        raise ValueError("eval_use_ema=True requires use_ema or an EMA object")

    state = TrainerState()
    moe_controller = MoEController(model)
    diagnostic_collector = KimiDiagnosticCollector(
        model,
        diagnostics_config,
        parameter_specs=parameter_registry.specs,
    )
    log_rank = distributed_config.diagnostics.log_rank
    is_log_rank = distributed_context.global_rank == log_rank
    printer = KimiTrainingPrinter() if verbose and is_log_rank else None
    output_dir = Path(checkpoint_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    owns_logger = logger is None
    if logger is None:
        logger = (
            JSONLLogger(output_dir / f"{checkpoint_config.run_name}.jsonl")
            if is_log_rank
            else MemoryLogger()
        )

    history: dict[str, list] = {
        "train": [],
        "validation": [],
        "predictions": [],
    }
    resume = checkpoint_config.resume_from
    if resume is not None:
        if distributed_config.enabled:
            resumed = load_distributed_checkpoint(
                resume,
                model=model,
                context=distributed_context,
                optimizer=(
                    optimizer if checkpoint_config.save_optimizer else None
                ),
                scheduler=scheduler,
                scaler=precision["scaler"],
                ema=ema,
                trainer_state=state,
                curriculum=curriculum,
                diagnostics=diagnostic_collector,
                sampler=None,
                restore_rng=checkpoint_config.save_rng_state,
            )
            history = resumed.get("extra", {}).get("history") or history
        else:
            resumed = load_checkpoint(
                resume,
                model,
                optimizer=(
                    optimizer if checkpoint_config.save_optimizer else None
                ),
                scheduler=scheduler,
                scaler=precision["scaler"],
                ema=ema,
                trainer_state=state,
                curriculum=curriculum,
                diagnostics=diagnostic_collector,
                map_location="cpu",
                strict=True,
                restore_rng=checkpoint_config.save_rng_state,
            )
            history = resumed.get("history") or history
        model.to(device_obj)
        if curriculum is not None:
            if curriculum.state.tokens_seen != state.tokens_seen:
                raise ValueError(
                    "trainer and context curriculum token counts disagree"
                )
            state.curriculum_stage_index = curriculum.stage_index
            if train_loader_factory is not None:
                train_loader = build_context_loader(
                    train_loader_factory,
                    curriculum.current_max_seq_len(),
                )
        if distributed_config.enabled:
            sampler_state = resumed.get("sampler_state")
            active_sampler = getattr(train_loader, "sampler", None)
            if sampler_state is not None:
                if not hasattr(active_sampler, "load_state_dict"):
                    raise ValueError(
                        "distributed checkpoint contains sampler state but "
                        "the active loader has no stateful sampler"
                    )
                active_sampler.load_state_dict(sampler_state)

    if printer is not None:
        if distributed_config.enabled:
            print_topology(
                distributed_context,
                log_rank=log_rank,
                per_rank_debug=(
                    distributed_config.diagnostics.per_rank_debug
                ),
                wrapper_mode=distributed_config.data_parallel.mode,
                checkpoint_format=distributed_config.checkpoint.format,
                global_batch_size=(
                    getattr(train_loader, "batch_size", 0)
                    * distributed_context.dp_size
                    * distributed_context.ep_size
                    * training_config.gradient_accumulation_steps
                ),
            )
        printer.print_run_header(
            run_name=checkpoint_config.run_name,
            model=model,
            device=device_obj,
            precision=training_config.precision,
            optimizer_kind=kimi_optimizer_config.kind,
            epochs=training_config.epochs,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            registry=parameter_registry,
        )

    last_checkpoint = None
    try:
        for epoch in range(state.epoch, training_config.epochs):
            epoch_sampler = getattr(train_loader, "sampler", None)
            if hasattr(epoch_sampler, "set_epoch"):
                epoch_sampler.set_epoch(epoch)

            def on_step(step_metrics):
                alerts = step_metrics.pop("_alerts", ())
                if (
                    printer is not None
                    and state.optimizer_step % training_config.log_every_steps == 0
                ):
                    printer.print_step(
                        state.optimizer_step, step_metrics, alerts
                    )

            train_segments = []
            remaining_batches = training_config.max_batches_per_epoch
            while True:
                segment_stats = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device=device_obj,
                    scheduler=scheduler,
                    scaler=precision["scaler"],
                    ema=ema,
                    amp_enabled=precision["amp_enabled"],
                    amp_dtype=training_config.precision,
                    grad_accum_steps=training_config.gradient_accumulation_steps,
                    grad_clip=training_config.grad_clip_norm,
                    max_batches=remaining_batches,
                    use_mtp=training_config.use_mtp,
                    state=state,
                    moe_controller=moe_controller,
                    curriculum=curriculum,
                    log_every=training_config.log_every_steps,
                    logger=logger,
                    diagnostics=diagnostic_collector,
                    on_optimizer_step=on_step,
                    max_seq_len=training_config.max_seq_len,
                    context_ignore_index=(
                        loss_config.ignore_index
                        if loss_config is not None
                        else (
                            int(
                                getattr(
                                    raw_model_config,
                                    "mtp",
                                    None,
                                ).ignore_index
                            )
                            if getattr(
                                raw_model_config,
                                "mtp",
                                None,
                            )
                            is not None
                            else -100
                        )
                    ),
                    image_token_id=getattr(
                        raw_model_config,
                        "image_token_id",
                        None,
                    ),
                    video_token_id=getattr(
                        raw_model_config,
                        "video_token_id",
                        None,
                    ),
                    distributed_context=distributed_context,
                    stop_on_context_transition=(
                        curriculum is not None
                        and curriculum.config.reset_dataloader_on_transition
                    ),
                )
                train_segments.append(segment_stats)
                if remaining_batches is not None:
                    remaining_batches -= int(segment_stats["num_batches"])
                transition = segment_stats.get("context_transition")
                if (
                    transition is None
                    or curriculum is None
                    or not curriculum.config.reset_dataloader_on_transition
                    or (
                        remaining_batches is not None
                        and remaining_batches <= 0
                    )
                ):
                    break
                train_loader = build_context_loader(
                    train_loader_factory,
                    curriculum.current_max_seq_len(),
                )
                transitioned_sampler = getattr(
                    train_loader, "sampler", None
                )
                if hasattr(transitioned_sampler, "set_epoch"):
                    transitioned_sampler.set_epoch(epoch)
            train_stats = _merge_train_segments(train_segments)
            history["train"].append(train_stats)

            eval_stats = None
            should_eval = (
                val_loader is not None
                and (epoch + 1) % training_config.eval_every_epochs == 0
            )
            if should_eval:
                eval_stats = eval_one_epoch(
                    model,
                    val_loader,
                    device=device_obj,
                    amp_enabled=precision["amp_enabled"],
                    amp_dtype=training_config.precision,
                    max_batches=training_config.max_eval_batches,
                    use_mtp=training_config.use_mtp,
                    ema=ema,
                    use_ema=eval_use_ema,
                    distributed_context=distributed_context,
                )
                history["validation"].append(eval_stats)
                eval_loss = float(eval_stats["loss"])
                if (
                    state.best_eval_loss is None
                    or eval_loss < state.best_eval_loss
                ):
                    state.best_eval_loss = eval_loss

            prediction = None
            prediction_every = training_config.prediction_every_epochs
            if prediction_every is not None and (epoch + 1) % prediction_every == 0:
                preview_loader = val_loader if val_loader is not None else train_loader
                try:
                    preview_batch = next(iter(preview_loader))
                except StopIteration:
                    preview_batch = None
                if preview_batch is not None:
                    prediction = next_token_preview(
                        model,
                        preview_batch,
                        device=device_obj,
                        amp_enabled=precision["amp_enabled"],
                        amp_dtype=training_config.precision,
                        use_mtp=training_config.use_mtp,
                        sample_index=prediction_config.sample_index,
                        max_tokens=prediction_config.max_tokens,
                        tokenizer=tokenizer,
                        id_to_text=id_to_text,
                    )
                    prediction["epoch"] = epoch
                    history["predictions"].append(prediction)
                    if verbose and is_log_rank:
                        print_next_token_preview(
                            prediction,
                            title=f"Kimi K3 next-token preview | epoch={epoch}",
                        )

            state.epoch = epoch + 1
            logger.log(
                state.optimizer_step,
                {
                    "epoch": epoch,
                    "train/loss": float(train_stats["loss"]),
                    "eval/loss": (
                        None if eval_stats is None else float(eval_stats["loss"])
                    ),
                    "tokens_seen": state.tokens_seen,
                    "context_stage": state.curriculum_stage_index,
                },
            )
            should_checkpoint = (
                (epoch + 1) % training_config.checkpoint_every_epochs == 0
                or epoch + 1 == training_config.epochs
            )
            if should_checkpoint:
                moe_controller.assert_clean()
                last_checkpoint = output_dir / (
                    f"{checkpoint_config.run_name}_epoch_{epoch:04d}"
                    + ("" if distributed_config.enabled else ".pt")
                )
                serialized_training_config = {
                    "training": training_config.to_dict(),
                    "loss": (
                        None
                        if loss_config is None
                        else loss_config.to_dict()
                    ),
                    "optimizer": optimizer_config.to_dict(),
                    "kimi_optimizer": kimi_optimizer_config.to_dict(),
                    "scheduler": scheduler_config.to_dict(),
                    "diagnostics": diagnostics_config.to_dict(),
                    "context_curriculum": (
                        curriculum.config.to_dict()
                        if curriculum is not None
                        else context_curriculum_config.to_dict()
                    ),
                    "checkpoint": checkpoint_config.to_dict(),
                    "distributed": distributed_config.to_dict(),
                }
                if distributed_config.enabled:
                    save_distributed_checkpoint(
                        last_checkpoint,
                        model=model,
                        context=distributed_context,
                        step=state.optimizer_step,
                        optimizer=(
                            optimizer
                            if checkpoint_config.save_optimizer
                            else None
                        ),
                        scheduler=scheduler,
                        scaler=precision["scaler"],
                        ema=ema,
                        trainer_state=state,
                        curriculum=curriculum,
                        diagnostics=diagnostic_collector,
                        sampler=getattr(train_loader, "sampler", None),
                        model_config=(
                            None
                            if raw_model_config is None
                            else raw_model_config.to_dict()
                        ),
                        training_config=serialized_training_config,
                        registry_fingerprint=parameter_registry.fingerprint,
                        save_rng=(
                            checkpoint_config.save_rng_state
                            and distributed_config.checkpoint.save_rng_per_rank
                        ),
                        extra={
                            "history": history,
                            "optimizer_type": type(optimizer).__name__,
                            "distributed_report": distributed_report,
                        },
                    )
                else:
                    save_checkpoint(
                        last_checkpoint,
                        model,
                        optimizer=(
                            optimizer
                            if checkpoint_config.save_optimizer
                            else None
                        ),
                        scheduler=scheduler,
                        scaler=precision["scaler"],
                        ema=ema,
                        trainer_state=state,
                        curriculum=curriculum,
                        epoch=epoch,
                        global_step=state.optimizer_step,
                        model_config=raw_model_config,
                        training_config=serialized_training_config,
                        history=history,
                        metadata={
                            "optimizer_type": type(optimizer).__name__
                        },
                        save_rng_state=checkpoint_config.save_rng_state,
                        diagnostics=diagnostic_collector,
                    )
                if is_log_rank:
                    _prune_epoch_checkpoints(
                        output_dir,
                        checkpoint_config.run_name,
                        checkpoint_config.keep_last_n,
                    )
            if printer is not None:
                printer.print_epoch_summary(
                    epoch=epoch,
                    state=state,
                    train_stats=train_stats,
                    eval_stats=eval_stats,
                    checkpoint=last_checkpoint if should_checkpoint else None,
                )
    finally:
        if owns_logger:
            logger.close()
        if owns_distributed_context:
            distributed_context.close()

    return {
        "model": model,
        "optimizer": optimizer,
        "optimizer_info": optimizer_info,
        "scheduler": scheduler,
        "ema": ema,
        "state": state,
        "history": history,
        "last_checkpoint": last_checkpoint,
        "precision": precision,
        "parameter_registry": parameter_registry,
        "diagnostics": diagnostic_collector,
        "curriculum": curriculum,
        "train_loader": train_loader,
    }


train_kimi_k3 = train_kimiK3


__all__ = ["train_kimiK3", "train_kimi_k3"]
