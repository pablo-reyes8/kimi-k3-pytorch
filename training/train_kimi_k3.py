"""High-level Kimi K3 pretraining orchestration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import torch

from .adam_optimizer import build_adamw_optimizer
from .autocast import resolve_device, setup_device_and_precision
from .checkpoints import load_checkpoint, save_checkpoint
from .config import (
    CheckpointConfig,
    OptimizerConfig,
    PredictionConfig,
    SchedulerConfig,
    TrainingConfig,
)
from .ema import EMA
from .eval_one_epoch import eval_one_epoch
from .logger import JSONLLogger
from .moe_control import MoEController
from .predictions import next_token_preview, print_next_token_preview
from .scheduler import build_warmup_cosine_scheduler
from .seed import set_seed
from .state import TrainerState
from .train_one_epoch import train_one_epoch


def _loader_batches(loader, maximum: int | None) -> int:
    try:
        batches = len(loader)
    except TypeError as error:
        raise ValueError(
            "total_steps is required for a dataloader without __len__"
        ) from error
    return min(batches, maximum) if maximum is not None else batches


def _prune_epoch_checkpoints(directory: Path, run_name: str, keep: int) -> None:
    paths = sorted(directory.glob(f"{run_name}_epoch_*.pt"))
    for path in paths[:-keep]:
        path.unlink()


def train_kimiK3(
    *,
    model,
    train_loader,
    val_loader=None,
    device: str | torch.device = "auto",
    training_config: TrainingConfig | None = None,
    optimizer_config: OptimizerConfig | None = None,
    scheduler_config: SchedulerConfig | None = None,
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
) -> dict[str, Any]:
    """Train Kimi K3 through modular train/eval/checkpoint components.

    Optimizer and scheduler may be injected for experiments. When omitted,
    the phase-1 AdamW and warmup-cosine baseline are constructed here.
    """

    training_config = training_config or TrainingConfig()
    optimizer_config = optimizer_config or OptimizerConfig()
    scheduler_config = scheduler_config or SchedulerConfig()
    checkpoint_config = checkpoint_config or CheckpointConfig()
    prediction_config = prediction_config or PredictionConfig()

    set_seed(training_config.seed, deterministic=training_config.deterministic)
    device_obj = resolve_device(device)
    if training_config.precision == "fp16" and device_obj.type != "cuda":
        raise RuntimeError("FP16 training requires CUDA")
    amp_enabled = training_config.precision != "fp32"
    precision = setup_device_and_precision(
        device=device_obj,
        amp_enabled=amp_enabled,
        amp_dtype=training_config.precision,
    )
    model.to(device_obj)

    optimizer_info = None
    if optimizer is None:
        optimizer, optimizer_info = build_adamw_optimizer(
            model,
            learning_rate=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
        )

    if total_steps is None:
        batches = _loader_batches(
            train_loader, training_config.max_batches_per_epoch
        )
        steps_per_epoch = math.ceil(
            batches / training_config.gradient_accumulation_steps
        )
        total_steps = max(steps_per_epoch * training_config.epochs, 1)
    if scheduler is None:
        warmup_steps = scheduler_config.resolve_warmup_steps(total_steps)
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr=(
                optimizer_config.learning_rate
                * scheduler_config.min_lr_ratio
            ),
        )

    if ema is None and use_ema:
        ema = EMA(model, decay=ema_decay, device="cpu")
    if eval_use_ema and ema is None:
        raise ValueError("eval_use_ema=True requires use_ema or an EMA object")

    state = TrainerState()
    moe_controller = MoEController(model)
    output_dir = Path(checkpoint_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    owns_logger = logger is None
    if logger is None:
        logger = JSONLLogger(output_dir / f"{checkpoint_config.run_name}.jsonl")

    history: dict[str, list] = {
        "train": [],
        "validation": [],
        "predictions": [],
    }
    resume = checkpoint_config.resume_from
    if resume is not None:
        resumed = load_checkpoint(
            resume,
            model,
            optimizer=optimizer if checkpoint_config.save_optimizer else None,
            scheduler=scheduler,
            scaler=precision["scaler"],
            ema=ema,
            trainer_state=state,
            curriculum=curriculum,
            map_location="cpu",
            strict=True,
            restore_rng=checkpoint_config.save_rng_state,
        )
        history = resumed.get("history") or history
        model.to(device_obj)

    if verbose:
        print(
            f"Kimi K3 training | device={device_obj} "
            f"precision={training_config.precision} "
            f"epochs={training_config.epochs} "
            f"start_epoch={state.epoch} total_steps={total_steps}"
        )

    last_checkpoint = None
    try:
        for epoch in range(state.epoch, training_config.epochs):
            train_stats = train_one_epoch(
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
                max_batches=training_config.max_batches_per_epoch,
                use_mtp=training_config.use_mtp,
                state=state,
                moe_controller=moe_controller,
                curriculum=curriculum,
                log_every=training_config.log_every_steps,
                logger=logger,
            )
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
                    if verbose:
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
            if verbose:
                message = (
                    f"epoch={epoch} train_loss={train_stats['loss']:.6f}"
                )
                if eval_stats is not None:
                    message += f" eval_loss={eval_stats['loss']:.6f}"
                print(message)

            should_checkpoint = (
                (epoch + 1) % training_config.checkpoint_every_epochs == 0
                or epoch + 1 == training_config.epochs
            )
            if should_checkpoint:
                moe_controller.assert_clean()
                last_checkpoint = output_dir / (
                    f"{checkpoint_config.run_name}_epoch_{epoch:04d}.pt"
                )
                save_checkpoint(
                    last_checkpoint,
                    model,
                    optimizer=(
                        optimizer if checkpoint_config.save_optimizer else None
                    ),
                    scheduler=scheduler,
                    scaler=precision["scaler"],
                    ema=ema,
                    trainer_state=state,
                    curriculum=curriculum,
                    epoch=epoch,
                    global_step=state.optimizer_step,
                    model_config=getattr(model, "config", None),
                    training_config={
                        "training": training_config.to_dict(),
                        "optimizer": optimizer_config.to_dict(),
                        "scheduler": scheduler_config.to_dict(),
                        "checkpoint": checkpoint_config.to_dict(),
                    },
                    history=history,
                    metadata={"optimizer_type": type(optimizer).__name__},
                    save_rng_state=checkpoint_config.save_rng_state,
                )
                _prune_epoch_checkpoints(
                    output_dir,
                    checkpoint_config.run_name,
                    checkpoint_config.keep_last_n,
                )
    finally:
        if owns_logger:
            logger.close()

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
    }


train_kimi_k3 = train_kimiK3


__all__ = ["train_kimiK3", "train_kimi_k3"]
