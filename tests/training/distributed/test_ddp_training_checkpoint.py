from __future__ import annotations

import copy
from pathlib import Path

import torch

from src import KimiK3
from src.kimi_k3.config import kimi_k3_cpu_tiny_config
from training import (
    CheckpointConfig,
    DiagnosticsConfig,
    KimiOptimizerConfig,
    PredictionConfig,
    SchedulerConfig,
    TrainerState,
    TrainingConfig,
    train_kimiK3,
    train_one_epoch,
)
from training.distributed import (
    DataParallelConfig,
    DistributedConfig,
    build_device_mesh,
    initialize_distributed,
    load_distributed_checkpoint,
    save_distributed_checkpoint,
    unwrap_model,
    wrap_data_parallel,
)

from tests.training.distributed.helpers import configure_rank, launch


def _rank_batch(rank: int) -> dict[str, torch.Tensor]:
    ids = (
        torch.tensor([[1, 5, 6, 2, 0]])
        if rank == 0
        else torch.tensor([[1, 7, 2, 0, 0]])
    )
    mask = ids.ne(0)
    return {"input_ids": ids, "labels": ids, "attention_mask": mask}


def _ddp_worker(rank, world_size, port, output, checkpoint_root):
    configure_rank(rank, world_size, port)
    distributed = DistributedConfig(
        enabled=True,
        backend="gloo",
        data_parallel=DataParallelConfig(
            mode="ddp",
            size=world_size,
            find_unused_parameters=True,
        ),
    )
    context = build_device_mesh(
        initialize_distributed(distributed), distributed
    )
    try:
        torch.manual_seed(41)
        config = kimi_k3_cpu_tiny_config(
            enable_vision=False, enable_mtp=False
        )
        base = KimiK3(config)
        reference = copy.deepcopy(base)
        ddp = wrap_data_parallel(
            base,
            distributed,
            context,
            training_precision="fp32",
        )
        optimizer = torch.optim.AdamW(
            ddp.parameters(), lr=1e-3, weight_decay=0.0
        )
        state = TrainerState()
        stats = train_one_epoch(
            ddp,
            [_rank_batch(rank)],
            optimizer,
            device="cpu",
            use_mtp=False,
            state=state,
            grad_clip=None,
            distributed_context=context,
        )
        assert stats["tokens"] == 7
        assert state.tokens_seen == 7

        torch.distributed.barrier()
        if rank == 0:
            reference_optimizer = torch.optim.AdamW(
                reference.parameters(), lr=1e-3, weight_decay=0.0
            )
            global_batch = {
                key: torch.cat(
                    [_rank_batch(0)[key], _rank_batch(1)[key]], dim=0
                )
                for key in _rank_batch(0)
            }
            train_one_epoch(
                reference,
                [global_batch],
                reference_optimizer,
                device="cpu",
                use_mtp=False,
                grad_clip=None,
            )
            for actual, expected in zip(
                unwrap_model(ddp).parameters(), reference.parameters()
            ):
                torch.testing.assert_close(
                    actual, expected, atol=2e-6, rtol=2e-6
                )

        checkpoint = Path(checkpoint_root) / "step_1"
        save_distributed_checkpoint(
            checkpoint,
            model=ddp,
            context=context,
            step=1,
            optimizer=optimizer,
            trainer_state=state,
            model_config=config.to_dict(),
            training_config={"precision": "fp32"},
        )
        train_one_epoch(
            ddp,
            [_rank_batch(rank)],
            optimizer,
            device="cpu",
            use_mtp=False,
            state=state,
            grad_clip=None,
            distributed_context=context,
        )
        expected_next_parameters = [
            parameter.detach().clone()
            for parameter in unwrap_model(ddp).parameters()
        ]
        expected_next_state = state.state_dict()
        restored = load_distributed_checkpoint(
            checkpoint,
            model=ddp,
            context=context,
            optimizer=optimizer,
            trainer_state=state,
        )
        assert state.tokens_seen == 7
        train_one_epoch(
            ddp,
            [_rank_batch(rank)],
            optimizer,
            device="cpu",
            use_mtp=False,
            state=state,
            grad_clip=None,
            distributed_context=context,
        )
        for actual, expected in zip(
            unwrap_model(ddp).parameters(), expected_next_parameters
        ):
            torch.testing.assert_close(actual, expected)
        assert state.state_dict() == expected_next_state
        assert restored["metadata"]["step"] == 1
        torch.manual_seed(43)
        master_model = KimiK3(config)
        master_result = train_kimiK3(
            model=master_model,
            train_loader=[_rank_batch(rank)],
            device="cpu",
            training_config=TrainingConfig(
                epochs=1,
                precision="fp32",
                use_mtp=False,
                max_batches_per_epoch=1,
                max_eval_batches=1,
                log_every_steps=1,
                eval_every_epochs=1,
                checkpoint_every_epochs=1,
                prediction_every_epochs=None,
                max_seq_len=5,
            ),
            kimi_optimizer_config=KimiOptimizerConfig(
                kind="adamw",
                adamw_lr=1e-3,
                weight_decay=0.0,
                per_head_qkv=False,
                qk_clip_enabled=False,
            ),
            scheduler_config=SchedulerConfig(
                warmup_ratio=0.0, min_lr_ratio=0.0
            ),
            diagnostics_config=DiagnosticsConfig(enabled=False),
            checkpoint_config=CheckpointConfig(
                output_dir=Path(checkpoint_root) / "master",
                run_name="ddp_master",
                keep_last_n=1,
            ),
            prediction_config=PredictionConfig(max_tokens=2),
            distributed_config=distributed,
            distributed_context=context,
            total_steps=1,
            verbose=False,
        )
        assert master_result["state"].tokens_seen == 7
        assert (
            Path(master_result["last_checkpoint"]) / "SUCCESS"
        ).is_file()
        if rank == 0:
            torch.save(
                {
                    "tokens": stats["tokens"],
                    "checkpoint_complete": (
                        checkpoint / "SUCCESS"
                    ).is_file(),
                },
                output,
            )
    finally:
        context.close()


def test_two_rank_ddp_matches_global_batch_and_checkpoint_roundtrip(tmp_path):
    output = tmp_path / "ddp.pt"
    launch(_ddp_worker, output, str(tmp_path / "checkpoint"))
    result = torch.load(output, weights_only=True)
    assert result == {"tokens": 7.0, "checkpoint_complete": True}
