import math

import torch
from torch.utils.data import DataLoader

from src import KimiK3, kimi_k3_cpu_tiny_config
from training import (
    CheckpointConfig,
    MoEController,
    PredictionConfig,
    TrainerState,
    TrainingConfig,
    build_adamw_optimizer,
    load_checkpoint,
    train_kimiK3,
    train_one_epoch,
)


def kimi_loader(num_samples=2):
    generator = torch.Generator().manual_seed(22)
    # 3 and 4 are reserved visual placeholders in the tiny Kimi config.
    ids = torch.randint(5, 100, (num_samples, 8), generator=generator)
    return DataLoader(
        [
            {
                "input_ids": row,
                "labels": row.clone(),
                "attention_mask": torch.ones(8, dtype=torch.bool),
            }
            for row in ids
        ],
        batch_size=1,
        shuffle=False,
    )


def test_real_kimi_one_step_updates_mtp_and_commits_every_qb_layer():
    torch.manual_seed(3)
    model = KimiK3(kimi_k3_cpu_tiny_config()).cpu()
    optimizer, _ = build_adamw_optimizer(
        model, learning_rate=2e-4, weight_decay=0
    )
    controller = MoEController(model)
    before_mtp = model.mtp.fusion.projection.weight.detach().clone()
    before_biases = [layer.routing_bias.clone() for layer in controller.layers]
    state = TrainerState()
    stats = train_one_epoch(
        model,
        kimi_loader(),
        optimizer,
        grad_accum_steps=2,
        grad_clip=1.0,
        use_mtp=True,
        state=state,
        moe_controller=controller,
    )
    assert math.isfinite(stats["loss"])
    assert stats["ntp_tokens"] == 14
    assert stats["mtp_tokens"] == 12
    assert stats["optimizer_steps"] == 1
    assert state.valid_ntp_tokens_seen == 14
    assert state.valid_mtp_tokens_seen == 12
    assert not torch.equal(
        before_mtp, model.mtp.fusion.projection.weight
    )
    assert all(
        not torch.equal(before, layer.routing_bias)
        for before, layer in zip(before_biases, controller.layers)
    )


def test_train_kimiK3_orchestrates_eval_preview_logging_and_checkpoint(tmp_path):
    torch.manual_seed(7)
    model = KimiK3(kimi_k3_cpu_tiny_config()).cpu()
    loader = kimi_loader(num_samples=1)
    result = train_kimiK3(
        model=model,
        train_loader=loader,
        val_loader=loader,
        device="cpu",
        training_config=TrainingConfig(
            epochs=1,
            precision="fp32",
            use_mtp=True,
            log_every_steps=1,
            prediction_every_epochs=1,
        ),
        checkpoint_config=CheckpointConfig(
            output_dir=tmp_path,
            run_name="integration",
        ),
        prediction_config=PredictionConfig(max_tokens=4),
        verbose=False,
    )
    assert result["state"].epoch == 1
    assert result["state"].optimizer_step == 1
    assert len(result["history"]["train"]) == 1
    assert len(result["history"]["validation"]) == 1
    assert len(result["history"]["predictions"]) == 1
    assert result["last_checkpoint"].exists()
    assert (tmp_path / "integration.jsonl").exists()

    restored = KimiK3(kimi_k3_cpu_tiny_config()).cpu()
    restored_state = TrainerState()
    loaded = load_checkpoint(
        result["last_checkpoint"],
        restored,
        trainer_state=restored_state,
    )
    assert loaded["format_version"] == 2
    assert restored_state.optimizer_step == 1
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)
