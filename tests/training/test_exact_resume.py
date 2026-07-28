import copy
import random

import numpy as np
import torch

from src import BaselineCausalLM, BaselineCausalLMConfig
from training import (
    TrainerState,
    WarmupCosineLR,
    build_adamw_optimizer,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)


def tiny_lm():
    return BaselineCausalLM(
        BaselineCausalLMConfig(
            vocab_size=20,
            d_model=12,
            n_layers=1,
            n_heads=3,
            mlp_hidden_dim=24,
            max_seq_len=6,
            dropout=0,
        )
    )


def batches():
    generator = torch.Generator().manual_seed(91)
    return [
        {
            "input_ids": torch.randint(0, 20, (2, 6), generator=generator),
            "labels": torch.randint(0, 20, (2, 6), generator=generator),
        }
        for _ in range(2)
    ]


def assert_nested_equal(expected, actual):
    if torch.is_tensor(expected):
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            assert_nested_equal(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            assert_nested_equal(left, right)
    else:
        assert expected == actual


def test_checkpoint_resume_matches_uninterrupted_cpu_trajectory(tmp_path):
    torch.manual_seed(17)
    initial = tiny_lm()
    continuous = copy.deepcopy(initial)
    split = copy.deepcopy(initial)
    data = batches()

    continuous_optimizer, _ = build_adamw_optimizer(
        continuous, learning_rate=1e-3
    )
    continuous_scheduler = WarmupCosineLR(
        continuous_optimizer, total_steps=2, warmup_steps=1
    )
    continuous_state = TrainerState()
    for batch in data:
        train_one_epoch(
            continuous,
            [batch],
            continuous_optimizer,
            scheduler=continuous_scheduler,
            state=continuous_state,
        )

    split_optimizer, _ = build_adamw_optimizer(split, learning_rate=1e-3)
    split_scheduler = WarmupCosineLR(
        split_optimizer, total_steps=2, warmup_steps=1
    )
    split_state = TrainerState()
    train_one_epoch(
        split,
        [data[0]],
        split_optimizer,
        scheduler=split_scheduler,
        state=split_state,
    )
    path = save_checkpoint(
        tmp_path / "resume.pt",
        split,
        optimizer=split_optimizer,
        scheduler=split_scheduler,
        trainer_state=split_state,
    )

    random.random()
    np.random.rand()
    torch.rand(3)
    resumed = tiny_lm()
    resumed_optimizer, _ = build_adamw_optimizer(
        resumed, learning_rate=1e-3
    )
    resumed_scheduler = WarmupCosineLR(
        resumed_optimizer, total_steps=2, warmup_steps=1
    )
    resumed_state = TrainerState()
    load_checkpoint(
        path,
        resumed,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        trainer_state=resumed_state,
        restore_rng=True,
    )
    train_one_epoch(
        resumed,
        [data[1]],
        resumed_optimizer,
        scheduler=resumed_scheduler,
        state=resumed_state,
    )

    assert_nested_equal(continuous.state_dict(), resumed.state_dict())
    assert_nested_equal(
        continuous_optimizer.state_dict(), resumed_optimizer.state_dict()
    )
    assert continuous_scheduler.state_dict() == resumed_scheduler.state_dict()
    assert continuous_state.state_dict() == resumed_state.state_dict()
