from __future__ import annotations

import subprocess
import sys

import pytest
import torch
import torch.nn as nn

from configuration import resolve_kimi_pipeline_profile
from data import load_data_config
from inference import load_inference_yaml_config
from src import load_model_config
from training import load_training_config, validate_pipeline_compatibility
from training.distributed import (
    ColumnParallelLinear,
    DataParallelConfig,
    DistributedConfig,
    DistributedContext,
    RowParallelLinear,
    VocabParallelEmbedding,
    VocabParallelLMHead,
    vocab_parallel_cross_entropy,
    wrap_data_parallel,
)


def test_world_size_one_parallel_primitives_are_exact():
    torch.manual_seed(3)
    linear = nn.Linear(6, 8)
    inputs = torch.randn(2, 4, 6)
    column = ColumnParallelLinear(linear, gather_output=True)
    row = RowParallelLinear(linear, input_is_parallel=False)
    torch.testing.assert_close(column(inputs), linear(inputs))
    torch.testing.assert_close(row(inputs), linear(inputs))

    embedding = nn.Embedding(12, 6)
    head = nn.Linear(6, 12, bias=False)
    head.weight = embedding.weight
    parallel_embedding = VocabParallelEmbedding(embedding)
    parallel_head = VocabParallelLMHead(
        head, tied_weight=parallel_embedding.weight
    )
    ids = torch.tensor([[0, 3, 11]])
    hidden = parallel_embedding(ids)
    torch.testing.assert_close(hidden, embedding(ids))
    torch.testing.assert_close(parallel_head(hidden), head(embedding(ids)))
    assert parallel_embedding.weight is parallel_head.weight

    logits = torch.randn(2, 3, 12, requires_grad=True)
    targets = torch.tensor([[1, 4, -100], [2, 8, 7]])
    expected = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 12),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(targets)
    actual = vocab_parallel_cross_entropy(
        logits,
        targets,
        vocab_start=0,
        vocab_end=12,
        ignore_index=-100,
    )
    torch.testing.assert_close(actual, expected)


def test_fsdp_cpu_failure_is_precise_for_this_torch_build():
    config = DistributedConfig(
        enabled=True,
        backend="gloo",
        data_parallel=DataParallelConfig(mode="fsdp", size=2),
    )
    context = DistributedContext(
        initialized=True,
        owns_process_group=False,
        backend="gloo",
        global_rank=0,
        local_rank=0,
        world_size=2,
        local_world_size=2,
        device=torch.device("cpu"),
        dp_size=2,
    )
    with pytest.raises(RuntimeError, match="requires an accelerator"):
        wrap_data_parallel(
            nn.Linear(2, 2),
            config,
            context,
            training_precision="fp32",
        )


@pytest.mark.parametrize(
    ("name", "world_size", "axes"),
    [
        ("distributed_ddp_2x_t4", 2, (2, 1, 1)),
        ("distributed_tp_2x_24gb", 2, (1, 2, 1)),
        ("distributed_tp_ep_4x_24gb", 4, (1, 2, 2)),
    ],
)
def test_distributed_profiles_validate_without_allocation(
    name, world_size, axes
):
    profile = resolve_kimi_pipeline_profile(
        f"config/kimi_full_pipeline/{name}"
    )
    data = load_data_config(profile.data)
    model = load_model_config(profile.model)
    training = load_training_config(profile.training)
    validate_pipeline_compatibility(
        training, model_config=model, data_config=data
    )
    assert training.distributed.logical_world_size == world_size
    assert (
        training.distributed.data_parallel.size,
        training.distributed.tensor_parallel.size,
        training.distributed.expert_parallel.size,
    ) == axes


def test_distributed_validation_cli_prints_torchrun_command():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.validate_distributed_config",
            "--profile",
            "config/kimi_full_pipeline/distributed_tp_2x_24gb",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "--nproc_per_node=2" in completed.stdout
    assert "DP=1 TP=2 EP=1" in completed.stdout


def test_distributed_inference_yaml_uses_same_strict_topology():
    config = load_inference_yaml_config(
        "config/inference/distributed_greedy_tp2.yaml"
    )
    assert config.distributed.logical_world_size == 2
    assert config.distributed.tensor_parallel.enabled
    assert config.generation.do_sample is False
