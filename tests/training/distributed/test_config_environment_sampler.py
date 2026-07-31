from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from configuration.yaml_utils import ConfigError
from data import load_data_config
from src import load_model_config
from src.kimi_k3.config import kimi_k3_cpu_tiny_config
from training import load_training_config, validate_pipeline_compatibility
from training.distributed import (
    DataParallelConfig,
    DistributedCheckpointConfig,
    DistributedConfig,
    DistributedEnvironment,
    InactiveParallelConfig,
    StatefulDistributedSampler,
    TensorParallelConfig,
    coordinates_from_rank,
    distributed_config_from_dict,
    rank_from_coordinates,
)
from training.distributed.parallelize import validate_distributed_model_config


def test_environment_parser_defaults_and_rejects_invalid_rank():
    environment = DistributedEnvironment.from_environ({})
    assert environment == DistributedEnvironment(0, 1, 0, 1)
    with pytest.raises(ValueError, match="RANK must be smaller"):
        DistributedEnvironment.from_environ(
            {"RANK": "2", "WORLD_SIZE": "2"}
        )
    with pytest.raises(ValueError, match="integer"):
        DistributedEnvironment.from_environ({"WORLD_SIZE": "two"})


def test_rank_coordinate_mapping_is_bijective():
    observed = set()
    for dp in range(2):
        for tp in range(3):
            for ep in range(2):
                rank = rank_from_coordinates(
                    dp, tp, ep, tp_size=3, ep_size=2
                )
                observed.add(rank)
                assert coordinates_from_rank(
                    rank, tp_size=3, ep_size=2
                ) == (dp, tp, ep)
    assert observed == set(range(12))


def test_strict_nested_config_and_model_divisibility():
    config = distributed_config_from_dict(
        {
            "enabled": True,
            "backend": "gloo",
            "data_parallel": {"mode": "none", "size": 1},
            "tensor_parallel": {"enabled": True, "size": 2},
        }
    )
    validate_distributed_model_config(
        config,
        kimi_k3_cpu_tiny_config(
            enable_vision=False, enable_mtp=False
        ),
    )
    with pytest.raises(ConfigError, match="unknown"):
        distributed_config_from_dict({"mystery": True})
    with pytest.raises(ValueError, match="exactly when"):
        distributed_config_from_dict(
            {
                "enabled": True,
                "tensor_parallel": {"enabled": False, "size": 2},
            }
        )
    with pytest.raises(ValueError, match="tied Kimi vocabulary"):
        TensorParallelConfig(
            enabled=True,
            size=2,
            shard_embeddings=True,
            shard_lm_head=False,
        )
    with pytest.raises(ValueError, match="must remain disabled"):
        InactiveParallelConfig(enabled=True, size=2)
    assert distributed_config_from_dict(config.to_dict()) == config

    reserved_norms = replace(
        config,
        tensor_parallel=replace(
            config.tensor_parallel, sequence_parallel_norms=True
        ),
    )
    with pytest.raises(ConfigError, match="sequence_parallel_norms"):
        validate_distributed_model_config(
            reserved_norms,
            kimi_k3_cpu_tiny_config(
                enable_vision=False, enable_mtp=False
            ),
        )
    export_only = replace(
        config,
        checkpoint=DistributedCheckpointConfig(format="rank0_full"),
    )
    with pytest.raises(ConfigError, match="export-only"):
        validate_distributed_model_config(
            export_only,
            kimi_k3_cpu_tiny_config(
                enable_vision=False, enable_mtp=False
            ),
        )


def test_fsdp_ema_and_sparse_ddp_fail_during_yaml_compatibility():
    profile = "config/kimi_full_pipeline/cpu_smoke"
    training = load_training_config(f"{profile}/training.yaml")
    model = load_model_config(f"{profile}/model.yaml")
    data = load_data_config(f"{profile}/data.yaml")
    fsdp = DistributedConfig(
        enabled=True,
        data_parallel=DataParallelConfig(mode="fsdp", size=2),
    )
    with pytest.raises(ConfigError, match="sharded EMA"):
        validate_pipeline_compatibility(
            replace(
                training,
                runtime=replace(training.runtime, use_ema=True),
                distributed=fsdp,
            ),
            model_config=model,
            data_config=data,
        )

    sparse_ddp = DistributedConfig(
        enabled=True,
        data_parallel=DataParallelConfig(mode="ddp", size=2),
    )
    with pytest.raises(ConfigError, match="find_unused_parameters"):
        validate_distributed_model_config(sparse_ddp, model)


def test_stateful_sampler_is_disjoint_and_resumes_exact_cursor():
    dataset = torch.arange(12)
    left = StatefulDistributedSampler(
        dataset, num_replicas=2, rank=0, shuffle=False
    )
    right = StatefulDistributedSampler(
        dataset, num_replicas=2, rank=1, shuffle=False
    )
    left_iterator = iter(left)
    assert next(left_iterator) == 0
    assert next(left_iterator) == 2
    state = left.state_dict()
    resumed = StatefulDistributedSampler(
        dataset, num_replicas=2, rank=0, shuffle=False
    )
    resumed.load_state_dict(state)
    assert list(resumed) == [4, 6, 8, 10]
    assert set([0, 2, 4, 6, 8, 10]).isdisjoint(set(iter(right)))
