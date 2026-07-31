"""Validate a complete distributed profile without allocating model weights."""

from __future__ import annotations

import argparse
from pathlib import Path

from configuration import resolve_kimi_pipeline_profile
from data import load_data_config
from src import load_model_config
from training import load_training_config, validate_pipeline_compatibility


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Validate Kimi DP/TP/EP YAML topology without training"
    )
    result.add_argument(
        "--profile",
        required=True,
        help="profile directory containing data/model/training YAMLs",
    )
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    profile = resolve_kimi_pipeline_profile(args.profile)
    data = load_data_config(profile.data)
    model = load_model_config(profile.model)
    training = load_training_config(profile.training)
    validate_pipeline_compatibility(
        training, model_config=model, data_config=data
    )
    distributed = training.distributed
    print("Distributed profile valid")
    print(
        f"  world_size={distributed.logical_world_size} "
        f"DP={distributed.data_parallel.size} "
        f"TP={distributed.tensor_parallel.size} "
        f"EP={distributed.expert_parallel.size}"
    )
    print(
        f"  wrapper={distributed.data_parallel.mode} "
        f"backend={distributed.backend} "
        f"checkpoint={distributed.checkpoint.format}"
    )
    root = Path(args.profile).as_posix()
    print("Launch")
    print(
        "  torchrun --standalone "
        f"--nproc_per_node={distributed.logical_world_size} "
        f"-m scripts.train_kimi --profile {root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
