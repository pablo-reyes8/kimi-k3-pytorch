"""Train Kimi K3 from one complete profile or three explicit YAML files.

Use ``--validate-only`` to check the complete contract without allocating
datasets, model weights, optimizer state or running training.
"""

from __future__ import annotations

import argparse

from configuration import resolve_kimi_pipeline_profile
from data import build_dataloaders_from_yaml, load_data_config
from src import build_model_from_yaml, load_model_config
from training import (
    load_training_config,
    train_kimi_from_yaml,
    validate_pipeline_compatibility,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Kimi K3 full-profile training entrypoint"
    )
    result.add_argument(
        "--profile",
        help="directory containing data.yaml, model.yaml and training.yaml",
    )
    result.add_argument("--data-config")
    result.add_argument("--model-config")
    result.add_argument("--training-config")
    result.add_argument(
        "--validate-only",
        action="store_true",
        help="validate all YAML contracts without building or training",
    )
    return result


def main(argv=None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    explicit = (
        args.data_config,
        args.model_config,
        args.training_config,
    )
    if args.profile:
        if any(explicit):
            argument_parser.error(
                "--profile cannot be combined with individual YAML paths"
            )
        profile = resolve_kimi_pipeline_profile(args.profile)
        data_path, model_path, training_path = (
            profile.data,
            profile.model,
            profile.training,
        )
    else:
        if not all(explicit):
            argument_parser.error(
                "pass --profile or all of --data-config, --model-config "
                "and --training-config"
            )
        data_path, model_path, training_path = explicit

    data_config = load_data_config(data_path)
    model_config = load_model_config(model_path)
    training_config = load_training_config(training_path)
    validate_pipeline_compatibility(
        training_config,
        model_config=model_config,
        data_config=data_config,
    )
    if args.validate_only:
        print("YAML pipeline valid")
        print(
            f"  data={data_config.name} "
            f"block_size={data_config.max_seq_len}"
        )
        print(
            f"  model=d_model:{model_config.d_model} "
            f"layers:{model_config.backbone.num_transformer_layers}"
        )
        print(
            f"  optimizer={training_config.optimizer.kind} "
            f"precision={training_config.training.precision}"
        )
        return 0

    data = build_dataloaders_from_yaml(data_path)
    model = build_model_from_yaml(
        model_path,
        data_bundle=data,
    )
    train_kimi_from_yaml(
        training_path,
        model=model,
        data=data,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
