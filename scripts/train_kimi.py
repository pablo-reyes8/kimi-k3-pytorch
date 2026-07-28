"""Train Kimi K3 from exactly three YAML files.

Use ``--validate-only`` to check the complete contract without allocating
datasets, model weights, optimizer state or running training.
"""

from __future__ import annotations

import argparse

from data import build_dataloaders_from_yaml, load_data_config
from src import build_model_from_yaml, load_model_config
from training import (
    load_training_config,
    train_kimi_from_yaml,
    validate_pipeline_compatibility,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Kimi K3 three-YAML training entrypoint"
    )
    result.add_argument("--data-config", required=True)
    result.add_argument("--model-config", required=True)
    result.add_argument("--training-config", required=True)
    result.add_argument(
        "--validate-only",
        action="store_true",
        help="validate all YAML contracts without building or training",
    )
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    data_config = load_data_config(args.data_config)
    model_config = load_model_config(args.model_config)
    training_config = load_training_config(args.training_config)
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

    data = build_dataloaders_from_yaml(args.data_config)
    model = build_model_from_yaml(
        args.model_config,
        data_bundle=data,
    )
    train_kimi_from_yaml(
        args.training_config,
        model=model,
        data=data,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
