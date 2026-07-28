"""Validate a data YAML, optionally building and inspecting its first batch."""

from __future__ import annotations

import argparse
import json

from data import build_dataloaders_from_yaml, load_data_config
from data.inspection import inspect_lm_dataloader


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Validate Kimi data YAML")
    result.add_argument("config")
    result.add_argument(
        "--build",
        action="store_true",
        help="build the dataset/loader; HF profiles may download data",
    )
    result.add_argument("--batches", type=int, default=1)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    config = load_data_config(args.config)
    print(
        f"Data YAML valid: {config.name} "
        f"kind={config.kind} block_size={config.max_seq_len}"
    )
    if args.build:
        if args.batches <= 0:
            raise ValueError("--batches must be positive")
        bundle = build_dataloaders_from_yaml(args.config)
        summary = inspect_lm_dataloader(
            bundle.train_loader,
            tokenizer=bundle.tokenizer,
            num_batches=args.batches,
        )
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
