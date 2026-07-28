"""Validate model YAML without allocating canonical weights by default."""

from __future__ import annotations

import argparse

from src import build_model_from_yaml, load_model_config


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Validate Kimi model YAML")
    result.add_argument("config")
    result.add_argument(
        "--instantiate",
        action="store_true",
        help="also allocate model parameters; never runs a forward",
    )
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    config = load_model_config(args.config)
    print("Model YAML valid")
    print(
        f"  d_model={config.d_model} "
        f"layers={config.backbone.num_transformer_layers} "
        f"experts={config.backbone.stable_latent_moe_config.num_routed_experts}"
    )
    print(
        f"  vision={config.enable_vision} mtp={config.enable_mtp}"
    )
    if args.instantiate:
        model = build_model_from_yaml(args.config)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f"  parameters={parameters:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
