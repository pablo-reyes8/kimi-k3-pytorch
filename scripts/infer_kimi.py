"""CLI for native-cache autoregressive Kimi K3 inference."""

from __future__ import annotations

import argparse
from dataclasses import replace

from configuration import resolve_kimi_pipeline_profile
from data import load_tokenizer_from_data_yaml
from inference import (
    ModelLoadConfig,
    inference_autoregressive,
    load_generation_config,
    load_kimi_checkpoint,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Generate text from a trained Kimi K3 checkpoint"
    )
    result.add_argument(
        "--profile",
        help="directory containing the profile data/model/training YAMLs",
    )
    result.add_argument("--model-config")
    result.add_argument("--data-config")
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--prompt", required=True)
    result.add_argument(
        "--inference-config",
        default="config/inference/greedy.yaml",
    )
    result.add_argument("--device", default=None)
    result.add_argument(
        "--precision",
        choices=("model", "fp32", "bf16", "fp16"),
        default="model",
    )
    result.add_argument("--max-new-tokens", type=int, default=None)
    result.add_argument("--temperature", type=float, default=None)
    result.add_argument("--top-k", type=int, default=None)
    result.add_argument("--top-p", type=float, default=None)
    result.add_argument("--repetition-penalty", type=float, default=None)
    sampling = result.add_mutually_exclusive_group()
    sampling.add_argument("--sample", action="store_true")
    sampling.add_argument("--greedy", action="store_true")
    result.add_argument("--seed", type=int, default=None)
    return result


def main(argv=None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.profile:
        if args.data_config or args.model_config:
            argument_parser.error(
                "--profile cannot be combined with --data-config or "
                "--model-config"
            )
        profile = resolve_kimi_pipeline_profile(args.profile)
        data_path, model_path = profile.data, profile.model
    else:
        if not args.data_config or not args.model_config:
            argument_parser.error(
                "pass --profile or both --data-config and --model-config"
            )
        data_path, model_path = args.data_config, args.model_config
    generation = load_generation_config(args.inference_config)
    overrides = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
        "device": args.device,
    }
    generation = replace(
        generation,
        **{
            name: value
            for name, value in overrides.items()
            if value is not None
        },
    )
    if args.sample:
        generation = replace(generation, do_sample=True)
    elif args.greedy:
        generation = replace(generation, do_sample=False)

    tokenizer = load_tokenizer_from_data_yaml(data_path)
    loaded = load_kimi_checkpoint(
        model_path,
        args.checkpoint,
        tokenizer=tokenizer,
        load_config=ModelLoadConfig(
            device=generation.device,
            precision=args.precision,
        ),
    )
    output = inference_autoregressive(
        loaded.model,
        args.prompt,
        tokenizer=tokenizer,
        generation_config=generation,
    )
    print("\n" + "=" * 88)
    print("Kimi K3 inference")
    print("=" * 88)
    print(f"Checkpoint : {loaded.checkpoint_path}")
    print(
        f"Prompt/generated tokens : "
        f"{output.prompt_tokens}/{output.generated_tokens}"
    )
    print(f"Finish reason : {output.finish_reason}")
    print(f"Decode tokens/s : {output.tokens_per_second:.2f}")
    print("-" * 88)
    print(output.completion_text)
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
