"""Master native-cache autoregressive generation for Kimi K3."""

from __future__ import annotations

from dataclasses import replace
import time
from typing import Any

import torch

from .cache import cache_summary
from .config import GenerationConfig
from .decode import decode_one_token
from .outputs import GenerationOutput
from .prefill import prefill_prompt
from .sampling import sample_next_token
from .tokenization import (
    decode_token_ids,
    encode_prompt,
    tokenizer_token_id,
)


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def resolve_generation_device(model, requested: str) -> torch.device:
    if requested == "auto":
        return _model_device(model)
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference requested but CUDA is unavailable")
    return device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _last_prompt_logits(logits, attention_mask):
    indices = attention_mask.long().sum(dim=1) - 1
    if torch.any(indices < 0):
        raise ValueError("every prompt must contain at least one valid token")
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, indices]


@torch.inference_mode()
def generate_tokens(
    model,
    input_ids: torch.Tensor,
    *,
    generation_config: GenerationConfig | None = None,
    attention_mask: torch.Tensor | None = None,
    prefill_kwargs: dict[str, Any] | None = None,
) -> GenerationOutput:
    """Generate tokens with one prefill followed only by cached decode steps."""
    config = generation_config or GenerationConfig()
    device = resolve_generation_device(model, config.device)
    if _model_device(model) != device:
        model.to(device)
    input_ids = input_ids.to(device=device, dtype=torch.long)
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have shape [B,T], T > 0")
    if attention_mask is None:
        pad_id = config.pad_token_id
        attention_mask = (
            torch.ones_like(input_ids, dtype=torch.bool)
            if pad_id is None
            else input_ids.ne(pad_id)
        )
    else:
        attention_mask = attention_mask.to(device=device)
    if attention_mask.dtype != torch.bool:
        raise TypeError("attention_mask must be boolean")
    prompt_lengths = attention_mask.sum(dim=1)
    if input_ids.shape[0] > 1 and not torch.equal(
        prompt_lengths, prompt_lengths[:1].expand_as(prompt_lengths)
    ):
        raise ValueError(
            "batched generation currently requires equal prompt lengths"
        )
    prompt_width = int(input_ids.shape[1])
    steps = config.max_new_tokens
    finish_reason = "length"
    if config.max_total_tokens is not None:
        available = max(config.max_total_tokens - prompt_width, 0)
        if available < steps:
            steps = available
            finish_reason = "max_total_tokens"

    was_training = model.training
    model.eval()
    generator = None
    if config.seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(config.seed)
    sequences = input_ids
    generated = []
    scores = []
    finished = torch.zeros(
        input_ids.shape[0], dtype=torch.bool, device=device
    )
    prefill_kwargs = dict(prefill_kwargs or {})
    _synchronize(device)
    total_started = time.perf_counter()
    prefill_started = time.perf_counter()
    try:
        output = prefill_prompt(
            model,
            input_ids,
            attention_mask,
            **prefill_kwargs,
        )
        _synchronize(device)
        prefill_seconds = time.perf_counter() - prefill_started
        cache = output.cache
        logits = _last_prompt_logits(output.logits, attention_mask)
        decode_started = time.perf_counter()
        eos_id = config.eos_token_id
        pad_id = (
            config.pad_token_id
            if config.pad_token_id is not None
            else eos_id
        )
        for _ in range(steps):
            if config.return_scores:
                scores.append(logits.detach().float().cpu())
            next_token = sample_next_token(
                logits,
                config,
                previous_token_ids=sequences,
                generator=generator,
            )
            active = ~finished
            if pad_id is not None:
                replacement = torch.full_like(next_token, int(pad_id))
                next_token = torch.where(
                    active[:, None], next_token, replacement
                )
            generated.append(next_token)
            sequences = torch.cat((sequences, next_token), dim=1)
            if eos_id is not None:
                finished |= active & next_token[:, 0].eq(int(eos_id))

            decoded = decode_one_token(
                model,
                next_token,
                cache,
                active[:, None],
            )
            cache = decoded.cache
            logits = decoded.logits[:, -1, :]
            if bool(finished.all()):
                finish_reason = "eos"
                break
        _synchronize(device)
        decode_seconds = time.perf_counter() - decode_started
    finally:
        if was_training:
            model.train()
    total_seconds = time.perf_counter() - total_started
    generated_ids = (
        torch.cat(generated, dim=1)
        if generated
        else input_ids.new_empty((input_ids.shape[0], 0))
    )
    generated_count = int(generated_ids.shape[1])
    return GenerationOutput(
        prompt_ids=input_ids,
        sequences=sequences,
        generated_ids=generated_ids,
        text=None,
        completion_text=None,
        finish_reason=finish_reason,
        cache=cache if config.return_cache else None,
        cache_stats=cache_summary(cache),
        scores=tuple(scores) if config.return_scores else None,
        prompt_tokens=prompt_width,
        generated_tokens=generated_count,
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        total_seconds=total_seconds,
        tokens_per_second=(
            generated_count / decode_seconds
            if decode_seconds > 0
            else 0.0
        ),
    )


@torch.inference_mode()
def inference_autoregressive(
    model,
    prompt: str | list[int] | torch.Tensor,
    *,
    tokenizer=None,
    generation_config: GenerationConfig | None = None,
    attention_mask: torch.Tensor | None = None,
    prefill_kwargs: dict[str, Any] | None = None,
    **generation_overrides,
) -> GenerationOutput:
    """Master API: encode a prompt, generate with cache, and decode text."""
    if generation_config is not None and generation_overrides:
        raise ValueError(
            "pass generation_config or keyword overrides, not both"
        )
    config = generation_config or GenerationConfig(**generation_overrides)
    if config.eos_token_id is None:
        eos = (
            tokenizer_token_id(tokenizer, "<eos>")
            if tokenizer is not None
            else getattr(getattr(model, "config", None), "eos_token_id", None)
        )
        if eos is not None:
            config = replace(config, eos_token_id=int(eos))
    if config.pad_token_id is None:
        pad = (
            tokenizer_token_id(tokenizer, "<pad>")
            if tokenizer is not None
            else getattr(getattr(model, "config", None), "pad_token_id", None)
        )
        if pad is not None:
            config = replace(config, pad_token_id=int(pad))
    input_ids = encode_prompt(
        prompt,
        tokenizer=tokenizer,
        add_bos_token=config.add_bos_token,
    )
    output = generate_tokens(
        model,
        input_ids,
        generation_config=config,
        attention_mask=attention_mask,
        prefill_kwargs=prefill_kwargs,
    )
    if tokenizer is not None:
        output.text = decode_token_ids(
            output.sequences,
            tokenizer=tokenizer,
            skip_special_tokens=config.skip_special_tokens,
        )
        output.completion_text = decode_token_ids(
            output.generated_ids,
            tokenizer=tokenizer,
            skip_special_tokens=config.skip_special_tokens,
        )
    return output


inference_autoregresive = inference_autoregressive
generate = generate_tokens


__all__ = [
    "generate",
    "generate_tokens",
    "inference_autoregressive",
    "inference_autoregresive",
    "resolve_generation_device",
]
