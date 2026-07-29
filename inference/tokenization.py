"""Tokenizer-neutral prompt encoding and generated-text decoding."""

from __future__ import annotations

from typing import Any

import torch


def tokenizer_token_id(tokenizer: Any, token: str) -> int | None:
    attribute = {
        "<bos>": "bos_token_id",
        "<eos>": "eos_token_id",
        "<pad>": "pad_token_id",
    }.get(token)
    if attribute is not None:
        value = getattr(tokenizer, attribute, None)
        if value is not None:
            return int(value)
    legacy = {
        "<bos>": "bos_id",
        "<eos>": "eos_id",
        "<pad>": "pad_id",
    }.get(token)
    if legacy is not None:
        value = getattr(tokenizer, legacy, None)
        if value is not None:
            return int(value)
    if hasattr(tokenizer, "token_to_id"):
        value = tokenizer.token_to_id(token)
        return None if value is None else int(value)
    return None


def encode_prompt(
    prompt: str | list[int] | torch.Tensor,
    *,
    tokenizer=None,
    add_bos_token: bool = True,
) -> torch.Tensor:
    """Return one right-unpadded prompt with shape ``[1,T]``."""
    if torch.is_tensor(prompt):
        ids = prompt.detach().clone().long()
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2:
            raise ValueError("prompt tensor must have shape [T] or [B,T]")
        if ids.shape[1] == 0:
            raise ValueError("prompt must contain at least one token")
        return ids
    if isinstance(prompt, list):
        if not prompt or not all(isinstance(value, int) for value in prompt):
            raise ValueError("prompt ID list must contain integers")
        return torch.tensor(prompt, dtype=torch.long).unsqueeze(0)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("text prompt must be a non-empty string")
    if tokenizer is None:
        raise ValueError("a tokenizer is required for text prompts")
    if callable(tokenizer) and not hasattr(tokenizer, "encode"):
        encoded = tokenizer(prompt, return_tensors="pt")
        ids = encoded["input_ids"]
    else:
        encoded = tokenizer.encode(prompt)
        ids = encoded.ids if hasattr(encoded, "ids") else encoded
        ids = torch.as_tensor(ids, dtype=torch.long).unsqueeze(0)
    if ids.ndim != 2:
        raise ValueError("tokenizer must produce token IDs with shape [1,T]")
    bos = tokenizer_token_id(tokenizer, "<bos>")
    if add_bos_token and bos is not None and (
        ids.shape[1] == 0 or int(ids[0, 0]) != bos
    ):
        ids = torch.cat(
            (torch.tensor([[bos]], dtype=torch.long), ids.long()), dim=1
        )
    if ids.shape[1] == 0:
        raise ValueError("tokenizer produced an empty prompt")
    return ids.long()


def decode_token_ids(
    token_ids: torch.Tensor | list[int],
    *,
    tokenizer,
    skip_special_tokens: bool = True,
) -> str | list[str]:
    if tokenizer is None:
        raise ValueError("a tokenizer is required to decode text")
    ids = (
        token_ids.detach().cpu().long()
        if torch.is_tensor(token_ids)
        else torch.tensor(token_ids, dtype=torch.long)
    )
    if ids.ndim == 1:
        rows = [ids.tolist()]
        single = True
    elif ids.ndim == 2:
        rows = ids.tolist()
        single = len(rows) == 1
    else:
        raise ValueError("token IDs must have shape [T] or [B,T]")
    decoded = []
    for row in rows:
        try:
            text = tokenizer.decode(
                row, skip_special_tokens=skip_special_tokens
            )
        except TypeError:
            text = tokenizer.decode(row)
        decoded.append(text)
    return decoded[0] if single else decoded


__all__ = [
    "decode_token_ids",
    "encode_prompt",
    "tokenizer_token_id",
]
