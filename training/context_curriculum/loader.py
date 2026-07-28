"""Loader-factory invocation with one explicit max-length contract."""

from __future__ import annotations

import inspect


def build_context_loader(factory, max_seq_len: int):
    if factory is None:
        raise ValueError("a train_loader_factory is required")
    signature = inspect.signature(factory)
    if "max_seq_len" in signature.parameters:
        return factory(max_seq_len=max_seq_len)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return factory(max_seq_len=max_seq_len)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if positional:
        return factory(max_seq_len)
    raise TypeError(
        "train_loader_factory must accept max_seq_len as an argument"
    )


__all__ = ["build_context_loader"]
