"""Safe model invocation shared by train, eval, and qualitative previews."""

from __future__ import annotations

import inspect
from typing import Any


def filter_forward_kwargs(model, values: dict[str, Any]) -> dict[str, Any]:
    """Drop trainer-only kwargs when a generic LM does not accept them."""

    raw_model = model.module if hasattr(model, "module") else model
    signature = inspect.signature(raw_model.forward)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return values
    return {
        name: value
        for name, value in values.items()
        if name in signature.parameters
    }


def call_model(model, batch: dict[str, Any], *, use_mtp: bool | None = None):
    values = dict(batch)
    if use_mtp is not None:
        values["use_mtp"] = use_mtp
        values["training_phase"] = "pretrain"
    return model(**filter_forward_kwargs(model, values))
