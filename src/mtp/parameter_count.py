from __future__ import annotations

import torch.nn as nn

from .head import KimiMTPHead


def _unique_parameter_count(*modules: nn.Module | None) -> int:
    parameters = {}
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            parameters[id(parameter)] = parameter
    return sum(parameter.numel() for parameter in parameters.values())


def mtp_parameter_counts(head: KimiMTPHead) -> dict[str, int]:
    """Count auxiliary and shared parameters without double-counting aliases."""

    fusion = _unique_parameter_count(head.fusion)
    block = _unique_parameter_count(head.block)
    input_embeddings = _unique_parameter_count(head.input_embeddings)
    lm_head = _unique_parameter_count(head.lm_head)
    all_referenced = _unique_parameter_count(
        head.fusion,
        head.block,
        head.input_embeddings,
        head.lm_head,
    )
    shared_overlap = (
        input_embeddings + lm_head
        - _unique_parameter_count(head.input_embeddings, head.lm_head)
    )
    return {
        "fusion": fusion,
        "block": block,
        "unique_mtp_output": 0,
        "unique_mtp_total": fusion + block,
        "shared_input_embeddings": input_embeddings,
        "shared_lm_head": lm_head,
        "shared_embedding_lm_overlap": shared_overlap,
        "all_referenced_unique": all_referenced,
    }
