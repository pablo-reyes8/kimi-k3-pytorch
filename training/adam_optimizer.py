"""AdamW construction with architecture-neutral parameter grouping."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


def build_adamw_parameter_groups(
    model: nn.Module, weight_decay: float = 0.1
) -> Tuple[list, Dict[str, Any]]:
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    decay, no_decay, decay_names, no_decay_names = [], [], [], []
    owner = {}
    for module_name, module in model.named_modules():
        for parameter_name, _ in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            owner[full_name] = module

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        module = owner.get(name)
        excluded = (
            parameter.ndim < 2
            or name.endswith(".bias")
            or isinstance(module, (nn.Embedding, nn.LayerNorm))
            or "norm" in name.lower()
            or module is getattr(model, "lm_head", None)
        )
        target, names = (
            (no_decay, no_decay_names) if excluded else (decay, decay_names)
        )
        target.append(parameter)
        names.append(name)

    groups = [
        {"params": decay, "weight_decay": weight_decay, "group_name": "decay"},
        {"params": no_decay, "weight_decay": 0.0, "group_name": "no_decay"},
    ]
    return groups, {
        "decay_names": decay_names,
        "no_decay_names": no_decay_names,
        "num_decay_params": sum(p.numel() for p in decay),
        "num_no_decay_params": sum(p.numel() for p in no_decay),
    }


def build_adamw_optimizer(
    model: nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> tuple[torch.optim.AdamW, Dict[str, Any]]:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    groups, info = build_adamw_parameter_groups(model, weight_decay)
    return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, eps=eps), info
