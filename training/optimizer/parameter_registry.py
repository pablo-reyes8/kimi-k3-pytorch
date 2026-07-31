"""Structural, auditable optimizer assignment for Kimi K3 parameters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import torch.nn as nn

from src.kda.projections import KDAProjections
from src.mla.latent_kv import LatentKVProjection
from src.mla.projections import MLAProjections

from .per_head_muon import HeadMatrixLayout


OptimizerFamily = Literal[
    "adamw_decay", "adamw_no_decay", "muon", "per_head_muon"
]


@dataclass(frozen=True)
class MatrixParameterSpec:
    parameter_name: str
    parameter: nn.Parameter
    owner_module_name: str
    role: str
    optimizer_family: OptimizerFamily
    apply_weight_decay: bool
    head_layout: HeadMatrixLayout | None = None

    def fingerprint_record(self) -> dict:
        layout = None
        if self.head_layout is not None:
            layout = {
                "num_heads": self.head_layout.num_heads,
                "head_dim": self.head_layout.head_dim,
                "head_axis": self.head_layout.head_axis,
                "input_dim": self.head_layout.input_dim,
                "output_dim": self.head_layout.output_dim,
                "packed_kind": self.head_layout.packed_kind,
            }
        return {
            "name": self.parameter_name,
            "shape": list(self.parameter.shape),
            "role": self.role,
            "family": self.optimizer_family,
            "weight_decay": self.apply_weight_decay,
            "head_layout": layout,
        }


@dataclass(frozen=True)
class ParameterAssignmentReport:
    specs: tuple[MatrixParameterSpec, ...]
    numel_by_family: dict[str, int]
    tensor_count_by_family: dict[str, int]
    percentages_by_family: dict[str, float]
    missing: tuple[str, ...]
    duplicates: tuple[str, ...]
    ambiguous: tuple[str, ...]
    fingerprint: str

    @property
    def total_numel(self) -> int:
        return sum(self.numel_by_family.values())

    def names_for(self, family: str) -> tuple[str, ...]:
        return tuple(
            spec.parameter_name
            for spec in self.specs
            if spec.optimizer_family == family
        )

    def to_dict(self) -> dict:
        return {
            "numel_by_family": dict(self.numel_by_family),
            "tensor_count_by_family": dict(self.tensor_count_by_family),
            "percentages_by_family": dict(self.percentages_by_family),
            "missing": list(self.missing),
            "duplicates": list(self.duplicates),
            "ambiguous": list(self.ambiguous),
            "fingerprint": self.fingerprint,
            "specs": [spec.fingerprint_record() for spec in self.specs],
        }


def _parent(module_name: str, modules: dict[str, nn.Module]):
    parent_name, _, attribute = module_name.rpartition(".")
    return modules.get(parent_name), attribute


def _qkv_role_and_layout(
    owner_name: str,
    modules: dict[str, nn.Module],
) -> tuple[str, HeadMatrixLayout] | None:
    owner = modules[owner_name]
    parallel_head_spec = getattr(owner, "_kimi_head_spec", None)
    if parallel_head_spec is not None:
        num_heads, head_dim = parallel_head_spec
        return getattr(owner, "_kimi_role"), HeadMatrixLayout(
            num_heads=num_heads,
            head_dim=head_dim,
            head_axis=0,
            input_dim=owner.in_features,
            output_dim=owner.local_out_features,
        )
    parent, attribute = _parent(owner_name, modules)
    if isinstance(parent, KDAProjections) and attribute in {
        "q_proj",
        "k_proj",
        "v_proj",
    }:
        role = {
            "q_proj": "attention_q",
            "k_proj": "attention_k",
            "v_proj": "attention_v",
        }[attribute]
        head_dim = (
            parent.config.value_head_dim
            if attribute == "v_proj"
            else parent.config.key_head_dim
        )
    elif isinstance(parent, MLAProjections) and attribute == "query":
        role = "attention_q"
        head_dim = parent.config.q_head_dim
    elif isinstance(parent, LatentKVProjection) and attribute in {
        "key_up",
        "value_up",
    }:
        role = (
            "attention_k" if attribute == "key_up" else "attention_v"
        )
        head_dim = (
            parent.config.q_head_dim
            if attribute == "key_up"
            else parent.config.v_head_dim
        )
    else:
        return None
    return role, HeadMatrixLayout(
        num_heads=parent.config.num_heads,
        head_dim=head_dim,
        head_axis=0,
        input_dim=owner.in_features,
        output_dim=owner.out_features,
    )


def _is_norm(name: str, owner: nn.Module) -> bool:
    return "norm" in name.lower() or isinstance(
        owner,
        (
            nn.LayerNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.GroupNorm,
        ),
    )


def _is_mtp_parameter(name: str) -> bool:
    lower = name.lower()
    return lower.startswith("mtp.") or ".mtp." in lower


def build_parameter_registry(
    model: nn.Module,
    *,
    kind: str = "per_head_muon_adamw",
    strict: bool = True,
) -> ParameterAssignmentReport:
    modules = dict(model.named_modules())
    owner_by_name: dict[str, tuple[str, nn.Module]] = {}
    for module_name, module in modules.items():
        for local_name, _ in module.named_parameters(recurse=False):
            full_name = (
                f"{module_name}.{local_name}" if module_name else local_name
            )
            owner_by_name[full_name] = (module_name, module)

    specs = []
    seen: dict[int, str] = {}
    duplicates = []
    ambiguous = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in seen:
            duplicates.append(name)
            continue
        seen[parameter_id] = name
        owner_name, owner = owner_by_name[name]
        qkv = _qkv_role_and_layout(owner_name, modules)

        role = "other"
        layout = None
        explicit_role = getattr(owner, "_kimi_role", None)
        parallel_linear = bool(
            getattr(owner, "_kimi_parallel_linear", False)
        )
        no_decay = (
            parameter.ndim < 2
            or name.endswith(".bias")
            or isinstance(owner, nn.Embedding)
            or _is_norm(name, owner)
            or "lm_head" in name
            or "pseudo_query" in name
            or explicit_role in {"embedding", "lm_head"}
        )
        if qkv is not None:
            role, layout = qkv
        elif explicit_role is not None:
            role = explicit_role
        elif isinstance(owner, nn.Embedding):
            role = "embedding"
        elif "lm_head" in name:
            role = "lm_head"
        elif _is_norm(name, owner):
            role = "norm"
        elif isinstance(owner, nn.Linear):
            lower = name.lower()
            if ".router." in lower:
                role = "moe_router"
            elif "shared_experts" in lower:
                role = "moe_shared_expert"
            elif "routed_experts" in lower:
                role = "moe_routed_expert"
            elif "output_proj" in lower:
                role = "attention_o"
            elif _is_mtp_parameter(name):
                role = "mtp_matrix"
            else:
                role = "dense_matrix"

        if kind == "adamw":
            family: OptimizerFamily = (
                "adamw_no_decay" if no_decay else "adamw_decay"
            )
        elif layout is not None and kind == "per_head_muon_adamw":
            family = "per_head_muon"
        elif (
            (isinstance(owner, nn.Linear) or parallel_linear)
            and parameter.ndim == 2
            and not no_decay
            and role != "moe_router"
        ):
            family = "muon"
        else:
            family = "adamw_no_decay" if no_decay else "adamw_decay"

        if (
            strict
            and parameter.ndim == 2
            and (isinstance(owner, nn.Linear) or parallel_linear)
            and role == "other"
        ):
            ambiguous.append(name)
        specs.append(
            MatrixParameterSpec(
                parameter_name=name,
                parameter=parameter,
                owner_module_name=owner_name,
                role=role,
                optimizer_family=family,
                apply_weight_decay=family
                in {"adamw_decay", "muon", "per_head_muon"},
                head_layout=layout if family == "per_head_muon" else None,
            )
        )

    trainable = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assigned = {id(spec.parameter) for spec in specs}
    missing = tuple(
        name for identifier, name in trainable.items() if identifier not in assigned
    )
    if strict and (missing or duplicates or ambiguous):
        raise RuntimeError(
            "invalid parameter registry: "
            f"missing={missing}, duplicates={duplicates}, ambiguous={ambiguous}"
        )

    families = (
        "adamw_decay",
        "adamw_no_decay",
        "muon",
        "per_head_muon",
    )
    numel = {
        family: sum(
            spec.parameter.numel()
            for spec in specs
            if spec.optimizer_family == family
        )
        for family in families
    }
    counts = {
        family: sum(spec.optimizer_family == family for spec in specs)
        for family in families
    }
    total = sum(numel.values())
    records = [spec.fingerprint_record() for spec in specs]
    fingerprint = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ParameterAssignmentReport(
        specs=tuple(specs),
        numel_by_family=numel,
        tensor_count_by_family=counts,
        percentages_by_family={
            family: 100.0 * value / max(total, 1)
            for family, value in numel.items()
        },
        missing=missing,
        duplicates=tuple(duplicates),
        ambiguous=tuple(ambiguous),
        fingerprint=fingerprint,
    )
