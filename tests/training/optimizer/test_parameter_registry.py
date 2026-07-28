import torch

from src import KimiK3, kimi_k3_cpu_tiny_config
from training.optimizer import build_parameter_registry


def test_registry_assigns_every_trainable_parameter_exactly_once():
    model = KimiK3(kimi_k3_cpu_tiny_config())
    registry = build_parameter_registry(model)
    expected = {id(parameter) for parameter in model.parameters()
                if parameter.requires_grad}
    assigned = [id(spec.parameter) for spec in registry.specs]
    assert set(assigned) == expected
    assert len(assigned) == len(set(assigned))
    assert registry.missing == ()
    assert registry.ambiguous == ()
    assert sum(registry.numel_by_family.values()) == sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_all_kda_mla_and_mtp_qkv_use_per_head_muon_only():
    model = KimiK3(kimi_k3_cpu_tiny_config())
    registry = build_parameter_registry(model)
    qkv = [
        spec for spec in registry.specs
        if spec.role in {"attention_q", "attention_k", "attention_v"}
    ]
    assert qkv
    assert all(spec.optimizer_family == "per_head_muon" for spec in qkv)
    assert all(spec.head_layout is not None for spec in qkv)
    names = {spec.parameter_name for spec in qkv}
    assert any("mtp" in name and "q_proj" in name for name in names)
    assert any("mtp" in name and "key_up" in name for name in names)
    assert any("backbone" in name and "q_proj" in name for name in names)
    assert any("backbone" in name and "key_up" in name for name in names)


def test_non_qkv_matrices_and_special_parameters_have_correct_families():
    registry = build_parameter_registry(
        KimiK3(kimi_k3_cpu_tiny_config())
    )
    for spec in registry.specs:
        if spec.role == "attention_o":
            assert spec.optimizer_family == "muon"
        if spec.role in {"embedding", "norm", "lm_head", "moe_router"}:
            assert spec.optimizer_family.startswith("adamw_")
    assert all(
        "routing_bias" not in spec.parameter_name
        for spec in registry.specs
    )
    fusion = next(
        spec for spec in registry.specs
        if spec.parameter_name == "mtp.fusion.projection.weight"
    )
    assert fusion.role == "mtp_matrix"
    assert fusion.optimizer_family == "muon"


def test_registry_fingerprint_is_stable_and_detects_architecture_change():
    first = build_parameter_registry(KimiK3(kimi_k3_cpu_tiny_config()))
    second = build_parameter_registry(KimiK3(kimi_k3_cpu_tiny_config()))
    changed_model = torch.nn.ModuleDict({
        "kimi": KimiK3(kimi_k3_cpu_tiny_config()),
        "new_projection": torch.nn.Linear(16, 16),
    })
    changed = build_parameter_registry(changed_model)
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != changed.fingerprint


def test_adamw_mode_puts_bias_norm_embedding_in_no_decay():
    registry = build_parameter_registry(
        KimiK3(kimi_k3_cpu_tiny_config()), kind="adamw"
    )
    for spec in registry.specs:
        if (
            spec.parameter.ndim < 2
            or spec.role in {"embedding", "norm", "lm_head"}
        ):
            assert spec.optimizer_family == "adamw_no_decay"
