"""Strict YAML schema and builders for complete Kimi K3 models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from configuration.yaml_utils import (
    ConfigError,
    dataclass_kwargs,
    expect_mapping,
    load_yaml_mapping,
    reject_unknown_keys,
)
from src.attention_residuals import AttentionResidualConfig
from src.kda import KDAConfig
from src.kimi_block import KimiBlockConfig
from src.kimi_k3.config import (
    KimiK3Config,
    VisionProjectorConfig,
    vision_output_dim,
)
from src.mla import GatedMLAConfig
from src.mtp import KimiMTPConfig
from src.stable_latent_moe import StableLatentMoEConfig
from src.vision import (
    HierarchicalVisionConfig,
    SwinVisionConfig,
    VisionEncoderConfig,
)


def _component(values, component_type, *, path: str, injected: dict):
    combined = {**injected, **values}
    return component_type(
        **dataclass_kwargs(combined, component_type, path=path)
    )


def _special_ids(model: dict[str, Any], data_bundle) -> dict[str, Any]:
    use_data = bool(model.pop("use_data_special_tokens", False))
    values = {
        name: model.pop(name, None)
        for name in (
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
            "image_token_id",
            "video_token_id",
        )
    }
    if use_data:
        if data_bundle is None:
            return values
        for name, token in (
            ("pad_token_id", "<pad>"),
            ("bos_token_id", "<bos>"),
            ("eos_token_id", "<eos>"),
        ):
            resolved = data_bundle.token_id(token)
            if resolved is not None:
                values[name] = int(resolved)
    return values


def build_model_config_from_mapping(
    values: dict[str, Any],
    *,
    data_bundle=None,
) -> KimiK3Config:
    """Build the complete typed architecture without allocating weights."""
    root = dict(values)
    model = expect_mapping(root, "model", path="root")
    reject_unknown_keys(root, path="root")
    if model.pop("architecture", None) != "kimi_k3":
        raise ConfigError("model.architecture must be 'kimi_k3'")
    model.pop("name", None)
    d_model = int(model.pop("d_model"))
    vocab_value = model.pop("vocab_size")
    if vocab_value == "auto":
        if data_bundle is None:
            raise ConfigError("vocab_size=auto requires data_bundle")
        vocab_size = data_bundle.vocab_size
    else:
        vocab_size = int(vocab_value)
        if data_bundle is not None and data_bundle.vocab_size > vocab_size:
            raise ConfigError(
                f"data vocabulary ({data_bundle.vocab_size}) exceeds model "
                f"vocab_size ({vocab_size})"
            )
    special_ids = _special_ids(model, data_bundle)
    num_heads = int(model.pop("num_heads"))
    if d_model % num_heads:
        raise ConfigError("model.d_model must be divisible by num_heads")
    value_head_dim = d_model // num_heads
    num_groups = int(model.pop("num_hybrid_groups"))
    attention_pattern = tuple(
        model.pop(
            "attention_pattern",
            ("kda", "kda", "kda", "gated_mla"),
        )
    )
    add_final = bool(model.pop("add_final_gated_mla", True))

    kda_values = expect_mapping(model, "kda", path="model")
    key_head_dim = int(kda_values.pop("key_head_dim"))
    kda = _component(
        kda_values,
        KDAConfig,
        path="model.kda",
        injected={
            "d_model": d_model,
            "num_heads": num_heads,
            "key_head_dim": key_head_dim,
            "value_head_dim": value_head_dim,
        },
    )
    mla_values = expect_mapping(model, "mla", path="model")
    q_head_dim = int(mla_values.pop("q_head_dim"))
    kv_latent_dim = int(mla_values.pop("kv_latent_dim"))
    mla = _component(
        mla_values,
        GatedMLAConfig,
        path="model.mla",
        injected={
            "d_model": d_model,
            "num_heads": num_heads,
            "q_head_dim": q_head_dim,
            "v_head_dim": value_head_dim,
            "kv_latent_dim": kv_latent_dim,
        },
    )
    moe_values = expect_mapping(model, "moe", path="model")
    moe = _component(
        moe_values,
        StableLatentMoEConfig,
        path="model.moe",
        injected={"d_model": d_model},
    )
    attnres_values = expect_mapping(
        model, "attention_residual", path="model"
    )
    attnres = _component(
        attnres_values,
        AttentionResidualConfig,
        path="model.attention_residual",
        injected={"d_model": d_model},
    )
    initializer_range = float(model.pop("initializer_range", 0.02))
    backbone = KimiBlockConfig(
        d_model=d_model,
        num_pattern_repeats=num_groups,
        kda_config=kda,
        mla_config=mla,
        stable_latent_moe_config=moe,
        attention_residual_config=attnres,
        attention_pattern=attention_pattern,
        add_final_gated_mla=add_final,
        rms_norm_eps=float(model.pop("rms_norm_eps", 1e-6)),
        residual_dropout=float(model.pop("residual_dropout", 0.0)),
        init_std=initializer_range,
    )

    vision_values = expect_mapping(
        model, "vision", path="model", required=False
    )
    enable_vision = bool(vision_values.pop("enabled", False))
    vision = None
    projector = None
    use_pixel_shuffle = bool(
        vision_values.pop("use_pixel_shuffle", True)
    )
    if enable_vision:
        vision_type = vision_values.pop("type", "moonvit")
        projector_values = vision_values.pop("projector", {})
        vision_classes = {
            "moonvit": VisionEncoderConfig,
            "hierarchical": HierarchicalVisionConfig,
            "swin": SwinVisionConfig,
        }
        if vision_type not in vision_classes:
            raise ConfigError(f"unsupported model.vision.type: {vision_type}")
        for key in ("embed_dims", "depths", "num_heads"):
            if isinstance(vision_values.get(key), list):
                vision_values[key] = tuple(vision_values[key])
        vision_cls = vision_classes[vision_type]
        vision = vision_cls(
            **dataclass_kwargs(
                vision_values, vision_cls, path="model.vision"
            )
        )
        width = vision_output_dim(vision)
        projector_defaults = {
            "input_dim": width * (4 if use_pixel_shuffle else 1),
            "hidden_dim": max(d_model, width),
            "output_dim": d_model,
        }
        if not isinstance(projector_values, dict):
            raise ConfigError("model.vision.projector must be a mapping")
        projector = VisionProjectorConfig(
            **dataclass_kwargs(
                {**projector_defaults, **projector_values},
                VisionProjectorConfig,
                path="model.vision.projector",
            )
        )
    else:
        reject_unknown_keys(vision_values, path="model.vision")

    mtp_values = expect_mapping(model, "mtp", path="model", required=False)
    enable_mtp = bool(mtp_values.pop("enabled", True))
    mtp = None
    if enable_mtp:
        mtp_attnres_values = mtp_values.pop(
            "attention_residual",
            {
                "mode": "block",
                "sublayers_per_depth_block": 8,
                "backend": "eager",
            },
        )
        if not isinstance(mtp_attnres_values, dict):
            raise ConfigError(
                "model.mtp.attention_residual must be a mapping"
            )
        mtp_attnres = _component(
            mtp_attnres_values,
            AttentionResidualConfig,
            path="model.mtp.attention_residual",
            injected={"d_model": d_model},
        )
        mtp = KimiMTPConfig(
            d_model=d_model,
            vocab_size=vocab_size,
            kda_config=kda,
            mla_config=mla,
            stable_latent_moe_config=moe,
            attention_residual_config=mtp_attnres,
            **dataclass_kwargs(
                mtp_values, KimiMTPConfig, path="model.mtp"
            ),
        )
    else:
        reject_unknown_keys(mtp_values, path="model.mtp")

    top_level_fields = {
        "tie_word_embeddings",
        "use_bias_in_lm_head",
        "vision_token_integration",
        "freeze_vision_encoder",
        "return_dict",
        "output_hidden_states",
        "output_attentions",
        "output_router_diagnostics",
        "output_attnres_diagnostics",
    }
    top_level = {
        key: model.pop(key)
        for key in tuple(model)
        if key in top_level_fields
    }
    reject_unknown_keys(model, path="model")
    return KimiK3Config(
        vocab_size=vocab_size,
        d_model=d_model,
        backbone=backbone,
        **special_ids,
        vision=vision,
        vision_projector=projector,
        mtp=mtp,
        enable_vision=enable_vision,
        enable_mtp=enable_mtp,
        vision_use_pixel_shuffle=use_pixel_shuffle,
        initializer_range=initializer_range,
        **top_level,
    )


def load_model_config(
    path: str | Path,
    *,
    data_bundle=None,
) -> KimiK3Config:
    """Load and validate model YAML without allocating model parameters."""
    _, values = load_yaml_mapping(path)
    return build_model_config_from_mapping(
        values, data_bundle=data_bundle
    )


def build_model_from_yaml(path: str | Path, *, data_bundle=None):
    """Instantiate KimiK3 from a fully validated YAML architecture."""
    from src.kimi_k3_mini import KimiK3

    return KimiK3(load_model_config(path, data_bundle=data_bundle))


__all__ = [
    "build_model_config_from_mapping",
    "build_model_from_yaml",
    "load_model_config",
]
