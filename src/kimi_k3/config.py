"""Configuration and multimodal integration helpers used by the KimiK3 orchestrator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from src.attention_residuals import AttentionResidualConfig
from src.kda import KDAConfig
from src.kimi_block import KimiBlockConfig
from src.mla import GatedMLAConfig
from src.mtp import KimiMTPConfig
from src.stable_latent_moe import StableLatentMoEConfig
from src.vision import (
    HierarchicalVisionConfig,
    SwinVisionConfig,
    VisionEncoderConfig,
)


VisionConfig = VisionEncoderConfig | HierarchicalVisionConfig | SwinVisionConfig


@dataclass(frozen=True)
class VisionProjectorConfig:
    """Configure the MLP that maps packed visual tokens into model width."""

    input_dim: int
    hidden_dim: int
    output_dim: int
    activation: Literal["gelu", "silu"] = "gelu"
    bias: bool = False

    def __post_init__(self) -> None:
        if min(self.input_dim, self.hidden_dim, self.output_dim) <= 0:
            raise ValueError("vision projector dimensions must be > 0")
        if self.activation not in ("gelu", "silu"):
            raise ValueError("unsupported vision projector activation")


def vision_output_dim(config: VisionConfig) -> int:
    """Return the final channel width produced by a vision configuration."""
    if isinstance(config, VisionEncoderConfig):
        return config.embed_dim
    return config.embed_dims[-1]


@dataclass(frozen=True)
class KimiK3Config:
    """Complete architecture configuration consumed by :class:`KimiK3`."""

    vocab_size: int
    d_model: int
    backbone: KimiBlockConfig
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    image_token_id: int | None = None
    video_token_id: int | None = None
    tie_word_embeddings: bool = True
    use_bias_in_lm_head: bool = False
    vision: VisionConfig | None = None
    vision_projector: VisionProjectorConfig | None = None
    mtp: KimiMTPConfig | None = None
    enable_vision: bool = True
    enable_mtp: bool = True
    vision_token_integration: Literal["replace"] = "replace"
    vision_use_pixel_shuffle: bool = True
    freeze_vision_encoder: bool = False
    initializer_range: float = 0.02
    return_dict: bool = True
    output_hidden_states: bool = False
    output_attentions: bool = False
    output_router_diagnostics: bool = False
    output_attnres_diagnostics: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.d_model <= 0:
            raise ValueError("vocab_size and d_model must be > 0")
        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be > 0")
        if self.backbone.d_model != self.d_model:
            raise ValueError("backbone.d_model must match d_model")
        special_ids = [
            value
            for value in (
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
                self.image_token_id,
                self.video_token_id,
            )
            if value is not None
        ]
        if any(not 0 <= value < self.vocab_size for value in special_ids):
            raise ValueError("special token IDs must be inside the vocabulary")
        if len(set(special_ids)) != len(special_ids):
            raise ValueError("non-None special token IDs must be distinct")
        if self.vision_token_integration != "replace":
            raise ValueError("only replacement visual integration is supported")
        if self.enable_vision:
            if self.vision is None or self.vision_projector is None:
                raise ValueError(
                    "vision and vision_projector are required when vision is enabled"
                )
            if self.image_token_id is None:
                raise ValueError("image_token_id is required when vision is enabled")
            encoder_width = vision_output_dim(self.vision)
            expected_input = (
                4 * encoder_width
                if self.vision_use_pixel_shuffle
                else encoder_width
            )
            if self.vision_projector.input_dim != expected_input:
                raise ValueError(
                    "vision_projector.input_dim does not match the visual path"
                )
            if self.vision_projector.output_dim != self.d_model:
                raise ValueError(
                    "vision_projector.output_dim must equal d_model"
                )
            if (
                self.vision_use_pixel_shuffle
                and isinstance(self.vision, VisionEncoderConfig)
                and self.vision.use_cls_token
            ):
                raise ValueError("pixel shuffle is incompatible with a CLS token")
        if self.enable_mtp:
            if self.mtp is None or not self.mtp.enabled:
                raise ValueError("an enabled MTP config is required")
            if (
                self.mtp.d_model != self.d_model
                or self.mtp.vocab_size != self.vocab_size
            ):
                raise ValueError("MTP dimensions must match the main model")

    def to_dict(self) -> dict:
        values = asdict(self)
        if isinstance(self.vision, VisionEncoderConfig):
            values["vision_type"] = "moonvit"
        elif isinstance(self.vision, HierarchicalVisionConfig):
            values["vision_type"] = "hierarchical"
        elif isinstance(self.vision, SwinVisionConfig):
            values["vision_type"] = "swin"
        return values

    @classmethod
    def from_dict(cls, values: dict) -> "KimiK3Config":
        values = dict(values)
        values["backbone"] = KimiBlockConfig.from_dict(values["backbone"])
        vision_values = values.get("vision")
        vision_type = values.pop("vision_type", None)
        if isinstance(vision_values, dict):
            vision_classes = {
                "moonvit": VisionEncoderConfig,
                "hierarchical": HierarchicalVisionConfig,
                "swin": SwinVisionConfig,
            }
            if vision_type not in vision_classes:
                raise ValueError("serialized vision_type is missing or invalid")
            for key in ("embed_dims", "depths", "num_heads"):
                if key in vision_values and isinstance(vision_values[key], list):
                    vision_values[key] = tuple(vision_values[key])
            values["vision"] = vision_classes[vision_type](**vision_values)
        if isinstance(values.get("vision_projector"), dict):
            values["vision_projector"] = VisionProjectorConfig(
                **values["vision_projector"]
            )
        if isinstance(values.get("mtp"), dict):
            values["mtp"] = KimiMTPConfig.from_dict(values["mtp"])
        return cls(**values)


def _build_text_configs(
    d_model: int,
    *,
    num_heads: int,
    key_head_dim: int,
    q_head_dim: int,
    latent_dim: int,
    shared_hidden: int,
    routed_hidden: int,
    num_routed: int,
    top_k: int,
    num_shared: int,
    groups: int,
    attnres_sublayers: int,
    target_depth_blocks: int | None = None,
) -> tuple[KimiBlockConfig, KDAConfig, GatedMLAConfig, StableLatentMoEConfig]:
    value_head_dim = d_model // num_heads
    kda = KDAConfig(
        d_model=d_model,
        num_heads=num_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        short_conv_kernel_size=4,
        chunk_size=64,
        secondary_tile_size=16,
    )
    mla = GatedMLAConfig(
        d_model=d_model,
        num_heads=num_heads,
        q_head_dim=q_head_dim,
        v_head_dim=value_head_dim,
        kv_latent_dim=min(
            latent_dim,
            num_heads * (q_head_dim + value_head_dim),
        ),
    )
    moe = StableLatentMoEConfig(
        d_model=d_model,
        latent_dim=latent_dim,
        num_shared_experts=num_shared,
        num_routed_experts=num_routed,
        routed_experts_per_token=top_k,
        shared_expert_hidden_dim=shared_hidden,
        routed_expert_hidden_dim=routed_hidden,
    )
    backbone = KimiBlockConfig(
        d_model=d_model,
        num_pattern_repeats=groups,
        kda_config=kda,
        mla_config=mla,
        stable_latent_moe_config=moe,
        attention_residual_config=AttentionResidualConfig(
            d_model,
            mode="block",
            sublayers_per_depth_block=attnres_sublayers,
            target_num_depth_blocks=target_depth_blocks,
            backend="eager",
        ),
    )
    return backbone, kda, mla, moe


def kimi_k3_cpu_tiny_config(
    *,
    enable_vision: bool = True,
    enable_mtp: bool = True,
) -> KimiK3Config:
    """Build a small but structurally complete configuration for CPU tests."""
    d_model, vocab_size = 16, 128
    backbone, kda, mla, moe = _build_text_configs(
        d_model,
        num_heads=2,
        key_head_dim=4,
        q_head_dim=4,
        latent_dim=8,
        shared_hidden=20,
        routed_hidden=16,
        num_routed=4,
        top_k=2,
        num_shared=1,
        groups=1,
        attnres_sublayers=4,
    )
    mtp = KimiMTPConfig(
        d_model=d_model,
        vocab_size=vocab_size,
        kda_config=kda,
        mla_config=mla,
        stable_latent_moe_config=moe,
        attention_residual_config=AttentionResidualConfig(
            d_model,
            mode="block",
            sublayers_per_depth_block=4,
            backend="eager",
        ),
    )
    return KimiK3Config(
        vocab_size=vocab_size,
        d_model=d_model,
        backbone=backbone,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        image_token_id=3,
        video_token_id=4,
        vision=VisionEncoderConfig(
            image_size=16,
            patch_size=4,
            embed_dim=16,
            depth=2,
            num_heads=4,
            use_cls_token=False,
        ),
        vision_projector=VisionProjectorConfig(64, 32, d_model),
        mtp=mtp,
        enable_vision=enable_vision,
        enable_mtp=enable_mtp,
    )


def kimi_k3_canonical_config() -> KimiK3Config:
    """Build canonical architecture metadata without allocating its weights."""
    d_model, vocab_size = 7168, 160000
    backbone, kda, mla, moe = _build_text_configs(
        d_model,
        num_heads=56,
        key_head_dim=128,
        q_head_dim=128,
        latent_dim=3584,
        shared_hidden=3072,
        routed_hidden=3072,
        num_routed=896,
        top_k=16,
        num_shared=2,
        groups=23,
        attnres_sublayers=24,
        target_depth_blocks=8,
    )
    mtp = KimiMTPConfig(
        d_model=d_model,
        vocab_size=vocab_size,
        kda_config=kda,
        mla_config=mla,
        stable_latent_moe_config=moe,
        attention_residual_config=AttentionResidualConfig(
            d_model,
            mode="block",
            sublayers_per_depth_block=8,
            target_num_depth_blocks=1,
            backend="two_phase",
        ),
    )
    return KimiK3Config(
        vocab_size=vocab_size,
        d_model=d_model,
        backbone=backbone,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        image_token_id=3,
        video_token_id=4,
        vision=VisionEncoderConfig(
            image_size=224,
            patch_size=14,
            embed_dim=1152,
            depth=27,
            num_heads=16,
            use_cls_token=False,
        ),
        vision_projector=VisionProjectorConfig(4608, 7168, d_model),
        mtp=mtp,
    )
