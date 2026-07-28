from __future__ import annotations

from dataclasses import asdict, dataclass

from src.attention_residuals import AttentionResidualConfig
from src.kda import KDAConfig
from src.mla import GatedMLAConfig
from src.stable_latent_moe import StableLatentMoEConfig


@dataclass(frozen=True)
class KimiMTPConfig:
    """Configuration for Kimi K3's single next-next-token prediction head."""

    d_model: int
    vocab_size: int
    kda_config: KDAConfig | None = None
    mla_config: GatedMLAConfig | None = None
    stable_latent_moe_config: StableLatentMoEConfig | None = None
    attention_residual_config: AttentionResidualConfig | None = None
    enabled: bool = True
    num_mtp_layers: int = 1
    future_offset: int = 2
    loss_weight: float = 0.1
    fusion_kind: str = "concat_project"
    normalize_hidden: bool = True
    normalize_future_embedding: bool = True
    fusion_bias: bool = False
    share_main_lm_head: bool = True
    detach_backbone_hidden: bool = False
    ignore_index: int = -100
    rms_norm_eps: float = 1e-6
    residual_dropout: float = 0.0
    init_std: float = 0.02
    return_logits_by_default: bool = True

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if self.num_mtp_layers != 1:
            raise ValueError("Kimi K3 reports exactly one MTP layer")
        if self.future_offset != 2:
            raise ValueError("canonical Kimi MTP predicts exactly x[t+2]")
        if self.loss_weight < 0:
            raise ValueError("loss_weight must be >= 0")
        if self.fusion_kind != "concat_project":
            raise ValueError("only fusion_kind='concat_project' is supported")
        if not self.normalize_hidden or not self.normalize_future_embedding:
            raise ValueError("canonical MTP requires both fusion inputs normalized")
        if self.fusion_bias:
            raise ValueError("canonical MTP fusion is bias-free")
        if not self.share_main_lm_head:
            raise ValueError("canonical MTP shares the main LM head")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be > 0")
        if not 0.0 <= self.residual_dropout < 1.0:
            raise ValueError("residual_dropout must satisfy 0 <= p < 1")
        if self.init_std <= 0:
            raise ValueError("init_std must be > 0")
        if not self.enabled:
            return
        required = (
            ("kda_config", self.kda_config),
            ("mla_config", self.mla_config),
            ("stable_latent_moe_config", self.stable_latent_moe_config),
            ("attention_residual_config", self.attention_residual_config),
        )
        for name, config in required:
            if config is None:
                raise ValueError(f"{name} is required when MTP is enabled")
            if config.d_model != self.d_model:
                raise ValueError(f"{name}.d_model must match d_model")
        if self.attention_residual_config.mode == "standard":
            raise ValueError("Kimi MTP requires Full or Block AttnRes")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "KimiMTPConfig":
        values = dict(values)
        converters = (
            ("kda_config", KDAConfig),
            ("mla_config", GatedMLAConfig),
            ("stable_latent_moe_config", StableLatentMoEConfig),
            ("attention_residual_config", AttentionResidualConfig),
        )
        for name, config_cls in converters:
            if isinstance(values.get(name), dict):
                values[name] = config_cls.from_dict(values[name])
        return cls(**values)
