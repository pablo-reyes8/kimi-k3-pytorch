from __future__ import annotations

from dataclasses import asdict, dataclass

from src.attention_residuals import AttentionResidualConfig
from src.kda import KDAConfig
from src.mla import GatedMLAConfig


CANONICAL_ATTENTION_PATTERN = ("kda", "kda", "kda", "gated_mla")


@dataclass(frozen=True)
class HybridBackboneConfig:
    """Configuration for the pre-AttnRes, pre-MoE Kimi K3 backbone."""

    d_model: int
    num_hybrid_groups: int
    attention_pattern: tuple[str, ...] = CANONICAL_ATTENTION_PATTERN
    add_final_gated_mla: bool = True
    add_ffn_after_final_global: bool = True
    rms_norm_eps: float = 1e-6
    residual_dropout: float = 0.0
    ffn_dropout: float = 0.0
    mlp_hidden_dim: int | None = None
    kda_config: KDAConfig | None = None
    mla_config: GatedMLAConfig | None = None
    use_dense_ffn: bool = True
    activation: str = "situ_glu"
    ffn_bias: bool = False
    init_std: float = 0.02
    attention_residual_config: AttentionResidualConfig | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "attention_pattern", tuple(self.attention_pattern))
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if self.num_hybrid_groups <= 0:
            raise ValueError("num_hybrid_groups must be > 0")
        if not self.attention_pattern:
            raise ValueError("attention_pattern must not be empty")
        unknown = set(self.attention_pattern) - {"kda", "gated_mla"}
        if unknown:
            raise ValueError(f"unsupported attention types: {sorted(unknown)}")
        if self.attention_pattern != CANONICAL_ATTENTION_PATTERN:
            raise ValueError(
                "the Kimi K3 profile requires the explicit 3:1 pattern "
                f"{CANONICAL_ATTENTION_PATTERN}"
            )
        if not self.add_final_gated_mla:
            raise ValueError("the Kimi K3 profile requires a final Gated MLA")
        if "kda" in self.attention_pattern and self.kda_config is None:
            raise ValueError("kda_config is required by attention_pattern")
        if (
            "gated_mla" in self.attention_pattern
            or self.add_final_gated_mla
        ) and self.mla_config is None:
            raise ValueError("mla_config is required by attention_pattern")
        for name, subconfig in (
            ("kda_config", self.kda_config),
            ("mla_config", self.mla_config),
        ):
            if subconfig is not None and subconfig.d_model != self.d_model:
                raise ValueError(
                    f"{name}.d_model must equal backbone d_model"
                )
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be > 0")
        for name in ("residual_dropout", "ffn_dropout"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must satisfy 0 <= p < 1")
        if self.resolved_mlp_hidden_dim <= 0:
            raise ValueError("mlp_hidden_dim must be > 0")
        if not self.use_dense_ffn:
            raise ValueError(
                "this pre-MoE phase requires use_dense_ffn=True"
            )
        if self.activation != "situ_glu":
            raise ValueError("only activation='situ_glu' is supported")
        if self.init_std <= 0:
            raise ValueError("init_std must be > 0")
        if self.attention_residual_config is not None:
            if self.attention_residual_config.d_model != self.d_model:
                raise ValueError(
                    "attention_residual_config.d_model must match backbone"
                )
            self.attention_residual_config.validate_topology(
                self.num_attention_layers,
                every_layer_has_ffn=self.add_ffn_after_final_global,
            )

    @property
    def depth_mixing(self) -> str:
        return (
            "standard"
            if self.attention_residual_config is None
            else self.attention_residual_config.mode
        )

    @property
    def resolved_mlp_hidden_dim(self) -> int:
        return 4 * self.d_model if self.mlp_hidden_dim is None else self.mlp_hidden_dim

    @property
    def num_group_layers(self) -> int:
        return self.num_hybrid_groups * len(self.attention_pattern)

    @property
    def num_attention_layers(self) -> int:
        return self.num_group_layers + int(self.add_final_gated_mla)

    @property
    def num_kda_layers(self) -> int:
        return self.num_hybrid_groups * self.attention_pattern.count("kda")

    @property
    def num_mla_layers(self) -> int:
        return (
            self.num_hybrid_groups
            * self.attention_pattern.count("gated_mla")
            + int(self.add_final_gated_mla)
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "HybridBackboneConfig":
        values = dict(values)
        if isinstance(values.get("kda_config"), dict):
            values["kda_config"] = KDAConfig.from_dict(values["kda_config"])
        if isinstance(values.get("mla_config"), dict):
            values["mla_config"] = GatedMLAConfig.from_dict(
                values["mla_config"]
            )
        if isinstance(values.get("attention_residual_config"), dict):
            values["attention_residual_config"] = (
                AttentionResidualConfig.from_dict(
                    values["attention_residual_config"]
                )
            )
        if "attention_pattern" in values:
            values["attention_pattern"] = tuple(values["attention_pattern"])
        return cls(**values)
