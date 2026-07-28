from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal


@dataclass(frozen=True)
class AttentionResidualConfig:
    d_model: int
    mode: Literal["standard", "full", "block"] = "block"
    rms_norm_eps: float = 1e-6
    query_init: Literal["zeros"] = "zeros"
    logits_in_fp32: bool = True
    weighted_sum_in_fp32: bool = True
    transformer_layers_per_depth_block: int | None = None
    sublayers_per_depth_block: int | None = 24
    target_num_depth_blocks: int | None = None
    add_final_output_mixer: bool = True
    include_embedding_source: bool = True
    backend: Literal["eager", "two_phase"] = "eager"
    return_depth_weights: bool = False
    return_depth_stats: bool = False

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if self.mode not in ("standard", "full", "block"):
            raise ValueError("mode must be 'standard', 'full', or 'block'")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be > 0")
        if self.query_init != "zeros":
            raise ValueError("canonical Kimi AttnRes requires query_init='zeros'")
        if not self.include_embedding_source:
            raise ValueError("canonical Kimi AttnRes requires the embedding source")
        if not self.add_final_output_mixer:
            raise ValueError("canonical Kimi AttnRes requires a final output mixer")
        for name in (
            "transformer_layers_per_depth_block",
            "sublayers_per_depth_block",
            "target_num_depth_blocks",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0 when provided")
        if self.mode == "block":
            if (
                self.transformer_layers_per_depth_block is None
                and self.sublayers_per_depth_block is None
            ):
                raise ValueError("block mode requires a depth block size")
            if (
                self.transformer_layers_per_depth_block is not None
                and self.sublayers_per_depth_block is not None
                and 2 * self.transformer_layers_per_depth_block
                != self.sublayers_per_depth_block
            ):
                raise ValueError(
                    "transformer-layer and sublayer block sizes contradict"
                )
        elif (
            self.transformer_layers_per_depth_block is not None
            or self.sublayers_per_depth_block is not None
            or self.target_num_depth_blocks is not None
        ):
            raise ValueError(
                "depth block sizes are only valid when mode='block'"
            )
        if self.backend not in ("eager", "two_phase"):
            raise ValueError("backend must be 'eager' or 'two_phase'")
        if self.backend == "two_phase" and self.mode != "block":
            raise ValueError("two_phase backend is only defined for block mode")

    @property
    def resolved_sublayers_per_depth_block(self) -> int:
        if self.mode != "block":
            raise ValueError("only block mode has a depth block size")
        if self.sublayers_per_depth_block is not None:
            return self.sublayers_per_depth_block
        return 2 * self.transformer_layers_per_depth_block

    def num_depth_blocks(self, num_transformer_layers: int) -> int:
        if num_transformer_layers <= 0:
            raise ValueError("num_transformer_layers must be > 0")
        if self.mode != "block":
            return 0
        return math.ceil(
            2 * num_transformer_layers
            / self.resolved_sublayers_per_depth_block
        )

    def validate_topology(
        self,
        num_transformer_layers: int,
        *,
        every_layer_has_ffn: bool,
    ) -> None:
        if self.mode == "standard":
            return
        if not every_layer_has_ffn:
            raise ValueError(
                "canonical AttnRes requires Attention and FFN in every "
                "transformer layer, including the final global MLA"
            )
        if self.target_num_depth_blocks is not None:
            observed = self.num_depth_blocks(num_transformer_layers)
            if observed != self.target_num_depth_blocks:
                raise ValueError(
                    "target_num_depth_blocks does not match topology: "
                    f"expected {self.target_num_depth_blocks}, observed {observed}"
                )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "AttentionResidualConfig":
        return cls(**values)
