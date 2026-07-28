"""Reusable neural-network primitives shared by Kimi attention implementations."""

from .head_utils import combine_heads, split_heads
from .headwise_rmsnorm import HeadwiseRMSNorm
from .output_gate import FullRankOutputGate
from .outputs import AttentionModuleOutput, KDAProjectionOutput
from .postprocess import PrimitiveAttentionPostprocess
from .short_conv import CausalShortConv1D
from .situ_glu import SiTUGLU, situ_glu_activation, softcap
from .states import ShortConvState

__all__ = [
    "AttentionModuleOutput",
    "CausalShortConv1D",
    "FullRankOutputGate",
    "HeadwiseRMSNorm",
    "KDAProjectionOutput",
    "PrimitiveAttentionPostprocess",
    "ShortConvState",
    "SiTUGLU",
    "combine_heads",
    "situ_glu_activation",
    "softcap",
    "split_heads",
]
