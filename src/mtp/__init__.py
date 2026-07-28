"""Multi-token prediction components used as an optional KimiK3 output head."""

from .alignment import MTPTrainingView, build_mtp_training_view
from .block import KimiMTPBlock
from .config import KimiMTPConfig
from .fusion import KimiMTPFusion
from .features import DraftFeatureProvider
from .head import KimiMTPHead
from .losses import combine_ntp_mtp_losses, masked_mtp_cross_entropy
from .outputs import KimiMTPOutput, MTPDiagnostics, MTPDraftOutput
from .parameter_count import mtp_parameter_counts

__all__ = [
    "DraftFeatureProvider",
    "KimiMTPBlock",
    "KimiMTPConfig",
    "KimiMTPFusion",
    "KimiMTPHead",
    "KimiMTPOutput",
    "MTPDiagnostics",
    "MTPDraftOutput",
    "MTPTrainingView",
    "build_mtp_training_view",
    "combine_ntp_mtp_losses",
    "masked_mtp_cross_entropy",
    "mtp_parameter_counts",
]
