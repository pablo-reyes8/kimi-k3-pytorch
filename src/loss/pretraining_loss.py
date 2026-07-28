"""High-level composition of Kimi next-token and MTP pretraining losses."""

from .pretraining.composite import KimiPretrainingLoss

__all__ = ["KimiPretrainingLoss"]
