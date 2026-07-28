from __future__ import annotations

from typing import Protocol

import torch


class DraftFeatureProvider(Protocol):
    """Future extension point; Phase 9 uses the final backbone hidden state."""

    def get_features(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        ...
