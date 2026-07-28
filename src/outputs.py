"""Small model-output contract shared by training and future Kimi components."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import torch


ScalarMetric = Union[torch.Tensor, float]


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    auxiliary_losses: Dict[str, torch.Tensor] = field(default_factory=dict)
    metrics: Dict[str, ScalarMetric] = field(default_factory=dict)
