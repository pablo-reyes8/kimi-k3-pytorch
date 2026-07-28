from dataclasses import dataclass

import torch


@dataclass
class VisionEncoderOutput:
    last_hidden_state: torch.Tensor
    grid_size: tuple[int, int]
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


@dataclass
class PixelShuffleOutput:
    last_hidden_state: torch.Tensor
    grid_size: tuple[int, int]

