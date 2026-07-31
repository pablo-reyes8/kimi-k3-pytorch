"""Serializable distributed sampling for deterministic resume."""

from __future__ import annotations

from torch.utils.data import DistributedSampler


class StatefulDistributedSampler(DistributedSampler):
    """DistributedSampler with an explicit epoch and consumed-index cursor."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cursor = 0

    def __iter__(self):
        indices = list(super().__iter__())
        start = min(self.cursor, len(indices))
        for index in indices[start:]:
            self.cursor += 1
            yield index

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        self.cursor = 0

    def state_dict(self) -> dict[str, int]:
        return {"epoch": int(self.epoch), "cursor": int(self.cursor)}

    def load_state_dict(self, state: dict[str, int]) -> None:
        epoch = int(state["epoch"])
        cursor = int(state["cursor"])
        if epoch < 0 or cursor < 0:
            raise ValueError("sampler epoch and cursor must be non-negative")
        self.epoch = epoch
        self.cursor = cursor


__all__ = ["StatefulDistributedSampler"]
