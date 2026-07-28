"""Dependency-free structured logging backends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Protocol


Scalar = int | float | bool | str | None


class TrainingLogger(Protocol):
    def log(self, step: int, metrics: Mapping[str, Scalar]) -> None: ...
    def close(self) -> None: ...


class JSONLLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def log(self, step: int, metrics: Mapping[str, Scalar]) -> None:
        record = {"step": int(step), **dict(metrics)}
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()


class MemoryLogger:
    """Small backend useful for tests and notebooks."""

    def __init__(self):
        self.records: list[dict] = []

    def log(self, step: int, metrics: Mapping[str, Scalar]) -> None:
        self.records.append({"step": int(step), **dict(metrics)})

    def close(self) -> None:
        return None
