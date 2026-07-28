"""Explicit Quantile Balancing lifecycle across optimizer windows."""

from __future__ import annotations

from contextlib import contextmanager

from src.stable_latent_moe import StableLatentMoE


class MoEController:
    """Coordinate every MoE layer without reaching into private buffers."""

    def __init__(self, model):
        raw_model = model.module if hasattr(model, "module") else model
        self.layers = tuple(
            module
            for module in raw_model.modules()
            if isinstance(module, StableLatentMoE)
            and module.config.enable_quantile_balancing
        )
        self.window_open = False

    def begin(self) -> None:
        if self.window_open:
            raise RuntimeError("a Quantile Balancing window is already open")
        for layer in self.layers:
            layer.begin_balance_accumulation()
        self.window_open = True

    def commit(self) -> tuple[object, ...]:
        if not self.window_open:
            raise RuntimeError("no Quantile Balancing window is open")
        updates = tuple(
            layer.finalize_and_commit_balance() for layer in self.layers
        )
        self.window_open = False
        return updates

    def discard(self) -> None:
        if not self.window_open:
            return
        for layer in self.layers:
            layer.discard_balance_accumulation()
        self.window_open = False

    @contextmanager
    def logical_batch(self):
        self.begin()
        try:
            yield self
        except BaseException:
            self.discard()
            raise

    def assert_clean(self) -> None:
        if self.window_open:
            raise RuntimeError("checkpoint requested with an open QB window")
