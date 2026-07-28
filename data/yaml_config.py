"""YAML-driven construction of reusable Kimi data pipelines."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader

from configuration.yaml_utils import (
    ConfigError,
    dataclass_kwargs,
    expect_mapping,
    load_yaml_mapping,
    reject_unknown_keys,
)
from training.context_curriculum import ProgressiveContextCollator

from .synthetic_long_context_retrieval import (
    SyntheticRetrievalConfig,
    create_synthetic_retrieval_dataloaders,
)
from .text_datasets import (
    TextDataloaderConfig,
    create_text_dataloaders,
)


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int = 8
    validation_batch_size: int | None = None
    num_workers: int = 0
    shuffle_train: bool = True
    pin_memory: bool | Literal["auto"] = "auto"
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    drop_last: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("loader.batch_size must be positive")
        if self.validation_batch_size is not None and (
            self.validation_batch_size <= 0
        ):
            raise ValueError(
                "loader.validation_batch_size must be None or positive"
            )
        if self.num_workers < 0:
            raise ValueError("loader.num_workers must be non-negative")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError(
                "persistent_workers requires loader.num_workers > 0"
            )
        if self.prefetch_factor is not None and self.prefetch_factor <= 0:
            raise ValueError(
                "loader.prefetch_factor must be None or positive"
            )
        if self.prefetch_factor is not None and self.num_workers == 0:
            raise ValueError(
                "loader.prefetch_factor requires loader.num_workers > 0"
            )
        if self.pin_memory not in (True, False, "auto"):
            raise ValueError("loader.pin_memory must be true, false or 'auto'")

    @property
    def eval_batch_size(self) -> int:
        return self.validation_batch_size or self.batch_size

    @property
    def resolved_pin_memory(self) -> bool:
        if self.pin_memory == "auto":
            return torch.cuda.is_available()
        return bool(self.pin_memory)


@dataclass(frozen=True)
class DataPipelineConfig:
    name: str
    kind: Literal["synthetic_retrieval", "hf_text"]
    dataset: SyntheticRetrievalConfig | TextDataloaderConfig
    loader: LoaderConfig
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("data.name must not be empty")
        if self.kind not in {"synthetic_retrieval", "hf_text"}:
            raise ValueError("unsupported data.kind")
    @property
    def max_seq_len(self) -> int:
        block_size = getattr(self.dataset, "block_size", None)
        if block_size is None:
            raise ValueError(
                "dataset.block_size must be explicit for YAML pipelines"
            )
        return int(block_size)


@dataclass
class DataBundle:
    """Loaders, tokenizer and context-aware factory passed to training."""

    train_loader: DataLoader
    val_loader: DataLoader | None
    tokenizer: Any
    config: DataPipelineConfig
    train_loader_factory: Any

    @property
    def vocab_size(self) -> int:
        value = getattr(self.tokenizer, "vocab_size", None)
        if value is None and hasattr(self.tokenizer, "get_vocab_size"):
            value = self.tokenizer.get_vocab_size()
        if value is None:
            raise ValueError("tokenizer does not expose a vocabulary size")
        return int(value)

    def token_id(self, token: str) -> int | None:
        if hasattr(self.tokenizer, "token_to_id"):
            return self.tokenizer.token_to_id(token)
        attribute = {
            "<pad>": "pad_id",
            "<bos>": "bos_id",
            "<eos>": "eos_id",
        }.get(token)
        return None if attribute is None else getattr(
            self.tokenizer, attribute, None
        )


def load_data_config(path: str | Path) -> DataPipelineConfig:
    """Parse and validate data YAML without downloading or building data."""
    source, root = load_yaml_mapping(path)
    data = expect_mapping(root, "data", path="root")
    reject_unknown_keys(root, path="root")
    name = str(data.pop("name", source.stem))
    kind = str(data.pop("kind", data.pop("type", "")))
    dataset_values = expect_mapping(data, "dataset", path="data")
    loader_values = expect_mapping(data, "loader", path="data")
    reject_unknown_keys(data, path="data")
    loader = LoaderConfig(
        **dataclass_kwargs(loader_values, LoaderConfig, path="data.loader")
    )
    misplaced = {"batch_size", "num_workers"} & set(dataset_values)
    if misplaced:
        names = ", ".join(sorted(misplaced))
        raise ConfigError(
            f"configure {names} only in data.loader, not data.dataset"
        )
    if kind == "synthetic_retrieval":
        dataset_values.setdefault("batch_size", loader.batch_size)
        dataset_values.setdefault("num_workers", loader.num_workers)
        dataset = SyntheticRetrievalConfig(
            **dataclass_kwargs(
                dataset_values,
                SyntheticRetrievalConfig,
                path="data.dataset",
            )
        )
        dataset.validate()
    elif kind == "hf_text":
        dataset_values.setdefault("batch_size", loader.batch_size)
        dataset_values.setdefault("num_workers", loader.num_workers)
        dataset = TextDataloaderConfig(
            **dataclass_kwargs(
                dataset_values,
                TextDataloaderConfig,
                path="data.dataset",
            )
        )
        if dataset.block_size is None:
            raise ConfigError(
                "data.dataset.block_size must be explicit in YAML"
            )
    else:
        raise ConfigError(
            "data.kind must be 'synthetic_retrieval' or 'hf_text'"
        )
    return DataPipelineConfig(
        name=name,
        kind=kind,
        dataset=dataset,
        loader=loader,
        source_path=source,
    )


def _rewrap_loader(
    dataset,
    loader: LoaderConfig,
    *,
    train: bool,
    max_seq_len: int,
    pad_token_id: int,
    ignore_index: int = -100,
) -> DataLoader:
    batch_size = loader.batch_size if train else loader.eval_batch_size
    generator = torch.Generator().manual_seed(loader.seed)
    collator = ProgressiveContextCollator(
        max_seq_len,
        pad_token_id=pad_token_id,
        ignore_index=ignore_index,
    )
    worker_kwargs = (
        {}
        if loader.num_workers == 0 or loader.prefetch_factor is None
        else {"prefetch_factor": loader.prefetch_factor}
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=loader.shuffle_train if train else False,
        num_workers=loader.num_workers,
        pin_memory=loader.resolved_pin_memory,
        persistent_workers=loader.persistent_workers,
        drop_last=loader.drop_last if train else False,
        generator=generator,
        collate_fn=collator,
        **worker_kwargs,
    )


def build_dataloaders_from_yaml(path: str | Path) -> DataBundle:
    """Build loaders once and expose a PCC-compatible loader factory."""
    config = load_data_config(path)
    if config.kind == "synthetic_retrieval":
        dataset_config = replace(
            config.dataset,
            batch_size=config.loader.batch_size,
            num_workers=config.loader.num_workers,
        )
        raw_train, raw_val, tokenizer = (
            create_synthetic_retrieval_dataloaders(
                dataset_config,
                use_mtp=False,
            )
        )
    else:
        dataset_config = replace(
            config.dataset,
            batch_size=config.loader.batch_size,
            num_workers=config.loader.num_workers,
        )
        raw_train, raw_val, tokenizer = create_text_dataloaders(
            dataset_config,
            use_mtp=False,
        )
    pad_id = (
        tokenizer.token_to_id("<pad>")
        if hasattr(tokenizer, "token_to_id")
        else tokenizer.pad_id
    )

    def make_train_loader(max_seq_len: int):
        if max_seq_len > config.max_seq_len:
            raise ValueError(
                f"requested context {max_seq_len} exceeds data block_size "
                f"{config.max_seq_len}"
            )
        return _rewrap_loader(
            raw_train.dataset,
            config.loader,
            train=True,
            max_seq_len=max_seq_len,
            pad_token_id=int(pad_id),
        )

    train_loader = make_train_loader(config.max_seq_len)
    val_loader = (
        None
        if raw_val is None
        else _rewrap_loader(
            raw_val.dataset,
            config.loader,
            train=False,
            max_seq_len=config.max_seq_len,
            pad_token_id=int(pad_id),
        )
    )
    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        config=config,
        train_loader_factory=make_train_loader,
    )


__all__ = [
    "DataBundle",
    "DataPipelineConfig",
    "LoaderConfig",
    "build_dataloaders_from_yaml",
    "load_data_config",
]
