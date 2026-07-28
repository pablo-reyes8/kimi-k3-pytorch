"""Architecture-neutral text data for Kimi-K3 Mini research."""

from .batch import normalize_lm_batch
from .synthetic_long_context_retrieval import (
    SimpleWordTokenizer,
    SyntheticRetrievalConfig,
    SyntheticRetrievalDataset,
    create_synthetic_retrieval_dataloaders,
)
from .text_datasets import TextDataloaderConfig, create_text_dataloaders

__all__ = [
    "SimpleWordTokenizer",
    "SyntheticRetrievalConfig",
    "SyntheticRetrievalDataset",
    "TextDataloaderConfig",
    "create_synthetic_retrieval_dataloaders",
    "create_text_dataloaders",
    "normalize_lm_batch",
]
