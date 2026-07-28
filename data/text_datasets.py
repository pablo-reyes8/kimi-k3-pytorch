"""General text dataset loaders for causal language modeling.

The project-specific synthetic retrieval and TinyStories loaders remain useful for
their specialized behavior. This module adds a small, uniform Hugging Face path:

    load_hf_text_dataset -> train/load tokenizer -> CausalTextDataset -> DataLoader

It intentionally returns batches compatible with ``normalize_lm_batch``:

    {"input_ids": [B,T], "labels": [B,T]}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from datasets import DatasetDict, IterableDatasetDict, load_dataset
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
except ImportError as exc:  # pragma: no cover - exercised only without optional deps.
    load_dataset = None
    DatasetDict = None
    IterableDatasetDict = None
    Tokenizer = None
    decoders = None
    models = None
    normalizers = None
    pre_tokenizers = None
    trainers = None
    _OPTIONAL_IMPORT_ERROR = exc
else:
    _OPTIONAL_IMPORT_ERROR = None


@dataclass(frozen=True)
class HFTextDatasetPreset:
    name: str
    dataset_name: str
    subset: Optional[str]
    text_field: str
    train_split: str
    validation_split: Optional[str]
    recommended_block_size: int
    notes: str


@dataclass(frozen=True)
class TextDataloaderConfig:
    preset_name: str = "wikitext2"
    block_size: Optional[int] = None
    batch_size: int = 8
    num_workers: int = 0
    tokenizer_path: Optional[str | Path] = None
    vocab_size: int = 16_000
    min_frequency: int = 2
    max_tokenizer_documents: Optional[int] = 50_000
    max_train_documents: Optional[int] = 20_000
    max_validation_documents: Optional[int] = 2_000


HF_TEXT_DATASETS: Dict[str, HFTextDatasetPreset] = {
    "wikitext2": HFTextDatasetPreset(
        name="wikitext2",
        dataset_name="Salesforce/wikitext",
        subset="wikitext-2-raw-v1",
        text_field="text",
        train_split="train",
        validation_split="validation",
        recommended_block_size=256,
        notes="Small Wikipedia long-form language-modeling benchmark.",
    ),
    "tinystories": HFTextDatasetPreset(
        name="tinystories",
        dataset_name="roneneldan/TinyStories",
        subset=None,
        text_field="text",
        train_split="train",
        validation_split="validation",
        recommended_block_size=256,
        notes="Simple-story corpus useful for tiny LMs and qualitative generation.",
    ),
    "ag_news": HFTextDatasetPreset(
        name="ag_news",
        dataset_name="fancyzhx/ag_news",
        subset=None,
        text_field="text",
        train_split="train",
        validation_split="test",
        recommended_block_size=128,
        notes="Short news articles; useful as a compact domain-shift corpus.",
    ),
    "imdb": HFTextDatasetPreset(
        name="imdb",
        dataset_name="stanfordnlp/imdb",
        subset="plain_text",
        text_field="text",
        train_split="train",
        validation_split="test",
        recommended_block_size=256,
        notes="Movie reviews with longer examples than news; good for medium context.",
    ),
    "minipile": HFTextDatasetPreset(
        name="minipile",
        dataset_name="JeanKaddour/minipile",
        subset=None,
        text_field="text",
        train_split="train",
        validation_split="validation",
        recommended_block_size=256,
        notes="Small diverse pretraining mix derived from The Pile-style sources.",
    ),
    "fineweb_edu_10bt_mincols": HFTextDatasetPreset(
        name="fineweb_edu_10bt_mincols",
        dataset_name="EliMC/fineweb-edu-10BT-mincols",
        subset=None,
        text_field="text",
        train_split="train",
        validation_split=None,
        recommended_block_size=512,
        notes="Educational web sample; use streaming or max_*_documents for local runs.",
    ),
}


def _require_optional_dependencies() -> None:
    if _OPTIONAL_IMPORT_ERROR is not None:
        raise ImportError(
            "Hugging Face dataset support requires optional dependencies. "
            'Install with: pip install -e ".[data]"'
        ) from _OPTIONAL_IMPORT_ERROR


def iter_texts(
    split,
    text_field: str = "text",
    max_documents: Optional[int] = None,
) -> Iterator[str]:
    count = 0
    for example in split:
        text = example.get(text_field, None)
        if isinstance(text, str) and text.strip():
            yield text
            count += 1
            if max_documents is not None and count >= max_documents:
                break


def train_byte_level_bpe_tokenizer(
    texts: Iterable[str],
    *,
    vocab_size: int = 16_000,
    min_frequency: int = 2,
    save_path: Optional[str | Path] = None,
) -> "Tokenizer":
    _require_optional_dependencies()

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=int(vocab_size),
        min_frequency=int(min_frequency),
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(save_path))

    return tokenizer


def load_or_train_byte_level_tokenizer(
    train_split,
    *,
    text_field: str = "text",
    tokenizer_path: str | Path,
    vocab_size: int = 16_000,
    min_frequency: int = 2,
    max_tokenizer_documents: Optional[int] = 50_000,
) -> "Tokenizer":
    _require_optional_dependencies()

    tokenizer_path = Path(tokenizer_path)
    if tokenizer_path.exists():
        return Tokenizer.from_file(str(tokenizer_path))

    return train_byte_level_bpe_tokenizer(
        iter_texts(train_split, text_field=text_field, max_documents=max_tokenizer_documents),
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        save_path=tokenizer_path,
    )


class CausalTextDataset(Dataset):
    """Tokenize text documents, concatenate them, and expose fixed LM blocks."""

    def __init__(
        self,
        texts: Iterable[str],
        tokenizer,
        *,
        block_size: int = 256,
        max_documents: Optional[int] = None,
    ):
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")

        self.block_size = int(block_size)

        bos_id = tokenizer.token_to_id("<bos>")
        eos_id = tokenizer.token_to_id("<eos>")
        pad_id = tokenizer.token_to_id("<pad>")

        if bos_id is None or eos_id is None or pad_id is None:
            raise ValueError("Tokenizer must define <bos>, <eos>, and <pad> tokens.")

        all_ids: List[int] = []
        for i, text in enumerate(texts):
            if max_documents is not None and i >= max_documents:
                break
            if not isinstance(text, str) or not text.strip():
                continue
            encoded = tokenizer.encode(text)
            all_ids.extend([bos_id] + encoded.ids + [eos_id])

        chunk_len = self.block_size + 1
        if len(all_ids) < chunk_len:
            raise ValueError(
                f"Not enough tokens ({len(all_ids)}) to build one block of length {chunk_len}."
            )

        n_chunks = len(all_ids) // chunk_len
        ids = torch.tensor(all_ids[: n_chunks * chunk_len], dtype=torch.long)
        ids = ids.view(n_chunks, chunk_len)

        self.input_ids = ids[:, :-1].contiguous()
        self.labels = ids[:, 1:].contiguous()

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
        }


def resolve_hf_text_preset(name: str) -> HFTextDatasetPreset:
    try:
        return HF_TEXT_DATASETS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown dataset preset {name!r}. Available: {sorted(HF_TEXT_DATASETS)}"
        ) from exc


def load_hf_text_splits(
    preset_name: str,
    *,
    streaming: bool = False,
):
    _require_optional_dependencies()

    preset = resolve_hf_text_preset(preset_name)
    if preset.subset is None:
        return load_dataset(preset.dataset_name, streaming=streaming)
    return load_dataset(preset.dataset_name, preset.subset, streaming=streaming)


def create_hf_text_dataloaders(
    preset_name: str,
    *,
    block_size: Optional[int] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    tokenizer_path: Optional[str | Path] = None,
    vocab_size: int = 16_000,
    min_frequency: int = 2,
    max_tokenizer_documents: Optional[int] = 50_000,
    max_train_documents: Optional[int] = 20_000,
    max_validation_documents: Optional[int] = 2_000,
) -> Tuple[DataLoader, Optional[DataLoader], "Tokenizer"]:
    """Create train/validation loaders for a preset HF text dataset.

    The max_* parameters are deliberately conservative by default so first-run
    experiments do not accidentally tokenize an entire web-scale corpus.
    """
    preset = resolve_hf_text_preset(preset_name)
    splits = load_hf_text_splits(preset_name, streaming=False)

    train_split = splits[preset.train_split]
    val_split = splits[preset.validation_split] if preset.validation_split else None

    block_size = int(block_size or preset.recommended_block_size)
    tokenizer_path = Path(tokenizer_path or f"data/cache/tokenizers/{preset.name}.json")

    tokenizer = load_or_train_byte_level_tokenizer(
        train_split,
        text_field=preset.text_field,
        tokenizer_path=tokenizer_path,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        max_tokenizer_documents=max_tokenizer_documents,
    )

    train_ds = CausalTextDataset(
        iter_texts(train_split, text_field=preset.text_field),
        tokenizer,
        block_size=block_size,
        max_documents=max_train_documents,
    )

    val_loader = None
    if val_split is not None:
        val_ds = CausalTextDataset(
            iter_texts(val_split, text_field=preset.text_field),
            tokenizer,
            block_size=block_size,
            max_documents=max_validation_documents,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, tokenizer


def create_text_dataloaders(
    cfg: TextDataloaderConfig,
    *,
    use_mtp: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader], "Tokenizer"]:
    """Uniform HF text dataloader entrypoint.

    This mirrors ``create_synthetic_retrieval_dataloaders(cfg=..., use_mtp=...)``
    for notebook ergonomics. The causal text batches already return shifted
    ``labels``; when ``use_mtp=True`` the model can derive MTP labels internally
    from those labels, so the dataloader output stays the same.
    """
    del use_mtp
    return create_hf_text_dataloaders(
        cfg.preset_name,
        block_size=cfg.block_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        tokenizer_path=cfg.tokenizer_path,
        vocab_size=cfg.vocab_size,
        min_frequency=cfg.min_frequency,
        max_tokenizer_documents=cfg.max_tokenizer_documents,
        max_train_documents=cfg.max_train_documents,
        max_validation_documents=cfg.max_validation_documents,
    )


def available_hf_text_dataset_presets() -> Sequence[str]:
    return tuple(sorted(HF_TEXT_DATASETS))
