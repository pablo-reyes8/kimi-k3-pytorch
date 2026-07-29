from pathlib import Path

import pytest
import torch

from data import text_datasets as module
from data.text_datasets import (
    CausalTextDataset,
    TextDataloaderConfig,
    available_hf_text_dataset_presets,
    create_text_dataloaders,
    iter_texts,
    load_or_train_byte_level_tokenizer,
    resolve_hf_text_preset,
    train_byte_level_bpe_tokenizer,
)


def test_presets_include_expected_research_corpora_and_unknown_rejected():
    presets = set(available_hf_text_dataset_presets())
    assert {
        "wikitext2",
        "tinystories",
        "ag_news",
        "imdb",
        "minipile",
        "fineweb_edu_10bt_mincols",
        "fineweb_10bt",
        "fineweb_100bt",
        "fineweb_350bt",
    }.issubset(presets)
    assert resolve_hf_text_preset("wikitext2").dataset_name == "Salesforce/wikitext"
    with pytest.raises(KeyError, match="Available"):
        resolve_hf_text_preset("missing")


def test_iter_texts_filters_invalid_documents_and_respects_limit():
    split = [
        {"text": ""},
        {"text": "first"},
        {"other": "ignored"},
        {"text": "second"},
        {"text": "third"},
    ]
    assert list(iter_texts(split, max_documents=2)) == ["first", "second"]


def test_tokenizer_training_defines_special_tokens_and_can_persist(tmp_path):
    path = tmp_path / "tokenizer.json"
    tokenizer = train_byte_level_bpe_tokenizer(
        ["alpha beta gamma", "delta epsilon"],
        vocab_size=64,
        min_frequency=1,
        save_path=path,
    )
    assert path.exists()
    for token in ("<unk>", "<pad>", "<bos>", "<eos>"):
        assert tokenizer.token_to_id(token) is not None
    assert tokenizer.decode(tokenizer.encode("alpha beta").ids)


def test_load_or_train_reuses_existing_tokenizer_file(tmp_path, monkeypatch):
    path = tmp_path / "tokenizer.json"
    train_byte_level_bpe_tokenizer(["one two three"], vocab_size=32, save_path=path)

    def forbidden(*args, **kwargs):
        raise AssertionError("existing tokenizer should be loaded, not trained")

    monkeypatch.setattr(module, "train_byte_level_bpe_tokenizer", forbidden)
    loaded = load_or_train_byte_level_tokenizer(
        [{"text": "unused"}], tokenizer_path=path
    )
    assert loaded.token_to_id("<bos>") is not None


def local_tokenizer():
    return train_byte_level_bpe_tokenizer(
        ["one two three four", "five six seven eight"], vocab_size=64, min_frequency=1
    )


def test_causal_text_dataset_blocks_and_exact_shift():
    dataset = CausalTextDataset(
        ["one two three four", "five six seven eight"],
        local_tokenizer(),
        block_size=5,
    )
    sample = dataset[0]
    assert set(sample) == {"input_ids", "labels"}
    assert sample["input_ids"].shape == sample["labels"].shape == (5,)
    assert sample["input_ids"].dtype == sample["labels"].dtype == torch.long
    torch.testing.assert_close(sample["input_ids"][1:], sample["labels"][:-1])


@pytest.mark.parametrize("block_size", [0, -1])
def test_causal_text_dataset_rejects_invalid_block_size(block_size):
    with pytest.raises(ValueError):
        CausalTextDataset(["enough text here"], local_tokenizer(), block_size=block_size)


def test_causal_text_dataset_rejects_insufficient_tokens():
    with pytest.raises(ValueError, match="Not enough tokens"):
        CausalTextDataset(["x"], local_tokenizer(), block_size=100)


def test_causal_text_dataset_requires_special_tokens():
    from tokenizers import Tokenizer, models

    tokenizer = Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>"))
    with pytest.raises(ValueError, match="<bos>"):
        CausalTextDataset(["hello world"], tokenizer, block_size=2)


def test_create_text_dataloaders_forwards_every_configuration_field(monkeypatch):
    calls = {}

    def fake(preset_name, **kwargs):
        calls["preset_name"] = preset_name
        calls.update(kwargs)
        return "train", "validation", "tokenizer"

    monkeypatch.setattr(module, "create_hf_text_dataloaders", fake)
    config = TextDataloaderConfig(
        preset_name="ag_news",
        block_size=64,
        batch_size=4,
        num_workers=2,
        tokenizer_path=Path("custom.json"),
        vocab_size=123,
        min_frequency=3,
        max_tokenizer_documents=10,
        max_train_documents=20,
        max_validation_documents=5,
        streaming=True,
    )
    assert create_text_dataloaders(config, use_mtp=True) == (
        "train",
        "validation",
        "tokenizer",
    )
    assert calls == {
        "preset_name": "ag_news",
        "block_size": 64,
        "batch_size": 4,
        "num_workers": 2,
        "tokenizer_path": Path("custom.json"),
        "vocab_size": 123,
        "min_frequency": 3,
        "max_tokenizer_documents": 10,
        "max_train_documents": 20,
        "max_validation_documents": 5,
        "streaming": True,
    }


@pytest.mark.parametrize(
    "name,subset",
    [
        ("fineweb_10bt", "sample-10BT"),
        ("fineweb_100bt", "sample-100BT"),
        ("fineweb_350bt", "sample-350BT"),
    ],
)
def test_fineweb_scale_presets_use_streamable_official_configs(
    name, subset
):
    preset = resolve_hf_text_preset(name)
    assert preset.dataset_name == "HuggingFaceFW/fineweb"
    assert preset.subset == subset
    assert preset.validation_split is None


def test_hf_split_loader_forwards_subset_and_streaming(monkeypatch):
    calls = {}

    def fake_load_dataset(dataset_name, subset, *, streaming):
        calls.update(
            dataset_name=dataset_name,
            subset=subset,
            streaming=streaming,
        )
        return {"train": []}

    monkeypatch.setattr(module, "load_dataset", fake_load_dataset)
    result = module.load_hf_text_splits(
        "fineweb_100bt", streaming=True
    )
    assert result == {"train": []}
    assert calls == {
        "dataset_name": "HuggingFaceFW/fineweb",
        "subset": "sample-100BT",
        "streaming": True,
    }
