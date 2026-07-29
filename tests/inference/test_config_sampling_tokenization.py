import pytest
import torch

from data import load_tokenizer_from_data_yaml
from inference import (
    GenerationConfig,
    apply_repetition_penalty,
    decode_token_ids,
    encode_prompt,
    load_generation_config,
    sample_next_token,
    top_k_filter,
    top_p_filter,
)


def test_generation_profiles_and_invalid_controls(tmp_path):
    greedy = load_generation_config("config/inference/greedy.yaml")
    creative = load_generation_config("config/inference/creative.yaml")
    assert not greedy.do_sample
    assert creative.do_sample
    assert creative.top_k == 50
    with pytest.raises(ValueError, match="temperature"):
        GenerationConfig(temperature=0)
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "inference:\n  max_new_tokens: 2\n  typo: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="typo"):
        load_generation_config(bad)


def test_top_k_top_p_and_repetition_penalty_have_exact_semantics():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    top_k = top_k_filter(logits, 2)
    assert torch.isfinite(top_k).tolist() == [[True, True, False, False]]
    top_p = top_p_filter(logits, 0.6)
    assert torch.isfinite(top_p).sum() == 1
    repeated = apply_repetition_penalty(
        torch.tensor([[2.0, -2.0, 1.0]]),
        torch.tensor([[0, 1]]),
        2.0,
    )
    torch.testing.assert_close(
        repeated, torch.tensor([[1.0, -4.0, 1.0]])
    )


def test_sampling_is_seeded_and_greedy_is_argmax():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    greedy = sample_next_token(
        logits, GenerationConfig(do_sample=False)
    )
    assert greedy.item() == 3
    config = GenerationConfig(
        do_sample=True, temperature=0.8, top_k=3, seed=9
    )
    first = torch.Generator().manual_seed(9)
    second = torch.Generator().manual_seed(9)
    assert torch.equal(
        sample_next_token(logits, config, generator=first),
        sample_next_token(logits, config, generator=second),
    )


def test_synthetic_tokenizer_roundtrips_prompt_without_data_build():
    tokenizer = load_tokenizer_from_data_yaml(
        "config/kimi_full_pipeline/cpu_smoke/data.yaml"
    )
    ids = encode_prompt(
        "key_1 is value_4", tokenizer=tokenizer, add_bos_token=True
    )
    assert ids.shape[0] == 1
    assert ids[0, 0].item() == tokenizer.bos_id
    text = decode_token_ids(ids, tokenizer=tokenizer)
    assert text.startswith("<bos> key_1 is value_4")
