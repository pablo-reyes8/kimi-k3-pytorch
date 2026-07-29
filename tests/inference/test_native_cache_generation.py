import torch

from data import load_tokenizer_from_data_yaml
from inference import (
    GenerationConfig,
    cache_summary,
    compare_full_vs_cached_logits,
    generate_tokens,
    inference_autoregressive,
)
from src import KimiK3, kimi_k3_cpu_tiny_config


def tiny_text_model():
    torch.manual_seed(91)
    return KimiK3(
        kimi_k3_cpu_tiny_config(enable_vision=False)
    ).cpu()


def test_prefill_decode_audit_matches_full_kimi_logits():
    model = tiny_text_model()
    ids = torch.tensor([[5, 6, 7, 8, 9, 10]])
    audit = compare_full_vs_cached_logits(model, ids, split=3)
    assert audit["allclose"]
    assert audit["max_abs_diff"] < 1e-5
    assert audit["cache_stats"]["sequence_length"] == 6
    assert audit["cache_stats"]["num_kda_layers"] == 3
    assert audit["cache_stats"]["num_mla_layers"] == 2


def test_generation_uses_and_returns_fully_advanced_native_cache():
    model = tiny_text_model().train()
    ids = torch.tensor([[5, 6, 7, 8]])
    output = generate_tokens(
        model,
        ids,
        generation_config=GenerationConfig(
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
            return_scores=True,
        ),
    )
    assert output.generated_ids.shape == (1, 3)
    assert output.sequences.shape == (1, 7)
    assert output.cache.sequence_length == 7
    assert output.cache.sequence_lengths.tolist() == [7]
    assert len(output.scores) == 3
    assert output.cache_stats == cache_summary(output.cache)
    assert output.cache_stats["memory_bytes"] > 0
    assert model.training


def test_master_prompt_api_encodes_generates_and_decodes_text():
    tokenizer = load_tokenizer_from_data_yaml(
        "config/kimi_full_pipeline/cpu_smoke/data.yaml"
    )
    model = tiny_text_model()
    output = inference_autoregressive(
        model,
        "key_1 is value_4",
        tokenizer=tokenizer,
        generation_config=GenerationConfig(
            max_new_tokens=1,
            do_sample=False,
            eos_token_id=127,
            pad_token_id=tokenizer.pad_id,
        ),
    )
    assert output.generated_tokens == 1
    assert isinstance(output.text, str)
    assert isinstance(output.completion_text, str)
    assert output.cache.sequence_length == output.sequences.shape[1]
