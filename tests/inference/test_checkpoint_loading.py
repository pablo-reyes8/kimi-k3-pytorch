import torch

from data import load_tokenizer_from_data_yaml
from inference import (
    GenerationConfig,
    ModelLoadConfig,
    inference_autoregressive,
    load_kimi_checkpoint,
)
from src import build_model_from_yaml


def test_training_checkpoint_format_restores_inference_model(tmp_path):
    torch.manual_seed(17)
    model = build_model_from_yaml(
        "config/kimi_full_pipeline/cpu_smoke/model.yaml"
    )
    checkpoint = tmp_path / "random_untrained.pt"
    torch.save(
        {
            "format_version": 2,
            "model_state_dict": model.state_dict(),
            "global_step": 12,
            "epoch": 3,
            "metadata": {"purpose": "inference-test"},
        },
        checkpoint,
    )
    tokenizer = load_tokenizer_from_data_yaml(
        "config/kimi_full_pipeline/cpu_smoke/data.yaml"
    )
    loaded = load_kimi_checkpoint(
        "config/kimi_full_pipeline/cpu_smoke/model.yaml",
        checkpoint,
        tokenizer=tokenizer,
        load_config=ModelLoadConfig(device="cpu", precision="fp32"),
    )
    assert loaded.format_version == 2
    assert loaded.global_step == 12
    assert loaded.epoch == 3
    assert not loaded.missing_keys and not loaded.unexpected_keys
    assert not loaded.model.training
    assert not any(
        parameter.requires_grad
        for parameter in loaded.model.parameters()
    )
    torch.testing.assert_close(
        loaded.model.embed_tokens.weight,
        model.embed_tokens.weight,
        rtol=0,
        atol=0,
    )
    output = inference_autoregressive(
        loaded.model,
        "key_1 is value_4",
        tokenizer=tokenizer,
        generation_config=GenerationConfig(
            max_new_tokens=1,
            do_sample=False,
            eos_token_id=127,
            pad_token_id=tokenizer.pad_id,
        ),
    )
    assert output.generated_ids.shape == (1, 1)
    assert output.cache.sequence_length == output.sequences.shape[1]
