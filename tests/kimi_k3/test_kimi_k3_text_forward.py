from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from src import KimiK3


def test_cpu_tiny_text_forward_is_finite_and_has_public_shapes(
    tiny_kimi_model,
    text_batch,
):
    ids, mask = text_batch
    tiny_kimi_model.eval()
    with torch.inference_mode():
        output = tiny_kimi_model(ids, mask)
    assert output.logits.shape == (2, 8, 128)
    assert output.last_hidden_state.shape == (2, 8, 16)
    assert output.cache is output.mtp_logits is None
    assert torch.isfinite(output.logits).all()
    assert not output.logits.requires_grad


def test_high_level_text_forward_matches_manual_composition(
    tiny_kimi_model,
    text_batch,
):
    ids, mask = text_batch
    tiny_kimi_model.eval()
    with torch.inference_mode():
        manual_hidden = tiny_kimi_model.backbone(
            tiny_kimi_model.embed_tokens(ids),
            attention_mask=mask,
        ).last_hidden_state
        manual_logits = tiny_kimi_model.lm_head(manual_hidden)
        actual = tiny_kimi_model(ids, mask)
    torch.testing.assert_close(actual.last_hidden_state, manual_hidden)
    torch.testing.assert_close(actual.logits, manual_logits)


def test_inputs_embeds_text_path_matches_input_ids_path(
    tiny_kimi_model,
    text_batch,
):
    ids, mask = text_batch
    tiny_kimi_model.eval()
    with torch.inference_mode():
        by_ids = tiny_kimi_model(ids, mask)
        by_embeds = tiny_kimi_model(
            inputs_embeds=tiny_kimi_model.embed_tokens(ids),
            attention_mask=mask,
        )
    torch.testing.assert_close(by_ids.logits, by_embeds.logits)


def test_mtp_is_explicit_and_never_changes_main_logits(
    tiny_kimi_model,
    text_batch,
):
    ids, mask = text_batch
    tiny_kimi_model.eval()
    with torch.inference_mode():
        plain = tiny_kimi_model(ids, mask)
        with_mtp = tiny_kimi_model(ids, mask, use_mtp=True)
    torch.testing.assert_close(plain.logits, with_mtp.logits, rtol=0, atol=0)
    assert plain.mtp_logits is None
    assert with_mtp.mtp_logits.shape == (2, 6, 128)


def test_mtp_matches_direct_head_call(tiny_kimi_model, text_batch):
    ids, mask = text_batch
    tiny_kimi_model.eval()
    with torch.inference_mode():
        integrated = tiny_kimi_model(ids, mask, use_mtp=True)
        direct = tiny_kimi_model.mtp(
            integrated.last_hidden_state,
            ids,
            attention_mask=mask,
        )
    torch.testing.assert_close(integrated.mtp_logits, direct.logits)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({}, "exactly one"),
        (
            {
                "input_ids": torch.ones(1, 3, dtype=torch.long),
                "inputs_embeds": torch.ones(1, 3, 16),
            },
            "exactly one",
        ),
        (
            {
                "input_ids": torch.ones(1, 3, dtype=torch.long),
                "position_ids": torch.arange(3).view(1, 3),
            },
            "NoPE",
        ),
        (
            {
                "input_ids": torch.ones(1, 3, dtype=torch.long),
                "cache_position": torch.arange(3),
            },
            "HybridBackboneCache",
        ),
    ],
)
def test_forward_rejects_ambiguous_or_unsupported_inputs(
    tiny_kimi_model, kwargs, match
):
    with pytest.raises(ValueError, match=match):
        tiny_kimi_model(**kwargs)


def test_wrong_shapes_and_mask_dtype_are_rejected(tiny_kimi_model):
    with pytest.raises(TypeError, match="boolean"):
        tiny_kimi_model(
            torch.ones(2, 4, dtype=torch.long),
            attention_mask=torch.ones(2, 4),
        )
    with pytest.raises(ValueError, match="inputs_embeds"):
        tiny_kimi_model(inputs_embeds=torch.randn(2, 4, 15))


def test_return_tuple_has_fixed_four_slot_order(tiny_kimi_model, text_batch):
    ids, mask = text_batch
    output = tiny_kimi_model(
        ids, mask, use_mtp=True, return_dict=False
    )
    assert len(output) == 4
    logits, cache, hidden_states, mtp_logits = output
    assert logits.shape == (2, 8, 128)
    assert cache is hidden_states is None
    assert mtp_logits.shape == (2, 6, 128)


def test_hidden_state_and_diagnostic_flags_are_opt_in(
    tiny_kimi_model,
    text_batch,
):
    ids, mask = text_batch
    with torch.inference_mode():
        plain = tiny_kimi_model(ids, mask)
        detailed = tiny_kimi_model(
            ids,
            mask,
            output_hidden_states=True,
            output_router_diagnostics=True,
            output_attnres_diagnostics=True,
        )
    assert plain.hidden_states is plain.backbone_diagnostics is None
    assert detailed.hidden_states is not None
    assert detailed.backbone_diagnostics is not None
    assert detailed.attnres_diagnostics is not None


def test_right_padding_works_and_left_padding_is_explicitly_rejected(
    tiny_kimi_model,
):
    ids = torch.tensor([[5, 6, 7, 8, 0, 0]])
    right = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    with torch.inference_mode():
        assert tiny_kimi_model(ids, right).logits.shape == (1, 6, 128)
    left = torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.bool)
    with pytest.raises(ValueError, match="left padding"):
        tiny_kimi_model(ids, left)


def test_all_padding_sequence_is_rejected(tiny_kimi_model):
    with pytest.raises(ValueError, match="all-padding"):
        tiny_kimi_model(
            torch.zeros(1, 4, dtype=torch.long),
            torch.zeros(1, 4, dtype=torch.bool),
        )


def test_embedding_and_output_setters_validate_and_retie_explicitly(
    tiny_kimi_model,
):
    replacement = nn.Embedding(128, 16)
    tiny_kimi_model.set_input_embeddings(replacement)
    assert tiny_kimi_model.get_input_embeddings() is replacement
    assert tiny_kimi_model.lm_head.weight is not replacement.weight
    tiny_kimi_model.tie_weights()
    assert tiny_kimi_model.lm_head.weight is replacement.weight
    assert tiny_kimi_model.mtp.input_embeddings is replacement
    with pytest.raises(ValueError):
        tiny_kimi_model.set_output_embeddings(nn.Linear(15, 128))


def test_untied_lm_head_has_independent_storage(tiny_kimi_config):
    model = KimiK3(replace(tiny_kimi_config, tie_word_embeddings=False))
    assert model.lm_head.weight.data_ptr() != model.embed_tokens.weight.data_ptr()
