import copy

import pytest
import torch

from src import KimiK3, KimiK3Config


def test_text_full_matches_prefill_plus_decode(
    tiny_kimi_model,
    text_batch,
):
    ids, mask = text_batch
    tiny_kimi_model.eval()
    with torch.inference_mode():
        full = tiny_kimi_model(ids, mask).logits
        prefill = tiny_kimi_model.prefill(ids[:, :3], mask[:, :3])
        pieces = [prefill.logits]
        cache = prefill.cache
        for position in range(3, ids.shape[1]):
            step = tiny_kimi_model.decode_step(
                ids[:, position : position + 1],
                cache,
                mask[:, position : position + 1],
            )
            pieces.append(step.logits)
            cache = step.cache
        incremental = torch.cat(pieces, dim=1)
    torch.testing.assert_close(
        incremental, full, atol=1e-6, rtol=1e-5
    )
    assert cache.sequence_length == ids.shape[1]


def test_multimodal_full_matches_visual_prefill_plus_decode(
    tiny_kimi_model,
    multimodal_batch,
):
    ids, mask, pixels, counts = multimodal_batch
    tiny_kimi_model.eval()
    with torch.inference_mode():
        full = tiny_kimi_model(
            ids, mask, pixel_values=pixels, image_counts=counts
        ).logits
        prefill = tiny_kimi_model.prefill(
            ids[:, :6],
            mask[:, :6],
            pixel_values=pixels,
            image_counts=counts,
        )
        pieces = [prefill.logits]
        cache = prefill.cache
        for position in range(6, ids.shape[1]):
            step = tiny_kimi_model.decode_step(
                ids[:, position : position + 1],
                cache,
                mask[:, position : position + 1],
            )
            pieces.append(step.logits)
            cache = step.cache
    torch.testing.assert_close(
        torch.cat(pieces, dim=1),
        full,
        atol=1e-6,
        rtol=1e-5,
    )


def test_prefill_helper_matches_direct_cached_forward(
    tiny_kimi_model,
    text_batch,
):
    ids, mask = text_batch
    tiny_kimi_model.eval()
    with torch.inference_mode():
        direct = tiny_kimi_model(ids, mask, use_cache=True)
        helper = tiny_kimi_model.prefill(ids, mask)
    torch.testing.assert_close(direct.logits, helper.logits)
    assert direct.cache.sequence_length == helper.cache.sequence_length == 8


def test_decode_requires_cache_and_one_token(tiny_kimi_model, text_batch):
    ids, mask = text_batch
    with pytest.raises(ValueError, match="non-empty cache"):
        tiny_kimi_model.decode_step(ids[:, :1], None, mask[:, :1])
    prefill = tiny_kimi_model.prefill(ids[:, :3], mask[:, :3])
    with pytest.raises(ValueError, match="exactly one"):
        tiny_kimi_model(
            ids[:, 3:5],
            mask[:, 3:5],
            cache=prefill.cache,
            use_cache=True,
        )


def test_mtp_and_visual_inputs_are_rejected_during_decode(
    tiny_kimi_model,
    text_batch,
):
    ids, mask = text_batch
    prefill = tiny_kimi_model.prefill(ids[:, :3], mask[:, :3])
    with pytest.raises(ValueError, match="full-sequence"):
        tiny_kimi_model.decode_step(
            ids[:, 3:4],
            prefill.cache,
            mask[:, 3:4],
            use_mtp=True,
        )
    with pytest.raises(ValueError, match="prefill"):
        tiny_kimi_model.decode_step(
            ids[:, 3:4],
            prefill.cache,
            mask[:, 3:4],
            pixel_values=torch.randn(2, 3, 16, 16),
        )


def test_state_dict_strict_roundtrip_preserves_logits_and_tying(
    tiny_kimi_model,
    text_batch,
):
    ids, mask = text_batch
    tiny_kimi_model.eval()
    clone = KimiK3(tiny_kimi_model.config).eval()
    clone.load_state_dict(copy.deepcopy(tiny_kimi_model.state_dict()), strict=True)
    assert clone.lm_head.weight is clone.embed_tokens.weight
    assert clone.mtp.lm_head is clone.lm_head
    with torch.inference_mode():
        expected = tiny_kimi_model(ids, mask, use_mtp=True)
        actual = clone(ids, mask, use_mtp=True)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.mtp_logits, expected.mtp_logits, rtol=0, atol=0)


def test_config_roundtrip_and_cache_absence_from_state_dict(tiny_kimi_model):
    assert KimiK3Config.from_dict(
        tiny_kimi_model.config.to_dict()
    ) == tiny_kimi_model.config
    assert not any("cache" in key for key in tiny_kimi_model.state_dict())


def test_eval_forward_is_deterministic(tiny_kimi_model, text_batch):
    ids, mask = text_batch
    tiny_kimi_model.eval()
    with torch.inference_mode():
        first = tiny_kimi_model(ids, mask, use_mtp=True)
        second = tiny_kimi_model(ids, mask, use_mtp=True)
    torch.testing.assert_close(first.logits, second.logits, rtol=0, atol=0)
    torch.testing.assert_close(first.mtp_logits, second.mtp_logits, rtol=0, atol=0)


def test_bfloat16_cpu_forward_preserves_public_dtypes(
    tiny_kimi_model,
    text_batch,
):
    ids, mask = text_batch
    model = tiny_kimi_model.to(torch.bfloat16).eval()
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(ids, mask)
    assert output.logits.dtype == torch.bfloat16
    assert output.last_hidden_state.dtype == torch.bfloat16
    assert torch.isfinite(output.logits).all()
