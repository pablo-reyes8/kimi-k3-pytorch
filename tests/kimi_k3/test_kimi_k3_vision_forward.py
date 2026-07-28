import pytest
import torch

from src import KimiK3


def test_cpu_tiny_multimodal_forward_shapes_and_metadata(
    tiny_kimi_model,
    multimodal_batch,
):
    ids, mask, pixels, counts = multimodal_batch
    tiny_kimi_model.eval()
    with torch.inference_mode():
        output = tiny_kimi_model(
            ids,
            mask,
            pixel_values=pixels,
            image_counts=counts,
        )
    assert output.logits.shape == (2, 8, 128)
    assert output.vision_outputs.images.last_hidden_state.shape == (2, 16, 16)
    assert output.multimodal_metadata.image_positions.shape == (8, 2)
    assert output.multimodal_metadata.image_token_counts.tolist() == [4, 4]


def test_high_level_multimodal_forward_matches_manual_composition(
    tiny_kimi_model,
    multimodal_batch,
):
    ids, mask, pixels, counts = multimodal_batch
    tiny_kimi_model.eval()
    with torch.inference_mode():
        encoded = tiny_kimi_model.vision_encoder(pixels)
        packed = tiny_kimi_model.vision_token_packer(
            encoded.last_hidden_state, encoded.grid_size
        )
        projected = tiny_kimi_model.vision_projector(
            packed.last_hidden_state
        )
        composed, _ = tiny_kimi_model.multimodal_composer(
            tiny_kimi_model.embed_tokens(ids),
            ids,
            mask,
            image_embeddings=projected,
            image_mask=torch.ones(
                projected.shape[:2], dtype=torch.bool
            ),
            image_counts=counts,
        )
        hidden = tiny_kimi_model.backbone(
            composed,
            attention_mask=mask,
        ).last_hidden_state
        expected = tiny_kimi_model.lm_head(hidden)
        actual = tiny_kimi_model(
            ids,
            mask,
            pixel_values=pixels,
            image_counts=counts,
        ).logits
    torch.testing.assert_close(actual, expected)


def test_multiple_images_are_assigned_only_to_their_declared_sample(
    tiny_kimi_model,
):
    ids = torch.tensor(
        [
            [3, 3, 3, 3, 3, 3, 3, 3, 5, 6],
            [3, 3, 3, 3, 7, 8, 9, 10, 11, 12],
        ]
    )
    pixels = torch.randn(3, 3, 16, 16)
    output = tiny_kimi_model(
        ids,
        torch.ones_like(ids, dtype=torch.bool),
        pixel_values=pixels,
        image_counts=torch.tensor([2, 1]),
    )
    assert output.multimodal_metadata.image_token_counts.tolist() == [8, 4]
    positions = output.multimodal_metadata.image_positions
    assert (positions[:, 0] == 0).sum() == 8
    assert (positions[:, 0] == 1).sum() == 4


def test_video_frames_are_flattened_into_one_visual_sequence(
    tiny_kimi_model,
):
    ids = torch.tensor([[4, 4, 4, 4, 4, 4, 4, 4, 7, 8]])
    videos = torch.randn(1, 2, 3, 16, 16)
    output = tiny_kimi_model(
        ids,
        torch.ones_like(ids, dtype=torch.bool),
        video_values=videos,
        video_counts=torch.ones(1, dtype=torch.long),
    )
    assert output.multimodal_metadata.video_token_counts.tolist() == [8]
    assert output.vision_outputs.videos.last_hidden_state.shape[0] == 2


def test_vision_encoder_is_called_once_and_not_during_decode(
    tiny_kimi_model,
    multimodal_batch,
):
    ids, mask, pixels, counts = multimodal_batch
    calls = 0
    original = tiny_kimi_model.vision_encoder.forward

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    tiny_kimi_model.vision_encoder.forward = counted
    prefill = tiny_kimi_model.prefill(
        ids[:, :6],
        mask[:, :6],
        pixel_values=pixels,
        image_counts=counts,
    )
    tiny_kimi_model.decode_step(
        ids[:, 6:7], prefill.cache, mask[:, 6:7]
    )
    assert calls == 1


@pytest.mark.parametrize(
    "kind,match",
    [
        ("count_mismatch", "sum to"),
        ("placeholder_mismatch", "placeholders"),
        ("embeds_with_vision", "inputs_embeds"),
    ],
)
def test_visual_contract_mismatches_raise(
    tiny_kimi_model,
    multimodal_batch,
    kind,
    match,
):
    ids, mask, pixels, counts = multimodal_batch
    kwargs = dict(
        input_ids=ids,
        attention_mask=mask,
        pixel_values=pixels,
        image_counts=counts,
    )
    if kind == "count_mismatch":
        kwargs["image_counts"] = torch.tensor([2, 1])
    elif kind == "placeholder_mismatch":
        kwargs["input_ids"] = ids.clone()
        kwargs["input_ids"][0, 1] = 5
    else:
        kwargs.pop("input_ids")
        kwargs["inputs_embeds"] = tiny_kimi_model.embed_tokens(ids)
    with pytest.raises(ValueError, match=match):
        tiny_kimi_model(**kwargs)


def test_pixel_values_with_disabled_vision_raise(config_no_vision):
    model = KimiK3(config_no_vision)
    with pytest.raises(ValueError, match="vision is disabled"):
        model(
            torch.ones(1, 4, dtype=torch.long),
            pixel_values=torch.randn(1, 3, 16, 16),
        )


def test_multimodal_mtp_branch_is_auxiliary_only(
    tiny_kimi_model,
    multimodal_batch,
):
    ids, mask, pixels, counts = multimodal_batch
    tiny_kimi_model.eval()
    with torch.inference_mode():
        plain = tiny_kimi_model(
            ids, mask, pixel_values=pixels, image_counts=counts
        )
        auxiliary = tiny_kimi_model(
            ids,
            mask,
            pixel_values=pixels,
            image_counts=counts,
            use_mtp=True,
        )
    torch.testing.assert_close(plain.logits, auxiliary.logits, rtol=0, atol=0)
    assert auxiliary.mtp_logits.shape == (2, 6, 128)


def test_multimodal_future_text_cannot_change_prefix_logits(
    tiny_kimi_model,
    multimodal_batch,
):
    ids, mask, pixels, counts = multimodal_batch
    changed = ids.clone()
    changed[:, 6:] = torch.tensor([[30, 31], [32, 33]])
    tiny_kimi_model.eval()
    with torch.inference_mode():
        first = tiny_kimi_model(
            ids, mask, pixel_values=pixels, image_counts=counts
        ).logits
        second = tiny_kimi_model(
            changed, mask, pixel_values=pixels, image_counts=counts
        ).logits
    torch.testing.assert_close(first[:, :6], second[:, :6], atol=1e-6, rtol=1e-5)
