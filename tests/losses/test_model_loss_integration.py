import pytest
import torch


def test_model_forward_labels_delegate_to_next_token_loss(loss_tiny_model):
    ids = torch.tensor([[5, 6, 7, 8, 9, 10]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    output = loss_tiny_model(ids, mask, labels=ids)
    direct = loss_tiny_model.pretraining_loss(
        logits=output.logits,
        labels=ids,
        attention_mask=mask,
    )
    torch.testing.assert_close(output.loss, direct.loss)
    torch.testing.assert_close(output.ntp_loss.loss, direct.ntp.loss)
    assert output.mtp_loss is None


def test_model_mtp_loss_uses_phase9_training_view_once(loss_tiny_model):
    ids = torch.tensor([[5, 6, 7, 8, 9, 10]])
    mask = torch.ones_like(ids, dtype=torch.bool)

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy internal MTP loss was called")

    loss_tiny_model.mtp.loss_fn.forward = forbidden
    # Kimi requests target alignment from the head with compute_loss=False,
    # then delegates exactly once to the composite phase-11 loss.
    output = loss_tiny_model(ids, mask, labels=ids, use_mtp=True)
    assert output.mtp_loss is not None
    assert output.mtp_loss.future_offsets == (2,)
    torch.testing.assert_close(
        output.loss,
        output.ntp_loss.loss
        + loss_tiny_model.config.mtp.loss_weight * output.mtp_loss.loss,
    )


def test_multimodal_ntp_backpropagates_to_vision_and_text_paths(loss_tiny_model):
    ids = torch.tensor([[1, 3, 3, 3, 3, 20, 21, 2]])
    labels = ids.clone()
    labels[:, 1:5] = -100
    mask = torch.ones_like(ids, dtype=torch.bool)
    pixels = torch.randn(1, 3, 16, 16)
    output = loss_tiny_model(
        ids,
        mask,
        pixel_values=pixels,
        image_counts=torch.ones(1, dtype=torch.long),
        labels=labels,
    )
    output.loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in loss_tiny_model.vision_encoder.parameters()
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in loss_tiny_model.vision_projector.parameters()
    )
    assert loss_tiny_model.embed_tokens.weight.grad is not None
    assert loss_tiny_model.lm_head.weight.grad is not None


def test_inference_and_decode_never_compute_training_losses(loss_tiny_model):
    ids = torch.tensor([[5, 6, 7, 8]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    inference = loss_tiny_model(ids, mask)
    assert inference.loss is inference.ntp_loss is inference.mtp_loss is None
    prefill = loss_tiny_model.prefill(ids, mask)
    decoded = loss_tiny_model.decode_step(
        torch.tensor([[9]]),
        prefill.cache,
        torch.ones(1, 1, dtype=torch.bool),
    )
    assert decoded.loss is None
    with pytest.raises(ValueError, match="cannot run"):
        loss_tiny_model(
            torch.tensor([[9]]),
            torch.ones(1, 1, dtype=torch.bool),
            cache=prefill.cache,
            use_cache=True,
            labels=torch.tensor([[9]]),
        )


def test_model_rejects_posttraining_loss_inside_forward(loss_tiny_model):
    ids = torch.tensor([[5, 6, 7]])
    with pytest.raises(ValueError, match="explicitly"):
        loss_tiny_model(ids, labels=ids, training_phase="sft")

