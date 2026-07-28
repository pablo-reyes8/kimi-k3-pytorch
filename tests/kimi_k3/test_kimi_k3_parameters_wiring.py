from dataclasses import replace

import torch

from src import KimiK3


def test_tied_parameter_occurs_once_in_optimizer_parameter_stream(
    tiny_kimi_model,
):
    parameter_ids = [id(parameter) for parameter in tiny_kimi_model.parameters()]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert tiny_kimi_model.lm_head.weight is tiny_kimi_model.embed_tokens.weight
    assert tiny_kimi_model.mtp.lm_head.weight is tiny_kimi_model.lm_head.weight


def test_parameter_report_matches_unique_parameter_identity(
    tiny_kimi_model,
):
    report = tiny_kimi_model.parameter_report()
    unique = {
        id(parameter): parameter
        for parameter in tiny_kimi_model.parameters()
    }
    assert report.total == sum(parameter.numel() for parameter in unique.values())
    assert report.total == tiny_kimi_model.num_parameters()
    assert report.lm_head_unique == 0
    assert report.mtp > 0 and report.vision > 0 and report.backbone > 0


def test_disabling_optional_paths_reduces_parameter_count(
    tiny_kimi_config,
):
    full = KimiK3(tiny_kimi_config)
    no_vision = KimiK3(replace(tiny_kimi_config, enable_vision=False))
    no_mtp = KimiK3(replace(tiny_kimi_config, enable_mtp=False))
    assert no_vision.num_parameters() < full.num_parameters()
    assert no_mtp.num_parameters() < full.num_parameters()


def test_forward_call_order_is_embedding_backbone_lm_then_mtp(
    tiny_kimi_model,
    text_batch,
):
    ids, mask = text_batch
    order = []
    hooks = [
        tiny_kimi_model.embed_tokens.register_forward_hook(
            lambda *args: order.append("embedding")
        ),
        tiny_kimi_model.backbone.register_forward_hook(
            lambda *args: order.append("backbone")
        ),
        tiny_kimi_model.lm_head.register_forward_hook(
            lambda *args: order.append("lm_head")
        ),
        tiny_kimi_model.mtp.register_forward_hook(
            lambda *args: order.append("mtp")
        ),
    ]
    tiny_kimi_model.eval()
    with torch.inference_mode():
        tiny_kimi_model(ids, mask, use_mtp=True)
    for hook in hooks:
        hook.remove()
    # MTP also calls the shared embedding and LM head. Its own call still
    # begins only after the main backbone and main LM projection.
    assert order[:3] == ["embedding", "backbone", "lm_head"]
    assert order[-1] == "mtp"


def test_main_lm_head_receives_exact_final_normalized_hidden(
    tiny_kimi_model,
    text_batch,
):
    ids, mask = text_batch
    received = []

    def capture(module, args):
        received.append(args[0])

    hook = tiny_kimi_model.lm_head.register_forward_pre_hook(capture)
    tiny_kimi_model.eval()
    with torch.inference_mode():
        output = tiny_kimi_model(ids, mask)
    hook.remove()
    assert received[0].data_ptr() == output.last_hidden_state.data_ptr()


def test_inference_output_has_empty_loss_contract(
    tiny_kimi_model,
    text_batch,
):
    ids, mask = text_batch
    with torch.inference_mode():
        output = tiny_kimi_model(ids, mask)
    assert output.loss is None
    assert output.ntp_loss is None
    assert output.mtp_loss is None
