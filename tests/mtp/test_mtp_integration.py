import torch
import torch.nn as nn

from src.kimi_block import KimiBlock
from src.mtp import KimiMTPHead, combine_ntp_mtp_losses, mtp_parameter_counts
from tests.stable_latent_moe.conftest import tiny_kimi_config

from .conftest import tiny_mtp_config, tiny_mtp_head


def test_kimi_block_to_mtp_end_to_end_backward_and_step():
    torch.manual_seed(811)
    embedding = nn.Embedding(23, 8)
    lm_head = nn.Linear(8, 23, bias=False)
    backbone = KimiBlock(tiny_kimi_config())
    mtp = KimiMTPHead(tiny_mtp_config(), embedding, lm_head)
    ids = torch.randint(0, 23, (2, 7))
    mask = torch.ones_like(ids, dtype=torch.bool)
    initial = embedding(ids)
    hidden = backbone(initial, attention_mask=mask).last_hidden_state
    main_logits = lm_head(hidden[:, :-1])
    ntp_loss = torch.nn.functional.cross_entropy(
        main_logits.flatten(0, 1), ids[:, 1:].flatten()
    )
    mtp_output = mtp(hidden, ids, mask, labels=ids)
    total = combine_ntp_mtp_losses(
        ntp_loss, mtp_output.loss, mtp.config.loss_weight
    )
    before = mtp.fusion.projection.weight.detach().clone()
    optimizer = torch.optim.SGD(
        list(backbone.parameters())
        + list(embedding.parameters())
        + list(lm_head.parameters())
        + list(mtp.fusion.parameters())
        + list(mtp.block.parameters()),
        lr=0.01,
    )
    total.backward()
    assert (
        backbone.layers[0].attention.projections.q_proj.weight.grad
        is not None
    )
    assert backbone.layers[-1].ffn.up_projection.weight.grad is not None
    optimizer.step()
    assert not torch.equal(before, mtp.fusion.projection.weight)


def test_mtp_block_matches_group_structure_without_parameter_storage_sharing():
    backbone = KimiBlock(tiny_kimi_config())
    mtp = tiny_mtp_head().block
    assert mtp.attention_types == backbone.attention_types[:4]
    main_group_parameters = {
        parameter.data_ptr()
        for layer in backbone.layers[:4]
        for parameter in layer.parameters()
    }
    mtp_parameters = {
        parameter.data_ptr()
        for parameter in mtp.parameters()
    }
    assert main_group_parameters.isdisjoint(mtp_parameters)


def test_parameter_counts_do_not_duplicate_tied_embedding_lm_head():
    head = tiny_mtp_head(tied=True)
    counts = mtp_parameter_counts(head)
    shared = head.input_embeddings.weight.numel()
    assert counts["fusion"] == 2 * 8 + 2 * 8 * 8
    assert counts["unique_mtp_output"] == 0
    assert counts["shared_embedding_lm_overlap"] == shared
    assert counts["all_referenced_unique"] == (
        counts["unique_mtp_total"] + shared
    )


def test_short_sequence_head_returns_zero_loss_without_nan():
    head = tiny_mtp_head()
    for tokens in (0, 1, 2):
        ids = torch.empty(2, tokens, dtype=torch.long)
        hidden = torch.empty(2, tokens, 8)
        output = head(hidden, ids, labels=ids)
        assert output.logits.shape == (2, 0, 23)
        assert output.loss.item() == 0.0
        assert torch.isfinite(output.loss)


def test_enabling_mtp_does_not_change_main_path_logits():
    torch.manual_seed(919)
    backbone = KimiBlock(tiny_kimi_config()).eval()
    embedding = nn.Embedding(23, 8)
    lm_head = nn.Linear(8, 23, bias=False)
    ids = torch.randint(0, 23, (2, 6))
    with torch.no_grad():
        hidden_before = backbone(embedding(ids)).last_hidden_state
        logits_before = lm_head(hidden_before)
        mtp = KimiMTPHead(tiny_mtp_config(), embedding, lm_head).eval()
        hidden_after = backbone(embedding(ids)).last_hidden_state
        logits_after = lm_head(hidden_after)
        _ = mtp(hidden_after, ids)
    torch.testing.assert_close(logits_after, logits_before, rtol=0, atol=0)
