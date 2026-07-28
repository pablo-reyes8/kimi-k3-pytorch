import copy

import pytest
import torch
import torch.nn.functional as F

from src.mtp import KimiMTPHead, combine_ntp_mtp_losses

from .conftest import tiny_mtp_config, tiny_mtp_head


def test_real_lm_head_and_tied_embedding_sharing():
    head = tiny_mtp_head(tied=True)
    assert head.lm_head.weight is head.input_embeddings.weight
    assert head.get_output_embeddings() is head.lm_head
    assert head.get_input_embeddings() is head.input_embeddings


def test_mtp_loss_reaches_fusion_shared_experts_embedding_and_lm_head():
    head = tiny_mtp_head(tied=False)
    ids = torch.randint(0, 23, (2, 7))
    hidden = torch.randn(2, 7, 8, requires_grad=True)
    head(hidden, ids, labels=ids).loss.backward()
    assert torch.isfinite(hidden.grad).all() and hidden.grad.abs().sum() > 0
    assert torch.isfinite(head.fusion.projection.weight.grad).all()
    assert head.fusion.projection.weight.grad.abs().sum() > 0
    assert torch.isfinite(head.input_embeddings.weight.grad).all()
    assert head.input_embeddings.weight.grad[ids[:, 1:-1]].abs().sum() > 0
    assert torch.isfinite(head.lm_head.weight.grad).all()
    for moe in head.block.moe_layers:
        for expert in moe.shared_experts:
            gradients = [p.grad for p in expert.parameters()]
            assert all(grad is not None and torch.isfinite(grad).all() for grad in gradients)


def test_detach_ablation_blocks_only_backbone_hidden_gradient():
    head = tiny_mtp_head(detach_backbone_hidden=True)
    hidden = torch.randn(1, 6, 8, requires_grad=True)
    ids = torch.randint(0, 23, (1, 6))
    head(hidden, ids, labels=ids).loss.backward()
    assert hidden.grad is None
    assert head.fusion.projection.weight.grad is not None
    assert head.input_embeddings.weight.grad is not None


def test_loss_weight_zero_preserves_mtp_gradient_when_loss_is_computed_separately():
    head = tiny_mtp_head(loss_weight=0.0)
    ids = torch.randint(0, 23, (1, 6))
    output = head(torch.randn(1, 6, 8), ids, labels=ids)
    total = combine_ntp_mtp_losses(torch.tensor(2.0, requires_grad=True), output.loss, 0.0)
    total.backward()
    assert head.fusion.projection.weight.grad is not None
    assert torch.count_nonzero(head.fusion.projection.weight.grad) == 0


def test_shared_lm_head_gradient_equals_sum_of_objective_contributions():
    head = tiny_mtp_head()
    ids = torch.randint(0, 23, (2, 6))
    hidden = torch.randn(2, 6, 8)
    main_features = torch.randn(2, 5, 8)
    main_targets = torch.randint(0, 23, (2, 5))
    mtp_loss = head(hidden, ids, labels=ids).loss
    ntp_loss = F.cross_entropy(
        head.lm_head(main_features).flatten(0, 1),
        main_targets.flatten(),
    )
    grad_ntp = torch.autograd.grad(ntp_loss, head.lm_head.weight, retain_graph=True)[0]
    grad_mtp = torch.autograd.grad(mtp_loss, head.lm_head.weight, retain_graph=True)[0]
    total = ntp_loss + 0.2 * mtp_loss
    grad_total = torch.autograd.grad(total, head.lm_head.weight)[0]
    torch.testing.assert_close(grad_total, grad_ntp + 0.2 * grad_mtp)


def test_state_dict_roundtrip_preserves_outputs_and_sharing():
    head = tiny_mtp_head(tied=True).eval()
    clone = tiny_mtp_head(tied=True, seed=999).eval()
    clone.load_state_dict(copy.deepcopy(head.state_dict()))
    assert clone.lm_head.weight is clone.input_embeddings.weight
    ids = torch.randint(0, 23, (2, 7))
    hidden = torch.randn(2, 7, 8)
    with torch.no_grad():
        expected = head(hidden, ids, return_diagnostics=True).logits
        actual = clone(hidden, ids, return_diagnostics=True).logits
    torch.testing.assert_close(actual, expected)
    assert any("routing_bias" in key for key in clone.state_dict())
    assert any(
        "final_output_attnres.pseudo_query" in key
        for key in clone.state_dict()
    )


def test_disabled_head_has_no_auxiliary_block_parameters():
    head = tiny_mtp_head(enabled=False)
    assert not any(name.startswith(("block.", "fusion.")) for name, _ in head.named_parameters())
    output = head(
        torch.randn(1, 4, 8),
        torch.randint(0, 23, (1, 4)),
    )
    assert output.logits is None and output.loss is None


def test_draft_one_step_cache_grows_and_is_separate():
    head = tiny_mtp_head().eval()
    with torch.no_grad():
        first = head.draft_one_step(
            torch.randn(2, 1, 8),
            torch.randint(0, 23, (2, 1)),
        )
        second = head.draft_one_step(
            torch.randn(2, 1, 8),
            torch.randint(0, 23, (2, 1)),
            cache=first.cache,
        )
    assert first.logits.shape == (2, 1, 23)
    assert first.cache.sequence_length == 1
    assert second.cache.sequence_length == 2
    assert len(second.cache.layer_caches) == 4


def test_teacher_forced_full_matches_incremental_draft_logits():
    head = tiny_mtp_head().eval()
    ids = torch.randint(0, 23, (2, 7))
    hidden = torch.randn(2, 7, 8)
    with torch.no_grad():
        full = head(hidden, ids).logits
        cache = None
        steps = []
        for position in range(ids.shape[1] - 2):
            output = head.draft_one_step(
                hidden[:, position : position + 1],
                ids[:, position + 1 : position + 2],
                cache=cache,
            )
            steps.append(output.logits)
            cache = output.cache
        incremental = torch.cat(steps, dim=1)
    torch.testing.assert_close(
        incremental, full, atol=1e-5, rtol=1e-5
    )


def test_draft_api_is_eval_only():
    head = tiny_mtp_head()
    with pytest.raises(RuntimeError, match="eval-only"):
        head.draft_one_step(
            torch.randn(1, 1, 8),
            torch.ones(1, 1, dtype=torch.long),
        )


def test_optimizer_step_changes_unique_mtp_parameter():
    head = tiny_mtp_head()
    optimizer = torch.optim.SGD(head.parameters(), lr=0.05)
    ids = torch.randint(0, 23, (2, 7))
    hidden = torch.randn(2, 7, 8)
    before = head.fusion.projection.weight.detach().clone()
    loss = head(hidden, ids, labels=ids).loss
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, head.fusion.projection.weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_fp16_forward_is_finite():
    head = tiny_mtp_head().cuda()
    ids = torch.randint(0, 23, (2, 7), device="cuda")
    hidden = torch.randn(2, 7, 8, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        output = head(hidden, ids, labels=ids)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.loss)


def test_cpu_bfloat16_autocast_is_finite():
    head = tiny_mtp_head()
    ids = torch.randint(0, 23, (2, 7))
    hidden = torch.randn(2, 7, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = head(hidden, ids, labels=ids)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.loss)
