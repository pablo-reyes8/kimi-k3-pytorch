import torch

from conftest import tiny_model


def test_causal_lm_contract_causality_and_forward_embeddings():
    model = tiny_model().eval()
    input_ids = torch.randint(1, 48, (2, 12))
    labels = torch.randint(1, 48, (2, 12))
    output = model(input_ids, labels=labels)
    assert output.logits.shape == (2, 12, 48)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.auxiliary_losses == {} and output.metrics == {}

    changed = input_ids.clone()
    changed[:, 8:] = torch.randint(1, 48, (2, 4))
    with torch.no_grad():
        first = model(input_ids).logits
        second = model(changed).logits
        embedded = model.embedding(input_ids)
        from_embeddings = model.forward_embeddings(embedded).logits
    torch.testing.assert_close(first[:, :8], second[:, :8], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(first, from_embeddings)
