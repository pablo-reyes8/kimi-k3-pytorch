import pytest
import torch

from src import VisualPlaceholderComposer


def composer_inputs():
    ids = torch.tensor([[5, 3, 3, 6], [3, 7, 8, 9]])
    embeds = torch.randn(2, 4, 4)
    mask = torch.ones(2, 4, dtype=torch.bool)
    projected = torch.tensor(
        [
            [[1.0] * 4, [2.0] * 4],
            [[3.0] * 4, [99.0] * 4],
        ]
    )
    projected_mask = torch.tensor([[True, True], [True, False]])
    counts = torch.tensor([1, 1])
    return ids, embeds, mask, projected, projected_mask, counts


def test_composer_replaces_placeholders_exactly_without_addition():
    ids, embeds, mask, projected, projected_mask, counts = composer_inputs()
    output, metadata = VisualPlaceholderComposer(4, 3, 4)(
        embeds,
        ids,
        mask,
        image_embeddings=projected,
        image_mask=projected_mask,
        image_counts=counts,
    )
    torch.testing.assert_close(output[0, 1:3], projected[0])
    torch.testing.assert_close(output[1, 0], projected[1, 0])
    torch.testing.assert_close(output[0, 0], embeds[0, 0])
    assert metadata.image_positions.tolist() == [[0, 1], [0, 2], [1, 0]]
    assert metadata.image_token_counts.tolist() == [2, 1]


def test_composer_supports_uneven_multiple_images_per_sample():
    ids = torch.tensor([[3, 3, 5, 6], [3, 7, 8, 9]])
    embeds = torch.zeros(2, 4, 2)
    projected = torch.tensor([[[1.0, 1]], [[2.0, 2]], [[3.0, 3]]])
    output, metadata = VisualPlaceholderComposer(2, 3, 4)(
        embeds,
        ids,
        torch.ones_like(ids, dtype=torch.bool),
        image_embeddings=projected,
        image_mask=torch.ones(3, 1, dtype=torch.bool),
        image_counts=torch.tensor([2, 1]),
    )
    assert output[:, :, 0].tolist() == [[1, 2, 0, 0], [3, 0, 0, 0]]
    assert metadata.image_token_counts.tolist() == [2, 1]


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("too_few", "placeholders"),
        ("masked_placeholder", "masked padding"),
        ("without_images", "require matching"),
    ],
)
def test_composer_rejects_placeholder_contract_violations(mutation, match):
    ids, embeds, mask, projected, projected_mask, counts = composer_inputs()
    if mutation == "too_few":
        projected_mask[0, 1] = False
    elif mutation == "masked_placeholder":
        mask[0, 1] = False
    elif mutation == "without_images":
        projected = projected_mask = counts = None
    with pytest.raises(ValueError, match=match):
        VisualPlaceholderComposer(4, 3, 4)(
            embeds,
            ids,
            mask,
            image_embeddings=projected,
            image_mask=projected_mask,
            image_counts=counts,
        )


def test_count_resolution_is_strict_and_per_sample():
    with pytest.raises(ValueError, match="sum"):
        VisualPlaceholderComposer.resolve_counts(
            "image_counts",
            torch.tensor([2, 1]),
            batch_size=2,
            item_count=2,
            device=torch.device("cpu"),
        )
    actual = VisualPlaceholderComposer.resolve_counts(
        "image_counts",
        None,
        batch_size=2,
        item_count=2,
        device=torch.device("cpu"),
    )
    assert actual.tolist() == [1, 1]
