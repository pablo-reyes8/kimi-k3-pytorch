import itertools

import pytest
import torch

from src.attention_residuals import (
    BlockAttentionResidualController,
    BlockAttentionResidualState,
    FullAttentionResidualController,
    FullAttentionResidualState,
)


def tensors(count, shape=(1, 2, 3), dtype=torch.float32):
    return [
        torch.full(shape, float(index + 1), dtype=dtype)
        for index in range(count)
    ]


def test_full_state_source_registry_embedding_index_and_order():
    embedding, *outputs = tensors(5)
    controller = FullAttentionResidualController()
    state = controller.initialize(embedding)
    assert state.sources[0] is embedding
    for index, output in enumerate(outputs, 1):
        controller.append_output(state, output)
        assert state.num_sublayer_outputs == index
        assert state.available_sources().shape[2] == 1 + index
        torch.testing.assert_close(
            state.available_sources()[:, :, index], output
        )


def test_full_state_functional_append_and_clone_do_not_alias():
    embedding, output = tensors(2)
    state = FullAttentionResidualState([embedding])
    appended = state.with_appended(output)
    assert state.num_sublayer_outputs == 0
    assert appended.num_sublayer_outputs == 1
    cloned = appended.clone(detach=True)
    cloned.sources[0].add_(100)
    assert not torch.equal(cloned.sources[0], appended.sources[0])
    assert all(not source.requires_grad for source in cloned.sources)


def test_full_state_rejects_broken_invariants():
    with pytest.raises(ValueError):
        FullAttentionResidualState([])
    with pytest.raises(ValueError):
        FullAttentionResidualState(tensors(2), num_sublayer_outputs=0)


def test_block_partial_sum_never_contains_embedding():
    embedding = torch.full((1, 2, 3), 100.0)
    outputs = tensors(3)
    state = BlockAttentionResidualState(embedding, 4)
    for index, output in enumerate(outputs, 1):
        state.append_output(output)
        expected = sum(outputs[:index])
        torch.testing.assert_close(state.partial_block, expected)
        assert not torch.equal(state.partial_block, embedding + expected)
        assert state.outputs_in_current_block == index


def test_first_site_has_no_partial_later_sites_have_exactly_one():
    embedding, first, second = tensors(3)
    state = BlockAttentionResidualState(embedding, 4)
    assert state.available_sources().shape[2] == 1
    state.append_output(first)
    sources = state.available_sources()
    assert sources.shape[2] == 2
    torch.testing.assert_close(sources[:, :, -1], first)
    state.append_output(second)
    sources = state.available_sources()
    assert sources.shape[2] == 2
    torch.testing.assert_close(sources[:, :, -1], first + second)


@pytest.mark.parametrize("block_size", [1, 2, 3, 4])
@pytest.mark.parametrize("outputs", range(1, 11))
def test_block_boundaries_match_exact_integer_partition(block_size, outputs):
    state = BlockAttentionResidualState(torch.zeros(1, 1, 2), block_size)
    for value in tensors(outputs, shape=(1, 1, 2)):
        state.prepare_for_site()
        state.append_output(value)
    state.finalize()
    expected = [block_size] * (outputs // block_size)
    if outputs % block_size:
        expected.append(outputs % block_size)
    assert state.block_sizes == expected
    assert len(state.completed_blocks) == len(expected)
    assert state.partial_block is None
    assert state.outputs_in_current_block == 0


def test_exact_division_does_not_create_empty_block():
    state = BlockAttentionResidualState(torch.zeros(1, 1, 2), 4)
    for output in tensors(8, shape=(1, 1, 2)):
        state.prepare_for_site()
        state.append_output(output)
    state.finalize()
    assert state.block_sizes == [4, 4]
    assert len(state.completed_blocks) == 2


def test_partial_final_block_sizes_are_4_4_2():
    state = BlockAttentionResidualState(torch.zeros(1, 1, 2), 4)
    for output in tensors(10, shape=(1, 1, 2)):
        state.prepare_for_site()
        state.append_output(output)
    state.finalize()
    assert state.block_sizes == [4, 4, 2]
    assert state.current_depth_block_index == 3


def test_completed_blocks_are_not_mutated_by_future_outputs():
    state = BlockAttentionResidualState(torch.zeros(1, 1, 2), 2)
    first_outputs = tensors(2, shape=(1, 1, 2))
    for output in first_outputs:
        state.prepare_for_site()
        state.append_output(output)
    state.prepare_for_site()
    snapshot = state.completed_blocks[0].clone()
    state.append_output(torch.full((1, 1, 2), 50.0))
    torch.testing.assert_close(state.completed_blocks[0], snapshot, rtol=0, atol=0)


def test_final_sources_are_embedding_plus_completed_blocks_only():
    embedding = torch.full((1, 1, 2), 100.0)
    state = BlockAttentionResidualState(embedding, 3)
    for output in tensors(5, shape=(1, 1, 2)):
        state.prepare_for_site()
        state.append_output(output)
    state.finalize()
    sources = state.available_sources()
    assert sources.shape[2] == 3
    torch.testing.assert_close(sources[:, :, 0], embedding)
    torch.testing.assert_close(sources[:, :, 1], sum(tensors(3, shape=(1, 1, 2))))
    torch.testing.assert_close(sources[:, :, 2], sum(tensors(5, shape=(1, 1, 2))[3:]))


def test_block_clone_has_no_aliasing():
    state = BlockAttentionResidualState(torch.randn(1, 2, 3), 2)
    for output in tensors(3):
        state.prepare_for_site()
        state.append_output(output)
    cloned = state.clone(detach=True)
    cloned.embedding.add_(100)
    cloned.completed_blocks[0].add_(100)
    cloned.partial_block.add_(100)
    assert not torch.equal(cloned.embedding, state.embedding)
    assert not torch.equal(cloned.completed_blocks[0], state.completed_blocks[0])
    assert not torch.equal(cloned.partial_block, state.partial_block)


def test_k3_block_partition_and_final_source_count_without_large_model():
    state = BlockAttentionResidualState(torch.zeros(1, 1, 1), 24)
    one = torch.ones(1, 1, 1)
    for _ in range(186):
        state.prepare_for_site()
        state.append_output(one)
    state.finalize()
    assert state.block_sizes == [24] * 7 + [18]
    assert state.available_sources().shape[2] == 9
