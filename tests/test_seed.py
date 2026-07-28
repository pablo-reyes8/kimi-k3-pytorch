import random

import numpy as np
import pytest
import torch

from training.seed import set_seed


def sample():
    return random.random(), np.random.rand(3), torch.rand(3)


def test_seed_reproduces_python_numpy_and_torch():
    set_seed(123)
    first = sample()
    set_seed(123)
    second = sample()
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    torch.testing.assert_close(first[2], second[2], atol=0, rtol=0)


def test_different_seeds_change_streams():
    set_seed(1)
    first = sample()
    set_seed(2)
    second = sample()
    assert first[0] != second[0]
    assert not np.array_equal(first[1], second[1])
    assert not torch.equal(first[2], second[2])


def test_non_integer_seed_rejected():
    with pytest.raises(TypeError):
        set_seed(1.5)


def test_deterministic_flag_controls_cudnn_settings():
    original = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        set_seed(1, deterministic=True)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        set_seed(1, deterministic=False)
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = original
