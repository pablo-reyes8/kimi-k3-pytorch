from dataclasses import replace

import pytest
import torch

from src import KimiK3, kimi_k3_cpu_tiny_config


@pytest.fixture
def tiny_kimi_config():
    return kimi_k3_cpu_tiny_config()


@pytest.fixture
def tiny_kimi_model(tiny_kimi_config):
    torch.manual_seed(1234)
    return KimiK3(tiny_kimi_config).cpu()


@pytest.fixture
def config_no_vision(tiny_kimi_config):
    return replace(tiny_kimi_config, enable_vision=False)


@pytest.fixture
def config_no_mtp(tiny_kimi_config):
    return replace(tiny_kimi_config, enable_mtp=False)


@pytest.fixture
def text_batch():
    ids = torch.tensor(
        [
            [5, 6, 7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18, 19, 20],
        ]
    )
    return ids, torch.ones_like(ids, dtype=torch.bool)


@pytest.fixture
def multimodal_batch(text_batch):
    ids, mask = text_batch
    ids = ids.clone()
    ids[:, 1:5] = 3
    pixels = torch.randn(2, 3, 16, 16)
    counts = torch.ones(2, dtype=torch.long)
    return ids, mask, pixels, counts
