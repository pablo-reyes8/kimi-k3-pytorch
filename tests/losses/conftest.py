import pytest
import torch

from src import KimiK3, kimi_k3_cpu_tiny_config


@pytest.fixture
def loss_tiny_model():
    torch.manual_seed(123)
    return KimiK3(kimi_k3_cpu_tiny_config()).cpu()

