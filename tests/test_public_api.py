import data
import src
import training
from src.kimi_block import KimiBlock
from src.kimi_k3_mini import BaselineCausalLM, BaselineCausalLMConfig
from src.mtp import KimiMTPHead
from src.outputs import CausalLMOutput


def test_public_exports_are_present_and_kimi_architecture_is_not_faked():
    assert src.BaselineCausalLM is BaselineCausalLM
    assert src.BaselineCausalLMConfig is BaselineCausalLMConfig
    assert src.CausalLMOutput is CausalLMOutput
    assert src.KimiBlock is KimiBlock
    assert src.KimiMTPHead is KimiMTPHead
    assert not hasattr(src, "KimiK3Mini")


def test_data_and_training_public_exports_resolve():
    for name in data.__all__:
        assert hasattr(data, name)
    for name in training.__all__:
        assert hasattr(training, name)
