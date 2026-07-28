import io

import torch

from src.mla import GatedMLA, MLACache
from tests.mla.conftest import tiny_config, tiny_mla


def test_state_dict_roundtrip_produces_exact_eval_output():
    source = tiny_mla().eval()
    target = GatedMLA(tiny_config()).eval()
    target.load_state_dict(source.state_dict())
    x = torch.randn(2, 7, 12)
    torch.testing.assert_close(
        source(x).hidden_states, target(x).hidden_states, rtol=0, atol=0
    )


def test_eval_is_deterministic_and_disables_both_dropouts():
    model = tiny_mla(attention_dropout=0.8, output_dropout=0.8).eval()
    x = torch.randn(2, 7, 12)
    first = model(x).hidden_states
    second = model(x).hidden_states
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_training_dropout_is_active():
    model = tiny_mla(attention_dropout=0.5, output_dropout=0.5).train()
    x = torch.randn(2, 16, 12)
    assert not torch.equal(model(x).hidden_states, model(x).hidden_states)


def test_cache_roundtrip_preserves_every_tensor():
    cache = tiny_mla().eval()(torch.randn(2, 5, 12), use_cache=True).cache
    buffer = io.BytesIO()
    torch.save(cache, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)
    assert isinstance(loaded, MLACache)
    torch.testing.assert_close(cache.latent_kv, loaded.latent_kv)
    torch.testing.assert_close(cache.attention_mask, loaded.attention_mask)
    torch.testing.assert_close(cache.sequence_offset, loaded.sequence_offset)


def test_cache_is_runtime_state_not_model_state_dict():
    model = tiny_mla().eval()
    _ = model(torch.randn(2, 5, 12), use_cache=True)
    assert not any("cache" in name for name in model.state_dict())


def test_to_dtype_moves_every_floating_parameter_and_buffer():
    model = tiny_mla().to(dtype=torch.float64)
    assert all(parameter.dtype == torch.float64 for parameter in model.parameters())
    assert all(
        not buffer.dtype.is_floating_point or buffer.dtype == torch.float64
        for buffer in model.buffers()
    )


def test_edge_dimensions_b1_h1_q1_v1_l1():
    model = GatedMLA(
        tiny_config(
            d_model=1,
            num_heads=1,
            q_head_dim=1,
            v_head_dim=1,
            kv_latent_dim=1,
        )
    )
    output = model(torch.randn(1, 1, 1), use_cache=True)
    assert output.hidden_states.shape == (1, 1, 1)
    assert output.cache.latent_kv.shape == (1, 1, 1)


def test_invalid_hidden_shapes_and_empty_sequence_are_rejected():
    model = tiny_mla()
    for shape in ((2, 3, 11), (2, 0, 12), (2, 12)):
        try:
            model(torch.randn(shape))
        except ValueError:
            pass
        else:
            raise AssertionError(f"shape {shape} should be rejected")
