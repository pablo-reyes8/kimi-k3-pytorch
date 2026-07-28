import torch

from src.mla import MLAProjections
from tests.mla.conftest import tiny_config


def test_projection_shapes_with_q_not_equal_v():
    projections = MLAProjections(tiny_config()).double()
    x = torch.randn(2, 7, 12, dtype=torch.float64)
    output = projections(x)
    assert output.query.shape == (2, 7, 3, 2)
    assert output.latent_kv.shape == (2, 7, 5)
    assert output.key.shape == (2, 7, 3, 2)
    assert output.value.shape == (2, 7, 3, 4)


def test_each_projection_matches_its_linear_equation_exactly():
    config = tiny_config(projection_bias=True)
    projections = MLAProjections(config).double()
    x = torch.randn(2, 4, 12, dtype=torch.float64)
    output = projections(x)
    expected_q = projections.query(x).reshape(2, 4, 3, 2)
    expected_c = projections.latent_kv.compression(x)
    expected_k = projections.latent_kv.key_up(expected_c).reshape(2, 4, 3, 2)
    expected_v = projections.latent_kv.value_up(expected_c).reshape(2, 4, 3, 4)
    torch.testing.assert_close(output.query, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(output.latent_kv, expected_c, rtol=0, atol=0)
    torch.testing.assert_close(output.key, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(output.value, expected_v, rtol=0, atol=0)


def test_key_and_value_reconstruct_from_the_same_latent():
    projections = MLAProjections(tiny_config()).double()
    x = torch.randn(1, 5, 12, dtype=torch.float64)
    latent = projections.compress_kv(x)
    key, value = projections.reconstruct_kv(latent)
    output = projections(x)
    torch.testing.assert_close(key, output.key, rtol=0, atol=0)
    torch.testing.assert_close(value, output.value, rtol=0, atol=0)


def test_query_compression_key_and_value_parameters_are_independent():
    projections = MLAProjections(tiny_config())
    pointers = [
        projections.query.weight.data_ptr(),
        projections.latent_kv.compression.weight.data_ptr(),
        projections.latent_kv.key_up.weight.data_ptr(),
        projections.latent_kv.value_up.weight.data_ptr(),
    ]
    assert len(set(pointers)) == len(pointers)


def test_identifiable_layout_has_no_silent_head_permutation():
    config = tiny_config(
        d_model=6, num_heads=2, q_head_dim=2, v_head_dim=3, kv_latent_dim=4
    )
    projections = MLAProjections(config)
    latent = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    with torch.no_grad():
        projections.latent_kv.key_up.weight.zero_()
        projections.latent_kv.key_up.weight[:, 0] = torch.arange(1, 5)
    key, _ = projections.reconstruct_kv(latent)
    assert key.flatten().tolist() == [1.0, 2.0, 3.0, 4.0]


def test_noncontiguous_hidden_states_are_supported():
    projections = MLAProjections(tiny_config())
    x = torch.randn(2, 5, 24)[..., ::2]
    assert not x.is_contiguous()
    assert projections(x).value.shape == (2, 5, 3, 4)
