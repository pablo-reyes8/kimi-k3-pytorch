import importlib.util

import pytest
import torch

from src.kda import recurrent_kda
from tests.kda.conftest import random_core


@pytest.mark.optional_dependency
@pytest.mark.skipif(
    importlib.util.find_spec("fla") is None,
    reason="optional fla-core oracle is unavailable",
)
def test_recurrent_core_matches_optional_fla_naive_oracle():
    from fla.ops.kda.naive import naive_recurrent_kda

    q, k, v, g, beta, state = random_core(
        batch=1, tokens=7, heads=2, key_dim=3, value_dim=4,
        dtype=torch.float32
    )
    ours = recurrent_kda(
        q, k, v, g, beta, state, output_final_state=True
    )
    # FLA defaults to an additional 1/sqrt(K) query scale; override with 1.
    expected_output, expected_state = naive_recurrent_kda(
        q, k, v, g, beta, scale=1.0, initial_state=state,
        output_final_state=True
    )
    torch.testing.assert_close(ours.hidden_states, expected_output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(ours.final_state, expected_state, rtol=1e-5, atol=1e-6)

