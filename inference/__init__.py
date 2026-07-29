"""Native-cache inference utilities for Kimi K3."""

from .audit import compare_full_vs_cached_logits
from .cache import (
    cache_memory_bytes,
    cache_summary,
    validate_kimi_cache,
)
from .config import GenerationConfig, ModelLoadConfig
from .decode import decode_one_token
from .generation import (
    generate,
    generate_tokens,
    inference_autoregressive,
    inference_autoregresive,
)
from .loading import LoadedKimiCheckpoint, load_kimi_checkpoint
from .outputs import GenerationOutput
from .prefill import prefill_prompt
from .sampling import (
    apply_repetition_penalty,
    sample_next_token,
    top_k_filter,
    top_p_filter,
)
from .tokenization import (
    decode_token_ids,
    encode_prompt,
    tokenizer_token_id,
)
from .yaml_config import load_generation_config

__all__ = [
    "GenerationConfig",
    "GenerationOutput",
    "LoadedKimiCheckpoint",
    "ModelLoadConfig",
    "apply_repetition_penalty",
    "cache_memory_bytes",
    "cache_summary",
    "compare_full_vs_cached_logits",
    "decode_one_token",
    "decode_token_ids",
    "encode_prompt",
    "generate",
    "generate_tokens",
    "inference_autoregressive",
    "inference_autoregresive",
    "load_generation_config",
    "load_kimi_checkpoint",
    "prefill_prompt",
    "sample_next_token",
    "tokenizer_token_id",
    "top_k_filter",
    "top_p_filter",
    "validate_kimi_cache",
]
