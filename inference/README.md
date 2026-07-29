# Kimi K3 inference

This package turns a trained Kimi checkpoint into autoregressive text while
using the architecture's real heterogeneous cache:

```text
prompt -> KimiK3.prefill()
       -> HybridBackboneCache
       -> KimiK3.decode_step() once per generated token
```

KDA layers retain recurrent matrix and short-convolution states. Gated MLA
layers retain compressed latent KV states. Inference never substitutes a
generic GPT `past_key_values` tuple and never recomputes the full prefix.

## Master API

```python
from configuration import resolve_kimi_pipeline_profile
from data import load_tokenizer_from_data_yaml
from inference import (
    GenerationConfig,
    ModelLoadConfig,
    inference_autoregressive,
    load_kimi_checkpoint,
)

profile = resolve_kimi_pipeline_profile(
    "config/kimi_full_pipeline/low_gpu"
)
tokenizer = load_tokenizer_from_data_yaml(profile.data)
loaded = load_kimi_checkpoint(
    profile.model,
    "checkpoints/t4_15gb/kimi_k3_t4_15gb_epoch_0002.pt",
    tokenizer=tokenizer,
    load_config=ModelLoadConfig(device="cuda", precision="fp16"),
)
output = inference_autoregressive(
    loaded.model,
    "Once upon a time",
    tokenizer=tokenizer,
    generation_config=GenerationConfig(
        max_new_tokens=64,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
    ),
)
print(output.completion_text)
```

## Modules

- `config.py`: generation and checkpoint-load controls.
- `tokenization.py`: tokenizer-neutral prompt encode/decode.
- `sampling.py`: greedy, temperature, top-k, top-p and repetition penalty.
- `prefill.py`: prompt processing and native cache creation.
- `decode.py`: one-token cached decode.
- `generation.py`: autoregressive loop and master prompt API.
- `cache.py`: KDA/MLA cache validation and memory summaries.
- `loading.py`: model YAML plus checkpoint restoration.
- `audit.py`: full-forward versus cached-decode parity.
- `yaml_config.py`: optional inference profile loader.

MTP is not used as speculative decoding here. The current Kimi MTP head is a
full-sequence auxiliary training branch and explicitly rejects cached decode.
Adding verified speculative acceptance belongs to a separate phase.

Not included: paged attention, continuous batching, quantized caches,
distributed serving or custom serving kernels.
