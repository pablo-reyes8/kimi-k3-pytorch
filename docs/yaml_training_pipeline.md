# YAML-controlled training pipeline

## Public API

```python
from data import build_dataloaders_from_yaml
from src import build_model_from_yaml
from training import train_kimi_from_yaml

data = build_dataloaders_from_yaml("config/data/tinystories.yaml")
model = build_model_from_yaml(
    "config/kimi_k3/t4_15gb.yaml",
    data_bundle=data,
)
result = train_kimi_from_yaml(
    "config/training/t4_15gb.yaml",
    model=model,
    data=data,
)
```

The third function validates cross-file invariants and delegates to
`train_kimiK3`. It does not implement a second training loop.

## Data contract

`build_dataloaders_from_yaml` returns `DataBundle`:

- `train_loader`;
- `val_loader`;
- `tokenizer`;
- validated `DataPipelineConfig`;
- `train_loader_factory(max_seq_len)` for PCC transitions.

Dataset configuration and loader configuration are separate. Batch size and
workers belong only to `data.loader`.

Currently supported executable dataset families are:

- deterministic synthetic long-context retrieval;
- registered Hugging Face causal-text presets.

Hugging Face profiles may download data and train/cache a tokenizer when first
built. Config-only validation never downloads anything.

## Model contract

The model YAML controls:

- vocabulary and special IDs;
- width, heads, hybrid groups and attention pattern;
- every KDA, MLA and Stable LatentMoE dataclass option;
- Attention Residual mode/backend;
- optional MoonViT/hierarchical/Swin vision path and projector;
- optional MTP and its AttnRes block;
- top-level tying, dropout and diagnostic output flags.

Passing `data_bundle` resolves tokenizer special IDs when
`use_data_special_tokens: true` and verifies that the dataset vocabulary fits
inside the model vocabulary.

`load_model_config` performs all structural validation without allocating
weights. This is safe even for the canonical metadata profile.

## Training contract

The training YAML controls:

- device and FP32/BF16/FP16;
- epochs, accumulation, clipping and evaluation cadence;
- NTP ignore index, label smoothing and MTP loss weight;
- AdamW, Muon or Per-Head Muon and all optimizer hyperparameters;
- cosine warmup/decay;
- diagnostic budgets and console blocks;
- Progressive Context Curriculum;
- EMA;
- checkpoints/resume;
- next-token previews.

If PCC is enabled, the adapter passes `train_loader_factory` to the master
trainer. Otherwise it passes the already-built loader. Model/data/training
lengths and MTP capabilities are checked before optimizer allocation.

## CLI

Configuration-only validation:

```bash
python -m scripts.train_kimi \
  --data-config config/data/tinystories.yaml \
  --model-config config/kimi_k3/t4_15gb.yaml \
  --training-config config/training/t4_15gb.yaml \
  --validate-only
```

Remove `--validate-only` to build data/model and start the master trainer.

Individual validation:

```bash
python -m scripts.validate_data_config config/data/tinystories.yaml
python -m scripts.validate_model_config config/kimi_k3/t4_15gb.yaml
```

`validate_data_config --build` and `validate_model_config --instantiate` are
explicit opt-ins because they can download data or allocate significant
memory.
