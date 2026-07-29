# Scripts

## Entry points

`train_kimi.py`
: Public CLI for a full-profile directory (or three explicit YAML paths). It
  is the only script that starts the high-level `train_kimiK3` orchestrator.
  `--validate-only` performs no dataset build, model allocation or training.

`validate_data_config.py`
: Validates the data schema. `--build` is opt-in because Hugging Face profiles
  can download/tokenize data; with it, the script prints shapes and a decoded
  first-batch preview.

`validate_model_config.py`
: Validates all architecture dataclasses without allocating parameters.
  `--instantiate` is opt-in and still never performs a forward or training.

`infer_kimi.py`
: Restores a model YAML plus trained checkpoint and runs the native
  KDA/MLA-cache autoregressive master function. Sampling can come from
  `config/inference/*.yaml` and be overridden from CLI.

## Research and legacy utilities

`cpu_smoke_train.py`
: Old download-free Phase-0 smoke script for the baseline Transformer. It is
  retained as a narrow regression/debugging utility, not as the Kimi training
  entrypoint.

`benchmark_training_diagnostics.py`
: Manual A/B benchmark for diagnostics overhead (`off`, `cheap`, `standard`,
  `deep`). It intentionally performs training steps and should only be run
  when benchmarking.

## Examples

Validate a complete T4 pipeline without running anything:

```bash
python -m scripts.train_kimi \
  --profile config/kimi_full_pipeline/low_gpu \
  --validate-only
```

Start it by removing `--validate-only`.

Inference:

```bash
python -m scripts.infer_kimi \
  --profile config/kimi_full_pipeline/low_gpu \
  --checkpoint checkpoints/t4_15gb/kimi_k3_t4_15gb_epoch_0002.pt \
  --prompt "Once upon a time"
```
