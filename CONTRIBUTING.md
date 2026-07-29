# Contributing to Kimi-K3 Mini

Thanks for helping improve the project. Contributions should preserve its main
goal: readable, paper-oriented PyTorch implementations backed by strong,
CPU-safe behavioral tests.

## Before opening a change

1. Search existing issues and pull requests.
2. For a large architectural change, open a feature request describing the
   public API, YAML impact and validation strategy.
3. Never attach private datasets, credentials or untrusted checkpoints.
4. Keep post-training and distributed execution changes clearly separated
   from the currently supported pretraining pipeline.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,data]"
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1`.

Useful commands:

```bash
make validate
make check
make test
```

`make validate PROFILE=config/kimi_full_pipeline/low_gpu` validates a complete
experiment profile without downloading data, allocating a model or training.

## Design rules

- Keep KDA, MLA, MoE, AttnRes, MTP, vision, training and inference concerns in
  their existing modules.
- Extend typed dataclasses and strict YAML parsers rather than reading loosely
  structured dictionaries throughout the codebase.
- Route training through `train_kimiK3`; route generation through
  `inference_autoregressive`.
- Preserve checkpoint compatibility or document and test the migration.
- Do not silently fall back from a Kimi-specific mechanism to a generic
  Transformer implementation.
- Avoid network access in unit tests. Use deterministic, realistically shaped
  tensors that remain practical on CPU.

## Testing expectations

Every behavioral change needs a test that measures the intended invariant, not
only tensor creation. Depending on the scope:

```bash
pytest tests/configuration
pytest tests/inference
pytest tests/training
pytest tests/kda tests/mla tests/hybrid_backbone
pytest
```

Numerical code should test shapes, dtypes, causality, masks, gradients,
full-vs-cached parity and failure cases where applicable. Do not start real
training as part of a normal test.

## Pull requests

Keep pull requests focused and explain:

- what changed and why;
- which public contracts or YAML fields changed;
- exact commands used for validation;
- numerical, memory, checkpoint or data risks;
- any work intentionally deferred.

By contributing, you agree that your contribution is licensed under the
project's [MIT License](LICENSE) and that community participation follows the
[Code of Conduct](CODE_OF_CONDUCT.md).
