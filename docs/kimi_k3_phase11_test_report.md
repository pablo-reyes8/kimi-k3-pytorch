# Phase 11 CPU test report

The phase-11 suite contains 64 CPU tests under `tests/losses/`. The complete
repository regression collects 1,276 tests: 1,261 pass and 15 are intentionally
skipped (`31.77s`, Python 3.10, PyTorch 2.8.0 CPU). It covers:

Before phase 11 the repository collected 1,212 tests (1,197 passing and 15
skipped). Phase 11 therefore adds 64 tests without removing earlier coverage.

- NTP manual references, alignment, masks, boundaries, weighting, BF16, and
  connected-zero behavior;
- phase-9 MTP targets, single/multi-depth reductions and empty futures;
- SFT roles, tool calls versus observations, component weights and reductions;
- exact Kimi-RL baselines, stable clipping, raw log-ratio L2 and detach rules;
- MOPD reward sign, stop-gradient, bilateral clipping, teacher routing,
  regularized policy gradients, and corrected Top-k reverse KL;
- global token-weighted reduction across simulated ranks;
- end-to-end tiny text, multimodal, and MTP backward paths;
- the absence of an MoE balancing loss.

Direct tests use batches of one to four samples, sequence lengths up to eight,
and vocabularies up to eleven. Integration uses the real 50,068-parameter
CPU-tiny KimiK3, `16x16` images, four projected visual tokens, and vocabulary
size 128. Sensitive calculations are FP32 even with BF16 inputs.
Manual reference comparisons use PyTorch's default
`torch.testing.assert_close` tolerances; exact scalar counts, masks, and
normalizers use equality checks.

There is no `MoEBalancingLoss`, router z-loss, Switch-style balance objective,
or differentiable Quantile Balancing term. Router bias remains a no-gradient
buffer owned by Stable LatentMoE.

`config/losses/cpu_tiny.yaml` contains test values only. Kimi K3 does not
publish the MTP weight, RL bounds/L2 coefficient, MOPD bound, SFT reduction,
or reward-source weights.
