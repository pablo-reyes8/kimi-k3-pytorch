# Kimi K3 primitives phase report

## Implemented APIs

- `SiTUGLU` and the independently testable `softcap` /
  `situ_glu_activation` equations.
- `CausalShortConv1D` with depthwise lag-ordered kernels and immutable
  `ShortConvState`.
- `HeadwiseRMSNorm` over the last `Dh` axis, with per-head affine scales by
  default.
- `FullRankOutputGate`, where the gate is generated from the residual input.
- `ShortConvState`, `KDAProjectionOutput`, and `AttentionModuleOutput`.
- `split_heads`, `combine_heads`, and `PrimitiveAttentionPostprocess`.

## Tensor convention

The entire primitive package uses:

```text
hidden states:    [B, T, D]
per-head tensors: [B, T, H, Dh]
D = H * Dh
```

No primitive silently accepts `[B,H,T,Dh]`.

## Numerical decisions

- SiTU softcaps use `beta * tanh(x / beta)`, never a hard clamp.
- Default SiTU bounds are `beta_gate=4` and `beta_up=25`.
- Headwise RMSNorm accumulates squared reductions in FP32 for FP16/BF16 and
  preserves FP32/FP64 accumulation otherwise. Output returns to input dtype.
- ShortConv stores weights by semantic lag: index zero is the current token.
  PyTorch's cross-correlation kernel is flipped only at the internal call.
- ShortConv state cloning preserves autograd and prevents in-place mutation or
  storage aliasing.
- Output gates and SiTU projections use normal initialization with standard
  deviation `0.02` and zero biases when enabled.

## Scope boundary

This phase does not implement KDA, Gated MLA, Attention Residuals,
LatentMoE, MTP, multimodal integration, or task-specific training. The
postprocessing composition exists only to freeze head reshape, normalization,
and gating conventions before KDA.

## Verification

The tests in `tests/kimi_primitives` include direct reference equations,
extreme-value bounds, gradcheck, strong prefix causality, impulse responses,
full/chunk/token decoding equivalence, state immutability, head/channel/batch
independence, BF16, serialization, state-dict roundtrips, and complete
end-to-end gradients.

The next architectural phase is `4_IMPLEMENTAR_KDA.md`.

