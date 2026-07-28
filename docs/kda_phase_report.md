# Kimi Delta Attention phase report

## Mathematical source

The K3-specific recurrence, lower-bounded decay, head-wise normalization, and
full-rank output gate follow Kimi K3 Technical Report §2.1.1, equations 1–6.
The UT/WY representation follows Kimi Linear §3.1 and was cross-checked against
the MIT-licensed FLA KDA reference implementation:

- `fla/ops/kda/naive.py`
- `fla/ops/kda/chunk.py`
- `fla/layers/kda.py`

FLA is not a runtime dependency.

## Implemented equations

The recurrent path applies:

```text
Sbar = Diag(exp(g_t)) S
error = v_t - Sbar^T k_t
S = Sbar + beta_t k_t error^T
o_t = S^T q_t
```

The current token writes before it is read. The chunkwise path computes:

```text
L = I + StrictTril(Diag(beta) K_gamma K_gamma_inverse^T)
M = solve_triangular(L, Diag(beta))
W = M K_gamma
U = M V
V_tilde = U - W S_in
O = (Gamma * Q) S_in + Tril((Gamma * Q)(K/Gamma)^T) V_tilde
```

and updates the cross-chunk state using relative log-decays.

## Shapes and precision

```text
hidden: [B,T,D]
q/k/g:  [B,T,H,K]
v/raw:  [B,T,H,V]
beta:   [B,T,H]
state:  [B,H,K,V]
D = H*V; K and V may differ
```

FP16/BF16 inputs accumulate recurrent states, reductions, UT matrices, and
triangular solves in FP32 by default. FP32 and FP64 retain their input
precision. Outputs return to activation dtype. No training state is detached.

## Decay initialization

`A_log` is initialized to zero as required by K3. The default
`official_fla` bias initializer samples `dt` log-uniformly in `[1e-3, 1e-1]`
and stores its inverse-softplus transform, matching the official FLA KDA layer.
The explicitly named `zeros` option is an experimental fallback, not an
official recipe.

## Execution modes

- `recurrent`: exact token recurrence and mathematical oracle.
- `chunkwise`: loop across chunks only; UT and causal interactions within each
  chunk are dense/tiled matrix operations.
- `decode`: exactly one token, updating recurrent and three independent
  ShortConv states without reprocessing the prefix.

Monotonic right-padding is supported. Invalid tokens use `g=0`, `beta=0`, and
`q=0`; their projected values never enter ShortConv cache state.

## Scope

This phase implements only KDA and its K3 postprocessing. It does not add a
residual connection, Gated MLA, Attention Residuals, MoE, or multimodal
integration.

