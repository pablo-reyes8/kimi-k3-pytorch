# Kimi K3 Mini — Phase 11 loss contract

## Public structure

The implementation lives in `src/loss/`. Its root contains the five public
differentiable objectives:

1. `NextTokenCrossEntropyLoss`
2. `MultiTokenPredictionLoss`
3. `SFTTrajectoryCrossEntropyLoss`
4. `KimiPolicyOptimizationLoss`
5. `MultiTeacherOnPolicyDistillationLoss`

Shared validation, reduction, reward processing, and per-loss mathematics live
under `common/`, `pretraining/`, `sft/`, `rl/`, and `distillation/`.
`src.kimi_k3.losses` is a compatibility re-export, not a second implementation.

## Pretraining

`NextTokenCrossEntropyLoss` accepts explicit causal shifting or already-aligned
targets. Its effective mask combines attention, semantic loss, packed-boundary,
and ignore-index masks. Visual placeholders should have `label=-100`; text
following visual tokens remains trainable and propagates gradient through the
projector and MoonViT.

`MultiTokenPredictionLoss` consumes the target IDs and validity mask constructed
by phase 9. It never reconstructs future shifts. Single-depth `[B,T,V]` and the
research extension `[B,H,T,V]` share one weighted token reduction.

`KimiPretrainingLoss` computes `NTP + lambda_mtp * MTP`. It contains no MoE
auxiliary loss, router z-loss, or entropy term.

## SFT

The collator must provide `assistant_mask`. System, user, developer, retrieved
context, tool observations, and visual placeholders are context by default.
Reasoning, tool calls, arguments, assistant text, final answers, and EOT can be
selected explicitly. Token mean is the default; sequence mean is an ablation.

## Kimi policy optimization

For each prompt group, rewards use the exact group-mean baseline. The loss is:

```text
-clip(exp(current_logp - old_logp), ratio_min, ratio_max) * advantage
+tau * (current_logp - old_logp)^2
```

The stable implementation clamps in log space before `exp`, but applies L2 to
the original log-ratio. This is not PPO's minimum surrogate. There is no value
head, GAE, critic loss, or default entropy bonus.

## Multi-teacher OPD

Each sample declares one teacher ID or one `(domain, effort)` pair. The reward
is `clamp(stop_gradient(teacher_logp - student_reference_logp), -Rmax, Rmax)`.
Modes are sampled-token policy gradient, Kimi-RL regularized stale rollouts,
and corrected Top-k reverse KL. Sampled mode never needs full teacher logits.

## Reduction and distributed training

Every primary loss returns `loss`, `loss_sum`, `normalizer`, and
`num_valid_tokens`. A future DDP trainer must all-reduce counts and backpropagate
`local_loss_sum * world_size / global_normalizer`. Loss modules perform no
collectives, backward calls, optimizer steps, model-mode changes, or mutation.

## Quantile Balancing

Quantile Balancing remains causal router control inside Stable LatentMoE. It is
a buffer update outside autograd, not a differentiable loss. The composite
output exposes `moe_aux_loss = None` deliberately.

## Decisions not published by Kimi K3

MTP coefficient, label smoothing, SFT reduction, RL clipping bounds, log-ratio
L2, MOPD reward bound, reward weights, and exact collator masks are project
configuration. `KimiLossConfig` requires explicit post-training values.
