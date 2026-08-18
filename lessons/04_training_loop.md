# Lesson 04 — Losses and the training loop

## Objective

Wire your autograd (lesson 03) into an actual learning loop. Concretely:
implement the two losses you will use for the rest of the curriculum —
**softmax cross-entropy** for classification/next-token and **MSE** for
regression — and an **SGD-with-momentum** optimizer; then drive them with a
minibatch loop to train a 2D 3-class **spiral classifier** to >95% accuracy.

By the end you have the universal shape of every training run you will ever
write: `forward → loss → backprop → step`, repeated over minibatches.

## Why this lesson

Everything from lesson 05 onward is "the same loop, different model, bigger
data." If the loop is right here on a tiny problem you can *see*, you can trust
it on a transformer. The spiral is the classic stress test: the three arms
collapse toward the origin, so the inner points of different classes sit
~0.005 apart in input space. A network that beats 95% on the spiral has
genuinely learned a non-linear boundary — it isn't memorizing a trivial
separation. When a real LM won't train later, you'll come back to *this* loop
to check the basics (loss going down, grads not NaN, lr sane).

## Part A — softmax cross-entropy, the right way

You have logits `z ∈ R^C` (one row per example, `C` classes). The probability of
class `i` is the softmax:
```
p_i = exp(z_i) / Σ_j exp(z_j)
```
Cross-entropy against the true class `t` (one-hot) is:
```
L = -log(p_t) = -z_t + log Σ_j exp(z_j)
```
That last form is the whole point: you never need to form `p` and then take a
log. The loss is **`logsumexp(z) - z_t`**, and the gradient w.r.t. `z_i` is:
```
dL/dz_i = p_i - 1[i == t]        // softmax minus the one-hot target
```

### Why subtract the max (numerical stability)

`exp(z)` overflows the instant a logit is large (exp(100) ≈ 1e43, exp(1000) =
inf in f32). The fix is identity-based, not a hack:
```
log Σ_j exp(z_j) = m + log Σ_j exp(z_j - m)      where m = max_j z_j
```
The max subtraction is *exact* — it's just factoring out `exp(m)`. With `m =
max`, every exponent is `≤ 0`, so `exp` is in `(0, 1]`: no overflow, no
underflow asymmetry worth caring about. **Always compute logsumexp as
`m + log(Σ exp(z - m))`.**

**Worked example.** `z = [2.0, 1.0, 0.1]`, true class `t = 0`.
```
m = 2.0
z - m = [0, -1, -1.9]
exp  = [1.0, 0.368, 0.150]      Σ = 1.518
log Σ = 0.417
L = 0.417 + 2.0 - 2.0 = 0.417   // = -log p_0 ; p_0 = 1/1.518 = 0.659, -ln 0.659 = 0.417 ✓
```
Gradient: `p = [0.659, 0.242, 0.099]`, minus one-hot `[1,0,0]` →
`[−0.341, 0.242, 0.099]`. The true-class gradient is negative (push its logit
*up*); the rest are positive (push them *down*). That's the entire learning
signal for classification.

### Fusing softmax and CE into one op

Implement `softmaxCE(logits, targets)` as a **single autograd op** whose
forward computes the stable loss above and whose backward is the one-line
`p - onehot`. Do *not* build it as `softmax → log → nll` from separate ops —
that route is numerically worse (a `log` of a tiny `softmax` output) and slower.
This fused op is the single most-used loss in the curriculum; get it right once.

The targets are integer class indices (a `[]usize`), not one-hot vectors —
index into the logits to get `z_t` and to place the `-1` in the gradient.

## Part B — MSE (one paragraph)

`mse(pred, target) = mean( (pred - target)^2 )`. Build it from autograd ops:
`vSub → vMul (square) → vSumAll → scale by 1/N`. Its gradient is
`2(p-t)/N`. Use it for the regression warm-up (fit `y = 3x + 1`) and anywhere
you predict a continuous target. No softmax, no stability tricks.

## Part C — SGD with momentum

Vanilla SGD: `θ ← θ - lr·g`. It works but crawls in ravines. **Heavy-ball
momentum** keeps a velocity:
```
v ← μ·v + g          // or v ← μ·v - lr·g ; pick a convention and stick to it
θ ← θ - lr·v
```
`μ ≈ 0.9` damps oscillations and accelerates in consistent-gradient directions.
**Sign convention is the #1 bug**: if you write `v ← μ·v + g` then `θ ← θ - lr·v`,
keep both signs; flipping one makes momentum *amplify* errors instead of damp
them. (You will meet Adam in lesson 08 — it's this plus per-parameter adaptive
scaling.)

## Part D — the loop and the spiral

The spiral dataset: 3 classes, 2 features. For each class `c ∈ {0,1,2}` and
point `j`, set radius `t = (j+0.5)/N`, angle `= t·6π + c·(2π/3)`, and
`(x, y) = (t·cos θ, t·sin θ)` plus small noise. The `+0.5` radius floor is
**critical**: without it, the innermost point of every class sits at the origin
with *different labels* — three contradictions at one point, which no model can
fit. (This exact bug is in the reference as a comment; don't repeat it.)

The loop, per step:
```
logits = mlp.forward(x)            // build the graph
loss  = softmaxCE(logits, targets) // scalar
loss.backprop()                    // fills .grad (autograd zeros first)
opt.step(params)                   // SGD-momentum update
```
A fresh graph each step is fine — the arena holds the old nodes, you just stop
referencing them. You do **not** call `zero_grad` manually here *if* your
`backprop` zeros the whole graph first (lesson 03); if it accumulates across
calls, you must zero. Know which one your engine does.

**Cosine LR decay** (one-line version; full scheduler in lesson 08): shrink `lr`
along `lr_min + 0.5·(lr0 - lr_min)·(1 + cos(π·step/total))`. Constant-lr Adam
*oscillates forever* around a minimum and never settles; the decay lets it
tighten up. This is what makes the spiral actually hit 95% rather than plateau
at 80%.

## Multiple representations of softmax-CE

- **Algebra:** `L = logsumexp(z) - z_t`, `∇L = softmax(z) - onehot(t)`.
- **Diagram:** logits `z` → (subtract max, exp, sum) → `p`; the loss picks
  `-log p_t`; the gradient flows back as `p` with a `-1` punched into slot `t`.
- **Analogy:** the classes are *competitors* for probability mass summing to 1.
  CE measures how much mass the wrong classes stole from the true one. Raising
  the true logit and lowering the others is exactly "give the mass back."
- **Physical:** `∇L = p - 1_t` is a *force* pushing the true class's logit up
  and the others down, with magnitude proportional to how wrong the current
  distribution is.

## Tasks

1. Implement `mse` from autograd primitives; test it on fitting `y = 3x + 1`
   with a single `Linear` + Adam — W→3, b→1, loss→~0. (This is the reference
   `test "regression: fit y = 3x + 1"`.)

2. Implement fused `softmaxCE(logits: *Var, targets: []const usize) *Var`:
   forward = stable `logsumexp - z_t` (mean over the batch), backward =
   `(softmax - onehot)/B`. Grad-check it as part of a 2-layer MLP chain (the
   reference `test "grad-check: full 2-layer MLP + softmax-CE chain"`).

3. Implement `SGD` with momentum + weight decay; `(lr, momentum, weight_decay)`
   params; a `step(params)` and `deinit`.

4. Build the spiral dataset (3 classes, the radius floor, noise) and a 2-hidden
   MLP `2 → 64 → 64 → 3` (two hidden layers — one is not enough to separate the
   inner arms; the reference notes why). Train with Adam + cosine LR ~6000
   steps. Print loss + accuracy every 500 steps. Accuracy is argmax over the
   3 logits.

5. Run it. Watch acc climb from ~0.33 (chance) past 0.95.

## Checkpoint: the test that must pass

`test "overfit: tiny clean spiral reaches ~100% train acc"` in the reference
integration suite (a clean, tiny spiral; `acc > 0.95` after 500 steps). And the
end-to-end `test "grad-check: full 2-layer MLP + softmax-CE chain"` proving the
fused softmax-CE backward is correct under composition with matmul, bias
broadcast, and relu.

The bar: **spiral accuracy > 95%**, and the MLP+CE grad-check is green.

## Common pitfalls

- **Naive softmax/log without the max trick** → `inf` / `NaN` on any
  confident logit. Always `m = max`, `logsumexp = m + log(Σ exp(z-m))`.
- **Forgetting to zero grads between steps** *if* your `backprop` accumulates.
  The reference `backprop` zeros the whole graph first, so the loop doesn't
  call `zero_grad` — but *know your own engine*.
- **Momentum sign convention** — pick one, keep both signs aligned. A flipped
  sign makes the model diverge.
- **Learning rate too high** → loss explodes to NaN in a few steps. If you see
  `loss=nan`, halve `lr`. For the spiral, `lr0=0.05` with cosine decay is
  stable; `0.5` is not.
- **Single hidden layer on the spiral** → plateaus ~70-80%. The inner arms need
  depth to compose sharp boundaries. Use two.
- **No radius floor** → three classes' first points collapse at the origin;
  the data is literally unlearnable; accuracy stuck at 33%.
- **ReLU + finite-difference grad-check with large eps** → units flip across
  the kink and the numeric gradient diverges from the analytic subgradient.
  That's a *methodology* artifact (ReLU isn't smooth), not a bug. Use a smooth
  activation or smaller eps for the grad-check (lesson 03's atol floor absorbs
  the rest).

## What good looks like

- `softmaxCE` passes a full-chain grad-check; the spiral hits >95%.
- You can derive `∇L = softmax - onehot` on a napkin and explain why the max
  subtraction is exact, not approximate.
- Your loop is `forward → loss → backprop → step` and you can swap the optimizer
  (SGD ↔ Adam) by changing one line.

Lesson 05 turns text into tokens, so this loop has real data to chew on.