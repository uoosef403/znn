# Lesson 08 — Optimizers: Adam, Muon, and Newton-Schulz

## Objective

Implement two optimizers and a scheduler that together cover everything the
rest of the curriculum uses: **Adam** (bias-corrected adaptive moments — the
reliable workhorse), and **Muon** (momentum + Newton-Schulz orthogonalization —
the genuinely modern piece that updates matrix parameters along an
*orthogonal* direction). Wire them into a **Hybrid** router (Muon for matrices,
Adam for everything else) with a warmup→cosine schedule and global grad clipping.
The gate: a small LM trained with Hybrid beats the same LM trained with plain
Adam at equal step budget, and the Newton-Schulz routine demonstrably
*improves conditioning* (is more orthogonal than its input).

## Why Muon

Plain gradient descent on a matrix parameter moves each entry by its own
gradient. But the *useful* geometry of a matrix is its singular-value structure:
the loss is far more sensitive along large singular directions than small
ones, so raw gradient steps overshoot on big directions and crawl on small
ones — the conditioning problem. Muon's fix: take the momentum buffer of the
gradient, then **orthogonalize** it (replace each singular value by its sign,
so every direction gets the same step size). An orthogonal update has unit
spectral norm, so the step is *balanced across directions* regardless of how
ill-conditioned the gradient was. On a fixed step budget this pulls ahead of
Adam on the matrix-heavy parameters (attention/FFN projections) that dominate
an LM's parameter count.

You don't compute an SVD (expensive). You approximate the orthogonal sign
factor with a **5-step Newton-Schulz matrix polynomial** — a few matmuls,
exact same shape in and out. That's the whole trick, and it's the first
non-obvious algorithm in the curriculum.

## Adam — the workhorse

Per parameter, maintain a first moment `m` (EMA of the gradient) and a second
moment `v` (EMA of squared gradient), both **bias-corrected** for the
warm-up bias of starting at 0:
```
t += 1
m = b1*m + (1-b1)*g            # b1 = 0.9
v = b2*v + (1-b2)*g*g          # b2 = 0.999
mhat = m / (1 - b1^t)          # bias correction
vhat = v / (1 - b2^t)
param -= lr * mhat / (sqrt(vhat) + eps)   # eps = 1e-8
```
The `sqrt(vhat)` divides the step by a per-parameter RMS scale: big-gradient
parameters get *relatively* smaller steps, small-gradient ones get bigger
steps — an automatic per-coordinate learning rate. The bias correction matters
only early (it shrinks toward 1 as `t` grows) but it's the difference between
sane early training and a divergent first 50 steps.

**Worked example, 1D quadratic, 3 steps.** Minimize `f(x) = (x-4)^2`, so
`g = 2(x-4)`. Start `x=0`. `lr=0.1`, `b1=0.9`, `b2=0.999`, `eps=1e-8`.
```
t=1: g=-8. m=0.1*(-8)=-0.8. v=0.001*64=0.064.
     mhat=-0.8/(1-0.9)=-8. vhat=0.064/(1-0.999)=64.
     x -= 0.1 * (-8)/(8+1e-8) = +0.1.   x=0.1
t=2: g=2(0.1-4)=-7.8. m=0.9*(-0.8)+0.1*(-7.8)=-1.5. v=0.999*0.064+0.001*60.84=0.1245.
     mhat=-1.5/(1-0.81)=-7.89. vhat=0.1245/(1-0.998)=62.25.
     x -= 0.1 * (-7.89)/7.89 = +0.1.   x=0.2
t=3: g=-7.6. ... step ~ +0.1 again. x≈0.3.
```
Note the bias-corrected step is ~`lr` per step on this well-scaled gradient —
without correction (`mhat=-0.8`), step 1 would be `0.1*0.8/8 = 0.01`, ten
times too small. **Bias correction is what makes the first steps not crawl.**
The `eps` keeps the denominator off zero when `v` is tiny.

## Muon — orthogonalize the momentum

```
g_scaled = grad * sqrt(max(m, n) / min(m, n))   # balance wide vs tall matrices
buf     = momentum * buf + g_scaled            # heavy-ball momentum
u       = newtonSchulz5(buf)                   # ORTHOGONAL sign factor
param  -= lr * u
```
(With Nesterov: orthogonalize `g_scaled + momentum*buf` instead of `buf` — the
lookahead version. Either is fine; the reference uses Nesterov.)

**Newton-Schulz (the orthogonalizer).** Given matrix `G` (`[m,n]`), produce an
*approximately orthogonal* `O` of the same shape: if `G = U S V^T` is the SVD,
`O = U (sign S) V^T` — keep the *directions*, throw away the *magnitudes*. The
5-step polynomial (coefficients `(a,b,c) = (3.44420, -4.77557, 2.03120)`,
tuned for exactly 5 steps):
```
X_0 = G / ||G||_F                       # normalize so singular values land in (0,1)
repeat 5 times:
    A  = X X^T                          # Gram matrix, [M,M] (M = min(m,n))
    AA = A A
    X  = a*X + b*(A X) + c*(AA X)        # cubic in the Gram matrix
O  = X                                  # ≈ sign(S) on the spectrum
```
Operate on the "short-and-fat" orientation (transpose if `m > n` so the Gram
matrix is the smaller `[M,M]`), then transpose back. Each iteration is a handful
of matmuls (`A = X X^T`, `AA = A A`, two `B @ X`). Five steps is enough to push
singular values into ~[0.7, 1.2] — **approximately orthogonal, not exact.**

> **Critical fact you will be tempted to forget:** 5 steps is *approximate*. An
> ill-conditioned input ends with singular values in a band around 1, never
> exactly 1. So `O @ O^T == I` is the **wrong** test — it will fail. The right
> test is *monotone improvement*: the orthogonality gap `||O O^T - I||/M` is
> strictly **smaller** than the gap of the Frobenius-normalized input, and
> bounded (no NaN, no explosion). Gate the routine on *that*, not on exactness.

**Worked example, intuition.** `G = diag([1, 0.02])` (ill-conditioned: one big
singular value, one tiny). Frobenius-normalized, `X_0 ≈ diag([0.9998, 0.02])`.
Raw, this matrix's two update directions are wildly unbalanced: direction 0 is
5000× the magnitude of direction 1. After 5 NS steps the singular values both
land near 1: the update now steps ~equally in both directions. **That
rebalancing is the entire point of Muon** — it's why the matrix parameters
descend in a better-conditioned way than Adam's per-entry scaling.

**Grad scaling** `sqrt(max(m,n)/min(m,n))`: balances wide vs tall matrices so
the orthogonalized step has comparable scale across shapes. Without it, a
`[4096, 128]` vs a `[128, 4096]` matrix get very different effective step sizes.

## The Hybrid router

Muon is only defined for **2D matrix** parameters. 1D params (biases, RMSNorm
gamma) have no singular values to orthogonalize, and 2D **embedding tables** /
the **output head** are updated *sparsely* (only looked-up rows, lesson 07) —
the orthogonal factor is ill-defined there. So route:

- 2D matrix params **not** in an override set → **Muon**
- 1D params + overridden 2D (token embedding `W`, LM head `W_head`) → **Adam**

Partition the parameter list **once** at init (by pointer identity against an
override set), then each `step()` calls `muon.step(muon_params)` and
`adam.step(adam_params)`. Typical base LRs: `lr_muon ≈ 0.02`, `lr_adam ≈ 0.003`
(Muon's orthogonal steps are larger per unit lr, so it wants a smaller lr).

## Scheduler and clipping

**Warmup→cosine:** multiply the base lr by a factor each step:
```
step < warmup:   factor = (step+1) / warmup           # linear ramp 0 -> 1
else:            progress = (step - warmup) / (total - warmup)
                 factor = min + 0.5*(1-min)*(1 + cos(pi*progress))   # cosine to `min`
```
Warmup avoids a giant first step on un-calibrated moments; cosine decays the lr
to refine the solution late. Guard the divisions: `max(warmup, 1)` and
`max(total - warmup, 1)`.

**Global grad clipping:** compute the global L2 norm over *all* params' grads,
and if it exceeds `max_norm`, rescale every grad by `max_norm / norm`. Call it
*before* the optimizer step. Returns the pre-clip norm (useful for logging
explosions). Guard `norm == 0` (don't divide by zero; a no-op when there's
nothing to clip).

## Multiple representations

- **Algebra (Adam):** `theta -= lr * EMA(grad) / (RMS(grad) + eps)`, both EMAs
  bias-corrected. Per-coordinate adaptive step.
- **Algebra (Muon):** `theta -= lr * sign-SVD(EMA(grad))` — step along the
  orthogonal direction of the momentum, magnitude equalized across singular
  directions.
- **Diagram (Muon):** `grad --EMA--> buf --normalize--> X0 --5x NS--> O --*lr--> step`.
  The NS block is `A=XX^T -> AA=A^2 -> X = aX + bAX + cAAX`.
- **Analogy:** Adam drives each wheel of a car at a speed proportional to how
  bumpy *that wheel's* road is. Muon first rotates the whole steering vector so
  no single wheel dominates, then drives all wheels at the same speed. Muon is
  for the *geometry* of the update; Adam is for per-coordinate *patience*.
- **Physical:** orthogonalizing removes magnitude (the "how hard" of each
  direction) and keeps direction (the "which way"). The loss surface is
  anisotropic; orthogonalization isotropizes the *step*, so you don't ricochet
  off the steep walls.

## Tasks

1. **`Adam`** with per-parameter state in an `AutoHashMap(usize, []f32)` keyed
   by `@intFromPtr(p)`. Track `t` globally. Implement the bias-corrected update
   above. `setLr`. `deinit` (free all moment buffers).

2. **`newtonSchulz5(alloc, g, m, n, out, steps)`** as a *free function* (no
   `Var`) so it's unit-testable in isolation. Handle the transpose-to-fat
   orientation, Frobenius normalize, run the cubic polynomial `steps` times,
   transpose back. Use the NS5 coefficients.

3. **`Muon`** with a per-parameter momentum buffer. Only step on 2D params
   (`p.value.ndim() == 2`); skip others. Apply grad scaling, heavy-ball (+
   Nesterov lookahead), NS, then `param -= lr * u`.

4. **`Hybrid`** — partition params once at init (override set by pointer), hold
   a `Muon` and an `Adam`, `step()` calls both. `setLr` scales both groups
   together by the scheduler factor.

5. **`Scheduler.factor(step)`** and **`clipGradNorm(params, max_norm)`**.

6. **The checkpoints** below — the conditioning test, the stability test, and
   the head-to-head LM test.

## Checkpoint: the tests that must pass

Three tests (names from the reference `integration.zig`):

1. `test "Muon: Newton-Schulz output is more orthogonal than the input"` — feed
   an *ill-conditioned* matrix of known SVD (one dominant singular value, the
   rest tiny) through `newtonSchulz5`. Assert: output not NaN, orthogonality
   gap `< 1.0` (bounded), gap **strictly less** than the input's gap
   (improvement, not no-op), and `< 0.3` in absolute terms. **Do not assert
   `O@O^T == I`** — 5 steps is approximate.

2. `test "Muon: momentum buffer -> NS path is stable across steps"` — take
   several Muon steps on a 2D param with fresh random grads; assert no NaN, no
   explosion (norm `< 1e3`).

3. `test "Gate 7: Muon-hybrid LM beats plain-Adam LM at equal steps"` — train
   the *same* small LM with Hybrid vs plain Adam, equal steps, same data/seed.
   Assert Hybrid's loss is lower. This is the end-to-end payoff.

Run `zig test src/integration.zig` (or `src/optim.zig` for the isolated NS
routine). All three green.

## Common pitfalls

- **Testing NS for exact orthogonality** (`O@O^T == I`) — fails; 5 steps is
  approximate (singular values land in a band, not exactly 1). Gate on
  *improvement* + boundedness.
- **Forgetting bias correction in Adam** — first ~50 steps crawl or diverge.
  `mhat = m/(1-b1^t)`, `vhat = v/(1-b2^t)` with `t` *global across all params*
  (one counter), recomputed each step.
- **Applying Muon to non-2D params** — undefined for 1D, ill-defined for sparse
  2D (embeddings/head). Skip non-2D in `Muon.step`; route overrides to Adam in
  `Hybrid`. Forgetting this produces NaNs in embedding rows that got no real
  gradient.
- **`eps` in the Adam denominator** — without it, a parameter whose `v` is
  exactly 0 divides by zero. Keep `eps=1e-8`.
- **Scheduler division by zero** — `warmup=0` or `total==warmup` divides by
  zero. Guard with `max(warmup, 1)` and `max(total - warmup, 1)`.
- **`clipGradNorm` when norm is 0** — early steps with zero grad give
  `total=0`; guard the scale (`total > 0`).
- **NS on the wrong orientation** — if `m > n` you must transpose so the Gram
  matrix `X X^T` is `[min, min]`, not `[max, max]` (else it's slower *and* the
  iteration's conditioning suffers). Transpose back at the end.
- **NS Frobenius normalization** — skip it and singular values start > 1, the
  polynomial diverges. Always `X /= (||X||_F + 1e-7)`.
- **Hash-map key** — key moment buffers by `@intFromPtr(p)`, *not* by index in
  the params slice (the slice order can change between calls; the pointer is
  stable for the life of the parameter). Free all buffers in `deinit` or you
  leak across optimizer recreations.
- **Per-step allocator churn in NS** — NS allocates scratch every call. In a
  hot training loop, allocate the scratch once and reuse, or accept the cost on
  CPU (the GPU version, lesson 12, must batch this).

## What good looks like

- The conditioning test shows the gap *shrink* (`in=0.4 → out=0.05`-ish), not
  stay flat (no-op NS) or explode (divergent NS).
- The head-to-head test has Hybrid *win*, and you can articulate *why*
  (orthogonal steps on the matrix-heavy projections), not just "it did".
- You can add a new optimizer (e.g. plain SGD with momentum, already in the
  reference as `SGD`) on the same `step(params)` interface in ~30 lines.

You now have the optimizer that trains the transformer in lesson 09. The
modern stack — RoPE, SwiGLU, KV cache, GQA — starts next, all of it updated by
this Hybrid.