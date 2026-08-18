# Lesson 03 — Tensor autograd and the grad-check harness

## Objective

Promote the scalar `Value` from lesson 01 to a tensor-valued `Var`: same
topo/backprop engine, same "each op records a closure, walk backwards" idea,
but every number is now a tensor and one new hard part appears — when an op
broadcast its inputs, the backward gradient must be **un-broadcast** back down
to each parent's real shape. Then build the **grad-check harness** that is the
objective arbiter for the rest of the curriculum (every backward you write from
here on is checked against finite differences). By the end, you can add a new
op in a forward line + a backward line and *prove* it's right.

## Why this

Lesson 01 was the chain rule. Lesson 02 was shapes. This is where they fuse:
the backward of a tensor op is *just the chain rule, with shape rules*. The one
thing that has no scalar analogue is un-broadcasting: if `c = a + b` with `a`
`[2,1]` and `b` `[1,3]` broadcast to `c` `[2,3]`, then `dc` is `[2,3]`, but
`a.grad` must stay `[2,1]` — so `a`'s backward has to **sum `dc` down** to
`[2,1]`. Get this one mechanic right and every broadcasting op's backward
becomes mechanical. Miss it and you get silent shape mismatches that only blow
up downstream, with grads that *look* plausible.

## The model

```zig
pub const Var = struct {
    value: Tensor,      // forward value
    grad: Tensor,       // gradient of the final loss w.r.t. this (same shape)
    parents: []const *Var,
    backward: ?*const fn (*const Var) void,
    // optional ctx pointer if your closure needs captured state
};
pub fn parameter(alloc, t: Tensor) !*Var;  // leaf we want grads for
pub fn constant(alloc, t: Tensor) !*Var;   // leaf, requires_grad = false
```

`backprop()` is **identical in spirit** to lesson 01: build a topo list
(parents-first), zero all grads, seed the root's grad (`self.grad.data[0] = 1`
when the root is a scalar loss — non-scalar roots must be seeded by the caller),
then walk the reversed list calling each `backward`.

The only difference from lesson 01: `grad` is a `Tensor`, and each op's backward
must produce a gradient *the same shape as that parent's `value`*. That's where
`accumulate` + `unbroadcast` come in.

### The two functions that make everything work

Every backward ends by adding a local gradient into each parent:
```zig
// accumulate `incoming` (already-shaped as the parent's grad contribution) into
// parent.grad, un-broadcasting if the shapes differ.
pub fn accumulate(alloc, target: *Tensor, incoming: Tensor) void {
    if (shapeEq(target.shape, incoming.shape)) {
        for (0..target.numel()) |i| target.data[i] += incoming.data[i];
        return;
    }
    const reduced = unbroadcast(alloc, incoming, target.shape) catch return;
    for (0..target.numel()) |i| target.data[i] += reduced.data[i];
}
```
and `unbroadcast(g, target_shape)` does the numpy "sum-to-shape": (1) while `g`
has *more* dims than the target, `sumAxis(g, 0)` and squeeze that leading dim
off; (2) for each remaining dim where the target is `1` but `g` is larger,
`sumAxis` along that dim. The result has exactly `target_shape`. This is the
*inverse* of the forward broadcast — broadcasting copies in the forward, summing
collapses in the backward.

### Worked example 1: matmul backward

`C = A @ B`, `A:[M,K]`, `B:[K,N]`, `C:[M,N]`. Given upstream `dC:[M,N]`:
```
dA = dC @ B^T      // [M,N]@[N,K] -> [M,K]   (same shape as A) ✓
dB = A^T @ dC      // [K,M]@[M,N] -> [K,N]   (same shape as B) ✓
```
So the matmul backward is *two matmuls* (with transposes) — the same primitive
you wrote in lesson 02, re-used. No broadcasting here (shapes already match),
so `accumulate` is just `+=`. Sanity-check the dimensions: `dA`'s shape must
equal `A.shape`, always. If it doesn't, you have a transpose backwards.

### Worked example 2: broadcasted add backward

`c = a + b`, `a:[2,1]`, `b:[1,3]`, `c:[2,3]`, upstream `dc:[2,3]`:
```
da = unbroadcast(dc, [2,1]) = sumAxis(dc, axis=1) -> [2,1]   ✓
db = unbroadcast(dc, [1,3]) = sumAxis(dc, axis=0) -> [1,3]   ✓
```
The forward *broadcast* `a` from `[2,1]` to `[2,3]` by repeating columns; the
backward *sums the columns back* to `[2,1]`. The forward broadcasts `b` from
`[1,3]` by repeating rows; the backward sums the rows back to `[1,3]`. One
rule, both directions. `accumulate` calls `unbroadcast` for you when the
shapes differ, so your backward can simply build `dc`-shaped contributions and
hand them to `accumulate`.

### The pattern for a new op

For `out = f(x)`:
1. forward: compute `out.value`, set `parents = &.{x}` (or `{x,y}`), stash any
   closure context you need (e.g. for relu, the mask of where `x>0`; for
   sigmoid, `out.value` itself).
2. backward: read `self.grad` (= upstream `dL/dout`), compute the local
   contribution to each parent's grad using the local derivative and the
   stashed context, and `accumulate(alloc, &parent.grad, contribution)`.
That's it. Adding `vRelu`, `vMul`, `vAdd`, `vSigmoid`, `vMatmul`, `vSumAll` is
this pattern repeated. The `makeVar(alloc, value, parents, ctx, back)` helper
exists to do the boilerplate.

## The grad-check harness (the arbiter)

This is the tool that makes the rest of the curriculum trustworthy, so build
it carefully. Given a `build(alloc, ctx) !*Var` that constructs the graph from
the current leaf values and returns a *scalar* root, and the list of `leaves`
to perturb:

1. **Analytic pass.** `build` once, `backprop()`, stash each leaf's `.grad` into
   a saved buffer.
2. **Numeric pass.** For each leaf element `e`: set `leaf.data[e] += eps`,
   `build`, read scalar `Lp`; set `leaf.data[e] -= eps`, `build`, read `Lm`;
   restore `leaf.data[e]`. Numeric grad `num = (Lp - Lm) / (2*eps)` (central
   difference).
3. **Compare.** `abs_err = |num - ana|`. `bound = tol*max(|num|,|ana|) + atol`.
   `scaled = abs_err / max(1e-12, bound)`. Track the worst `scaled` over all
   leaves/elements.
4. **Pass** when `max_scaled <= 1.0`.

Two non-negotiable details (from lessons 00/01, and you will be burned by
forgetting them later):

- **The `atol` floor is baked into `bound`.** `bound = tol*max(...) + atol`
  with `atol = 1e-4`. A pure-relative `|num-ana|/|num|` would scream at a
  gradient of `1e-6`; the `+atol` term absorbs f32 finite-difference roundoff
  for tiny gradients while still flagging real bugs on large gradients (where
  `tol*max(...)` dominates). The reference `gradCheck` uses exactly
  `tol=1e-2, atol=1e-4`. Don't "simplify" the `atol` away.
- **Rebuild the graph from scratch each perturbation.** You must call `build`
  *after* mutating `leaf.data[e]`, not reuse a stale graph — the forward values
  in the saved graph would be wrong. The harness rebuilds twice per element.

**The `1/sqrt(numel)` trick** for nonlinear ops: a plain `sumAll` loss over a
big tensor has tiny per-element grads that drown in f32 noise, but a `sum` loss
blows the magnitude up. For grad-checking nonlinear ops, define a *weighted*
scalar loss — multiply the output by a fixed random weight tensor and sum, or
sum then scale by `1/sqrt(numel)` — so the per-element grad sits comfortably
above `atol` without exploding. You'll meet this again in Gate 11 (gated-delta)
where it's the difference between a green and a red grad-check.

## Multiple representations

- **Algebra:** each backward is `dL/d(parent) = (dL/d(out)) · (∂out/∂parent)`,
  with `∂out/∂parent` summed over any broadcast axes to match the parent's
  shape.
- **Diagram:** same DAG as lesson 01; now each edge carries a *tensor* of
  gradient. The shapes must shrink back to each parent as you walk upstream —
  `accumulate`/`unbroadcast` are the "shape-fixing" step at each node.
- **Analogy:** forward broadcast *blew up* a `[2,1]` into a `[2,3]`; backward
  *folds* the `[2,3]` back into a `[2,1]` by summing the blown-up axis. Inflate
  going down, deflate coming back.
- **Physical intuition:** grad-check is a *caliper*. You never trust a
  hand-derived backward on the first try; you measure it against the only
  ground truth that can't have a sign or shape error — the actual numerical
  slope of the loss.

## Tasks

1. **`Var` + `parameter`/`constant` + `backprop`** (topo + zero + seed + walk),
   mirroring lesson 01 but tensor-valued. Reuse your lesson-02 `Tensor`.

2. **`accumulate` and `unbroadcast`.** Make `unbroadcast` the *only* place
   shape reduction happens. Test it directly: `unbroadcast` a `[2,3]` of ones
   to `[2,1]` gives `[[3],[3]]`; to `[1,3]` gives `[[6,6,6]]`.

3. **Ops:** `vAdd` (broadcast), `vMul` (broadcast), `vMatmul`, `vRelu`,
   `vSigmoid`, `vSumAll`, `vSoftmaxCE` (do softmax-CE *fused* — see the hand
   case below; don't softmax-then-CE separately, it's numerically worse). Each
   forward + backward via `accumulate`.

4. **`gradCheck`** as specified above, with `atol=1e-4, tol=1e-2`, central
   differences, rebuild-per-perturbation.

5. **Run every op through gradCheck.** Build a small `Ctx` struct holding the
   leaves, a `build` fn, and call `gradCheck(Ctx, ..., &leaves, eps, tol)`.

## Checkpoint: the tests that must pass

`zig test src/autograd.zig`. In the reference suite these are the gate:
- `scalar Value: a*b + a, backward` → `dz/dx = y+1 = 4`, `dz/dy = x = 2`.
  (This is the *scalar* Value from lesson 01 still living here — keep it; it's
  the warm-up.)
- `grad-check: matmul + relu + sumAll` — `z = relu(a@b)` summed, `a:[2,3]`,
  `b:[3,2]`, grad-checks with `eps=1e-3, tol=1e-2`. **Must pass.**
- `softmax-CE: exact gradient on a hand-computed case` — logits `[0,0,0]`,
  target `0`: loss `= ln 3`, `dlogits = [-2/3, 1/3, 1/3]`. Match to `1e-4`.
- `grad-check: softmax cross-entropy`
- `grad-check: indexRows (embedding gather)` — the backward of a row-gather is
  *scatter-add*; grads accumulate into the gathered rows.
- `grad-check: broadcasted mul`
- `grad-check: sigmoid + silu`

The bar for every `grad-check` test: `gradCheck` reports
`max_scaled=...` and it must be `<= 1.0` (the harness returns
`Error.GradientMismatch` otherwise, failing the test). Typical good values are
well under `0.5`; you'll see the harness print `max_scaled=0.0696`, `0.0125`,
etc. If you see `max_scaled=42`, you have a bug — usually a missing
un-broadcast or a transposed backward.

## Common pitfalls

- **Forgetting to un-broadcast in a backward.** The classic: `vAdd`'s backward
  does `a.grad += dc` directly when `a` is `[2,1]` and `dc` is `[2,3]` → either
  a shape crash or, worse, writing a `[2,3]` into a `[2,1]` buffer. Always go
  through `accumulate`, which un-broadcasts for you. Never index a parent's
  grad assuming it's the output's shape.
- **Not zeroing grads before backprop.** `backprop` zeroes the whole graph
  first (it walks the topo and `zerosGrad`s every node). If you bypass that
  and call backward fns manually, you'll accumulate stale grads across checks.
- **Reusing the forward graph for the numeric perturbation.** The harness
  must `build` *after* changing `leaf.data[e]`. A stale graph gives a numeric
  grad of ~0 for everything and the check "passes" for the wrong reason (both
  sides near 0) on some leaves while failing on others. If numeric grads are
  all ~0, you forgot to rebuild.
- **Pure-relative tolerance on f32.** `|num-ana|/|num|` fails on tiny grads.
  Use `bound = tol*max(|num|,|ana|) + atol` and pass when `scaled <= 1`.
  This is the *same* atol-floor lesson from lesson 00, now in code.
- **Sign error in a backward.** e.g. `dB = dC^T @ A` instead of `A^T @ dC`,
  or `dRelu = grad * (x>0)` vs `(x<0)`. grad-check catches these instantly
  (`max_scaled` huge) — that's the point. When it screams, trust it and recheck
  the one local derivative, not the whole graph.
- **`sumAxis` not keeping the dim.** `unbroadcast` relies on `sumAxis`
  *keeping* the reduced dim as size 1 (lesson 02). If your `sumAxis` collapses
  it, `unbroadcast`'s shape math breaks. Keep the dim.
- **Non-scalar root not seeded.** `backprop` seeds `self.grad[0]=1` only for a
  scalar root. If you grad-check a non-scalar output, seed it yourself (e.g.
  a random `+/-1` tensor) before calling backprop, and define the loss as the
  dot of output and that seed — or just always reduce to a scalar in `build`.

## What good looks like

- `gradCheck` passes on matmul+relu+sum, softmax-CE, indexRows, broadcasted mul,
  sigmoid+silu — all `max_scaled <= 1`, typically `<< 1`.
- You can add a brand-new op (`vTanh`, `vSilu`) in a forward line + a backward
  line, run grad-check, and get green on the first or second try — because you
  route the backward through `accumulate` and you trust the harness.
- You can explain, with the inflate/deflate analogy, why `a:[2,1]`'s
  add-backward sums axis 1, without hesitating.
- The harness's `atol` floor is in your muscle memory; you will reach for it
  again in lessons 08 (RMSNorm), 10 (attention), and 11 (gated-delta).

You now have a *verified* autograd. From here on, "it works" is decided by
`gradCheck`, not by your confidence. Lesson 04 puts a training loop on top of
it and watches something actually learn.