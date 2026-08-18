# Lesson 11 — Gated Delta Attention / DeltaNet (the hero)

## Objective

Replace softmax attention (the `QK^T → softmax → @V` you built in lessons 09–10)
with a *linear* attention built on the **delta rule** under two learned gates.
You will implement the **recurrent form** (forward + backprop-through-time, the
training path), grad-check it, derive and implement the **chunked-parallel form**
(the inference/throughput path) and prove it matches the recurrent form, then
build a **DeltaNet LM** and show it trains and beats the unigram baseline — at
parity with your softmax-attention LM.

This is the hardest lesson and the most important. It is where "modern linear
attention" stops being a buzzword and becomes a recurrence you can write,
differentiate, and parallelize.

## Why this instead of (or alongside) softmax

Softmax attention builds an `L × L` score matrix and softmaxes over it. That
quadratic-in-`L` cost is the wall large-context models hit. **Linear attention**
rewrites attention as a *recurrence on a small per-head state* `S ∈ ℝ^{Dh×Dh}`:
time is `O(L·Dh²)`, memory is `O(Dh²)` — **no `L × L` matrix**. The catch: plain
linear attention (just `S += v k^T`, `o = S q`) is weak — it never *corrects*
what it already stored. The **delta rule** fixes that: at each step it writes a
*correction* `d_t = v_t − α_t·S_{t-1}k_t`, the new value *minus* the part of it
the memory already predicts. That correction term is the whole idea. Add a
forget gate `α` (how much old memory survives) and a write gate `β` (how hard
the correction lands) and you have **Gated DeltaNet**.

## The model (the recurrence)

Per head, per batch — a state `S` of shape `[Dh, Dh]`, updated token by token.
With `α_t = exp(g_t)` (forget) and `β_t = sigmoid(b_t)` (write):

```
r_t   = S_{t-1} k_t                       (recall: what does memory say for key k_t?)
d_t   = v_t − α_t r_t                     (the DELTA: new value minus recalled part)
S_t   = α_t S_{t-1} + β_t (d_t) k_t^T     (decay old, write the correction)
o_t   = S_t q_t                           (read)
```

Inputs (per head): `q, k, v ∈ [L, Dh]`, `α, β ∈ [L]` (one scalar gate per
position). The op takes the *gate values* `α, β` as inputs — the **nn layer**
applies `exp`/`sigmoid` to learned projections first, so the autograd op itself
is pure recurrence. `S_{-1} = 0`.

### Multiple representations

- **Algebra:** it is linear attention whose write is a rank-1 *correction*
  `β_t d_t k_t^T` instead of a raw rank-1 write `v_t k_t^T`. The `−α_t r_t` term
  is what makes it a *delta* (error-correcting) update rather than an
  accumulating one.
- **Diagram:** `S` is a little `Dh×Dh` matrix living *between* tokens. Each step
  it shrinks by `α` (forget), then gets a rank-1 nudge `β d k^T` added. The query
  `q_t` reads one row-combination out. Draw it as: tokens flow left→right; at
  each token the state matrix is touched in place; `o_t` peels off a vector.
- **Analogy:** `S` is a notebook. `α` is how much of yesterday's notes you keep
  (forget gate — `exp` keeps it positive, so "keep 90%" is clean). `β` is how
  firmly you write. But you don't transcribe `v_t` verbatim: you first check
  what you *already* have for key `k_t` (that's `r_t = S_{t-1}k_t`), and you
  write only the **difference** `d_t = v_t − α_t r_t` — the part of the new
  value your notes *don't* already capture. That is the delta. A notebook that
  only ever *appends* drifts; a notebook that *corrects* stays sharp.
- **Physical:** it is a *data-dependent* linear RNN on a matrix-valued state.
  The gates make the recurrence learnable rather than fixed. Because the read
  `o_t = S_t q_t` and the write are both linear in `S`, the whole thing can be
  *parallelized across time within a chunk* (see part C) — the property that
  makes it trainable at scale where a general nonlinear RNN cannot.

### A worked numeric example (Dh = 2, three tokens, fixed gates)

Let `α_t = 0.5` for all `t` (forget half each step), `β_t = 1.0` (full write),
`S_{-1} = 0`. Inputs:
```
k_0 = [1, 0],  v_0 = [2, 0],  q_0 = [1, 0]
k_1 = [0, 1],  v_1 = [0, 3],  q_1 = [0, 1]
k_2 = [1, 1],  v_2 = [1, 1],  q_2 = [1, 0]
```

**t=0:** `r_0 = S_{-1} k_0 = [0,0]`. `d_0 = v_0 − α r_0 = [2,0]`.
`S_0 = 0.5·0 + 1·d_0 k_0^T = [[2,0],[0,0]]`. `o_0 = S_0 q_0 = [2,0]`.

**t=1:** `r_1 = S_0 k_1 = [[2,0],[0,0]]·[0,1]^T = [0,0]`. `d_1 = [0,3]−0 = [0,3]`.
`S_1 = 0.5·S_0 + d_1 k_1^T = [[1,0],[0,0]] + [[0,0],[0,3]] = [[1,0],[0,3]]`.
`o_1 = S_1 q_1 = [0,3]`.

**t=2:** `r_2 = S_1 k_2 = [[1,0],[0,3]]·[1,1]^T = [1,3]`. `d_2 = v_2 − α r_2 =
[1,1] − 0.5·[1,3] = [0.5, −0.5]`. **← notice the correction:** `v_2=[1,1]` but
memory already *recalls* `[1,3]` for this key, so the delta only writes the
residual `[0.5,−0.5]`, not the raw value. `S_2 = 0.5·S_1 + d_2 k_2^T =
[[0.5,0],[0,1.5]] + [[0.5,0.5],[−0.5,−0.5]] = [[1,0.5],[−0.5,1]]`.
`o_2 = S_2 q_2 = [1, −0.5]`.

That `d_2 = [0.5, −0.5]` step is the whole intuition in one line: it is **not**
`v_2`, it is `v_2` *minus what was already recalled*. Re-derive these three
steps by hand before you write any code. If your `S_2` doesn't match, you have
the orientation of `k_t^T` wrong (it's an outer product `d k^T`, rows indexed by
`d`, columns by `k`).

## Tasks — part A: the recurrent forward

1. **Define the op** `vGatedDelta(alloc, q, k, v, alpha, beta)` returning a
   `*Var` of shape `[Bh, L, Dh]`. The inputs are flat-heads layout: `q,k,v` are
   `[Bh, L, Dh]`, `alpha, beta` are `[Bh, L]`. `Bh` = batch×heads (you merge
   them before the op, like you did for MHA).

2. **Forward:** loop `h` over `Bh`, loop `t` over `L`, carry a `Dh×Dh` state in
   a scratch buffer. Compute `r`, `d`, update `S`, read `o`. **Stash every
   `S_t`** (for `t = 0..L-1`) into a `[Bh, L, Dh, Dh]` tensor — the backward
   needs all intermediate states (BPTT). This is the autograd op's `ctx`.

3. **The nn layer** `GatedDeltaAttn`: project `x → q,k,v` with `Wq, Wk, Wv`,
   scale `k ← k / sqrt(Dh)`, split heads. Compute the gate *logits* via
   `W_alpha, W_beta` (both `[C, H]`), then `α = exp(x @ W_alpha)`,
   `β = sigmoid(x @ W_beta)` — reshape to `[Bh, L]`. Call `vGatedDelta`. Merge
   heads, project out with `Wo`. **No RoPE here** (the recurrence carries
   position; that is the DeltaNet design — RoPE would fight the delta structure).
   Expose `params()` returning the six weight vars.

4. **Block + LM:** `GatedDeltaBlock` (norm1 → attn → +residual → norm2 → ffn →
   +residual) mirroring your transformer block, and `GatedDeltaLM` (embed →
   blocks → final norm → head). Give it a `forwardIncr` that carries one
   per-head state across positions (a `Cache` holding `S`), and a `forwardGen`
   that loops positions against the carried state — the recurrent form *is* the
   generation path, no KV cache needed (state is `Dh²` per head, constant in `L`).

## Tasks — part B: the recurrent backward (BPTT)

Backward walks `t` from `L-1` down to `0`, carrying `G = dL/dS_t` (a `Dh×Dh`).
Let `g_t = dL/do_t` (the incoming output grad). Per step (you derive these
from the forward by the product rule — don't take them on faith):

```
G     += g_t q_t^T                  (o_t = S_t q_t  -> dL/dS_t gets g_t q_t^T)
dq_t   = S_t^T g_t                  (o_t = S_t q_t  -> dL/dq)
Gk     = G k_t                      (used three ways below)
dv_t   = β_t Gk                     (d_t = v_t - α r_t  -> dL/dv)
dk_t   = β_t G^T v_t − α_t β_t S_{t-1}^T Gk − α_t β_t G^T r_t
dα_t   = <G, S_{t-1}> − β_t (r_t · Gk)        (scalar; alpha gate)
dβ_t   = d_t . Gk                              (scalar; beta gate)
G      <- α_t G − α_t β_t Gk k_t^T             (propagate G to t-1)
```

The `dα_t` term has two pieces: the direct effect of `α` on the decay
(`<G, S_{t-1}>`) and its effect *through* the delta correction
(`−β_t (r_t · Gk)`). The `dβ_t` is beautifully simple: just the dot of the
correction `d_t` with `Gk`. Sanity: if `d_t` were zero (memory already perfect),
`dβ_t` is zero — no reason to change the write strength. Nice gradient.

5. **Implement `gatedDeltaBack`** following the recipe above. You have `S_{t-1}`
   from the stash (zero for `t=0`). **Mind the braceless-for rule:** every
   `for` body that does `G[i] += ...` or `state[i*Dh+j] = ...` must be wrapped in
   braces; nested `for (i) for (j) {...}` needs a trailing `;` after the closing
   brace. This is the #1 compile-error source in this op.

6. **Wire it as a `Var`** with five parents `(q,k,v,alpha,beta)` and the stashed
   `S` as `ctx`. `accumulate` the five output grads into the parents' `.grad`.

## Tasks — part C: the chunked-parallel form (the hero derivation)

The recurrence is sequential in `t`. For long sequences that is slow and can't
use the matmul hardware. The **chunked-parallel form** computes a chunk of `Cc`
tokens at once with matmuls + a triangular solve, then carries a small state to
the next chunk. Recurrent *within* a chunk would still be sequential; the trick
is to rewrite the recurrence so the chunk is a pure linear-algebra problem.

Here is the derivation. Work it on paper before coding — the lesson is in the
algebra, not the code.

**Step 1 — absorb the write gate.** Scale `k̃ = √β k`, `ṽ = √β v`. Then the
update becomes `S_t = α_t S_{t-1}(I − k̃_t k̃_t^T) + ṽ_t k̃_t^T`. (Multiply it
out: the `β` and the `−α r` both collapse into the `√β` factors. The point:
`β` no longer appears as a separate coefficient — it is absorbed into `k̃, ṽ`.)

**Step 2 — go to the decay-free frame.** Define the cumulative decay
`G_t = ∏_{m≤t} α_m` (`G_{-1}=1`) and the decay-free state `S̃_t = S_t / G_t`.
Dividing the recurrence through by `G_t`, the `α_t` decays cancel (because
`G_t = α_t G_{t-1}`), leaving a **plain delta rule with no decay**:
```
S̃_t = S̃_{t-1} + (ṽ_t/G_t − S̃_{t-1} k̃_t) k̃_t^T.
```
This is the key move: a data-dependent decay recurrence has been turned into a
*decay-free* delta recurrence on `(k̃, ṽ/G)` with state `S̃`. Decay-free delta
is a linear system you can solve with a triangular matrix.

**Step 3 — the chunk as a triangular solve.** For a chunk of size `Cc` with
incoming state `S̃_in`, stack the chunk keys/values into `K̃, Ṽ/G ∈ [Cc, Dh]`
and queries `Q ∈ [Cc, Dh]`. The strict-lower-triangular Gram matrix
`tril(K̃ K̃^T, −1)` encodes "past keys only". Define
```
T = (I + tril(K̃ K̃^T, −1))^{-1}        (lower-tri, unit diagonal — forward substitution)
W = T K̃                                [Cc, Dh]
U = T (Ṽ/G)                             [Cc, Dh]
```
`(I + tril(K̃K̃^T,−1))` is unit lower-triangular, so its inverse `T` is too, and
you compute it by **forward substitution**, not a general inverse. Then the
plain-delta parallel identities give:
```
S̃_out = S̃_in + (U − W S̃_in^T)^T K̃^T          (carry to next chunk)
õ     = Q S̃_in^T + tril(Q K̃^T, 0) (U − W S̃_in^T)   (diagonal INCLUDED)
```
and the real (decay-restored) output is `o_t = G_t · õ_t`.

**Why diagonal included here but `−1` for the solve:** the solve `T` models
causality *strictly* (token `t` is influenced only by `t' < t`), so the Gram is
`tril(·, −1)`. The output mask `tril(QK̃^T, 0)` *includes* the diagonal because
`o_t` reads the state *after* token `t`'s own write. Off-by-one on this diagonal
is the most common bug and produces a small-but-systematic mismatch.

7. **Implement `gatedDeltaChunkedForward`** as a pure-tensor function (no
   autograd — it is the verified reference for the structure; training backward
   goes through the recurrent BPTT op). Loop chunks; carry `S̃` and `g_acc` (the
   cumulative `G`) between chunks; per chunk: build `k̃, ṽ/G, q`, the strict-lower
   Gram, forward-substitute `T`, `W = T k̃`, `U = T (ṽ/G)`, the correction
   `corr = U − W S̃_in^T`, the output `õ = Q S̃_in^T + tril(QK̃^T,0)·corr`, scale
   by `G_t`, and update `S̃_out = S̃_in + corr^T K̃^T`. Requires `L % Cc == 0`.

## Checkpoint: the tests that must pass

1. **`grad-check: gated delta attention (recurrent, 5 inputs)`** — your BPTT
   backward against finite differences on all five inputs `(q,k,v,α,β)`. Bar:
   `max_scaled` under the grad-check tolerance (`rtol 2e-2`, `atol` floor — see
   the pitfall below). Config: `Bh=2, L=4, Dh=4`, `α ≈ 0.8`, `β ≈ 0.5`.

2. **`Gate 10: chunked parallel gated-delta forward == recurrent forward`** —
   the chunked form vs the recurrent form. Bar: `max|diff| < 1e-3`. Config:
   `Bh=2, L=16, Dh=6, Cc=4` (so `L % Cc == 0`).

3. **`Gate 10: DeltaNet recurrent-state forward == non-cached forward`** — your
   `forwardGen` (carried state) matches the full `forward`. Bar: `max|diff| <
   1e-3`.

4. **`Gate 10: DeltaNet LM trains and beats unigram baseline`** — a DeltaNet LM
   (`V=16, L=8, C=32, H=4, 2 layers, ffn=64`) trained 300 steps Adam `lr=3e-3` on
   a bigram task beats the unigram baseline on val. (In the reference, DeltaNet
   val ≈ 0.73 vs unigram ≈ 2.33.)

Run: `zig test src/znn/autograd.zig` (1, 2) and `zig test src/znn/integration.zig`
(3, 4). All four green = the gate.

## Common pitfalls

- **The delta is a *correction*, not a write.** The single most common bug is
  writing `S_t = α S_{t-1} + β v_t k_t^T` (raw write) and dropping the `−α r_t`.
  That is plain gated linear attention, *not* DeltaNet — it'll train worse and
  the chunked form won't match. `d_t = v_t − α_t r_t` must use the recalled
  `r_t = S_{t-1} k_t`. The worked example above exists to nail this down.
- **`exp` for `α`, `sigmoid` for `β` — and they are *gate values*, fed to the
  op.** The nn layer applies `exp`/`sigmoid`; the autograd op receives `α, β`
  already in `[0,∞)` and `[0,1]`. Don't `exp` inside the op. `exp` for the forget
  gate is *why* the chunked form gets clean cumulative ratios `G_t = ∏α_m` — a
  `sigmoid` forget gate would not factor into a product so cleanly.
- **`α < 1` for stability.** In the grad-check, `α ≈ 0.8` (not 1.0). A forget
  gate near 1 makes `G_t` explode over a long sequence and the decay-free frame
  `ṽ/G` blows up. Keep `α < 1` in tests; the model learns appropriate gates.
- **Chunked `tril` diagonal conventions.** Solve uses `tril(K̃K̃^T, −1)`
  (strict); output mask uses `tril(QK̃^T, 0)` (includes diagonal). Swap them and
  you get a small systematic error that fails the `<1e-3` bar.
- **Forward substitution for `T`, not a general inverse.** `(I + strict-lower)`
  is unit-lower-triangular; compute `T = (I + N)^{-1}` by forward substitution
  (`T = I − N·T` row by row). A generic inverse "works" but is slower and
  numerically worse.
- **`ṽ/G` divides by the *within-chunk cumulative* `G`, and `G_t` scales the
  output back.** Forget either and the magnitudes drift across chunks.
- **BPTT braceless-for rule.** Every `for` that writes `G[i*Dh+j] += ...` or
  `state[..] = ...` must be braced; nested `for (i) for (j) {…};` needs the
  trailing `;`. You will hit `invalid left-hand side to assignment` here — add
  the braces (lesson 00).
- **Grad-check loss scaling for this nonlinear op.** The reference grad-check
  scales the weighted loss `W` by `1/√(Bh·L·Dh)` so the per-element signal stays
  above the f32 noise floor without blowing up the loss magnitude. Use the
  `atol` floor (`1e-4`) — a pure-relative check will false-fail on the tiny
  `dα/dβ` gradients. (Lesson 00, again.)

## What good looks like

- The worked-example numbers above match your forward by hand and by code.
- The BPTT backward grad-checks all five inputs green with the atol floor.
- The chunked form matches the recurrent form to `<1e-3` — which means your
  derivation of steps 1–3 is right; if it's off, the bug is almost always the
  `tril` diagonal convention or a forgotten `ṽ/G` or `G_t` rescale.
- A DeltaNet LM trains and beats unigram, at parity with your softmax LM at
  equal params — and you can explain *why* the delta correction is the thing
  that makes linear attention competitive.

You have now built the hero. Lesson 12 takes one of these ops (the matmul that
feeds it, and the recurrence itself) onto the GPU.