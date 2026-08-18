# Lesson 10 — Efficient attention: GQA + a flash-style fused kernel

## Objective

Take the causal multi-head attention from lesson 09 and make it (a) **cheaper**
with **Grouped-Query Attention (GQA)** — fewer key/value heads, shared across
query heads — and (b) **memory-light** with a **flash-style fused kernel** that
never materializes the `L×L` score matrix, using the **online-softmax** trick.
Prove both produce *the same numbers* as the naive reference to `max|diff <
1e-3`, and that the fused kernel's backward grad-checks against finite
differences.

## Why

Attention's cost is two-fold. *Compute:* `O(L·L·Dh)` for the score matmul.
*Memory:* the `L×L` probability matrix, which for long `L` dominates and forces
materialization. Two engineering moves cut these down without changing the math:

- **GQA** shrinks the `K,V` projections — and the **KV cache** at generation —
  by a factor of the group size, with negligible quality loss. This is the
  single most impactful inference-cost reduction in modern open LLMs.
- **Flash** keeps the working set at `O(L)` per query row instead of `O(L²)`,
  by streaming the softmax. Same result, far less memory pressure → longer
  contexts fit. You'll implement the CPU analog; the GPU kernel is lesson 12.

## Part A — Grouped-Query Attention (GQA)

In lesson 09's MHA, `n_heads == n_kv_heads`: every query head has its own
key/value head. **GQA** decouples them: `n_kv_heads < n_heads`, and each
key/value head is *shared* by a *group* of query heads:
```
group = n_heads / n_kv_heads          // must divide evenly
Wq: [C, C]                            // C = n_heads * Dh
Wk, Wv: [C, C_kv]                      // C_kv = n_kv_heads * Dh   <-- SMALLER
```
The query side splits into `B*H` rows; the key/value side into `B*Hkv` rows.
Query head `qi` uses kv head `kv = qi / group`. The attention computation is
identical to MHA *except* the kv index is `qi / group` instead of `qi`.

**Worked example.** `n_heads = 4`, `n_kv_heads = 2` → `group = 2`.
```
query heads:  q0 q1 q2 q3
kv heads:      k0     k1     (k0 shared by q0,q1; k1 shared by q2,q3)
qi -> kv:      0  0  1  1   (qi / 2)
KV cache size: 2 heads instead of 4 -> HALVED
```
`Bq = group * Bkv` is the invariant (e.g. `Bq=4, Bkv=2`): the number of query
"rows" is exactly `group` times the number of kv rows. Assert it.

**Why it works at all:** attention quality is dominated by the *queries*
(deciding what to look for); the keys/values just need enough *capacity* to
represent what's being retrieved. A handful of kv heads suffices, so we
over-provision queries and under-provision kv. The KV cache — the dominant
inference cost — scales with `n_kv_heads`, so cutting it 2× or 4× cuts
inference memory/time by the same factor with tiny quality cost.

### GQA in the cache

`forwardIncr` stores only `Hkv` heads of `k,v` (not `H`), in `[B*Hkv, cap, Dh]`.
For each query head `qi`, fetch the matching kv head `qi/group` from the cache.
This is *where* GQA pays off: the cache is `group`× smaller.

## Part B — Flash-style fused attention (online softmax)

Naive attention (lesson 09) does:
```
scores = q @ k^T / sqrt(Dh)            // [Bq, L, L]   <-- the L×L materializes
scores = mask_causal(scores)            // future -> -inf
P      = softmax(scores, axis=-1)       // another L×L
out    = P @ v                          // [Bq, L, Dh]
```
The `L×L` matrix lives in memory. **Flash** computes the *same* `out` without
ever holding the full `L×L`: for each query row `i`, stream over keys `j =
0..i`, maintaining a **running max `m`**, a **running denominator `l`**, and a
**running output `o[Dh]`**. This is the **online softmax**.

### Online softmax — the worked example

You want `out = softmax(s) · v` where `s` is the score row, without ever
materializing the whole row's exps at once. Keep running stats `m` (max so
far) and `l` (sum of exps so far) and the weighted output `o`. When you see a
new score `s_j` with value `v_j`:
```
m_new = max(m, s_j)
l_new = l * exp(m - m_new) + exp(s_j - m_new)      // rescale the old sum
o_new = o * exp(m - m_new) + exp(s_j - m_new) * v_j  // rescale the old output
m = m_new
```
At the end: `out = o / l`.

**Worked example**, two keys, `q=[1,0]`, `k0=[1,0]` (s=1), `k1=[0,1]` (s=0),
`v0=[10,0]`, `v1=[0,20]`, scale 1:
```
j=0: s=1.   m=-inf->1   l = 0*exp(-inf... ) [0] + exp(1-1)=1   o = 0*.. + exp(0)*v0 = [10,0]
j=1: s=0.   m=max(1,0)=1 l = 1*exp(1-1) + exp(0-1) = 1 + 0.3679 = 1.3679
                          o = [10,0]*exp(0) + exp(-1)*[0,20] = [10, 7.358]
out = o / l = [10/1.3679, 7.358/1.3679] = [7.310, 5.378]
```
Check naively: `softmax([1,0]) = [0.7311, 0.2689]`; `out = 0.7311*[10,0] +
0.2689*[0,20] = [7.311, 5.378]`. **Match.** The rescale factor `exp(m_old -
m_new)` is the whole trick: when the max grows, you discount everything you've
already accumulated so the denominator and output stay consistent.

### The block algorithm

The CPU analog here streams *one key at a time* (per query row), so the working
set is `O(Dh)` per row, not `O(L)`. A real GPU flash kernel streams *blocks* of
keys at a time (tiling for SRAM) — same recurrence, batched. The point for us:
**no `[Bq, L, L]` allocation**, ever.

### Backward

The forward didn't stash the probability matrix (that was the point). So the
backward **recomputes** the softmax probs for one query row at a time from `q`
and `k`, then applies the standard softmax-attention gradient:
```
d_scores[i,j] = P[i,j] * (d_attn[i] · v[j]  -  sum_j' P[i,j'] * (d_attn[i] · v[j']))
d_q += d_scores @ k ;  d_k += d_scores^T @ q ;  d_v += P^T @ d_attn
```
with the kv-grouping (`kv = qi / group`) folded into every index. Recompute is
the flash-attention trade: pay compute back to save memory. Grad-check it.

## Multiple representations of the online softmax

- **Algebra:** maintain `(m, l, o)`; on new `s`: `m'=max(m,s)`,
  `l'=l·e^{m-m'}+e^{s-m'}`, `o'=o·e^{m-m'}+e^{s-m'}·v`; finalize `o/l`.
- **Diagram:** a cursor walking the key axis; three registers (max, denom,
  output) that rescale as the cursor advances; no matrix, just a streaming dot.
- **Analogy:** computing a weighted average where you might discover a much
  larger value later — you keep the *running* average but, when a new max
  appears, you "renormalize" everything seen so far downward by how much the
  scale reference shifted. The `e^{m-m'}` factor *is* that renormalization.
- **Physical:** it is the numerically-stable softmax (subtract the max) made
  *incremental*; the running max is the stability trick and the recurrence is
  the streaming.

## Tasks

1. **`vCausalGQA(q, k, v, group)`** — the naive (matrix-materializing) GQA
   attention. `q: [Bq, L, Dh]`, `k, v: [Bkv, L, Dh]`, `Bq == group*Bkv`.
   Implement scores, causal mask, softmax, `@v`, with the `kv = qi/group`
   index. Grad-check it (`grad-check: GQA causal attention (group=2, shared
   kv)`).

2. **`vFlashAttn(q, k, v, group)`** — the online-softmax fused version. Forward
   with running `(m, l, o)` per query row, no `L×L` buffer. Backward
   recomputes probs per row and applies the softmax-attn gradient. Grad-check
   it (`grad-check: flash fused attention (online softmax, group=2)`).

3. **Equivalence test.** With `group=1`, `vFlashAttn` and `vCausalGQA` must
   match the lesson-09 `vCausalMHA` to `max|diff < 1e-3`. With `group=2`, flash
   must match GQA to `1e-3`. Reference test: `Gate 9: flash & GQA match
   reference attention (max-abs < 1e-3)`.

4. **`GqaSelfAttention`** layer (Wq `[C,C]`, Wk/Wv `[C,C_kv]`, Wo `[C,C]`),
   with `forward` (autograd, calls `vFlashAttn`) and `forwardIncr` (tensor, KV
   cache of `Hkv` heads). **`GqaTransformerLM`** with `forward` + `forwardGen`.

5. **The checkpoint.** GQA KV-cache == non-cached forward (`max|diff < 1e-3`);
   the GQA LM trains and beats the unigram baseline. Reference tests:
   `Gate 9: GQA KV-cache forward == non-cached forward` and `Gate 9: GQA LM
   (flash attn) trains and beats unigram baseline`.

## Checkpoint: the tests that must pass

```
zig test src/znn/autograd.zig     # grad-check: GQA causal attention
                                 # grad-check: flash fused attention
                                 # Gate 9: flash & GQA match reference attention (max-abs < 1e-3)
zig test src/znn/integration.zig # Gate 9: GQA KV-cache forward == non-cached forward
                                 # Gate 9: GQA LM (flash attn) trains and beats unigram baseline
```
Bars: all grad-checks use the atol-floor rule; the equivalence test requires
`max|diff < 1e-3` for *both* group=1 (vs MHA) and group=2 (flash vs GQA); the
LM's val loss must beat unigram.

## Common pitfalls

- **The group index `qi / group`, not `qi % group`.** `%` would *tile* query
  heads to kv heads; `/` *groups contiguous* query heads under one kv head.
  Both are valid GQA conventions but they are *different* — pick one and use it
  everywhere (forward, backward, cache). Mixing them silently breaks the
  equivalence test. The reference uses integer division `/ group`.
- **`Bq == group * Bkv` not asserted.** A shape mismatch here gives a wrong
  but plausible-looking result. Assert it up front.
- **Online-softmax rescale forgotten.** If you forget `l = l * exp(m - m_new)`
  (or the matching `o = o * exp(m - m_new)`), the running stats are
  inconsistent and `out` is wrong by exactly the missing rescale — the
  equivalence test catches it, but it's subtle because the *first* row (where
  the max never grows) looks fine.
- **Numerical drift in the running max** over a long row in f32. The
  `exp(m_old - m_new)` factors are all ≤ 1, so they don't blow up, but a very
  long row with a slowly drifting max can accumulate rounding. For the lesson's
  short `L` it's negligible; for the real kernel you'd use f32 with the
  stable rescale (as here) or a partial f32/bf16 split. Don't "optimize" the
  rescale away.
- **Causal mask inside the streaming loop.** `if (j > i) continue;` *before*
  you compute `s_j` — you must not let a future key update the running max/sum,
  or row `i`'s softmax sees the future and the equivalence test fails by a lot.
- **Backward recomputes probs with the SAME causal mask and SAME `kv=qi/group`
  index** as the forward. A mismatch between forward and backward kv indexing
  passes the forward equivalence test but fails the grad-check.
- **f32 vs the online rescale for the grad-check.** The flash grad-check, like
   all the attention checks, uses a weighted linear loss scaled by
   `1/sqrt(numel)` (lesson 00's pitfall) to keep `|L| ~ O(1)` so f32
   finite-difference roundoff stays below the tolerance bar. The reference uses
   `eps=1e-3, rtol=2e-2` for flash. Don't tighten past what f32 supports.
- **Cache stores `Hkv` heads, not `H`.** A lesson-09 habit (cache `H` heads)
   applied to a GQA layer wastes the whole point of GQA and breaks the
   `Bkv`-shaped cache indexing in `forwardIncr`.

## What good looks like

- `vFlashAttn` matches `vCausalGQA` matches `vCausalMHA` (group=1) to <1e-3 —
  three different implementations of the same math, provably equal.
- The fused op allocates **no `L×L` buffer** (grep your code; the only
  per-row allocation is `O(Dh)` or `O(L)` scratch, not `O(L²)`).
- The GQA LM trains and beats unigram; its KV cache is `group`× smaller than
  the MHA LM's at the same context.
- You can recompute the online-softmax worked example by hand and explain why
  `exp(m_old - m_new)` is the rescale that keeps the running average correct.

Lesson 11 replaces the *entire softmax attention* with a *linear* recurrence —
Gated Delta Attention / DeltaNet — that is `O(L·Dh²)` with no `L×L` matrix at
all, and carries state instead. The online-softmax streaming discipline you
just learned is the direct ancestor of the chunked-parallel form there.