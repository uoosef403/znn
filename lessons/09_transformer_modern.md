# Lesson 09 — The modern transformer block

## Objective

Assemble a *modern* decoder block — the same skeleton production LLMs use — out
of four pieces you build and verify one at a time: **RMSNorm**, **RoPE**, a
**SwiGLU** feed-forward net, and **causal multi-head self-attention with a KV
cache**. Stack a couple of these blocks into a decoder LM, train it on tokens,
and generate. The gate: the incremental (one-token-at-a-time) forward with the
KV cache produces *the same logits* as the full training forward, and the LM
trains to beat a unigram baseline.

## Why this shape

A "transformer block" is just a sandwich: normalize, attend, add residual;
normalize, feed-forward, add residual. The four ingredients each fix a real
problem in the naive 2017 transformer:

- **RMSNorm** is a cheaper, mean-free LayerNorm. No mean subtraction, no bias.
- **RoPE** injects position *without a learned embedding table* and *without
  it being thrown away at longer contexts* — positions are encoded as rotations,
  so relative position falls out of the dot product.
- **SwiGLU** replaces the `Linear → ReLU → Linear` MLP with a *gated* MLP that
  learns to suppress features, not just threshold them.
- **The KV cache** turns autoregressive generation from O(L²) recomputation
  into O(L) — generate one token against all past keys/values without
  recomputing them.

Build each as an autograd op (forward + backward, grad-checked), plus a
*forward-only* tensor version for the cache path. This dual implementation is
the lesson's real lesson: **training builds a graph; generation just runs
numbers.**

## 0. RMSNorm — normalize by the root-mean-square

For a vector `x` of dim `D`:
```
rms = sqrt(mean(x^2) + eps)         // mean over D, NOT mean(x)
out  = (x / rms) * gamma             // gamma is learned, shape [D], init 1
```
Compare LayerNorm: `out = (x - mean(x)) / sqrt(var(x)+eps) * gamma + beta`.
RMSNorm drops the `mean(x)` subtraction and the `beta`. Empirically
near-identical quality, one fewer pass over the data, one fewer parameter.

**Worked example**, `x = [3, 4]`, `eps = 1e-5`, `gamma = [1, 1]`:
```
mean(x^2) = (9 + 16)/2 = 12.5
rms = sqrt(12.5 + 1e-5) ≈ 3.5355
out = [3/3.5355, 4/3.5355] ≈ [0.8486, 1.1314]
```
Note it is *not* mean-subtracted: `out` still carries the sign and rough scale
of `x`. The backward: the gradient is `(gamma/rms) * (g - x * mean(x*g/rms^2) -
mean(x*g)*x/rms^2 ...)` — the cleanest way is to derive it as the chain rule
through `rms` being a function of `x`. Grad-check it; don't trust a memorized
formula.

## 1. RoPE — position as a rotation

For each *pair* of feature dimensions `(x[2i], x[2i+1])` at sequence position
`p`, rotate by angle `θ_i = p · base^{-2i/D}` (`base = 10000`):
```
out[2i]   = x[2i]   * cos(θ_i) - x[2i+1] * sin(θ_i)
out[2i+1] = x[2i]   * sin(θ_i) + x[2i+1] * cos(θ_i)
```
This is just 2×2 rotation matrices block-diagonally applied. **Why it's magic:**
the dot product between a query at position `p` and a key at position `q`
depends only on `p - q` (rotations compose: `R(p)^T R(q) = R(q - p)`). So the
model attends by *relative* position for free — no learned position table, and
it generalizes to longer contexts than it trained on (the angles just keep
spinning).

**Worked example**, `D=4` (two pairs), `base=10000`, `p=1`:
```
θ_0 = 1 · 10000^{-0/4} = 1.000         θ_1 = 1 · 10000^{-2/4} = 1/100 = 0.01
pair0: cos(1)=0.5403, sin(1)=0.8415
pair1: cos(0.01)=1.0000, sin(0.01)=0.0100
x = [1, 0, 1, 0]  -> out ≈ [0.5403, 0.8415, 1.0000, 0.0100]
```
The pair `(1,0)` rotated by ~57°; the low-frequency pair barely moved. High
dimensions = fast-spinning (encode short-range), low dims = slow (long-range).

**Apply RoPE to Q and K, never V.** V is the *content* being retrieved, not a
query/key; rotating it has no positional interpretation and breaks the
"relative-position dot product" property. This is a classic bug.

## 2. SwiGLU — a gated feed-forward net

```
gate = x @ W_gate           // [B, L, H]
up   = x @ W_up             // [B, L, H]
h    = silu(gate) * up      // elementwise; silu(x) = x * sigmoid(x)
out  = h @ W_down           // [B, L, C]
```
where `W_gate, W_up: [C, H]` and `W_down: [H, C]`. Compare the classical MLP
`Linear → ReLU → Linear`: SwiGLU's `silu(gate) * up` is a *learnable gate* —
`up` is the candidate, `silu(gate)` decides how much of each feature passes.
ReLU can only zero things; a sigmoid gate can also *partially* pass and *invert*
(negative gate → negative pass-through after `x·σ(x)`).

**Worked example**, one feature, `gate=2, up=3`:
```
silu(2) = 2 * sigmoid(2) = 2 * 0.8808 = 1.7616
h = 1.7616 * 3 = 5.2848     // vs ReLU(gate)*up = 2*3 = 6; the gate softens it
```

## 3. Causal multi-head attention + the KV cache

Full training forward for `x: [B, L, C]`, `C = n_heads · head_dim`:
```
q,k,v = x @ Wq,Wk,Wv                         // each [B, L, C]
split heads: [B,L,C] -> [B,L,H,Dh] -> permute -> [B*H, L, Dh]
q,k = RoPE(q,k)                              // position = the L index
attn = causal_sdpa(q, k, v)                  // [B*H, L, Dh], scores masked j>i
merge heads -> [B, L, C]
out = attn @ Wo
```
`causal_sdpa`: `scores = q@k^T / sqrt(Dh)`; mask future positions (`j > i`) to
`-inf`; `softmax` over `j`; `out = probs @ v`. The scale `1/sqrt(Dh)` keeps the
score variance ~1 regardless of head width (otherwise dot products of large
vectors explode and softmax saturates).

**The KV cache** is the same computation, *incremental*: during generation you
have one new token `x: [B, 1, C]` at absolute position `p`. You project its
`q,k,v`, apply RoPE to its `q,k` at position `p`, and **append** its `k,v` to a
buffer holding all past `k,v`. Then attention is one query row against `n = p+1`
cached keys. You never recompute past keys/values. The cache is `[B*H, cap, Dh]`
preallocated to a max capacity `cap`.

**Worked example** of why it's equivalent: full forward at step 3 (tokens
0,1,2,3) computes 4 query rows, each attending to keys 0..i. The incremental
path at step 3 computes *the 4th query row only*, against keys 0,1,2,3 already
in the cache. Row 3's scores and softmax are *identical* — same q, same keys,
same scale. The cache just avoids recomputing rows 0,1,2 (already emitted).

## The block and the LM

```
block(x):
    h = x + attn(rmsnorm1(x))        // residual around attention
    return h + ffn(rmsnorm2(h))       // residual around FFN
lm(ids):
    x = tok_emb[ids]                  // [B, L, C], NO learned position emb
    for blk in blocks: x = blk(x)
    x = rmsnorm_f(x)
    return x @ W_head                  // [B, L, V]
```
Generation (`forwardGen`): for each position, run the block in *incremental*
mode (`forwardIncr` with each layer's cache), produce that position's logits
from the final hidden state, keep the caches warm for the next step.

## Multiple representations of the cache

- **Algebra:** `attn_i = softmax(q_i · K_{0:i+1} / √Dh) · V_{0:i+1}`; the cache
  stores `K_{0:i}, V_{0:i}`, so step `i` only computes `q_i, k_i, v_i` and
  appends.
- **Diagram:** a growing column of keys/values; each new token's query reaches
  across the whole column; the column persists across steps.
- **Analogy:** the cache is a *tape* of everything the model has already read;
  generating the next word reads the new word and the whole tape, then writes
  the new word's k/v onto the end.
- **Physical:** it converts the O(L²) cost of regenerating the prefix every
  step into O(L) per step.

## Tasks

1. **`vRmsNorm`** autograd op + a `rmsNormTensor` forward-only version. The
   forward-only one is what the cache path calls (no graph during generation).
   Grad-check it (the reference suite has `grad-check: RMSNorm`).

2. **`vRoPE`** autograd op + `ropeTensor(t, pos)` forward-only (rotates a
   single-row `[B*H, 1, Dh]` at absolute position `pos`). Watch the **interleaved
   pair** layout `(x[2i], x[2i+1])` — the reference pairs dim `i` with dim
   `i + Dh/2` (a "half-split" layout); pick one and be consistent. Grad-check
   it (`grad-check: RoPE (rotary position embedding)`).

3. **`vSilu`** and **`SwiGLU`** (three matrices). The backward of `silu` is
   `sigmoid(x) + x·sigmoid(x)·(1-sigmoid(x))`. Grad-check silu
   (`grad-check: sigmoid + silu`).

4. **`vCausalMHA`** autograd op: scaled dot-product, causal mask, softmax, `@v`,
   with the head-split/merge reshape+permute dance. Grad-check it
   (`grad-check: causal MHA`). This is the most error-prone op in the curriculum
   — its backward is the softmax-attention gradient:
   `d_scores = P * (d_attn @ V^T - rowsum(P * (d_attn @ V^T)))` where `P` is the
   attention probability matrix.

5. **`TransformerBlock`** (norm1, attn, norm2, ffn) with `forward` (autograd)
   and `forwardIncr` (tensor, uses caches). **`TransformerLM`** with
   `forward(ids)` and `forwardGen(alloc, ids)` that loops positions calling
   `forwardIncr` per layer.

6. **The checkpoint.** Two tests:
   - KV-cache forward == non-cached forward (the bar: `max|diff| < 1e-3`, the
     reference hits ~1e-6 — the small residual is f32 summation-order
     difference between the batched and incremental paths). Reference test:
     `Gate 8: KV-cache forward == non-cached forward`.
   - The transformer LM trains and beats the unigram baseline on a bigram
     task. Reference test: `Gate 8: transformer LM trains and beats unigram
     baseline`.

## Checkpoint: the tests that must pass

```
zig test src/znn/autograd.zig     # grad-check: RMSNorm, RoPE, causal MHA, silu
zig test src/znn/integration.zig # Gate 8: KV-cache forward == non-cached forward
                                 # Gate 8: transformer LM trains and beats unigram baseline
```
Bars: `grad-check` uses the atol-floor rule (lesson 00); KV-cache `max|diff <
1e-3` (expect ~1e-6); the LM's val loss strictly less than the unigram
baseline's. Print the KV-cache diff so you can see how close you are.

## Common pitfalls

- **RoPE on V.** Don't. Q and K only. If you do, the cache-equality test will
  still *pass* for the full-vs-cached comparison (both paths would apply it
  identically) but the model is semantically broken and won't train well —
  rely on the *trains-and-beats-unigram* test too, not just the diff test.
- **Cache not reset between sequences.** `forwardGen` must give each layer a
  fresh `Cache{}` (or zero `len`). Reusing a warm cache across two sequences
  makes sequence 2 attend to sequence 1's keys → garbage. The reference
  allocates caches fresh per `forwardGen` call.
- **Cache capacity (`cap`).** Set it to the generation length *before* you
  start generating (the reference sets `blk.attn.cap = L` at the top of
  `forwardGen`). The buffer is preallocated to `cap`; if you forget, the
  append writes out of bounds or never allocates.
- **Position passed to RoPE in `forwardIncr`** is the *absolute* position
  `pos`, not 0. The full forward uses `0..L`; the incremental path must use the
  real index or the rotations won't match → cache-equality test fails by miles.
- **`head_dim = C / n_heads` must divide evenly**; `assert` it. A
  non-integer head dim silently misshapes the reshape.
- **The braceless-for rule** (lesson 00) bites hardest in the attention loops —
  the score accumulation `for (j) s += q[..]*k[..];` inside another loop is fine
  *if* the outer body is braced; a braceless `for (i) for (j) s += ...;` will
  not parse. Wrap assign-bodies in braces.
- **Scale `1/sqrt(Dh)`**, not `1/sqrt(C)` or `1/Dh`. Wrong scale → softmax
  saturates or collapses → gradients vanish/explode.
- **Causal mask `j > i → -inf` before softmax**, and skip those entries in the
  normalizer sum. Including them (even as 0) shifts the softmax.

## What good looks like

- RMSNorm, RoPE, SwiGLU, causal MHA each grad-check green.
- `forwardGen(ids)` matches `forward(ids)` to ~1e-6 — i.e. your incremental
  path is provably the same computation as your training path, just faster.
- The LM trains and its val loss drops below the unigram baseline; sampling
  produces locally-coherent (not random) text.
- You can explain, without notes, why RoPE goes on Q and K but not V, and why
  the cache makes generation O(L) per step instead of O(L²).

Lesson 10 takes this attention op and makes it (a) cheaper with **GQA** and
(b) memory-light with a **flash-style fused kernel** — the online-softmax
trick — and proves both still match this reference to <1e-3.