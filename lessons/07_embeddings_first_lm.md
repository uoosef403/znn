# Lesson 07 — Embeddings and your first language model

## Objective

Build a language model that actually *learns* on real text: token embeddings +
positional embeddings, an MLP context model, the causal next-token objective,
training on the tokenized FineWeb corpus from lessons 05–06, and autoregressive
sampling. The bar it must clear: **validation loss strictly below the unigram
baseline** (a model that just predicts each token's marginal frequency). That
is the first honest proof that your stack — tensors, autograd, losses,
optimizer, data — composes into something that learns structure, not noise.

## Why this gate

Everything before this was infrastructure: you can compute a gradient, take a
step, fetch a batch. None of that *does* anything until you point it at data
and watch loss fall. This lesson is the first end-to-end "it learns" gate. It
also introduces the two ideas that the rest of the curriculum rests on:

1. **An embedding is a learned lookup table — and a lookup is a matmul.** Token
   id `7` maps to row 7 of a `[vocab, dim]` matrix. Equivalently: one-hot vector
   `e_7` (1 at position 7, 0 elsewhere) times the matrix. *Same math.* This
   unification ("lookup = `onehot @ W`") is why embeddings get gradients for
   free through autograd, and why the GPU path (lesson 12) treats them as
   matmuls. Internalize it now.
2. **The causal LM objective: predict the next token.** A window of `L` tokens
   predicts token `L+1`. No future leaks. This is *the* objective of every
   decoder LM you'll build, upgraded per-position in lesson 09.

## The model

`MLPLM` — a makemore-style context model:
```
ids: [B*L]            (row-major: row b, position j -> ids[b*L + j])
tok_emb = Embedding(vocab, dim)(ids)   -> [B*L, dim]   # one row per token id
pos_emb = Embedding(L,     dim)(pos)   -> [B*L, dim]   # one row per position 0..L-1
h       = tok_emb + pos_emb            -> [B*L, dim]   # broadcast add
flat    = reshape(h, [B, L*dim])       -> [B, L*dim]   # pure view, no copy
logits  = MLP([L*dim, H, H, vocab])(flat) -> [B, vocab]
```
It predicts **one token per window** (the token right after the window). The
transformer in lesson 09 will upgrade this to *per-position* prediction
(every position predicts its successor), but the one-per-window version is the
cleanest first "it learns" loop and already beats the unigram baseline.

### Embedding lookup = one-hot matmul (the key unification)

A `Embedding(vocab, dim)` is a `Var` holding a `[vocab, dim]` matrix `W`.
Looking up ids `[i_0, i_1, ...]` returns rows `[W[i_0]; W[i_1]; ...]` — shape
`[len(ids), dim]`. Two equivalent views:

- **Indexing:** `out[k] = W[ids[k]]` (gather rows).
- **Matmul:** build one-hot `E` of shape `[len(ids), vocab]` where
  `E[k, ids[k]] = 1`. Then `out = E @ W`, since `(E @ W)[k, d] =
  sum_v E[k,v]*W[v,d] = W[ids[k], d]`. Exactly the gathered row.

The backward is the same in both views: the gradient flows to *only the rows
that were looked up* (one-hot has a single 1 per row). So
`dW[ids[k]] += dout[k]`. Rows never touched in this batch get zero gradient —
that's "sparse update", and it's why you route embeddings to Adam (not Muon) in
lesson 08: the orthogonalization factor is ill-defined for a matrix whose rows
are updated at different rates.

**Worked example, tiny.** `vocab=3`, `dim=2`:
```
W = [[0.1, 0.2],     # row 0 = token 0
     [0.3, 0.4],     # row 1 = token 1
     [0.5, 0.6]]     # row 2 = token 2
ids = [2, 0]         # look up token 2 then token 0
```
- As gather: `out = [W[2], W[0]] = [[0.5,0.6],[0.1,0.2]]`.
- As one-hot matmul: `E = [[0,0,1],[1,0,0]]`, `E @ W = [[0.5,0.6],[0.1,0.2]]`.
  Same. The backward `dW[2] += dout[0]`, `dW[0] += dout[1]`, `dW[1]` untouched.

### The causal objective and the unigram baseline

For a window of `L` input tokens, the **target is the token at position L**
(the one right after the window). Loss is softmax cross-entropy over `vocab`
classes, one loss per window.

The **unigram baseline** is the dumbest possible model that isn't broken: it
predicts each token by its *training* marginal frequency `p(t)`, and scores the
val set at `loss = -mean(log p(t))` (Laplace-smoothed so unseen tokens don't
blow up to `-log 0`). If your model can't beat that, it has learned nothing
about *context* — it just memorized frequencies. **Beating the unigram baseline
is the gate.** It's the bar every later LM test uses too (GQA, DeltaNet).

## Multiple representations

- **Algebra:** `logits = MLP(reshape(E_tok @ W_tok + E_pos @ W_pos, [B, L*dim]))`.
- **Diagram:** `ids --tok_emb--> [B*L,C] --+pos_emb--> --reshape[B,L*C]--> MLP --> [B,V]`.
  The `+pos_emb` broadcasts a per-position row across the batch; the reshape is
  a free view (strides, lesson 02); the MLP flattens context into one vector.
- **Analogy:** the embedding table is a *codebook* — every token id is a
  codeword, and the row is its meaning-vector. Positional embeddings are
  *timestamps* painted onto each meaning so the model can tell "the 3rd word"
  from "the 1st word" even if the token is the same. The MLP is a reader that
  looks at the whole timestamped window and guesses the next word.
- **Physical:** without positional embeddings, swapping two tokens in the
  window produces the *same* flattened vector after summing — the model is
  order-blind. Positional embeddings inject order. (In lesson 09 you'll replace
  them with RoPE, which rotates rather than adds — same goal, cleaner
  extrapolation.)

## Tasks

1. **`Embedding`** — a `Var` parameter `[vocab, dim]`, init small (`N(0, 0.02)`).
   `forward(ids: []const usize)` gathers rows (use an `vIndexRows` autograd op,
   or build a one-hot and `vMatmul` — either works; the gather is cheaper).
   `params()` returns the one weight. Make sure the backward only updates
   looked-up rows (grad-check it).

2. **`MLP`** — a stack of `Linear` layers with activations
   (`relu`/`tanh`/`none`). `forward(x: *Var)` chains `linear -> act -> linear ...`.
   `params(alloc)` flattens all `[W, b]` from every layer.

3. **`MLPLM`** — wire `tok_emb + pos_emb` (broadcast add over `[B*L, dim]`),
   `reshape` to `[B, L*dim]`, then `MLP([L*dim, H, H, vocab])`. Build the
   per-window position index `0..L-1` repeated across the batch. `forward(ids)`
   returns logits `[B, vocab]`; `params(alloc)` returns all trainable params.

4. **Training loop** (reuse your Adam from lesson 08 / SGD from 04): fetch a
   batch from the FineWeb dataset (lesson 06), ids = `b.x`, target = the token
   *after* each window = `b.y[bi*L + (L-1)]` (one target per window). Forward,
   `softmaxCE`, `backprop`, `step`. Reset the iterator at epoch boundaries.

5. **Unigram baseline:** count training token frequencies, Laplace-smooth,
   compute `-mean(log p(t))` over val tokens. Print it.

6. **Val loss:** run the model over val batches (no grad), average per-window
   next-token CE. Assert `model_val < unigram_val`.

7. **Sampling:** greedy — seed with `L` tokens (pad/clip a prompt to length
   `L`), `forward`, take argmax of the last row, append, slide the window one
   step, repeat. Print the continuation.

## Checkpoint: the test that must pass

`test "first LM: MLPLM val loss beats the unigram baseline"` (in the reference
`integration.zig`). The reference runs on a synthetic bigram sequence (V=16,
L=4, dim=24, hidden=64, 4000 tokens, 400 Adam steps) and asserts
`model_val < unigram_val`. Your bar: the same inequality, on your dataset. Run
`zig test src/integration.zig`. On real FineWeb (lesson 06's corpus), also
train briefly and confirm val loss drops below the unigram line and that a
greedy sample is *locally coherent* (repeats real words, not pure noise — full
fluency comes with the transformer, but "the the the" → "the quick brown" is
the direction).

## Common pitfalls

- **Embedding not a `Var.parameter`** → no grad, learns nothing. The table must
  be a trainable parameter (with a `.grad`), not a constant.
- **Positional embedding off-by-one** → position index must be `0..L-1`
  *repeated per row of the batch*, shape `[B*L, dim]` to add against `tok_emb`.
  Forgetting the repetition, or using `1..L`, breaks ordering silently.
- **Target shift** → the target for window `b` is the token *after* the window:
  `b.y[bi*L + (L-1)]`, not `b.x[bi*L + L-1]` (that's the last input, leakage-free
  but trivial) and not a per-position target (this model predicts one token
  per window). Get the index exactly right; the loss looks "fine" with the
  wrong shift but the model learns a degenerate mapping.
- **Reshape that copies when it shouldn't (or doesn't when it should)** → the
  `[B*L, dim] -> [B, L*dim]` reshape is a *pure view* (same buffer, new
  shape/strides) only if the source is contiguous in the right order. If your
  tensor's strides don't permit the view, you must copy; if you copy when you
  could view, you waste memory. Know which your `reshape` does.
- **Sampling out-of-vocab** → the seed prompt must BPE-encode (lesson 05) into
  ids all `< vocab`. Pad/clip the encoded prompt to exactly `L` tokens before
  the first forward.
- **Window in generation** → after each predicted token, *slide* the window
  (drop the oldest, append the new) and re-run forward on the full `L`-length
  window. Don't feed a growing buffer; the model was trained on fixed-`L`
  windows. (The KV cache in lesson 09 fixes the wasteful recompute here.)
- **Forgetting to reset the train iterator** at epoch end → `nextBatch` returns
  null; handle it by `reset()` and re-fetching, don't crash.
- **f32 loss magnitude** → average the loss (don't sum a huge window loss) so
  the per-token signal stays above f32 noise (lesson 00).

## What good looks like

- `model_val < unigram_val` holds, robustly, on more than one seed.
- You can explain why a lookup is a matmul and *why* that makes embedding
  gradients sparse — and predict, before reading lesson 08, that this sparsity
  is *why* embeddings won't go through Muon.
- A greedy sample from a briefly-trained model contains real words from the
  corpus, not gibberish — the model has latched onto local statistics. Full
  coherence is lesson 09's job; "it's reading the text, not the labels" is the
  bar here.

Lesson 08 makes the optimizer that trains *this* model faster — and introduces
Muon, the first genuinely modern piece of the stack.