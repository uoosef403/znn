# Lesson 06 — The FineWeb data pipeline

## Objective

Fetch a slice of **FineWeb-Edu** to a raw-bytes file (Python, one-time, data
prep only), then — in **Zig** — turn that text into next-token-prediction
batches: tokenize with your BPE (lesson 05), split train/val leakage-free, cut
sliding-window `(input, target)` pairs with the causal one-token shift,
shuffle start indices deterministically, and stream minibatches with zero IO
in the hot loop.

This is the conveyor belt that feeds every language model from lesson 07 on.
If it's right, training is pure compute. If the split leaks or the shift is
off-by-one, your "val loss" lies to you and you won't know why.

## Why two languages

The *one-time* fetch of a multi-GB dataset is exactly what Python's HuggingFace
`datasets` library is built for (streaming, no full-corpus RAM). The *runtime*
— tokenizing, batching, training — is what Zig is for. The contract between
them is a single dumb file: `data/text.bin` = raw UTF-8 text, newline-separated
documents. Zig doesn't care how it got there. This keeps your framework pure-Zig
at runtime while you still get the real FineWeb-Edu corpus. (On a 6 GB machine,
stream the fetch with a `--bytes` budget; for "real deal" scale, run the fetch
on RunPod — the file is just bytes.)

## Part A — fetch (Python, run once)

`scripts/fetch_fineweb.py` streams `HuggingFaceFW/fineweb-edu` and writes each
example's `text` field, newline-separated, to `data/text.bin` until a byte
budget. It has a `--fallback` that emits a bundled prose corpus so the whole
downstream pipeline is smoke-testable offline. You will *not* write the
fetcher's HF logic yourself — use the provided script — but you should
understand the contract: **one document per line, raw bytes, no framing**.

> **Why a byte budget, not a doc count:** streaming with a byte cap keeps the
> file under your disk/RAM limit and works on a 6 GB box. For a real run, raise
> `--bytes` or fetch on a bigger box.

## Part B — tokenize → ids → splits (Zig)

1. Read `data/text.bin` with the 0.16 IO API:
   `std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(1 << 30))`.
2. BPE-encode the whole text to `[]u32` ids. (Train the BPE on a *slice* of the
   corpus — training is O(V·N) per merge — but encode the *whole* corpus; BPE
   falls back to bytes for unseen pairs, so coverage is total.)
3. **Split by position, not by random scatter.** The val block is a contiguous
   *tail* of the token stream: `train = ids[0..train_n]`, `val =
   ids[train_n..]`, where `train_n = N - round(val_frac·N)`. This is the
   leakage-free split: no val window can reach back into train. A random
   per-example split shuffles near-duplicate passages across the boundary and
   inflates val loss as a fake metric of generalization.

## Part C — the causal window and the shift

A language model is trained to predict the next token. A window starting at
index `t`:
```
input  x = tokens[t .. t+L]          (L tokens)
target y = tokens[t+1 .. t+L+1]       (L tokens, shifted by one)
```
So position `j` in the input predicts position `j` in the target, which is the
*next* token. A split of `N` tokens yields `N - L` windows (start `t ∈
[0, N-L-1]`; the last window needs `t+L+1 ≤ N`, i.e. `t ≤ N-L-1`). The targets
are still token ids (`[]u32`), not floats — they're *indices* the loss indexes
into (lesson 04's `softmaxCE` takes `[]const usize`). The embedding layer
(lesson 07) lifts them into the f32 graph.

**Worked example.** 20 tokens `100..119`, `seq_len = 4`, `batch = 3`, `val_frac = 0`.
```
N=20, L=4  ->  numSeqs = 20-4 = 16 windows, numBatches = 16/3 = 5
window at t=0:  x = [100,101,102,103]   y = [101,102,103,104]
window at t=1:  x = [101,102,103,104]   y = [102,103,104,105]
...
```
The invariant the tests pin: **`y[i] == x[i] + 1`** for every row (because the
ids were a contiguous ramp here; in real text it's "`y[i]` is the token after
`x[i]` in the original stream"). If your `y` is `x` shifted by the wrong amount
or copied from the wrong window, this catches it instantly.

## Part D — deterministic shuffling + streaming

Generate the window-start indices `[0, N-L)` and **Fisher-Yates shuffle** them
with a *seeded* PRNG. Same seed → same order → reproducible `zig build` runs.
This is what makes bugs reproducible and the determinism test pass.

A `Split` holds `tokens`, `seq_len`, `batch_size`, the shuffled `starts`, and a
`pos` cursor. `nextBatch`:
```
if pos + B > starts.len: return null       // epoch end
for bi in 0..B:  t = starts[pos+bi]
                 x[bi] = tokens[t .. t+L]   y[bi] = tokens[t+1 .. t+L+1]
pos += B
return batch
```
`reset()` sets `pos = 0` for a new epoch; one full epoch returns exactly
`numBatches()` batches then `null`. The whole corpus lives in memory once, so
`nextBatch` is pure pointer arithmetic — **no IO in the loop**. (For a corpus
that doesn't fit in RAM, memory-map the token file; the start-index scheme is
unchanged.)

## Multiple representations

- **Algebra:** window `t` = `(tokens[t..t+L], tokens[t+1..t+L+1])`; count =
  `N - L`; batch `b` = rows for `starts[pos..pos+B]`.
- **Diagram:** the token stream as a ruler; a window is an L-wide slider; the
  target slider is the input slider nudged one token right. Batches grab `B`
  non-adjacent windows (shuffled) per step.
- **Analogy:** a conveyor belt of tokens; each window is "read L, predict the
  next L." Shuffling start indices = randomizing *which* L-frames come next,
  not *what order tokens appear within* a frame (that's always left-to-right).
- **Physical:** the split is a single cut of the stream into two contiguous
  pieces; val never reaches left of the cut, so it can't "see" train data.

## Tasks

1. **`Config` + `Batch` + `Split`.** `Split` has `tokens, seq_len,
   batch_size, starts: []usize, pos`. Methods: `numSeqs` (`N-L`, 0 if `N ≤ L`),
   `numBatches` (`numSeqs / B`), `reset`, `nextBatch`.

2. **`Dataset.fromIds`**: copy the ids, slice into train/val by the contiguous
   tail split, build shuffled `starts` for each. Use a *different* seed for val
   (e.g. `seed ^ golden`) so its order differs from train's. `fromText` /
   `fromTokenFile` tokenize then delegate.

3. **`makeStarts`**: Fisher-Yates with a seeded `DefaultPrng`; the result is
   the reproducible window order.

4. **`nextBatch`**: build flat `x`, `y` of length `B·L`; fill row-by-row from
   `starts`; the `+1` shift. Return `null` at epoch end.

5. **Tests** (mirroring the reference suite):
   - counts + no out-of-range starts (`Split: counts...`);
   - batches are the shifted windows, `numBatches` correct, exactly that many
     then `null` (`Dataset: batches are the shifted causal windows...`,
     `Dataset: full epoch returns numBatches batches then null; reset re-epochs`);
   - same seed → same starts and same first batch (`Dataset: deterministic
     order`);
   - train/val disjoint, no window crosses the cut (`Dataset: train and val do
     not share windows`).

## Checkpoint: the tests that must pass (all in `data.zig`)

- `test "Split: counts and no out-of-range starts"`
- `test "Dataset: batches are the shifted causal windows (x[t..t+L], y[t+1..t+L+1])"`
- `test "Dataset: deterministic order (same seed -> same batches)"`
- `test "Dataset: train and val do not share windows (leakage-free split)"`
- `test "Dataset: full epoch returns numBatches batches then null; reset re-epochs"`

The bar: all five green. The key invariants — `y` is `x` shifted by exactly one
token in the *original* stream; same seed → byte-identical batches; no train
window reaches into val.

## Common pitfalls

- **Off-by-one in the causal shift.** `y = tokens[t..t+L]` (no shift) or
  `y = tokens[t+2..]` (shift of 2) both train silently wrong objectives. The
  ramp test (`y[i] == x[i]+1`) catches it; without it, your LM learns to copy
  or skip and val loss looks "fine" while generation is gibberish.
- **Window count `N - L` vs `N - L + 1`.** You need `t + L + 1 ≤ N` for the
  target, so the last valid start is `N - L - 1`, giving `N - L` windows. Off
  by one here → an out-of-bounds read on the last window.
- **Random per-example split.** Scatters near-duplicate passages across the
  cut; val loss becomes a lie. Always split by contiguous position.
- **Non-deterministic shuffling.** If your PRNG isn't seeded (or you re-seed
  randomly), `zig build` runs differ and bugs become unreproducible. Seed it,
  and pin determinism with a test.
- **Same seed for train and val.** Harmless functionally, but it makes the two
  orders identical; use a different seed for val so a buggy test that compares
  them doesn't trivially pass.
- **Reading the whole corpus vs streaming.** Loading `data/text.bin` into RAM
  once is fine for ≤ a few GB (the whole point of the byte budget). For
  larger, memory-map; don't re-read per batch.
- **Zig 0.16 read API.** `readFileAlloc` needs `.limited(...)` — an
  `AllocLimite`. Omitting it is a compile error, not a runtime one. And
  `std.fs.cwd()` is gone — use `std.Io.Dir.cwd()` with `std.Options.debug_io`.
- **Token ids are `u32`, not `f32`.** They're indices. The loss indexes into
  them; the embedding layer (next lesson) lifts them to floats. Don't store
  batches as `f32` — you'll lose the ability to use them as indices and pay
  4× memory for nothing.

## What good looks like

- All five `data.zig` tests green; the ramp invariant `y[i] == x[i]+1` holds
  on every row of every batch.
- Two `Dataset`s on the same ids + seed produce byte-identical `starts` and
  identical first batches.
- A 10% val split: train windows all lie in `[0, train_n)`, val windows all in
  `[train_n, N)`; none cross the cut.
- One full epoch yields exactly `numBatches()` batches then `null`; `reset`
  re-epochs.
- Your `nextBatch` does no file IO — pure pointer math into the in-memory token
  buffer.

Lesson 07 finally builds a model: embeddings + an MLP context LM that eats
these batches and predicts the next token.