# Lesson 13 — The capstone: config, checkpoint, sampling, scaling

## Objective

Tie every gate into one runnable artifact: a **configurable** language model
that trains on FineWeb, **checkpoints** its weights to disk and **resumes**,
**samples** autoregressively with temperature / top-k / top-p, and has a clear
path to a real RunPod run. You prove the whole stack works end-to-end: loss
drops, a checkpoint round-trips *exactly*, and sampling produces text.

This is the graduation. Nothing here is a new neural-network idea — it is
*engineering* on top of the components you already built and verified.

## Why this lesson exists

A model that trains but can't be checkpointed is a model you can only run once.
A model that can't sample is a loss curve, not a language model. A config that's
hardcoded in `main` can't be scaled. The capstone is where "I built a
transformer" becomes "I have a small, reproducible training system."

## The model (what you assemble)

- **Architecture:** the GQA transformer LM from lesson 09 (RMSNorm + RoPE +
  SwiGLU + GQA + KV-cache). This is your strongest verified model.
- **Optimizer:** the **Hybrid** from lesson 08 — Muon for the 2D matrix weights,
  Adam for the token-embedding table, the output head, and all 1D params
  (biases, RMSNorm gains). Route the embedding + head to Adam explicitly because
  the Newton-Schulz factor is ill-defined on sparse-updated / projection
  matrices.
- **Schedule:** linear **warmup → cosine decay** (lesson 08's `Scheduler`), plus
  **global-norm gradient clipping** (`clipGradNorm`) before each step. Scale the
  two optimizer groups *independently* by the schedule factor: `opt.muon.lr =
  base_muon * f; opt.adam.lr = base_adam * f;` (Muon wants a higher base lr than
  Adam — a single shared `setLr` would flatten them; set the fields directly).
- **Config:** one struct holding *every* hyperparameter (architecture, data,
  optimization, sampling, IO paths). Two presets: a tiny **SMOKE** (finishes in
  seconds, verifiable locally) and a bigger **REAL** (the RunPod starting point).
- **Checkpointing:** a small module (`checkpoint`) that serializes the full
  parameter list to a binary file and restores it.
- **Sampling:** autoregressive generation over the **KV-cache forward path**
  (lesson 09's `forwardGen`), with temperature / top-k / top-p.

### The checkpoint format (and its honest limitation)

Serialize in the order `model.params()` returns, with shapes so the loader can
verify a structural match:
```
b"ZNN1"              4-byte magic
u32   param_count
per parameter:
  u8   ndim
  u64  shape[ndim]     (little-endian)
  f32  data[numel]     (native-endian; x86 == little-endian)
```
Save and load use the *same model config*, so `params()` returns parameters in
the same order on both sides — there is **no name table**. The loader verifies
each param's shape matches before copying (catches config drift).

**What it deliberately does NOT store:** optimizer state (Adam moments, Muon
momentum buffers), the RNG state, and the data-stream position. So **resume
restarts the optimizer moments** from zero — the model weights (the expensive
part) are fully restored, but the first few resumed steps are slightly noisier
as Adam's running stats re-warm. A production trainer persists optimizer state
too; for a from-scratch framework, weights-only is the right first cut.

### Sampling: three knobs, each with a worked example

Given the logits `l[V]` for the next token:

**Temperature** `T` (sharpen / flatten). Divide logits by `T` before softmax.
`T → 0` → argmax (greedy); `T = 1` → the model's raw distribution; `T > 1` →
flatter, more random. *Worked:* logits `[2, 1, 0]` softmax to ≈ `[0.67, 0.24,
0.09]`; at `T = 0.5` they become `[4,2,0]` softmax ≈ `[0.94, 0.06, 0.0]` —
sharper, picks the top more. Implement: `l[i] /= T`, and treat `T == 0` (or
`top_k == 1`) as greedy argmax to avoid `÷0`.

**Top-k** (truncate to the k best). Keep only the `k` highest logits, set the
rest to `−∞` before softmax (so they get zero probability). *Worked:* `V=5`,
`top_k=2` on logits `[1, 3, 2, 0, 4]` → keep ranks for logits `4` and `3`, mask
the other three → sample only between those two. Cuts the long tail of bad
tokens. Sort an index array by logit descending, mask everything past rank `k`.

**Top-p / nucleus** (truncate to the smallest set summing to ≥ p). Sort by
probability descending, walk the cumulative sum, keep the smallest prefix with
`cum ≥ p`, zero the tail, **renormalize over the nucleus**. *Worked:* probs
`[0.5, 0.3, 0.15, 0.04, 0.01]`, `top_p = 0.85`: cum reaches `0.8` after 2, then
`0.95 ≥ 0.85` after 3 → keep the first 3, renormalize to `[0.5, 0.3,
0.15]/0.95`. Unlike top-k, the *number* of candidates adapts to the
distribution's shape. Order matters: apply temperature first, then top-k, then
top-p (top-p over an already-truncated set).

Then **softmax over the survivors** (max-subtract for stability, treat `−∞` as
0) and a **categorical draw** by inverse-CDF against a uniform random.

### SMOKE vs REAL, and reading argv

Two configs, one app. The SMOKE config (`dim=64, 2 layers, GQA 4/2, seq=16,
50 steps`) finishes in seconds on CPU — it is sized to *verify the gate
locally*. The REAL config (`dim=384, 8 layers, GQA 6/2, seq=256, 4000 steps`)
is the RunPod starting point and is **not** run locally.

The mode (`train` / `resume` / `sample` / `smoke` / `real`) comes from argv. In
Zig 0.16 the main signature takes the process init:
```zig
pub fn main(init: std.process.Init.Minimal) void {
    var it = std.process.Args.Iterator.initAllocator(init.args, std.heap.smp_allocator) catch return;
    defer it.deinit();
    _ = it.next();                       // exe name
    if (it.next()) |arg| { ... }         // the mode
}
```
Keep the mode and the config *independent*: `sample` should sample from the
*smoke* checkpoint you just trained, not demand the (nonexistent) real one. So
the mode selects behavior *within* the active config; only a `real` arg swaps to
the heavy config.

### RunPod: a launch script, and an honest note on the GPU

Provide `scripts/runpod_capstone.sh` with `prep` (fetch FineWeb → `data/text.bin`
via the Python data-prep script, one-time), `train`, `resume`, `sample`, and
`bench` modes. **Pick a high-core CPU instance.** The capstone's *training loop*
builds the autograd graph on CPU — lesson 12's GPU backend verifies matmul +
the gated-delta kernel in isolation, but the *whole autograd graph* is not yet
on the GPU. So an A100/H100 helps only once you port the graph across (the
frontier exercise). Until then: more CPU cores + RAM is what speeds the REAL
run. The script documents this and points at `zig build bench -Dgpu` for the
GPU smoke.

## Tasks

1. **`checkpoint.zig`** with `save(alloc, params, path)` and
   `load(alloc, params, path)`. Build an `ArrayList(u8)`; write magic + u32
   count; per param write `u8 ndim`, `u64 shape[]` (little-endian via
   `std.mem.writeInt`), then the raw `f32` bytes (`@ptrCast` the `.data.ptr` to
   `[*]const u8`). Load reads back, verifies magic + count + each shape, and
   `@memcpy`s the bytes into `p.value.data`. Use `std.Io.Dir.cwd()` +
   `std.Options.debug_io` (lesson 00).

2. **A checkpoint round-trip test.** Train model A one step (so weights leave
   the init distribution), save; build model B with a *different* seed;
   load A's checkpoint into B; forward both on the same input; assert
   `max|diff| < 1e-5` (ideally exactly 0). This proves resume and sample rest on
   a correct foundation.

3. **The capstone app:** the `Config` struct + SMOKE/REAL presets, the training
   loop (Hybrid + scheduler + grad clip), checkpoint-on-best-val + periodic
   checkpoint, the sampling function (temp/top-k/top-p over `forwardGen`),
   and the `main(init)` arg parsing. Log loss every N steps; save the tokenizer
   too (load if present, else train+save) so resume/sample don't retrain BPE.

4. **The RunPod script** with the five modes and the high-core-CPU note.

## Checkpoint: the tests/bars that must pass

- **`Gate 12: checkpoint round-trip reproduces forward exactly`** — a fresh
  model loaded from a trained model's checkpoint produces identical forward
  output. Bar: `max|diff| < 1e-5` (reference hits exactly `0.000000`). In
  `checkpoint.zig`; run `zig test src/znn/checkpoint.zig`.
- **The smoke run** (`zig build run-13-capstone`): loss drops monotonically
  over the 50 steps (reference: train 6.5 → 1.8, val 6.3 → 2.2) and writes
  `data/ckpt_smoke.bin`.
- **`zig build run-13-capstone -- sample`**: loads the smoke checkpoint and
  prints generated text with temp/top-k/top-p applied. (The smoke model is
  barely trained, so the text is gibberish — the *gate* is that sampling
  *runs and applies the knobs*, not that the prose is good.)
- The full `zig build test` stays green (this adds no regressions).

## Common pitfalls

- **Resume does not restore optimizer state.** Documented, not a bug. The
  checkpoint stores weights only; Adam's `m,v` and Muon's buffer restart at
  zero. If you want true resume, extend the format to persist them (and the
  data-stream position + RNG) — a real follow-up.
- **`resume` is a reserved word in Zig 0.16** (it is the async-resume keyword).
  You cannot name an enum field or a variable `resume`. Use `cont` / `.cont`
  for the mode and `do_resume` for the flag.
- **`main(init: std.process.Init.Minimal)`** is how you get argv in 0.16, not a
  bare `pub fn main() void`. The arg comes from
  `std.process.Args.Iterator.initAllocator(init.args, ...)`. If you keep the
  no-arg signature you simply can't read the mode.
- **Sample mode must reuse the smoke checkpoint, not demand the real one.**
  If `sample` swaps to the REAL config, it will look for `data/ckpt.bin` which
  doesn't exist locally and fail. Keep mode and config independent.
- **Top-p must renormalize over the nucleus.** After zeroing the tail, the
  probabilities no longer sum to 1 — renormalize the survivors, or your
  categorical draw is biased toward the head. Same for top-k.
- **Temperature 0 / top_k 1 → greedy**, not `÷0`. Guard it.
- **Order of sampling operations:** temperature → top-k → top-p → softmax →
  draw. Doing top-p before top-k double-truncates in a confusing way.
- **The `atol` floor in any new grad-check** (lesson 00) — and here, no new
  grad-check is needed; the checkpoint test is a *forward* equality, not a
  gradient check, so `1e-5` absolute is fine.

## What good looks like

- `zig build run-13-capstone` finishes in seconds, prints a loss curve that
  drops, writes a checkpoint.
- `zig build run-13-capstone -- sample` loads that checkpoint and prints
  text; changing `temperature`/`top_k`/`top_p` visibly changes the output
  (greedy when `top_k=1`, wilder when `temperature` is high).
- The checkpoint round-trip test is exactly 0 (or `< 1e-5`).
- `scripts/runpod_capstone.sh` documents the honest CPU-bound nature of the
  training loop and points at the GPU frontier — and you can explain *why*
  wiring the autograd graph onto the GPU is the next step (and what it would
  take: a Compute interface the ops dispatch through, like the matmul/kernels
  in lesson 12, but for the whole graph).

You started in lesson 00 with `sum = 10` and the chain rule by hand. You are
ending with a checkpointable, sampleable, GQA-attention language model trained
on a real corpus, a Muon optimizer, gated-delta attention implemented and
parallelized, and a GPU backend — all written from scratch in Zig 0.16.0, every
backward grad-checked against finite differences. That is zero to hero.

What to study next (the honest "now go further" list): **muP** (learning-rate
transfer across model sizes), **RoPE variants** (NTK-aware, long-rope), the
**chunked-parallel gated-delta backward** (you did the forward; the analytic
backward is the capstone exercise of the hero), **porting the autograd graph
to GPU** (the frontier lesson 12 points at), and **mixed precision + tensor
cores** on an A100/H100. The framework you built is the scaffolding for all of
them.