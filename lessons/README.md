# Zero-to-Hero Neural Networks in Zig 0.16.0 — The Curriculum

You're going to write a neural-network framework from scratch — tensors,
autograd, optimizers (including **Muon**), a modern transformer with **RoPE /
SwiGLU / GQA / KV-cache**, **Gated Delta Attention / DeltaNet** (the hero),
a GPU backend (cuBLAS + hand-written CUDA kernels), and training on a real
corpus (**FineWeb**). In Zig 0.16.0. From zero.

This is a *firm-teacher* curriculum. That means:

- **I will not hand you the answers.** Each lesson gives you the objective, the
  intuition in several representations (algebra, diagram, analogy, a worked
  numeric example), scaffolded tasks, and the **exact checkpoint test your code
  must pass**. You write the code. The reference implementation exists (see
  below) but you are *not* allowed to look until you have attempted a step.
- **I will give you everything you need to figure it out.** Worked examples
  with concrete numbers, the math stated plainly, the pitfalls named up front,
  and a "what good looks like" rubric so you know when you're done.
- **The checkpoint is the arbiter.** Each lesson ends with a test. You run it
  (`zig test ...` or `zig build test`). Green = you understand it. If you can't
  get green, re-read the lesson, not the reference.

## How this folder relates to the reference implementation

There is a complete, verified reference implementation at
`/home/lenovo/ocd/llm` (the repo this `lessons/` folder lives inside). It is
the **teacher's edition**. Its structure:

```
src/znn/
  tensor.zig       # Gate 1
  autograd.zig     # Gate 2 (scalar Value + Tensor autograd + grad-check)
  loss.zig         # Gate 3
  tokenizer.zig    # Gate 4
  data.zig         # Gate 5
  nn.zig           # Gates 6,8,9,10 (embeddings, transformer, GQA, DeltaNet)
  optim.zig        # Gate 7 (Adam, Muon, Newton-Schulz, Hybrid, scheduler)
  integration.zig  # end-to-end training tests (the real gates)
  backend_cuda.zig # Gate 11 (cuBLAS + CUDA kernels)
  checkpoint.zig   # Gate 12
  kernels/znn_kernels.cu
apps/             # one runnable milestone per gate (02..13)
build.zig
```

**The rule:** you work in a *fresh, separate folder* (e.g. `~/myznn/`), building
your own version lesson by lesson. You may copy `starter/` (build template +
grad-check utility). You may **not** open `src/znn/*.zig` or `apps/*.zig` until
you have a working attempt at that lesson's checkpoint and either (a) it passes
or (b) you are genuinely stuck and have spent real effort. Reading the
reference *before* trying is how you learn nothing. Use it to *unstick*
yourself, not to skip the thinking.

## The path (13 lessons, each a "gate")

| # | Lesson | The gate (what proves you got it) |
|---|--------|-----------------------------------|
| 00 | Setup + the math you actually need | environment builds; you can run a tiny Zig program |
| 01 | Scalar autograd (micrograd) | `grad_check` on a scalar `Value` graph matches finite differences |
| 02 | Tensors (strided storage, broadcast, matmul) | matmul matches a naive triple loop; broadcasting works |
| 03 | Tensor autograd + grad-check | every op's analytic grad matches finite-diff (rel 1e-3) |
| 04 | Losses + training loop | a 2D spiral classifier trains to >95% |
| 05 | BPE tokenizer from scratch | encode/decode roundtrip; byte-level merges correct |
| 06 | FineWeb pipeline | deterministic batch streaming; train/val split |
| 07 | Embeddings + first LM | val loss < unigram baseline; samples are locally coherent |
| 08 | Optimizers (Adam, Muon, Newton-Schulz) | Muon update is (near-)orthogonal; small LM trains faster |
| 09 | Modern transformer (RMSNorm, RoPE, SwiGLU, KV-cache) | KV-cache forward == non-cached forward; samples coherent |
| 10 | Efficient attention (GQA, flash-style fused kernel) | max-abs diff < 1e-3 vs naive MHA |
| 11 | Gated Delta Attention / DeltaNet | grad-check passes; recurrent == chunked-parallel; a DeltaNet LM trains |
| 12 | GPU backend (cuBLAS + CUDA kernels) | GPU matmul matches CPU <1e-4 and is faster; kernel matches CPU |
| 13 | Capstone (config, checkpoint, RunPod, sampling) | end-to-end run: loss drops, checkpoint round-trips, sampling works |

Lessons build on each other. **Do not skip.** Gate 9 needs Gate 8; Gate 11
needs Gate 10; Gate 12 needs everything.

> **Note on numbering:** the **lesson number** (00–13, the `#` column) is *not*
> the same as the **test "Gate N" label** you'll see in the reference suite.
> The lessons are split more finely than the test gates (e.g. lesson 11 —
> Gated Delta — is verified by the `Gate 10: ...` tests; lesson 12 — GPU — by
> `Gate 11`; lesson 13 — capstone — by `Gate 12`). Each lesson's **Checkpoint**
> section names the *exact* test string to run, so you never have to guess —
> trust the checkpoint, not the global number.

## Prerequisites

- **You can code.** Any language. Zig specifics are taught inline.
- **Basic math.** You know what a derivative is (informally) and can read
  `y = mx + b`. Lesson 00 builds the rest: the chain rule with concrete numbers,
  what a matrix multiply actually *is*, and why floating point lies to you.
- **No deep linear algebra required.** We introduce exactly what we need
  (shapes, broadcasting, the matmul, an outer product, orthogonal matrices)
  when we need it. You will *learn* linear algebra by implementing it, not by
  being lectured at.

## How to use a lesson

1. Read it fully, including the worked numeric example. Don't skim.
2. Set up your fresh folder, copy `starter/`.
3. Do the tasks in order. Each task is small. Commit after each.
4. Run the checkpoint test. Iterate until green.
5. Only if stuck-after-real-effort: open the matching `src/znn/*.zig` to
   *unstick*, then close it and write your own.
6. Tick the box in `progress.md`.

The checkpoints reference real test names from the reference suite (e.g.
`test "Gate 10: ..."`) so you can see the exact bar.

## The firm-teacher contract, in one line

**I give you the map and a compass, not a ride.** Walk it yourself.