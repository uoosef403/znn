# znn — Zero-to-Hero Neural Networks in Zig 0.16.0

A from-scratch neural-network framework — tensors, autograd, optimizers
(**Adam + Muon + Newton-Schulz**), a modern transformer (**RMSNorm / RoPE /
SwiGLU / GQA / KV-cache**), **Gated Delta Attention / DeltaNet** (the hero), a
**GPU backend** (cuBLAS + hand-written CUDA kernels), and training on
**FineWeb** — all written from scratch in Zig 0.16.0, and verified by
finite-difference gradient checks + short end-to-end training runs.

## Two ways to use this repo

1. **Study it (recommended start):** read [`lessons/README.md`](lessons/README.md)
   and work through the 14 lessons (`lessons/00_setup_math.md` …
   `lessons/13_scaling_capstone.md`) in a *fresh, separate folder*, rebuilding
   everything yourself. Each lesson is firm-teacher style: objectives, intuition
   in multiple representations, a worked numeric example, scaffolded tasks (no
   handed-over answers), and a **checkpoint test you must pass**. Track yourself
   in [`lessons/progress.md`](lessons/progress.md).

2. **Read the reference (teacher's edition):** the verified implementation
   lives in [`src/znn/`](src/znn/) and [`apps/`](apps/). It is the **answer
   key** — see [`_reference_README.md`](_reference_README.md). Don't open it
   until you've attempted a lesson's checkpoint.

## Verify it works

```bash
zig build test                       # all grad-checks + end-to-end training tests
zig build bench -Dgpu                # CPU vs cuBLAS matmul + gated-delta kernel
zig build run-13-capstone -- smoke   # tiny capstone: loss drops, checkpoint, sampling
zig build run-13-capstone -- sample  # load checkpoint + generate (temp/top-k/top-p)
```

Requires Zig 0.16.0. The GPU bench additionally needs `nvcc` + cuBLAS
(`-Dgpu`). Long/large FineWeb runs are launched via
[`scripts/runpod_capstone.sh`](scripts/runpod_capstone.sh) on a high-core
instance (the training loop is CPU-autograd; the GPU backend verifies matmul +
the gated-delta kernel — porting the whole autograd graph to GPU is the named
frontier in lesson 12).

## What's inside

```
src/znn/        the framework: tensor, autograd, loss, optim, nn, tokenizer,
                data, integration (the gate tests), backend_cuda, checkpoint
src/kernels/    hand-written CUDA kernels (nvcc-compiled, linked from Zig)
apps/           one runnable milestone per gate (02..13)
scripts/        fetch_fineweb.py (one-time data prep), runpod_capstone.sh
lessons/        the curriculum (14 lessons + README + progress + starter/)
build.zig
```

## The gates (all green)

Tensor core → autograd (grad-checked) → losses/SGD spiral → BPE → FineWeb
pipeline → first LM (beats unigram) → Adam/Muon → modern transformer + KV-cache
→ GQA + flash-style fused attention → **Gated Delta / DeltaNet** (recurrent ==
chunked-parallel <1e-3, grad-checked) → GPU backend (cuBLAS matmul 1717×
speedup, custom kernel matches CPU) → capstone (config, checkpoint, RunPod,
sampling).

Start at [`lessons/README.md`](lessons/README.md).