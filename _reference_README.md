# ⚠️ Teacher's edition — read this first

This directory (`/home/lenovo/ocd/llm`) is the **reference implementation** of the
Zero-to-Hero neural-network curriculum: a complete, verified-from-scratch
neural-network framework in Zig 0.16.0 — tensors, autograd, optimizers
(**Adam + Muon + Newton-Schulz**), a modern transformer (**RMSNorm / RoPE /
SwiGLU / GQA / KV-cache**), **Gated Delta Attention / DeltaNet** (the hero),
a **GPU backend** (cuBLAS + hand-written CUDA kernels), and training on
**FineWeb**. Everything is grad-checked against finite differences and verified
end-to-end by short training runs.

**It is the answer key.** If you are a student working through
[`lessons/`](lessons/README.md), **do not read the source in
[`src/znn/`](src/znn/) or [`apps/`](apps/) until you have attempted that
lesson's checkpoint and either passed it or gotten genuinely stuck after real
effort.** Reading the reference first is how you learn nothing.

The intended workflow:
1. Work in a **fresh, separate folder** (not here).
2. Copy [`lessons/starter/`](lessons/starter/) for your build template + the
   grad-check harness.
3. Follow [`lessons/README.md`](lessons/README.md) lesson by lesson. Each
   lesson names the exact checkpoint test you must pass.
4. Only open the matching `src/znn/*.zig` here to *unstick* yourself — then
   close it and write your own version.

## What's verified here (the gates, all green)

```
zig build test       # all unit + grad-check + end-to-end training tests
zig build bench -Dgpu   # CPU vs cuBLAS matmul + gated-delta kernel correctness/throughput
zig build run-13-capstone -- smoke     # tiny capstone: loss drops, checkpoint, sampling
zig build run-13-capstone -- sample    # load checkpoint + generate with temp/top-k/top-p
```

Gate summary: tensor core → autograd (grad-checked) → losses/SGD spiral → BPE
→ FineWeb pipeline → first LM (beats unigram) → Adam/Muon → modern transformer
+ KV-cache → GQA + flash-style fused attention → **Gated Delta / DeltaNet**
(recurrent == chunked-parallel <1e-3, grad-checked) → GPU backend (cuBLAS 1717×
matmul speedup, custom kernel matches CPU) → capstone (config, checkpoint,
RunPod, sampling).

The long/large FineWeb runs are the **student's** job on RunPod, using
[`scripts/runpod_capstone.sh`](scripts/runpod_capstone.sh) (smoke-tested
configs provided here).