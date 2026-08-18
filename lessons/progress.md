# Progress — tick a box only when the checkpoint is green

Name: __________________  Start date: __________

- [ ] 00 — Setup + math. `zig version` = 0.16.0; `zig build` runs a hello exe.
- [ ] 01 — Scalar autograd. `grad_check` on a `Value` graph matches finite-diff.
- [ ] 02 — Tensors. matmul == naive triple loop; broadcasting add `[2,1]+[1,3]`.
- [ ] 03 — Tensor autograd. every op grad-checks (rel tol 1e-3, atol floor 1e-4).
- [ ] 04 — Training loop. spiral classifier >95% accuracy.
- [ ] 05 — BPE. encode/decode roundtrip; merges match reference on fixed corpus.
- [ ] 06 — FineWeb pipeline. deterministic batches; train/val split; no IO stalls.
- [ ] 07 — First LM. val loss < unigram baseline; samples locally coherent.
- [ ] 08 — Optimizers. Muon update near-orthogonal; Hybrid trains LM faster than Adam.
- [ ] 09 — Transformer. KV-cache fwd == non-cached fwd; loss sane; samples coherent.
- [ ] 10 — Efficient attention. GQA + fused kernel max|diff| < 1e-3 vs naive MHA.
- [ ] 11 — Gated Delta / DeltaNet. grad-check passes; recurrent == chunked-parallel <1e-3; DeltaNet LM trains & beats unigram.
- [ ] 12 — GPU backend. cuBLAS matmul matches CPU <1e-4 & faster; gated-delta kernel matches CPU <1e-3.
- [ ] 13 — Capstone. smoke run loss drops; checkpoint round-trips exact; temp/top-k/top-p sampling works.

## Notes (what surprised you, what you'd study next)

_

