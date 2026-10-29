# znn

A neural network framework and learning curriculum written in Zig. The project
builds from tensors and automatic differentiation to a transformer and a small
language model.

## Contents

- Tensors, automatic differentiation, loss functions, and optimizers
- A BPE tokenizer and a FineWeb data pipeline
- Transformer components, including gated delta attention
- A CUDA backend for matrix multiplication and gated delta kernels
- Fourteen lessons with exercises and checkpoint tests

The training loop uses CPU automatic differentiation. The CUDA backend is used
by the GPU benchmark; it does not run the full training loop.

## Start

Read the [lessons](lessons/README.md) and complete the exercises in a separate
directory. The implementation in `src/znn/` and `apps/` is the reference
solution. See the [reference guide](_reference_README.md) for details.

## Build and test

The project requires Zig 0.16.0.

```sh
zig build test
zig build run-13-capstone -- smoke
```

To run the CPU and GPU benchmark, install the CUDA toolkit and cuBLAS, then run:

```sh
zig build bench -Dgpu
```

The `scripts/` directory contains tools to prepare FineWeb data and launch a
training run. See the lessons for their prerequisites and usage.
