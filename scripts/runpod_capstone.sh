#!/usr/bin/env bash
# scripts/runpod_capstone.sh — launch the capstone training run on RunPod.
#
# This is the "real deal" config: the REAL preset in apps/13_capstone.zig
# (dim=384, 8 layers, GQA 6/2 heads, seq_len=256, ~4000 steps) on a bigger
# FineWeb slice. It is a CPU-autograd training run — the framework's training
# loop builds the autograd graph on CPU (Gate 11's GPU backend is verified for
# matmul + the gated-delta kernel via `zig build bench -Dgpu`, but porting the
# whole autograd graph to GPU is the next frontier, lesson 12). So pick a
# RunPod instance with many CPU cores + RAM; an A100/H100 helps only once the
# GPU path is wired into training. Until then: high-core CPU.
#
# PREREQ on the RunPod instance (do once):
#   sudo apt-get update && sudo apt-get install -y build-essential python3.13
#   curl -LsSf https://astral.sh/uv/install.sh | sh   # for the one-time data prep
#   # Zig 0.16.0:
#   curl -L https://ziglang.org/download/0.16.0/zig-linux-x86_64-0.16.0.tar.xz | \
#     sudo tar -xJ -C /usr/local --strip-components=1
#   zig version   # -> 0.16.0
#
#   # CUDA (only if you also want `zig build bench -Dgpu` to work on the box):
#   sudo apt-get install -y nvidia-cuda-toolkit libcublas-dev
#
# USAGE
#   bash scripts/runpod_capstone.sh prep    # fetch FineWeb -> data/text.bin (once)
#   bash scripts/runpod_capstone.sh train   # fresh REAL run (ReleaseFast)
#   bash scripts/runpod_capstone.sh resume  # load data/ckpt.bin, keep training
#   bash scripts/runpod_capstone.sh sample  # load ckpt, generate text
#   bash scripts/runpod_capstone.sh bench   # `zig build bench -Dgpu` smoke
#
# All training runs checkpoint to data/ckpt.bin (and the tokenizer to
# data/bpe.bin). Resuming restarts optimizer state (the checkpoint stores
# weights only — see src/znn/checkpoint.zig).

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-train}"

case "$MODE" in
  prep)
    echo "== fetching FineWeb-Edu slice -> data/text.bin =="
    mkdir -p data
    uv run scripts/fetch_fineweb.py --out data/text.bin --bytes $((1<<28))  # 256 MB
    echo "== done. now: bash scripts/runpod_capstone.sh train =="
    ;;
  train)
    echo "== capstone TRAIN (REAL config, ReleaseFast) =="
    zig build run-13-capstone -- train -Doptimize=ReleaseFast
    ;;
  resume)
    echo "== capstone RESUME from data/ckpt.bin =="
    zig build run-13-capstone -- resume -Doptimize=ReleaseFast
    ;;
  sample)
    echo "== capstone SAMPLE from data/ckpt.bin =="
    zig build run-13-capstone -- sample -Doptimize=ReleaseFast
    ;;
  bench)
    echo "== GPU smoke: cuBLAS matmul + gated-delta kernel vs CPU =="
    zig build bench -Dgpu
    ;;
  *)
    echo "unknown mode: $MODE"; echo "use: prep | train | resume | sample | bench"; exit 1
    ;;
esac