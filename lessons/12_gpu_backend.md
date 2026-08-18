# Lesson 12 — GPU backend: cuBLAS + CUDA kernels from Zig

## Objective

Make your framework actually fast on hardware. You will link cuBLAS (for
matmul) and compile your own CUDA kernels (nvcc → object file → linked into the
Zig executable) — including a GPU mirror of the gated-delta recurrent forward
from lesson 11 — and prove each GPU op matches your CPU reference in numbers and
beats it in wall-clock. Everything through Zig 0.16's build system and `@cImport`.

The GPU path is **opt-in** (`-Dgpu`) and *isolated*: the CPU framework never
touches a CUDA header. The backend lives in one file imported only by the GPU
build path.

## Why this matters

CPU matmul of `512³` is ~200 ms on this machine. cuBLAS on the GTX 1660 Ti does
it in ~0.13 ms — a **~1700× speedup**. Every linear layer, every attention
projection, every score is a matmul; putting matmul on the GPU is the single
highest-leverage optimization in the whole stack. And the gated-delta recurrence
is embarrassingly parallel across `(batch, head)` — one CUDA kernel turns a
serial loop over `L` (per head) into thousands of concurrent threads.

## The model

### The link pattern (the whole game)

Zig cannot call a `__global__` CUDA kernel directly — kernels are launched with
the `<<<grid,block>>>` syntax, which is CUDA-specific. So every kernel gets a
**host launcher**: an `extern "C"` function that Zig *can* call, which performs
the `<<<>>>` launch. The chain:

```
znn_kernels.cu:                       build.zig:                    backend_cuda.zig:
  __global__ void k(...) {...}         nvcc -c -O2 -arch=sm_75       extern fn launch_k(...);
  extern "C" void launch_k(...)        k.cu  ->  k.o   (nvcc step)   ... call launch_k(...) ...
     { k<<<grid,blk>>>(...); }          link k.o + cublas + cudart
```

The `.cu` file has `extern "C"` on both the kernel (so the name is unmangled
for the launcher) and the launcher (so Zig can `extern` it). Zig side:
`@cImport(@cInclude("cuda_runtime.h")) + @cInclude("cublas_v2.h")` to get the
runtime + cuBLAS types, and an `extern fn launch_k(...)` declaration matching
the launcher's signature exactly.

### cuBLAS is column-major — the row-major matmul trick

cuBLAS expects matrices in **column-major** order. Your tensors are row-major
(a row is contiguous). You do *not* transpose your data. Instead you ask cuBLAS
to compute the transpose of what you want, exploiting that **`A` row-major is
exactly `A^T` column-major in the same memory**. For `C[M,N] = A[M,K] @ B[K,N]`
(row-major), ask cuBLAS for `C_col[N,M] = B_col[N,K] @ A_col[K,M]` with
`OP_N, OP_N`:

```
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,            // rows,cols,k of the COLUMN-MAJOR problem
            &alpha,
            dB, N,              // B row-major == B^T col-major, leading dim N
            dA, K,              // A row-major == A^T col-major, leading dim K
            &beta,
            dC, N);             // C^T col-major == C row-major, leading dim N
```

The result `C_col[N,M]` in memory is `C^T` column-major, which is `C` row-major
`[M,N]` — exactly what you want, with no data movement. The two things to
memorize: the first two cuBLAS args are `(N, M)` not `(M, N)`, and the leading
dimensions are `dB→N`, `dA→K`, `dC→N` (each matrix's *row* stride, because
column-major "leading dim" is the stride between columns = row count).

**Worked example (2×2).** `A=[[1,2],[3,4]]` (row-major bytes: `1 2 3 4`),
`B=[[5,6],[7,8]]` (bytes: `5 6 7 8`). You want `C=[[19,22],[43,50]]` (bytes
`19 22 43 50`). Call `sgemm(OP_N, OP_N, N=2, M=2, K=2, &1, dB, ldb=2, dA,
lda=2, &0, dC, ldc=2)`. cuBLAS reads `dB` as `B^T=[[5,7],[6,8]]` col-major,
`dA` as `A^T=[[1,3],[2,4]]` col-major, computes `B^T @ A^T = [[19,43],[22,50]]`
col-major, which in memory is `19 22 43 50` — exactly your row-major `C`. The
transposition is *simulated* by the argument order, not by moving bytes.

### The gated-delta kernel (your first real kernel)

Mirror of lesson 11's recurrent forward, but parallel across `(batch, head)`:
one CUDA thread per `(batch, head)` carries a `Dh×Dh` state in **local memory**
(a `float state[MAXD*MAXD]` array), and loops `t = 0..L-1` sequentially (the
recurrence is inherently sequential within a head — this is the *recurrent*
kernel; the chunked-parallel kernel is the next exercise). Launch
`<<<ceil(Bh/128), 128>>>`. The kernel body is line-for-line the recurrence from
lesson 11's worked example, just in C with `__restrict__` pointers. Keep
`MAXD` as a compile-time cap (`#define MAXD 32`) and assert `Dh ≤ MAXD`.

### Device memory + the `?*anyopaque` convention

GPU pointers are opaque to Zig. Hold them as `?*anyopaque`. At the cuBLAS /
kernel boundary, cast: `@ptrCast(@alignCast(dA))` (the `@alignCast` is required
— `cudaMalloc` returns suitably aligned pointers and the compiler must be told).
`cudaMalloc/cudaFree/cudaMemcpy` (H↔D) are the only memory ops you need.

## Tasks

1. **`backend_cuda.zig`** with `@cImport` of the two headers, an `Error` enum,
   and a `CudaBackend` struct holding one `cublasHandle_t`. Implement
   `init/deinit/sync` (`cudaDeviceSynchronize`), `alloc/free/h2d/d2h`, and a
   row-major `matmul(dA, dB, dC, M, N, K)` using the transposed-call trick above.

2. **A batched matmul** (`matmulBatched`) as a loop of `sgemm` over a leading
   batch dim with the right strides. (cuBLAS `SgemmStridedBatched` is the
   obvious optimization — leave it as an exercise; a loop is correct first.)

3. **The gated-delta kernel** in `znn_kernels.cu` (`gated_delta_fwd_kernel` +
   `launch_gated_delta_fwd`), exact mirror of lesson 11's forward. Wire it from
   `backend_cuda.zig` via an `extern fn launch_gated_delta_fwd(...)` and a
   `gatedDeltaFwd` method that asserts `Dh ≤ 32` and checks `cudaGetLastError`.

4. **`build.zig` wiring** (the part with the most hidden footguns):
   - `const gpu = b.option(bool, "gpu", ...) orelse false;`
   - Only when `gpu`: create a `backend_cuda` module, add it to bench's imports.
   - Compile the kernel: `b.addSystemCommand(&.{ "nvcc", "-c", "-O2",
     "-arch=sm_75" })`, `addFileArg` the `.cu`, then **`addArg("-o")` followed
     by `addOutputFileArg("znn_kernels.o")`** (see pitfall), `addObjectFile`
     into the bench module, `linkSystemLibrary("cublas"/"cudart"/"c", .{})`,
     and `bench.step.dependOn(&nvcc.step)` so the `.o` exists before the link.
   - Pass the `gpu` flag into the app at comptime via `b.addOptions()` +
     `addOption(bool, "gpu", gpu)` + `module.addOptions("znn_opts", opts)`; the
     app does `const GPU = @import("znn_opts").gpu;` and `if (!GPU) return;`.

5. **`bench.zig`** (a runnable app, not a `test`): CPU matmul baseline (512³, a
   few reps, time it), then (if GPU) cuBLAS matmul — correctness
   (`max|diff|` vs CPU) + throughput + speedup — and the gated-delta kernel vs
   the CPU recurrent reference (`vGatedDelta`). Use `std.Io.Clock.now(.awake,
   io).nanoseconds` for timing (lesson 00).

## Checkpoint: the bars `zig build bench -Dgpu` must hit

- **cuBLAS matmul** `512³`: `max|diff|` vs CPU **< 1e-4** (right at the FP32
  accumulation boundary; the bench warns only above 1e-3) **and** a large
  speedup (on the 1660 Ti, ~0.13 ms vs ~228 ms CPU → ~1700×). The speedup is
  from the GPU's many cores, *not* tensor cores — the 1660 Ti is compute 7.5
  (Turing) with **no tensor cores**, so this is a pure FP32 path.
- **Gated-delta kernel** vs CPU recurrent: `max|diff| < 1e-3` (config `Bh=4,
  L=16, Dh=8`). The kernel is line-for-line the recurrence, so this is really
  "did you transcribe it correctly" — but a wrong `tril`/orientation would
  show up here too.

There are no `test "..."` blocks here — the gate is the bench *app's printed
output*. Green = the two `max|diff|` lines are under their bars and the speedup
is large.

## Common pitfalls

- **nvcc `-o` argument concatenation (the worst one).** Zig's
  `addPrefixedOutputFileArg("-o", "x.o")` produces a *single* argv `-o/abs/path/x.o`.
  nvcc (unlike gcc/clang) **rejects** the attached `-o<path>` form with
  `Unknown option '-o/abs/path/x.o'`. Pass them as **two** separate argv:
  `nvcc.addArg("-o"); const obj = nvcc.addOutputFileArg("znn_kernels.o");`. This
  one mistake is the entire difference between "links in an hour" and "links in
  a day."
- **A kernel needs a host launcher, not a direct call.** Declaring
  `extern fn gated_delta_fwd_kernel(...)` and calling it from Zig will *compile*
  but call the `__global__` symbol as an ordinary function (wrong ABI, often a
  crash returning 0). Always go through an `extern "C" void launch_...(...)`
  that does the `<<<>>>` launch.
- **Link `libc` (`-lc` / `linkSystemLibrary("c", .{})`), not just cublas+cudart.**
  CUDA's static registration runs `atexit`, which lives in libc. Without `-lc`
  you get `undefined symbol: atexit` at link time.
- **`linkSystemLibrary` takes two args.** `linkSystemLibrary("cublas", .{})`
  — the options struct is required in Zig 0.16.
- **Row-major transposed call: dims are `(N, M, K)`, leads are `N, K, N`.** Get
  any of these backwards and the result is a silently-wrong matrix that only a
  `max|diff|` check catches. The CPU-vs-GPU comparison *is* your safety net —
  never ship a GPU op without the CPU reference diff.
- **`@ptrCast` without `@alignCast`.** CUDA pointers are aligned; the compiler
  needs to be told, or you get "`@ptrCast` increases alignment" errors. Use
  `@ptrCast(@alignCast(dA))` at every cuBLAS/kernel boundary.
- **Device pointers are `?*anyopaque`.** Don't try to type them as `[*]f32`
  early — keep them opaque until the cast at the call. C `float` ↔ Zig `f32`
  (not `c.float`).
- **The nvcc object step must be a dependency.** `bench.step.dependOn(&nvcc.step)`
  or the `.o` won't exist at link time (`FileNotFound`).
- **No tensor cores on this GPU.** Don't expect BF16/FP16 speedups or an
  A100-class number. The 1660 Ti is FP32-only; that's the bar you're hitting.
  Real mixed-precision (and tensor cores) is a RunPod/A100 exercise.

## What good looks like

- `zig build bench` (no `-Dgpu`) runs the CPU baseline and prints "GPU path
  skipped" — the CPU build never touches a CUDA header.
- `zig build bench -Dgpu` compiles the `.cu` with nvcc, links cublas+cudart+libc,
  and prints both `max|diff|` lines under their bars with a large speedup.
- You can recite the row-major trick: "ask for the transpose, lead dims are the
  row strides, dims are `(N,M,K)`."
- You can add a new kernel (e.g. a flash-attention forward) by copying the
  pattern: `__global__` + `extern "C" launch_` + `extern fn` + a CPU diff check.

The autograd *training loop* is still on CPU here — the GPU path verifies the
heavy ops in isolation. Wiring the whole graph onto the GPU is the natural next
frontier (and the honest thing to say about the capstone in lesson 13). Lesson
13 ties everything into one runnable, checkpointable, sampleable model.