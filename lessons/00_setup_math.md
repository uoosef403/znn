# Lesson 00 — Setup and the math you actually need

## Objective

By the end of this lesson you can: build and run a Zig 0.16.0 program, use the
0.16 allocator/IO APIs without fighting them, and — more importantly — explain
in plain words what a tensor *is*, what a matrix multiply *does* (and why it's
the only linear-algebra primitive the whole field runs on), what the chain rule
*means* with concrete numbers, and why floating-point arithmetic lies to you in
a way that will bite your gradient checks later.

This lesson has **no checkpoint test of its own** — its checkpoint is that the
*next* lesson's environment builds. But the math here is load-bearing for every
gate after it. Do not skim.

---

## 0.1 Environment

### Zig 0.16.0

```bash
zig version   # must print 0.16.0
```

0.16 is a *moving* language; APIs you find in 0.13 blog posts will be wrong.
Trust `lib/std` on your machine, not the internet. The std you'll touch most:

**Allocators.** There is no `GeneralPurposeAllocator` anymore (it's
`DebugAllocator`). For apps and tests use the arena pattern:
```zig
var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
defer arena.deinit();
const a = arena.allocator();
```
`smp_allocator` is the fast default. The arena frees everything at once on
`deinit` — perfect for a training step where you build a throwaway graph each
step and don't want to track each allocation.

**ArrayList.** `std.ArrayList(T)` is now unmanaged:
```zig
var list: std.ArrayList(f32) = .empty;
defer list.deinit(a);          // takes the allocator
list.append(a, 3.0) catch ...;
```
Note `deinit` and `append` now take the allocator as the first arg.

**File I/O.** `std.fs.cwd()` is **gone**. Use `std.Io.Dir.cwd()` with an `io`
handle:
```zig
const io = std.Options.debug_io;
const text = try std.Io.Dir.cwd().readFileAlloc(io, "data/text.bin", a, .limited(1 << 30));
std.Io.Dir.cwd().writeFile(io, .{ .sub_path = "out.bin", .data = bytes }) catch {};
```
The `.limited(n)` is a `limite: AllocLimite` — required on reads.

**Timing.** `std.time.milliTimestamp()` is gone-ish; use the Io clock:
```zig
const ns = std.Io.Clock.now(.awake, io).nanoseconds;  // i128
```

**A build file.** Use `lessons/starter/build.zig`. Verify it sees your files.

> **Pitfall (you will hit this in Gate 1):** `zig build test` in 0.16 may print
> `failed command:` and omit `Build Summary` *even on success*. Trust the exit
> code (0 = pass) and, when in doubt, run `zig test path/to/file.zig` directly
> and look for `All N tests passed`. Don't panic at the `failed command` line.

> **Pitfall (you will hit this in every gate):** Zig 0.16 forbids *braceless
> for-bodies that assign*. `if (c) for (j) |x| v += x;` fails to parse. `for
> (i) |_,| for (j) |_| x = ...;` fails. **Wrap any for-body that assigns in
> braces.** A bare `for (0..n) |i| v += arr[i];` *inside a braced block* is fine;
> it's the *braceless compound* form at statement level that breaks. When you
> get `invalid left-hand side to assignment` or `expected ';' or 'else' after
> statement`, you have this. Add the braces.

### A sanity program

Make `src/main.zig`:
```zig
const std = @import("std");
pub fn main() void {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var xs: std.ArrayList(f32) = .empty;
    for (0..5) |i| xs.append(a, @floatFromInt(i)) catch return;
    var s: f32 = 0;
    for (xs.items) |x| s += x;
    std.debug.print("sum = {d}\n", .{s});  // 0+1+2+3+4 = 10
}
```
Build and run it. If `sum = 10` prints, your environment is ready.

---

## 0.2 The math you actually need

### A tensor is memory + a shape

Forget the "multidimensional array" framing for a moment. A tensor is:

1. a flat block of `f32` numbers in memory (a `[]f32`), and
2. a **shape** (a list of dimension sizes), and
3. **strides** (how far in the flat buffer you step to move one along each dim).

That's it. Everything — broadcasting, transposing, reshaping, even matmul — is
*just choosing how to index into the flat buffer*. This is the single most
important idea in the whole curriculum. If you implement it once, in lesson 02,
the rest of the field stops being magic.

**Worked example.** A 2×3 matrix stored row-major:
```
shape = [2, 3]
flat  = [1, 2, 3, 4, 5, 6]
        =  row0: 1 2 3
           row1: 4 5 6
```
To read element `[i, j]`: index = `i * 3 + j`. The stride for dim 0 is `3` (step
3 to go down a row), for dim 1 is `1` (step 1 to go across a column). So:
`flat[i*stride0 + j*stride1]`. A *transpose* doesn't move any data — it just
**swaps the strides**. A *reshape* doesn't move data — it changes the shape (and
recomputes strides) as long as the total count is unchanged. This is why
"transposing is free" and "reshaping is free" in real frameworks: no copying.

> **Task 00.a** (no code yet, just convince yourself): if `shape=[2,3]`,
> `strides=[3,1]`, what flat index is `[1,2]`? What about `[0,0]`? If I transpose
> to `shape=[3,2]` with `strides=[1,3]`, what flat index is now `[2,0]`? (Same
> buffer — which element is it?) Answers you should reach: 5, 0, and 5 again.
> Transposing `[1,2]`-of-the-original lands on the *same number*, just indexed
> differently. That's the whole trick.

### Matmul by hand — the one primitive to rule them all

A matrix multiply `C = A @ B` where `A` is `[M,K]`, `B` is `[K,N]`, gives `C`
`[M,N]`:
```
C[i,j] = sum over k of A[i,k] * B[k,j]
```
That's it. One number `C[i,j]` is a dot product of a *row of A* and a *column of
B*. The whole shape contract: the inner dimension `K` must match; the outer
dims `M` and `N` become `C`'s shape.

**Why it matters so much:** *every linear layer, every attention score, every
projection, every embedding lookup-as-matmul* is a matmul. The entire field is
"matmul, then a nonlinearity, then matmul, ...". If you implement matmul
correctly and fast (lesson 02, and the GPU version in lesson 12), you have
built 80% of the engine.

**Worked example, tiny.** `A=[[1,2],[3,4]]` (2×2), `B=[[5,6],[7,8]]` (2×2).
```
C[0,0] = 1*5 + 2*7 = 19
C[0,1] = 1*6 + 2*8 = 22
C[1,0] = 3*5 + 4*7 = 43
C[1,1] = 3*6 + 4*8 = 50
```
So `C = [[19,22],[43,50]]`. Implement a triple loop in lesson 02 and check it
against these numbers. (Note the **inner index `k` sums a row of A against a
column of B** — the orientation is the #1 shape bug: people sum a row against a
row and get silently wrong answers. A `grad_check` and a known-answer test like
this one catch it.)

**Batched matmul** (you'll need it for attention): `A[B,M,K] @ B[K,N]` (or
`[B,K,N]`) → `[B,M,N]`. Just loop the outer batch dim and do a matmul per slice.
The GPU version (lesson 12) uses `cublasSgemmStridedBatched` — same idea,
hardware-fast.

### The derivative, with concrete numbers

The derivative `f'(x)` is "how much `f` moves when `x` moves a tiny bit", as
that tiny bit shrinks to zero. Concretely:
```
f(x) = x^2
f'(3) = ?     ->  "if x goes 3 -> 3.001, f goes 9 -> 9.006001; slope ~ 6"
f'(x) = 2x, so f'(3) = 6.   matches.
```
That's all a gradient is: the derivative, but for a function with *many*
inputs — one derivative per input, telling you how the output wiggles when that
one input wiggles. The **gradient is a vector** the same length as the input.

**The chain rule is the whole engine of learning.** When you compose
functions `L = f(g(h(x)))`, the derivative w.r.t. the deepest input is:
```
dL/dx = (dL/df) * (df/dg) * (dg/dh) * (dh/dx)
```
Each function only needs to know *its own local derivative* and *the gradient
flowing into it from above*; it multiplies them and passes the result down.
**This is why autograd works.** You never compute `dL/dx` globally. You compute
it one node at a time, backwards, each node doing one local multiply-and-pass.

**Worked example.** `L = ((x * y) + z) ^ 2`, with `x=2, y=3, z=1`.
```
a = x*y = 6        (da/dx = y = 3,  da/dy = x = 2)
b = a+z = 7        (db/da = 1,      db/dz = 1)
L = b^2 = 49       (dL/db = 2b = 14)
```
Backward (seed `dL/dL = 1`):
```
dL/db = 1 * 2b = 14
dL/da = dL/db * db/da = 14 * 1 = 14
dL/dz = dL/db * db/dz = 14 * 1 = 14
dL/dx = dL/da * da/dx = 14 * 3 = 42
dL/dy = dL/da * da/dy = 14 * 2 = 28
```
Check by finite difference: bump `x` by `0.001` → `L` goes from 49 to
`((2.001*3)+1)^2 = 7.003^2 = 49.042009`, delta `0.042009` over `0.001` ≈ `42.0`.
**Exactly what the chain rule gave.** That's the whole miracle. In lesson 01
you build exactly this — but the computer does the bookkeeping.

> **Task 00.b:** recompute the worked example above changing `z` from 1 to 4.
> What is `dL/dx` now? (You should get `2*(x*y+z)*y` evaluated = `2*(6+4)*3 =
> 60`.) The chain rule is mechanical. Let it be mechanical.

### Floating point lies (and this *will* bite you)

`f32` has ~7 decimal digits. So:
```
1e8 + 1.0 == 1e8        // true. the 1.0 is below the precision at that scale
0.1 + 0.2 != 0.3        // true. it's 0.3000000119...
```
Two consequences for us:

1. **Finite-difference gradient checks are noisy.** `(L(x+eps) - L(x-eps)) /
   (2*eps)` with `eps=1e-3` in `f32` has roundoff ~`1e-4` in the *gradient*, not
   the loss. So a gradient of `0.0007` is mostly noise. **Pure relative
   error** `|num-ana|/|num|` will scream at tiny gradients. The fix — and you
   will implement this in lesson 03 — is an **absolute tolerance floor**: a
   gradient passes if `|num - ana| < atol` (say `1e-4`) *or* the relative error
   is small. Never use a pure-relative check on f32.

2. **Loss magnitude matters.** A mean over `N` elements scales the loss by
   `1/N`, but the gradient of the *sum* scales by `N`. If you take a mean loss
   over a huge tensor, the per-element signal drops toward f32 noise. For
   *nonlinear* ops in a grad-check, scale a weighted loss by `1/sqrt(numel)` so
   the signal stays above the noise floor but doesn't blow up. (You'll meet
   this in lesson 03 and again in lesson 11 for gated-delta.)

> **Task 00.c:** in the sanity program, add `(1e8 + 1.0) - 1e8` and print it.
> Expect not `1.0`. Then try `1e6 + 1.0 - 1e6`. Find the threshold where `f32`
> starts losing the `+1`. This is the *resolution* of the format, and it's the
> reason your grad checks need an absolute floor.

---

## What good looks like (lesson 00)

- `zig version` → 0.16.0; your sanity program prints `sum = 10`.
- You can recite the chain-rule worked example from memory and re-derive
  `dL/dx` for a small composed function by hand.
- You can explain why a transpose is free (strides, not data) and why a
  pure-relative grad check is a trap on f32.
- You have the `starter/build.zig` in a fresh folder and `zig build` runs (even
  with no apps yet, it should not error on the build file itself).

When that's true, go to lesson 01. The chain rule is about to become code.