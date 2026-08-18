# Lesson 02 — Tensors (strided storage, broadcasting, matmul)

## Objective

Build a `Tensor` type: a flat `[]f32` plus a `shape` and `strides`, with the
operations the whole framework rests on — elementwise add/mul, broadcasting,
reduce (`sumAxis`), transpose/permute, reshape, and **matmul** (the one
primitive to rule them all). By the end, you have a working linear-algebra
core, verified against known answers. No autograd yet — that's lesson 03.

## Why this

Lesson 00 said it and it bears repeating: a tensor is **memory + a shape**. If
you internalize that *strides do all the work* — transpose, reshape, even
broadcasting are index arithmetic, not data movement — then everything later
(attention's `Q @ K^T`, embedding gather, the batched matmuls) stops being
mysterious shape-juggling and becomes "the same flat buffer, indexed
differently". Get this right once and the rest of the curriculum is
*applications* of it.

## The model

```zig
pub const Tensor = struct {
    data: []f32,
    shape: []const usize,
    strides: []const usize,
    // (an allocator, or an offset, if you support views/slices later)
};
```

- `numel()` = product of `shape`.
- `flat(idx: []const usize) usize` = `sum_k idx[k] * strides[k]` — the **one**
  indexing function. Every op eventually calls this.
- `isContiguous()` = strides are the "row-major" product (last dim stride 1,
  each earlier dim stride = product of later dims). Most ops are easiest if you
  produce a fresh contiguous buffer.

The first decision: do ops produce *views* (share data, set strides) or *new
contiguous buffers* (copy)? **Start with new contiguous buffers for
everything.** Views are an optimization; they introduce aliasing bugs and
stride-math that will eat your time. You can add `transpose2D`-as-a-view later
once the tests are green. The reference impl actually materializes transposes
into a new buffer to keep the indexing simple — do the same until you have a
reason not to.

### Broadcasting — right-align, match-or-1

Two shapes are broadcast-compatible if, after **right-aligning** them, every
paired dim is either equal or one of them is `1`. The result dim is the `max`.
This is the *only* rule. People get it wrong by left-aligning.

**Worked example.** `[2,1] + [1,3]`:
```
right-align:  [2,1]
             [1,3]
pair dims:   (2 vs 1 -> max 2) , (1 vs 3 -> max 3)  -> result [2,3]
```
`[2,1]` is a *column*, `[1,3]` is a *row*; the sum is the full `2×3` outer-sum.
Concretely, `x=[[1],[10]]`, `y=[[1,2,3]]`:
```
z = [[1+1, 1+2, 1+3],     = [[2, 3, 4],
     [10+1,10+2,10+3]]      [11,12,13]]
```
That is the reference test `"broadcasting add [2,1]+[1,3]"`. Hit those exact
numbers.

**Implementation tip:** the cleanest way to broadcast two operands is to write
a helper that, for each output multi-index, computes the corresponding input
multi-index by *clamping* each axis to its real size when the operand's size
there is `1` (i.e. `in_idx[k] = out_idx[k] % op.shape[k]`). That `mod-by-size`
trick is the whole of broadcasting in one line, and it falls out of the
"right-align, 1 means repeat" rule.

### Matmul — the shape contract and the inner index

`C = A @ B`: `A` is `[M,K]`, `B` is `[K,N]`, `C` is `[M,N]`, and
```
C[i,j] = Σ_k A[i,k] * B[k,j]
```
The shape contract: the **inner dims must match** (`A`'s last == `B`'s first);
the **outer dims become C**. The inner index `k` walks a *row of A* against a
*column of B*. Get this orientation wrong (row against row) and you get
silently-wrong numbers that still look plausible.

**Worked example (memorize it).** `A=[[1,2,3],[4,5,6]]` (2×3), `B=[[7,8],[9,10],[11,12]]` (3×2):
```
C[0,0] = 1*7 + 2*9 + 3*11  = 58
C[0,1] = 1*8 + 2*10 + 3*12 = 64
C[1,0] = 4*7 + 5*9 + 6*11  = 139
C[1,1] = 4*8 + 5*10 + 6*12 = 154
C = [[58,64],[139,154]]
```
That is the reference test `"matmul 2x3 @ 3x2"` — match it to `1e-4`.

### Batched matmul (you'll need it in lesson 09/10)

A leading batch dimension: `A[B,M,K] @ B[K,N] -> [B,M,N]`, and you can also
broadcast the batch (a `[K,N]` `B` applied to every slice of `A`). Loop the
batch and call your 2D matmul per slice; the broadcasting case is "B has no
batch dim, so use the same B for every slice." The reference test
`"matmul batched with broadcast"` does exactly `[2,2,3] @ [3,2] -> [2,2,2]` and
expects `[[[4,5],[10,11]],[[2,2],[4,4]]]`. (Note the off-by-one in the reference's
own comment vs its asserts — *trust the asserts*, which are `[4,5,10,11]` for
batch 0: `[[1,2,3],[4,5,6]] @ [[1,0],[0,1],[1,1]] = [[1+0+3, 0+2+3],[4+0+6,0+5+6]] = [[4,5],[10,16]]`...
compute it yourself and match the test's `expectApproxEqAbs` lines exactly,
since one batch's second row number is the lesson's test of *your* reading.)

### transpose2D and permute

`transpose2D` on a `[m,n]` matrix gives `[n,m]` with `out[i,j] = in[j,i]`. The
reference test `"transpose2D and permute"`: `[[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]`.

`permute(x, dims)` is the n-d generalization: `out[idx] = in[permuted idx]`,
where `out_shape[k] = in_shape[dims[k]]`. The test permutes `[1,2,3]` by
`[0,2,1]` to `[1,3,2]` and expects data `[1,4,2,5,3,6]`. Implement it by walking
every output multi-index, computing the input multi-index via the `dims`
permutation, and indexing once. (Slow, correct, fine. Optimize later, or never
— transpose-as-stride-view is an optional extra-credit that the rest of the
framework does not require.)

### reduce: `sumAxis`

`sumAxis(x, axis)` sums along one axis and **keeps that dim as size 1** (so
shapes stay composable). `[2,3]` along axis 0 -> `[1,3] = [[5,7,9]]`; along
axis 1 -> `[2,1] = [[6],[15]]`. That keeping-the-dim convention is what makes
backward broadcasting (lesson 03) clean: a reduce in the forward is an
un-broadcast in the backward, and vice versa.

## Multiple representations

- **Algebra:** a tensor is an element of `R^{shape product}`; broadcasting is
  implicit replication via index-clamping; matmul is the bilinear form
  `Σ A_{ik}B_{kj}`.
- **Diagram:** draw `[2,3]` as a 2×3 grid sitting on a 6-long line; strides are
  the arrows between adjacent-in-each-axis cells. A transpose redraws the grid
  but the line (the data) is untouched.
- **Analogy:** the flat `[]f32` is a warehouse of numbered crates; the shape
  and strides are the *map* the forklift driver uses. Transpose/reshape hand
  the driver a new map over the same crates — no crates move.
- **Physical intuition:** broadcasting is "stretch to fit" — a `[1,3]` row and
  a `[2,1]` column don't *store* a 2×3; they store a row and a column and the
  op pretends each is replicated, on demand, via index mod.

## Tasks

1. **`Tensor` core.** `ones`, `zeros`, `fromSlice(shape, data)`, `numel`,
   `ndim`, `flat(idx)`, `isContiguous`, and a `reshape` (contiguous only, check
   product). Test: `"zeros/ones/shape/numel"`.

2. **Elementwise add/mul with broadcasting.** Right-align shapes; for each
   output index, clamp each input index by `% op.shape[k]`. Test:
   `"broadcasting add [2,1]+[1,3]"` → `[[2,3,4],[11,12,13]]`.

3. **`matmul`** (2D + batched with broadcast). Triple loop, inner index `k`.
   Tests: `"matmul 2x3 @ 3x2"` → `[[58,64],[139,154]]`; `"matmul batched with
   broadcast"` → the numbers above.

4. **`transpose2D` and `permute`.** New contiguous buffers. Test:
   `"transpose2D and permute"`.

5. **`sumAxis` (keeps dim), `softmax` (max-subtract for stability), and a couple
   of activations (`sigmoid`, `gelu`).** Tests: `"sumAxis keeps dim"`,
   `"softmax sums to 1 and is correct"` (note the `[1000,0,0]` row — if your
   softmax isn't max-subtracted, you get `exp(1000) = inf` and fail),
   `"gelu and sigmoid sanity"`.

> Remember the **braceless-for rule** (lesson 00): every `for` body that
> *assigns* (`v += ...`, `out[...] = ...`) must be braced. The matmul triple
> loop and the broadcast walker are where you'll hit it first. Wrap them.

## Checkpoint: the tests that must pass

Run `zig test src/tensor.zig` and get, in order:
- `zeros/ones/shape/numel`
- `matmul 2x3 @ 3x2`  (→ `[[58,64],[139,154]]`, abs tol `1e-4`)
- `matmul batched with broadcast`
- `broadcasting add [2,1]+[1,3]`  (→ `[[2,3,4],[11,12,13]]`)
- `transpose2D and permute`
- `sumAxis keeps dim`
- `softmax sums to 1 and is correct`
- `gelu and sigmoid sanity`

All green with `expectApproxEqAbs(..., 1e-4)`. That's the gate. (In the reference
suite these are exactly the tests in `src/znn/tensor.zig`.)

## Common pitfalls

- **Row-vs-column inner product.** The #1 matmul bug: summing `A[i,k]*B[j,k]`
  (row·row) instead of `A[i,k]*B[k,j]` (row·col). Results look plausible and are
  wrong. The known-answer `[[58,64],[139,154]]` test exists to kill this.
- **Left-aligning broadcast shapes.** `[2,1]` and `[3]` broadcast to `[2,3]`
  (right-align the `[3]` to `[1,3]`); if you left-align you'd think it's
  incompatible or get the wrong axis. Always right-align, match-or-1.
- **softmax without max-subtraction.** `exp(1000)` overflows to `inf`. Subtract
  the per-row max before `exp`. The `[1000,0,0]` test row is there to catch
  exactly this.
- **`sumAxis` collapsing the dim instead of keeping it.** If you return `[3]`
  instead of `[1,3]`, downstream reshapes and broadcasts break silently. Keep
  the reduced dim as size `1`.
- **Zig 0.16 braceless-for.** `for (0..n) |i| out[i] = ...;` at statement level
  with a compound body fails to parse. Brace it. You will hit this in matmul.
- **`reshape` on a non-contiguous tensor.** Either disallow it (assert
  contiguous) or make it copy. Silent data corruption otherwise.
- **Strides computed wrong for 3D+.** Stride of the last axis is 1; each
  earlier axis's stride is the product of all later dims. Off-by-one here
  scrambles `permute` and `matmul` batch loops in a way that's painful to debug.
  Unit-test `flat(idx)` on a `[2,3,4]` tensor before trusting it.

## What good looks like

- All eight tests green at `1e-4`.
- You can add a new elementwise op (`tanh`, `exp`, `neg`) in a few lines using
  your broadcast walker, and it just works for any compatible shapes.
- You can explain, with the warehouse/strides analogy, why transpose is "free"
  and why broadcasting is `idx % size`.
- The `flat(idx)` helper is the *only* indexing code in the whole file;
  everything else composes it. (If you have copy-pasted index math in five
  places, refactor — it's a bug farm.)

You now have the linear-algebra core. Lesson 03 puts gradients on it: same
topo/backprop pattern as lesson 01, but tensor-valued — and the one new hard
part, *un-broadcasting* gradients back to each parent's shape.