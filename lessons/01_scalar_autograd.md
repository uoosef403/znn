# Lesson 01 — Scalar autograd (micrograd)

## Objective

Build the smallest possible automatic differentiation engine: a single scalar
`Value` that knows its own gradient, its parents, and a *local backward
function*. With it, you compute gradients of any composed function by calling
`.backprop()` once. This is the whole of deep learning, in 60 lines, before you
generalize to tensors in lesson 02.

## Why this first

Every neural network, no matter how large, is "a function built out of small
pieces, whose gradient you need". If you understand autograd on *scalars*, you
understand autograd on *tensors* — the tensor version is just "do the scalar
thing, batched, with a couple of shape rules for broadcasting". The scary part
of "how does PyTorch know the gradient?" collapses to: **each op records a
little closure, and you walk the graph backwards multiplying.** That's lesson 01.

## The model

A `Value` holds:
- `data: f32` — the forward value,
- `grad: f32` — the gradient of the *final loss* w.r.t. this value (accumulated),
- `parents: []*Value` — the inputs that produced it,
- `backward: ?*const fn(*Value) void` — a closure that, given this value's
  `.grad`, pushes the *local* gradients into each parent's `.grad`.

The forward pass *builds the graph*: `a + b` returns a new `Value` whose
parents are `a, b` and whose backward adds `1*self.grad` to each parent (because
`d(a+b)/da = 1`). The backward pass *walks* the graph in reverse topological
order, calling each node's backward closure once.

### The chain rule, as a closure

For `c = a * b`: locally, `dc/da = b` and `dc/db = a`. When `c.grad` is known
(grad of the *loss* w.r.t. `c`), the contribution to `a.grad` is `c.grad * b`,
and to `b.grad` is `c.grad * a`. The closure captures `a` and `b` (by pointer)
and does:
```
a.grad += self.grad * b.data
b.grad += self.grad * a.data
```
Note **`+=`**, not `=`: a node can be used multiple times, so gradients
**accumulate**. This is essential and the #1 bug when you generalize: forgetting
to accumulate, or forgetting to *zero* grads before a new backward.

**Worked example** (the one from lesson 00, now as code): `L = ((x*y)+z)^2`,
`x=2,y=3,z=1`. Build:
```
a = x * y            // a.data=6,  parents=[x,y], back: x.grad+=a.grad*y, y.grad+=a.grad*x
b = a + z            // b.data=7,  parents=[a,z], back: a.grad+=b.grad,    z.grad+=b.grad
L = b * b            // L.data=49, parents=[b,b], back: b.grad += L.grad*b, b.grad += L.grad*b
                     //   (b used twice -> its grad accumulates from both edges)
```
Seed `L.grad = 1`, walk reverse-topo (`L, b, a, ...`):
```
L.back: b.grad += 1*b + 1*b = 14       // b.grad = 14
b.back: a.grad += 14; z.grad += 14      // a.grad=14, z.grad=14
a.back: x.grad += 14*3 = 42; y.grad += 14*2 = 28
```
**Exactly the hand-computed answer.** That's the engine.

## Multiple representations

- **Algebra:** `dL/d(in) = prod of local partials along the path` (chain rule).
- **Diagram:** a DAG, arrows point from parents to children during forward; the
  backward walk flows grads *upstream* (children → parents), each edge scaling
  by the local partial. Draw `x → a → b → L` and `y ↗`, `z ↗` and see grads flow
  back: `L→b→a→{x,y}` and `L→b→z`.
- **Analogy:** forward is *assembling* a machine from parts and recording the
  assembly manual; backward is *reading the manual in reverse*, each part
  handing blame ("you contributed this much to the error") to the part below
  it. The `backward` closure is one line of the manual.
- **Physical intuition:** the gradient is "how much the final loss wiggles per
  unit wiggle of this input". Each op just relays the wiggle, scaled by how
  sensitive it is locally.

## Tasks

1. **Define `Value`** with `data, grad, parents, backward` and an allocator
   (parents is a slice you allocate). Make `Value.constant(a, v)` and
   `Value.parameter(a, v)` for leaves (no parents, no backward).

2. **Implement `+`, `*`, and a unary `pow2` (or a general `pow`/`relu`/`exp`)**
   that each allocate a child `Value`, wire parents, and set a `backward`
   closure. Remember the **braceless-for rule** from lesson 00 if you loop.

3. **Topological sort.** Write `buildTopo(v, &list, &visited)` that DFS-walks
   parents-first so the list is in forward order; reverse it for backward.

4. **`backprop()` on the root:** zero all grads in the graph first, seed
   `self.grad = 1` (the root is a scalar loss), then walk the reversed topo list
   calling each node's backward.

5. **The checkpoint:** write a finite-difference grad-check. For each leaf,
   perturb `±eps`, recompute the loss numerically, compare to the analytic grad
   from `.backprop()`. Use **central differences** and the **atol floor** from
   lesson 00. Test on the worked example above and on a small `relu`-and-`exp`
   graph.

## Checkpoint: the test that must pass

A test (call it `test "Gate 1: scalar grad-check"`) that builds a non-trivial
scalar graph — e.g. `L = relu(a*b + c)^2 + exp(a)` for `a,b,c` random — runs
`.backprop()`, and checks each leaf's analytic grad against central-difference
with `eps=1e-3`, `atol=1e-4`, `rtol=1e-2`. **All leaves must pass.** In the
reference suite this is the first `grad_check` in `autograd.zig`.

Run it: `zig test src/autograd.zig` (or your equivalent). The bar: every leaf's
`|num - ana| < atol` *or* relative error `< rtol`.

## Common pitfalls

- **Forgetting to accumulate (`+=`)** in backward closures → a value used
  twice gets only the last contribution. The `b*b` example above is the trap:
  `b` has two incoming edges; its grad must sum both.
- **Forgetting to zero grads** before backward → grads leak across steps and
  across checks. `backprop` should zero every node's grad first (or you provide
  a `zero_grad` and call it).
- **Topo order wrong** → grads computed before their upstream grad is ready.
  Parents must come *before* children in the forward list; reverse for
  backward. DFS-walk parents-first, then reverse.
- **Closures capturing the wrong thing** in Zig: a closure can't capture
  mutable locals by value the way Python lambdas do. Store the parents on the
  child `Value` and reference them through `self.parents[i]` inside the
  backward fn (which takes `*Value` as its arg). Don't try to make a closure
  over `a`/`b` — wire them as parents and index.
- **Pure-relative tolerance** on f32 → false failures on tiny gradients. Use
  the atol floor. (Lesson 00; you will hit this again in lessons 03 and 11.)

## What good looks like

- A `Value` graph of ~5 ops, `.backprop()` fills every leaf's `.grad` to match
  central differences to within atol/rtol — green test.
- You can add a new op (`tanh`, `sigmoid`) in ~5 lines: one forward line, one
  backward line capturing the local derivative.
- You can explain *why* `b*b` needs accumulation without looking it up.

You now have autograd. Lesson 02 makes the numbers tensors, and lesson 03
brings this same graph back — but batched, with shapes.