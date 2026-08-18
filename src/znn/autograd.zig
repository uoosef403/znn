// autograd.zig — reverse-mode automatic differentiation, from scratch.
//
// Two layers, mirroring the curriculum:
//   1. `Value`  — a scalar (f32) node: the micrograd engine. Teaches the chain
//                  rule with no array machinery in the way.
//   2. `Var`   — a tensor node: the real engine the rest of the framework uses.
//                  Each op records its inputs and a backward function; calling
//                  `backward()` topo-sorts the graph and accumulates gradients
//                  into every leaf's `.grad`.
//
// Correctness is pinned by `gradCheck`, which compares analytic gradients to
// finite-difference numeric gradients. That harness is the arbiter: if an op's
// backward is wrong, gradCheck catches it.

const std = @import("std");
const mem = std.mem;
const ten = @import("tensor.zig");
const Tensor = ten.Tensor;
const TError = ten.Error;

pub const Error = error{
    GraphError,
    GradientMismatch,
    OutOfMemory,
} || TError;

// ===========================================================================
// 1. Scalar autograd (micrograd)
// ===========================================================================

pub const Value = struct {
    alloc: mem.Allocator,
    data: f32,
    grad: f32 = 0,
    parents: []const *Value = &.{},
    backward: ?*const fn (*const Value) void = null,
    ctx: ?*anyopaque = null,

    pub fn leaf(alloc: mem.Allocator, x: f32) *Value {
        const v = alloc.create(Value) catch unreachable;
        v.* = .{ .alloc = alloc, .data = x };
        return v;
    }

    // build the graph and propagate grads from this node to all leaves
    pub fn backprop(self: *Value) void {
        // topological order via DFS
        var topo: std.ArrayList(*Value) = .empty;
        defer topo.deinit(self.alloc);
        var visited = std.AutoHashMap(*Value, void).init(self.alloc);
        defer visited.deinit();
        buildTopo(self.alloc, self, &topo, &visited) catch return;
        // seed
        self.grad = 1.0;
        // reverse order
        var i: usize = topo.items.len;
        while (i > 0) {
            i -= 1;
            if (topo.items[i].backward) |f| f(topo.items[i]);
        }
    }
};

fn buildTopo(alloc: mem.Allocator, v: *Value, topo: *std.ArrayList(*Value), visited: *std.AutoHashMap(*Value, void)) !void {
    if (visited.contains(v)) return;
    try visited.put(v, {});
    for (v.parents) |p| try buildTopo(alloc, p, topo, visited);
    try topo.append(alloc, v);
}

// --- scalar ops (methods on Value, micrograd style: x.mul(y).add(x)) ---
const AddV = struct { a: *Value, b: *Value };
fn addBackV(self: *const Value) void {
    const c: *AddV = @ptrCast(@alignCast(self.ctx.?));
    c.a.grad += self.grad;
    c.b.grad += self.grad;
}
pub fn vAddS(alloc: mem.Allocator, a: *Value, b: *Value) *Value {
    const out = Value.leaf(alloc, a.data + b.data);
    out.parents = blk: {
        const ps = alloc.alloc(*Value, 2) catch unreachable;
        ps[0] = a;
        ps[1] = b;
        break :blk ps;
    };
    const ctx = alloc.create(AddV) catch unreachable;
    ctx.* = .{ .a = a, .b = b };
    out.ctx = ctx;
    out.backward = addBackV;
    return out;
}

const MulV = struct { a: *Value, b: *Value };
fn mulBackV(self: *const Value) void {
    const c: *MulV = @ptrCast(@alignCast(self.ctx.?));
    c.a.grad += self.grad * c.b.data;
    c.b.grad += self.grad * c.a.data;
}
pub fn vMulS(alloc: mem.Allocator, a: *Value, b: *Value) *Value {
    const out = Value.leaf(alloc, a.data * b.data);
    const ps = alloc.alloc(*Value, 2) catch unreachable;
    ps[0] = a;
    ps[1] = b;
    out.parents = ps;
    const ctx = alloc.create(MulV) catch unreachable;
    ctx.* = .{ .a = a, .b = b };
    out.ctx = ctx;
    out.backward = mulBackV;
    return out;
}

const ReluV = struct { a: *Value };
fn reluBackV(self: *const Value) void {
    const c: *ReluV = @ptrCast(@alignCast(self.ctx.?));
    if (c.a.data > 0) c.a.grad += self.grad;
}
pub fn vReluS(alloc: mem.Allocator, a: *Value) *Value {
    const out = Value.leaf(alloc, if (a.data > 0) a.data else 0);
    const ps = alloc.alloc(*Value, 1) catch unreachable;
    ps[0] = a;
    out.parents = ps;
    const ctx = alloc.create(ReluV) catch unreachable;
    ctx.* = .{ .a = a };
    out.ctx = ctx;
    out.backward = reluBackV;
    return out;
}

// ===========================================================================
// 2. Tensor autograd (Var)
// ===========================================================================

pub const Var = struct {
    alloc: mem.Allocator,
    value: Tensor,
    grad: Tensor,
    parents: []const *Var = &.{},
    backward: ?*const fn (*const Var) void = null,
    ctx: ?*anyopaque = null,
    requires_grad: bool = true,

    // a leaf whose gradient we want (a parameter)
    pub fn parameter(alloc: mem.Allocator, t: Tensor) !*Var {
        const v = try alloc.create(Var);
        v.* = .{ .alloc = alloc, .value = t, .grad = try ten.zeros(alloc, t.shape) };
        return v;
    }
    // a leaf constant (no grad needed, but we keep a zero grad buffer for uniformity)
    pub fn constant(alloc: mem.Allocator, t: Tensor) !*Var {
        const v = try alloc.create(Var);
        v.* = .{ .alloc = alloc, .value = t, .grad = try ten.zeros(alloc, t.shape), .requires_grad = false };
        return v;
    }

    pub fn zerosGrad(self: *Var) void {
        @memset(self.grad.data, 0);
    }

    // propagate grads from self to all reachable leaves
    pub fn backprop(self: *Var) void {
        var topo: std.ArrayList(*Var) = .empty;
        defer topo.deinit(self.alloc);
        var visited = std.AutoHashMap(*Var, void).init(self.alloc);
        defer visited.deinit();
        buildTopoVar(self.alloc, self, &topo, &visited) catch return;
        // zero all grads in the graph, then seed self.grad = 1
        for (topo.items) |n| n.zerosGrad();
        // seed: scalar loss -> grad = 1
        if (self.value.numel() == 1) {
            self.grad.data[0] = 1.0;
        } else {
            // non-scalar backward: caller must seed self.grad before calling
        }
        var i: usize = topo.items.len;
        while (i > 0) {
            i -= 1;
            if (topo.items[i].backward) |f| f(topo.items[i]);
        }
    }
};

fn buildTopoVar(alloc: mem.Allocator, v: *Var, topo: *std.ArrayList(*Var), visited: *std.AutoHashMap(*Var, void)) !void {
    if (visited.contains(v)) return;
    try visited.put(v, {});
    for (v.parents) |p| try buildTopoVar(alloc, p, topo, visited);
    try topo.append(alloc, v);
}

// ---- gradient accumulation with un-broadcasting ----
// Sum `incoming` (which may be broadcast-larger than target) down to target's
// shape and add it into target in place. This is the core of every backward.
pub fn accumulate(alloc: mem.Allocator, target: *Tensor, incoming: Tensor) void {
    if (shapeEq(target.shape, incoming.shape)) {
        for (0..target.numel()) |i| target.data[i] += incoming.data[i];
        return;
    }
    const reduced = unbroadcast(alloc, incoming, target.shape) catch return;
    for (0..target.numel()) |i| target.data[i] += reduced.data[i];
}

fn shapeEq(a: []const usize, b: []const usize) bool {
    if (a.len != b.len) return false;
    for (a, b) |x, y| if (x != y) return false;
    return true;
}

// reduce `g` to `target_shape` by summing over broadcast dims (numpy sum-to-shape)
pub fn unbroadcast(alloc: mem.Allocator, g: Tensor, target_shape: []const usize) !Tensor {
    var cur = g;
    // drop leading extra dims by summing axis 0 (keeps dim=1) then reshaping it off
    while (cur.ndim() > target_shape.len) {
        const summed = try ten.sumAxis(alloc, cur, 0);
        var sh: [8]usize = undefined;
        @memcpy(sh[0 .. summed.ndim() - 1], summed.shape[1..]);
        cur = try ten.reshape(alloc, summed, sh[0 .. summed.ndim() - 1]);
    }
    // same ndim now; sum dims where target is 1 but cur > 1
    var d: usize = 0;
    while (d < cur.ndim()) : (d += 1) {
        if (target_shape[d] == 1 and cur.shape[d] != 1) {
            cur = try ten.sumAxis(alloc, cur, d);
        }
    }
    return cur;
}

// ===========================================================================
// Var ops
// ===========================================================================

fn makeVar(alloc: mem.Allocator, value: Tensor, parents: []const *Var, ctx: anytype, back: *const fn (*const Var) void) *Var {
    const v = alloc.create(Var) catch unreachable;
    v.* = .{
        .alloc = alloc,
        .value = value,
        .grad = ten.zeros(alloc, value.shape) catch unreachable,
        .parents = parents,
        .backward = back,
        .ctx = ctx,
    };
    return v;
}

// ---- add ----
const AddCtx = struct {};
fn addBack(self: *const Var) void {
    const g = self.grad;
    accumulate(self.alloc, &self.parents[0].grad, g);
    accumulate(self.alloc, &self.parents[1].grad, g);
}
pub fn vAdd(alloc: mem.Allocator, a: *Var, b: *Var) !*Var {
    const val = try ten.add(alloc, a.value, b.value);
    const ps = try alloc.alloc(*Var, 2);
    ps[0] = a;
    ps[1] = b;
    return makeVar(alloc, val, ps, @as(*AddCtx, undefined), addBack);
}

// ---- sub ----
fn subBack(self: *const Var) void {
    const g = self.grad;
    accumulate(self.alloc, &self.parents[0].grad, g);
    // b gets -g
    const ng = ten.mulScalar(self.alloc, g, -1.0) catch return;
    accumulate(self.alloc, &self.parents[1].grad, ng);
}
pub fn vSub(alloc: mem.Allocator, a: *Var, b: *Var) !*Var {
    const val = try ten.sub(alloc, a.value, b.value);
    const ps = try alloc.alloc(*Var, 2);
    ps[0] = a;
    ps[1] = b;
    return makeVar(alloc, val, ps, @as(*AddCtx, undefined), subBack);
}

// ---- mul (elementwise, broadcast) ----
const MulCtx = struct {};
fn mulBack(self: *const Var) void {
    const g = self.grad;
    const a = self.parents[0];
    const b = self.parents[1];
    const ga = ten.mul(self.alloc, g, b.value) catch return;
    accumulate(self.alloc, &a.grad, ga);
    const gb = ten.mul(self.alloc, g, a.value) catch return;
    accumulate(self.alloc, &b.grad, gb);
}
pub fn vMul(alloc: mem.Allocator, a: *Var, b: *Var) !*Var {
    const val = try ten.mul(alloc, a.value, b.value);
    const ps = try alloc.alloc(*Var, 2);
    ps[0] = a;
    ps[1] = b;
    return makeVar(alloc, val, ps, @as(*MulCtx, undefined), mulBack);
}

// ---- mulScalar ----
const MulScalarCtx = struct { s: f32 };
fn mulScalarBack(self: *const Var) void {
    const c: *const MulScalarCtx = @ptrCast(@alignCast(self.ctx.?));
    const g = self.grad;
    // d/dx (s*x) = s ; a.grad += g * s
    const gs = ten.mulScalar(self.alloc, g, c.s) catch return;
    accumulate(self.alloc, &self.parents[0].grad, gs);
}
pub fn vMulScalar(alloc: mem.Allocator, a: *Var, s: f32) !*Var {
    const val = try ten.mulScalar(alloc, a.value, s);
    const ps = try alloc.alloc(*Var, 1);
    ps[0] = a;
    const ctx = try alloc.create(MulScalarCtx);
    ctx.* = .{ .s = s };
    return makeVar(alloc, val, ps, ctx, mulScalarBack);
}

// ---- addScalar ----
fn addScalarBack(self: *const Var) void {
    accumulate(self.alloc, &self.parents[0].grad, self.grad);
}
pub fn vAddScalar(alloc: mem.Allocator, a: *Var, s: f32) !*Var {
    const val = try ten.addScalar(alloc, a.value, s);
    const ps = try alloc.alloc(*Var, 1);
    ps[0] = a;
    return makeVar(alloc, val, ps, @as(*AddCtx, undefined), addScalarBack);
}

// ---- neg ----
fn negBack(self: *const Var) void {
    const ng = ten.mulScalar(self.alloc, self.grad, -1.0) catch return;
    accumulate(self.alloc, &self.parents[0].grad, ng);
}
pub fn vNeg(alloc: mem.Allocator, a: *Var) !*Var {
    const val = try ten.neg(alloc, a.value);
    const ps = try alloc.alloc(*Var, 1);
    ps[0] = a;
    return makeVar(alloc, val, ps, @as(*AddCtx, undefined), negBack);
}

// ---- matmul ----
const MatmulCtx = struct {};
fn matmulBack(self: *const Var) void {
    const g = self.grad;
    const a = self.parents[0];
    const b = self.parents[1];
    // da = g @ b^T
    const bt = ten.transpose2D(self.alloc, b.value) catch return;
    const da = ten.matmul(self.alloc, g, bt) catch return;
    accumulate(self.alloc, &a.grad, da);
    // db = a^T @ g
    const at = ten.transpose2D(self.alloc, a.value) catch return;
    const db = ten.matmul(self.alloc, at, g) catch return;
    accumulate(self.alloc, &b.grad, db);
}
pub fn vMatmul(alloc: mem.Allocator, a: *Var, b: *Var) !*Var {
    const val = try ten.matmul(alloc, a.value, b.value);
    const ps = try alloc.alloc(*Var, 2);
    ps[0] = a;
    ps[1] = b;
    return makeVar(alloc, val, ps, @as(*MatmulCtx, undefined), matmulBack);
}

// ---- relu ----
fn reluBack(self: *const Var) void {
    const a = self.parents[0];
    const g = self.grad;
    // grad where a.value > 0
    const out = ten.zeros(self.alloc, a.value.shape) catch return;
    const total = a.value.numel();
    for (0..total) |i| {
        const av = a.value.data[ten.physOffset(a.value, i)];
        out.data[i] = if (av > 0) g.data[i] else 0;
    }
    accumulate(self.alloc, &a.grad, out);
}
pub fn vRelu(alloc: mem.Allocator, a: *Var) !*Var {
    const val = try ten.relu(alloc, a.value);
    const ps = try alloc.alloc(*Var, 1);
    ps[0] = a;
    return makeVar(alloc, val, ps, @as(*AddCtx, undefined), reluBack);
}

// ---- sumAll -> scalar ----
fn sumAllBack(self: *const Var) void {
    const a = self.parents[0];
    const s = self.grad.data[0];
    const out = ten.full(self.alloc, a.value.shape, s) catch return;
    accumulate(self.alloc, &a.grad, out);
}
pub fn vSumAll(alloc: mem.Allocator, a: *Var) !*Var {
    const val = try ten.sumAll(alloc, a.value);
    const ps = try alloc.alloc(*Var, 1);
    ps[0] = a;
    return makeVar(alloc, val, ps, @as(*AddCtx, undefined), sumAllBack);
}

// ---- transpose2D (swap last two) ----
fn transposeBack(self: *const Var) void {
    const a = self.parents[0];
    const g = self.grad;
    // inverse of swap-last-2 is swap-last-2
    const out = ten.transpose2D(self.alloc, g) catch return;
    accumulate(self.alloc, &a.grad, out);
}
pub fn vTranspose2D(alloc: mem.Allocator, a: *Var) !*Var {
    const val = try ten.transpose2D(alloc, a.value);
    const ps = try alloc.alloc(*Var, 1);
    ps[0] = a;
    return makeVar(alloc, val, ps, @as(*AddCtx, undefined), transposeBack);
}

// ---- reshape (view): same elements, new shape. backward reshapes grad back.
// The forward copies (autograd tensors are independently owned); the backward
// reshapes the gradient to the parent's original shape. Since reshape is a pure
// memory reinterpretation when contiguous, this is exact. Needed to turn a flat
// [B*L, dim] embedding into [B, L*dim] for a context MLP, etc.
fn reshapeBack(self: *const Var) void {
    const a = self.parents[0];
    const g = self.grad;
    // g has this node's shape; reshape it back to the parent's shape
    const out = ten.reshape(self.alloc, g, a.value.shape) catch return;
    accumulate(self.alloc, &a.grad, out);
}
pub fn vReshape(alloc: mem.Allocator, a: *Var, new_shape: []const usize) !*Var {
    const val = try ten.reshape(alloc, a.value, new_shape);
    const ps = try alloc.alloc(*Var, 1);
    ps[0] = a;
    return makeVar(alloc, val, ps, @as(*AddCtx, undefined), reshapeBack);
}

// ---- indexRows: gather rows of `table` by `idx` ----
// table: [V, D], idx: [T] -> out: [T, D] where out[t] = table[idx[t]]
// The node OWNS a copy of `idx` (stored in IndexCtx) so the backward can read it
// regardless of what the caller does with its slice afterward — important under
// Zig 0.16's ArenaAllocator, whose `free` reclaims the most-recent allocation
// (shrinking the bump pointer), so a caller `defer free(idx)` could invalidate
// a borrowed idx before backward runs.
const IndexCtx = struct { idx: []usize };
fn indexBack(self: *const Var) void {
    const c: *const IndexCtx = @ptrCast(@alignCast(self.ctx.?));
    const table = self.parents[0];
    const g = self.grad; // [T, D]
    const D = table.value.shape[1];
    for (c.idx, 0..) |tok, t| {
        const row = if (tok < table.value.shape[0]) tok else 0; // match forward clamp
        for (0..D) |d| {
            table.grad.data[row * D + d] += g.data[t * D + d];
        }
    }
}
pub fn vIndexRows(alloc: mem.Allocator, table: *Var, idx: []const usize) !*Var {
    const V = table.value.shape[0];
    const D = table.value.shape[1];
    const val = try ten.zeros(alloc, &.{ idx.len, D });
    for (idx, 0..) |tok, t| {
        const row = if (tok < V) tok else 0;
        for (0..D) |d| val.data[t * D + d] = table.value.data[row * D + d];
    }
    const ps = try alloc.alloc(*Var, 1);
    ps[0] = table;
    const ctx = try alloc.create(IndexCtx);
    ctx.* = .{ .idx = try alloc.dupe(usize, idx) }; // owned copy
    return makeVar(alloc, val, ps, ctx, indexBack);
}

// ---- fused softmax cross-entropy loss ----
// logits: [B, C], targets: [B] (class indices) -> scalar loss
// loss = mean_b( -log softmax(logits[b])[targets[b]] )
// backward: dlogits = (softmax - onehot) / B
const SoftmaxCECtx = struct { targets: []const usize, probs: Tensor };
fn softmaxCEBack(self: *const Var) void {
    const c: *const SoftmaxCECtx = @ptrCast(@alignCast(self.ctx.?));
    const logits = self.parents[0];
    const B = c.targets.len;
    const C = logits.value.shape[1];
    const scale: f32 = 1.0 / @as(f32, @floatFromInt(B));
    // dlogits = (probs - onehot) / B
    const out = ten.zeros(self.alloc, logits.value.shape) catch return;
    for (0..B) |b| {
        const tgt = c.targets[b];
        for (0..C) |k| {
            const p = c.probs.data[b * C + k];
            const one: f32 = if (k == tgt) 1.0 else 0.0;
            out.data[b * C + k] = (p - one) * scale * self.grad.data[0];
        }
    }
    accumulate(self.alloc, &logits.grad, out);
}
pub fn vSoftmaxCE(alloc: mem.Allocator, logits: *Var, targets: []const usize) !*Var {
    const B = targets.len;
    const C = logits.value.shape[1];
    const probs = try ten.softmax(alloc, logits.value); // [B, C]
    var loss: f32 = 0;
    for (0..B) |b| {
        const tgt = targets[b];
        const p = @max(probs.data[b * C + tgt], 1e-12);
        loss -= @log(p);
    }
    loss /= @as(f32, @floatFromInt(B));
    const val = try ten.fromSlice(alloc, &.{}, &.{loss});
    const ps = try alloc.alloc(*Var, 1);
    ps[0] = logits;
    const ctx = try alloc.create(SoftmaxCECtx);
    ctx.* = .{ .targets = targets, .probs = probs };
    return makeVar(alloc, val, ps, ctx, softmaxCEBack);
}

// ===========================================================================
// Gate 8: transformer ops — sigmoid, silu, rmsnorm, permute, causal attention
// ===========================================================================

// ---- sigmoid (elementwise) ----
// out = 1 / (1 + exp(-x));  backward: din = out*(1-out)*grad
fn sigmoidBack(self: *const Var) void {
    const a = self.parents[0];
    const g = self.grad;
    const n = a.value.numel();
    const out = ten.zeros(self.alloc, a.value.shape) catch return;
    for (0..n) |i| {
        const s = self.value.data[i];
        out.data[i] = s * (1.0 - s) * g.data[i];
    }
    accumulate(self.alloc, &a.grad, out);
}
pub fn vSigmoid(alloc: mem.Allocator, a: *Var) !*Var {
    const val = try ten.sigmoid(alloc, a.value);
    const ps = try alloc.alloc(*Var, 1);
    ps[0] = a;
    return makeVar(alloc, val, ps, @as(*AddCtx, undefined), sigmoidBack);
}

// ---- silu / swish (elementwise): out = x * sigmoid(x) ----
// backward: din = (sig + x*sig*(1-sig)) * grad
fn siluBack(self: *const Var) void {
    const a = self.parents[0];
    const g = self.grad;
    const n = a.value.numel();
    const out = ten.zeros(self.alloc, a.value.shape) catch return;
    for (0..n) |i| {
        const x = a.value.data[i];
        const s = 1.0 / (1.0 + @exp(-x)); // sigmoid(x), recomputed (NOT silu/|x| — that flips sign for x<0)
        // d/dx [x*sig(x)] = sig(x) + x*sig(x)*(1-sig(x)) = sig(x)*(1 + x*(1-sig(x)))
        out.data[i] = s * (1.0 + x * (1.0 - s)) * g.data[i];
    }
    accumulate(self.alloc, &a.grad, out);
}
pub fn vSilu(alloc: mem.Allocator, a: *Var) !*Var {
    const n = a.value.numel();
    const val = try ten.zeros(alloc, a.value.shape);
    for (0..n) |i| {
        const x = a.value.data[i];
        const s = 1.0 / (1.0 + @exp(-x));
        val.data[i] = x * s;
    }
    const ps = try alloc.alloc(*Var, 1);
    ps[0] = a;
    return makeVar(alloc, val, ps, @as(*AddCtx, undefined), siluBack);
}

// ---- exp (elementwise): out = exp(x); backward: din = out * grad (= exp(x)*grad) ----
fn expBack(self: *const Var) void {
    const a = self.parents[0];
    const g = self.grad;
    const n = a.value.numel();
    const out = ten.zeros(self.alloc, a.value.shape) catch return;
    for (0..n) |i| out.data[i] = self.value.data[i] * g.data[i]; // self.value = exp(x)
    accumulate(self.alloc, &a.grad, out);
}
pub fn vExp(alloc: mem.Allocator, a: *Var) !*Var {
    const n = a.value.numel();
    const val = try ten.zeros(alloc, a.value.shape);
    for (0..n) |i| val.data[i] = @exp(a.value.data[i]);
    const ps = try alloc.alloc(*Var, 1);
    ps[0] = a;
    return makeVar(alloc, val, ps, @as(*AddCtx, undefined), expBack);
}

// ---- RMSNorm over the LAST axis: out = x / sqrt(mean(x^2) + eps) * gamma ----
// x: [..., D], gamma: [D] (broadcast). The normalization is per-row along D.
// This is LayerNorm without the mean-subtraction (and without beta). The
// backward is the standard RMSNorm gradient:
//   let r = 1/sqrt(ms + eps), ms = mean(x^2)
//   dout_x_i = gamma_i * r * (grad_i - x_i * mean(x * (gamma*grad)) * r^2)
//             where mean is over the last axis.
const RmsNormCtx = struct { gamma: *Var };
fn rmsNormBack(self: *const Var) void {
    const c: *const RmsNormCtx = @ptrCast(@alignCast(self.ctx.?));
    const x = self.parents[0];
    const g = self.grad; // [..., D], same shape as out
    const shape = x.value.shape;
    const D = shape[shape.len - 1];
    const nrows = x.value.numel() / D;
    const gamma = c.gamma.value.data;
    const dx = ten.zeros(self.alloc, shape) catch return;
    const r = self.alloc.alloc(f32, nrows) catch return;
    defer self.alloc.free(r);
    // precompute r[row] = 1/sqrt(mean(x_row^2)+eps) (matches forward)
    for (0..nrows) |row| {
        var ms: f32 = 0;
        for (0..D) |i| {
            const xv = x.value.data[row * D + i];
            ms += xv * xv;
        }
        ms /= @as(f32, @floatFromInt(D));
        r[row] = 1.0 / @sqrt(ms + 1e-5);
    }
    for (0..nrows) |row| {
        // c = mean over D of (x_i * gamma_i * grad_i)
        var c_sum: f32 = 0;
        for (0..D) |i| {
            c_sum += x.value.data[row * D + i] * gamma[i] * g.data[row * D + i];
        }
        c_sum /= @as(f32, @floatFromInt(D));
        const rr = r[row];
        for (0..D) |i| {
            // dx_i = gamma_i * rr * (g_i - x_i * c_sum * rr^2)
            dx.data[row * D + i] = gamma[i] * rr * (g.data[row * D + i] - x.value.data[row * D + i] * c_sum * rr * rr);
        }
    }
    accumulate(self.alloc, &x.grad, dx);
    // dgamma = sum over rows of (x_norm_i * grad_i) where x_norm_i = x_i * rr
    const dgamma = ten.zeros(self.alloc, c.gamma.value.shape) catch return;
    for (0..nrows) |row| {
        const rr = r[row];
        for (0..D) |i| {
            dgamma.data[i] += x.value.data[row * D + i] * rr * g.data[row * D + i];
        }
    }
    accumulate(self.alloc, &c.gamma.grad, dgamma);
}
pub fn vRmsNorm(alloc: mem.Allocator, a: *Var, gamma: *Var) !*Var {
    const shape = a.value.shape;
    const D = shape[shape.len - 1];
    const nrows = a.value.numel() / D;
    const val = try ten.zeros(alloc, shape);
    const gd = gamma.value.data;
    for (0..nrows) |row| {
        var ms: f32 = 0;
        for (0..D) |i| {
            const xv = a.value.data[row * D + i];
            ms += xv * xv;
        }
        ms /= @as(f32, @floatFromInt(D));
        const r = 1.0 / @sqrt(ms + 1e-5);
        for (0..D) |i| val.data[row * D + i] = a.value.data[row * D + i] * r * gd[i];
    }
    const ps = try alloc.alloc(*Var, 2);
    ps[0] = a;
    ps[1] = gamma;
    const ctx = try alloc.create(RmsNormCtx);
    ctx.* = .{ .gamma = gamma };
    return makeVar(alloc, val, ps, ctx, rmsNormBack);
}

// ---- permute (general transpose): reorder axes. backward = inverse permute. ----
const PermuteCtx = struct { order: []usize, inv: []usize };
fn permuteBack(self: *const Var) void {
    const c: *const PermuteCtx = @ptrCast(@alignCast(self.ctx.?));
    const a = self.parents[0];
    const g = self.grad;
    // invert the grad back to the parent's axis order
    const out = ten.permute(self.alloc, g, c.inv) catch return;
    accumulate(self.alloc, &a.grad, out);
}
pub fn vPermute(alloc: mem.Allocator, a: *Var, order: []const usize) !*Var {
    const val = try ten.permute(alloc, a.value, order);
    const ps = try alloc.alloc(*Var, 1);
    ps[0] = a;
    // inverse permutation
    const inv = try alloc.alloc(usize, order.len);
    for (order, 0..) |ax, i| inv[ax] = i;
    const ctx = try alloc.create(PermuteCtx);
    ctx.* = .{ .order = try alloc.dupe(usize, order), .inv = inv };
    return makeVar(alloc, val, ps, ctx, permuteBack);
}

// ---- causal scaled-dot-product attention (single custom op, grad-checkable) ----
// q, k, v: [B, L, D] (batched; for multi-head the caller reshapes so B = B*H and
// D = head_dim). Returns out: [B, L, D] = softmax_causal(q @ k^T / sqrt(D)) @ v.
// The causal mask zeros out positions j > i (future), so row i attends only to
// columns 0..=i. This is the "manual" reference attention that Gate 9's flash
// kernel must match, and the softmax-attention baseline Gate 10's DeltaNet is
// compared against.
const AttnCtx = struct { attn: Tensor, D: usize };
fn attnBack(self: *const Var) void {
    const c: *const AttnCtx = @ptrCast(@alignCast(self.ctx.?));
    const q = self.parents[0];
    const k = self.parents[1];
    const v = self.parents[2];
    const g = self.grad; // [B, L, D]
    const B = q.value.shape[0];
    const L = q.value.shape[1];
    const D = c.D;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    const attn = c.attn.data; // [B, L, L]

    const dq = ten.zeros(self.alloc, q.value.shape) catch return;
    const dk = ten.zeros(self.alloc, k.value.shape) catch return;
    const dv = ten.zeros(self.alloc, v.value.shape) catch return;
    for (0..B) |b| {
        for (0..L) |i| {
            for (0..L) |j| {
                if (j > i) continue; // masked, no gradient flows
                const a = attn[b * L * L + i * L + j];
                // dattn[i,j] = sum_d g[i,d] * v[j,d]
                var dattn: f32 = 0;
                for (0..D) |d| dattn += g.data[b * L * D + i * D + d] * v.value.data[b * L * D + j * D + d];
                // dscores[i,j] = attn[i,j] * (dattn[i,j] - rowsum(attn*dattn))
                // rowsum over j of attn[i,j]*dattn[i,j]:
                var rowsum: f32 = 0;
                for (0..L) |jj| {
                    if (jj > i) continue;
                    const aj = attn[b * L * L + i * L + jj];
                    var dj2: f32 = 0;
                    for (0..D) |d| dj2 += g.data[b * L * D + i * D + d] * v.value.data[b * L * D + jj * D + d];
                    rowsum += aj * dj2;
                }
                const dscores = a * (dattn - rowsum);
                // dv[j,d] += attn[i,j] * g[i,d]
                for (0..D) |d| dv.data[b * L * D + j * D + d] += a * g.data[b * L * D + i * D + d];
                // dq[i,d] += dscores * k[j,d] * scale
                for (0..D) |d| dq.data[b * L * D + i * D + d] += dscores * k.value.data[b * L * D + j * D + d] * scale;
                // dk[j,d] += dscores * q[i,d] * scale
                for (0..D) |d| dk.data[b * L * D + j * D + d] += dscores * q.value.data[b * L * D + i * D + d] * scale;
            }
        }
    }
    accumulate(self.alloc, &q.grad, dq);
    accumulate(self.alloc, &k.grad, dk);
    accumulate(self.alloc, &v.grad, dv);
}
pub fn vCausalMHA(alloc: mem.Allocator, q: *Var, k: *Var, v: *Var) !*Var {
    const B = q.value.shape[0];
    const L = q.value.shape[1];
    const D = q.value.shape[2];
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    // scores [B,L,L] with causal mask, attn = softmax over last axis (per row)
    const attn = try ten.zeros(alloc, &.{ B, L, L });
    for (0..B) |b| {
        for (0..L) |i| {
            var mx: f32 = -1e30;
            for (0..L) |j| {
                if (j > i) continue;
                var s: f32 = 0;
                for (0..D) |d| s += q.value.data[b * L * D + i * D + d] * k.value.data[b * L * D + j * D + d];
                s *= scale;
                if (s > mx) mx = s;
                // stash raw score temporarily in attn (overwrite with prob below)
                attn.data[b * L * L + i * L + j] = s;
            }
            var z: f32 = 0;
            for (0..L) |j| {
                if (j > i) continue;
                const p = @exp(attn.data[b * L * L + i * L + j] - mx);
                attn.data[b * L * L + i * L + j] = p;
                z += p;
            }
            const invz = if (z > 0) 1.0 / z else 0.0;
            for (0..L) |j| {
                if (j > i) {
                    attn.data[b * L * L + i * L + j] = 0.0;
                } else {
                    attn.data[b * L * L + i * L + j] *= invz;
                }
            }
        }
    }
    // out [B,L,D] = attn @ v
    const val = try ten.zeros(alloc, &.{ B, L, D });
    for (0..B) |b| {
        for (0..L) |i| {
            for (0..L) |j| {
                const a = attn.data[b * L * L + i * L + j];
                if (a == 0) continue;
                for (0..D) |d| val.data[b * L * D + i * D + d] += a * v.value.data[b * L * D + j * D + d];
            }
        }
    }
    const ps = try alloc.alloc(*Var, 3);
    ps[0] = q;
    ps[1] = k;
    ps[2] = v;
    const ctx = try alloc.create(AttnCtx);
    ctx.* = .{ .attn = attn, .D = D };
    return makeVar(alloc, val, ps, ctx, attnBack);
}

// ---- Grouped-Query Attention (GQA) ------------------------------------------
// Same scaled-dot-product causal attention as vCausalMHA, but several query
// heads share one key/value head. Inputs are already in the "flat heads" layout
// produced by splitHeads: q [Bq, L, D], k [Bkv, L, D], v [Bkv, L, D] with
// Bq = group * Bkv. The [B*H] indexing is interleaved (batch then head), and
// because H_q = group*H_kv, the kv head for query row qi is exactly qi / group
// (integer division distributes across the batch stride — see comment above).
// With group = 1 this reduces bit-for-bit to vCausalMHA. We keep the explicit
// [Bq, L, L] attention matrix so the math is easy to read; vFlashAttn below is
// the memory-fused version that never materializes it.
const GqaCtx = struct { attn: Tensor, D: usize, group: usize };
fn gqaBack(self: *const Var) void {
    const c: *const GqaCtx = @ptrCast(@alignCast(self.ctx.?));
    const q = self.parents[0];
    const k = self.parents[1];
    const v = self.parents[2];
    const g = self.grad; // [Bq, L, D]
    const Bq = q.value.shape[0];
    const L = q.value.shape[1];
    const D = c.D;
    const group = c.group;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    const attn = c.attn.data; // [Bq, L, L]

    const dq = ten.zeros(self.alloc, q.value.shape) catch return;
    const dk = ten.zeros(self.alloc, k.value.shape) catch return; // [Bkv, L, D]
    const dv = ten.zeros(self.alloc, v.value.shape) catch return;
    for (0..Bq) |qi| {
        const kv = qi / group; // shared kv head index
        for (0..L) |i| {
            for (0..L) |j| {
                if (j > i) continue;
                const a = attn[qi * L * L + i * L + j];
                // dattn[i,j] = sum_d g[qi,i,d] * v[kv,j,d]
                var dattn: f32 = 0;
                for (0..D) |d| dattn += g.data[qi * L * D + i * D + d] * v.value.data[kv * L * D + j * D + d];
                // rowsum over j of attn[i,j]*dattn[i,j] (softmax backward)
                var rowsum: f32 = 0;
                for (0..L) |jj| {
                    if (jj > i) continue;
                    const aj = attn[qi * L * L + i * L + jj];
                    var dj2: f32 = 0;
                    for (0..D) |d| dj2 += g.data[qi * L * D + i * D + d] * v.value.data[kv * L * D + jj * D + d];
                    rowsum += aj * dj2;
                }
                const dscores = a * (dattn - rowsum);
                // dv[kv,j,d] += attn[i,j] * g[qi,i,d]   (accumulates across the group)
                for (0..D) |d| dv.data[kv * L * D + j * D + d] += a * g.data[qi * L * D + i * D + d];
                // dq[qi,i,d] += dscores * k[kv,j,d] * scale
                for (0..D) |d| dq.data[qi * L * D + i * D + d] += dscores * k.value.data[kv * L * D + j * D + d] * scale;
                // dk[kv,j,d] += dscores * q[qi,i,d] * scale  (accumulates across the group)
                for (0..D) |d| dk.data[kv * L * D + j * D + d] += dscores * q.value.data[qi * L * D + i * D + d] * scale;
            }
        }
    }
    accumulate(self.alloc, &q.grad, dq);
    accumulate(self.alloc, &k.grad, dk);
    accumulate(self.alloc, &v.grad, dv);
}
pub fn vCausalGQA(alloc: mem.Allocator, q: *Var, k: *Var, v: *Var, group: usize) !*Var {
    const Bq = q.value.shape[0];
    const Bkv = k.value.shape[0];
    const L = q.value.shape[1];
    const D = q.value.shape[2];
    std.debug.assert(Bq == group * Bkv);
    std.debug.assert(D == k.value.shape[2] and D == v.value.shape[2]);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    const attn = try ten.zeros(alloc, &.{ Bq, L, L });
    for (0..Bq) |qi| {
        const kv = qi / group;
        for (0..L) |i| {
            var mx: f32 = -1e30;
            for (0..L) |j| {
                if (j > i) continue;
                var s: f32 = 0;
                for (0..D) |d| s += q.value.data[qi * L * D + i * D + d] * k.value.data[kv * L * D + j * D + d];
                s *= scale;
                if (s > mx) mx = s;
                attn.data[qi * L * L + i * L + j] = s;
            }
            var z: f32 = 0;
            for (0..L) |j| {
                if (j > i) continue;
                const p = @exp(attn.data[qi * L * L + i * L + j] - mx);
                attn.data[qi * L * L + i * L + j] = p;
                z += p;
            }
            const invz = if (z > 0) 1.0 / z else 0.0;
            for (0..L) |j| {
                if (j > i) {
                    attn.data[qi * L * L + i * L + j] = 0.0;
                } else {
                    attn.data[qi * L * L + i * L + j] *= invz;
                }
            }
        }
    }
    const val = try ten.zeros(alloc, &.{ Bq, L, D });
    for (0..Bq) |qi| {
        const kv = qi / group;
        for (0..L) |i| {
            for (0..L) |j| {
                const a = attn.data[qi * L * L + i * L + j];
                if (a == 0) continue;
                for (0..D) |d| val.data[qi * L * D + i * D + d] += a * v.value.data[kv * L * D + j * D + d];
            }
        }
    }
    const ps = try alloc.alloc(*Var, 3);
    ps[0] = q;
    ps[1] = k;
    ps[2] = v;
    const ctx = try alloc.create(GqaCtx);
    ctx.* = .{ .attn = attn, .D = D, .group = group };
    return makeVar(alloc, val, ps, ctx, gqaBack);
}

// ---- Flash-style fused causal attention (online softmax, no score matrix) ----
// Computes the SAME result as vCausalGQA but never allocates the [Bq, L, L]
// attention matrix: each query row is produced with the online (running-max)
// softmax, accumulating the output incrementally. This is the CPU analog of
// the flash-attention kernel idea — trade a little recomputation for a large
// memory saving (L*L -> O(L) working set per query row). Supports GQA via the
// same group parameter. The backward RECOMPUTES the softmax probabilities from
// q and k (we chose not to stash the matrix, so we pay it back in the backward)
// and then applies the standard softmax-attention gradient with the kv grouping.
const FlashCtx = struct { D: usize, group: usize };
fn flashBack(self: *const Var) void {
    const c: *const FlashCtx = @ptrCast(@alignCast(self.ctx.?));
    const q = self.parents[0];
    const k = self.parents[1];
    const v = self.parents[2];
    const g = self.grad; // [Bq, L, D]
    const Bq = q.value.shape[0];
    const L = q.value.shape[1];
    const D = c.D;
    const group = c.group;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));

    const dq = ten.zeros(self.alloc, q.value.shape) catch return;
    const dk = ten.zeros(self.alloc, k.value.shape) catch return;
    const dv = ten.zeros(self.alloc, v.value.shape) catch return;
    // recompute one row of attention probs at a time (no [Bq,L,L] buffer)
    const row = self.alloc.alloc(f32, L) catch return;
    defer self.alloc.free(row);
    for (0..Bq) |qi| {
        const kv = qi / group;
        for (0..L) |i| {
            // recompute softmax probs for query row qi, position i (j in 0..=i)
            var mx: f32 = -1e30;
            for (0..L) |j| {
                if (j > i) {
                    row[j] = 0;
                    continue;
                }
                var s: f32 = 0;
                for (0..D) |d| s += q.value.data[qi * L * D + i * D + d] * k.value.data[kv * L * D + j * D + d];
                s *= scale;
                row[j] = s;
                if (s > mx) mx = s;
            }
            var z: f32 = 0;
            for (0..L) |j| {
                if (j > i) continue;
                row[j] = @exp(row[j] - mx);
                z += row[j];
            }
            const invz = if (z > 0) 1.0 / z else 0.0;
            for (0..L) |j| {
                if (j > i) continue;
                row[j] *= invz; // row[j] = attn[qi,i,j]
            }
            // standard softmax-attention backward with these probs
            for (0..L) |j| {
                if (j > i) continue;
                const a = row[j];
                var dattn: f32 = 0;
                for (0..D) |d| dattn += g.data[qi * L * D + i * D + d] * v.value.data[kv * L * D + j * D + d];
                var rowsum: f32 = 0;
                for (0..L) |jj| {
                    if (jj > i) continue;
                    const aj = row[jj];
                    var dj2: f32 = 0;
                    for (0..D) |d| dj2 += g.data[qi * L * D + i * D + d] * v.value.data[kv * L * D + jj * D + d];
                    rowsum += aj * dj2;
                }
                const dscores = a * (dattn - rowsum);
                for (0..D) |d| dv.data[kv * L * D + j * D + d] += a * g.data[qi * L * D + i * D + d];
                for (0..D) |d| dq.data[qi * L * D + i * D + d] += dscores * k.value.data[kv * L * D + j * D + d] * scale;
                for (0..D) |d| dk.data[kv * L * D + j * D + d] += dscores * q.value.data[qi * L * D + i * D + d] * scale;
            }
        }
    }
    accumulate(self.alloc, &q.grad, dq);
    accumulate(self.alloc, &k.grad, dk);
    accumulate(self.alloc, &v.grad, dv);
}
pub fn vFlashAttn(alloc: mem.Allocator, q: *Var, k: *Var, v: *Var, group: usize) !*Var {
    const Bq = q.value.shape[0];
    const Bkv = k.value.shape[0];
    const L = q.value.shape[1];
    const D = q.value.shape[2];
    std.debug.assert(Bq == group * Bkv);
    std.debug.assert(D == k.value.shape[2] and D == v.value.shape[2]);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    const val = try ten.zeros(alloc, &.{ Bq, L, D });
    // per-query-row online softmax: running max m, running denom l, running output o[D]
    const o = try alloc.alloc(f32, D);
    defer alloc.free(o);
    for (0..Bq) |qi| {
        const kv = qi / group;
        for (0..L) |i| {
            var m: f32 = -1e30;
            var l: f32 = 0;
            for (0..D) |d| o[d] = 0;
            for (0..L) |j| {
                if (j > i) continue;
                var s: f32 = 0;
                for (0..D) |d| s += q.value.data[qi * L * D + i * D + d] * k.value.data[kv * L * D + j * D + d];
                s *= scale;
                const m_new = if (s > m) s else m;
                const exp_m = @exp(m - m_new); // rescale factor for the running stats
                const exp_s = @exp(s - m_new);
                l = l * exp_m + exp_s;
                for (0..D) |d| o[d] = o[d] * exp_m + exp_s * v.value.data[kv * L * D + j * D + d];
                m = m_new;
            }
            const invl = if (l > 0) 1.0 / l else 0.0;
            for (0..D) |d| val.data[qi * L * D + i * D + d] = o[d] * invl;
        }
    }
    const ps = try alloc.alloc(*Var, 3);
    ps[0] = q;
    ps[1] = k;
    ps[2] = v;
    const ctx = try alloc.create(FlashCtx);
    ctx.* = .{ .D = D, .group = group };
    return makeVar(alloc, val, ps, ctx, flashBack);
}

// ---- Gated Delta Attention (recurrent form) --------------------------------
// The Gated DeltaNet recurrence (Yang et al. 2024, "Gated Delta Networks").
// Per head, a Dh x Dh associative-memory state S is updated token by token with
// the delta rule under a forget gate (alpha) and a write gate (beta):
//   r_t = S_{t-1} k_t                 (recall the value currently stored for k_t)
//   S_t = alpha_t S_{t-1} + beta_t (v_t - alpha_t r_t) k_t^T   (decay + delta write)
//   o_t = S_t q_t                     (read)
// alpha_t = exp(g_t) is the data-dependent forget gate (exp so the chunked
// parallel form gets clean cumulative ratios); beta_t = sigmoid(b_t) is the
// write strength. Inputs are in the "flat heads" layout: q,k,v [Bh, L, Dh] and
// alpha,beta [Bh, L] (one scalar gate per head per position). The op is purely
// the recurrence with GIVEN alpha,beta; the nn layer applies exp/sigmoid first.
//
// Backward: BPTT through the state. We stash every S_t (t=0..L-1; S_{-1}=0) and
// walk t from L-1 down to 0. Let G = dL/dS_t (Dh x Dh), g_t = dL/do_t. Per step:
//   G += g_t q_t^T                       (o_t = S_t q_t)
//   dq_t   = S_t^T g_t
//   Gk     = G k_t
//   dv_t   = beta_t Gk
//   dk_t   = beta_t G^T v_t - alpha_t beta_t S_{t-1}^T Gk - alpha_t beta_t G^T r_t
//   dalpha_t = <G, S_{t-1}> - beta_t (r_t . Gk)
//   dbeta_t  = (v_t - alpha_t r_t) . Gk
//   G <- alpha_t G (I - beta_t k_t k_t^T) = alpha_t G - alpha_t beta_t Gk k_t^T
const GatedDeltaCtx = struct { S: Tensor, Dh: usize, L: usize, Bh: usize };
fn gatedDeltaBack(self: *const Var) void {
    const c: *const GatedDeltaCtx = @ptrCast(@alignCast(self.ctx.?));
    const q = self.parents[0];
    const k = self.parents[1];
    const v = self.parents[2];
    const alpha = self.parents[3];
    const beta = self.parents[4];
    const g = self.grad; // [Bh, L, Dh]
    const Bh = c.Bh;
    const L = c.L;
    const Dh = c.Dh;
    const S = c.S.data; // [Bh, L, Dh, Dh]

    const dq = ten.zeros(self.alloc, q.value.shape) catch return; // [Bh,L,Dh]
    const dk = ten.zeros(self.alloc, k.value.shape) catch return;
    const dv = ten.zeros(self.alloc, v.value.shape) catch return;
    const da = ten.zeros(self.alloc, alpha.value.shape) catch return; // [Bh,L]
    const db = ten.zeros(self.alloc, beta.value.shape) catch return;
    const G = self.alloc.alloc(f32, Dh * Dh) catch return; // dL/dS_t
    const Gk = self.alloc.alloc(f32, Dh) catch return;
    const Gv = self.alloc.alloc(f32, Dh) catch return;
    const SprevGk = self.alloc.alloc(f32, Dh) catch return;
    const Gr = self.alloc.alloc(f32, Dh) catch return;
    const r = self.alloc.alloc(f32, Dh) catch return;
    const d = self.alloc.alloc(f32, Dh) catch return;
    defer self.alloc.free(G);
    defer self.alloc.free(Gk);
    defer self.alloc.free(Gv);
    defer self.alloc.free(SprevGk);
    defer self.alloc.free(Gr);
    defer self.alloc.free(r);
    defer self.alloc.free(d);

    const ST = L * Dh * Dh; // stride per head in S
    const LT = L * Dh; // stride per head in q/k/v/g
    for (0..Bh) |h| {
        for (0..Dh * Dh) |i| G[i] = 0;
        var t: usize = L;
        while (t > 0) {
            t -= 1;
            const a = alpha.value.data[h * L + t];
            const b = beta.value.data[h * L + t];
            const kt = k.value.data[h * LT + t * Dh .. h * LT + (t + 1) * Dh];
            const vt = v.value.data[h * LT + t * Dh .. h * LT + (t + 1) * Dh];
            const qt = q.value.data[h * LT + t * Dh .. h * LT + (t + 1) * Dh];
            const gt = g.data[h * LT + t * Dh .. h * LT + (t + 1) * Dh];
            const St = S[h * ST + t * Dh * Dh .. h * ST + (t + 1) * Dh * Dh]; // S_t [Dh,Dh]
            // S_{t-1} (or zero for t=0)
            const Sp_off: usize = if (t > 0) (h * ST + (t - 1) * Dh * Dh) else 0;
            const Sp = S[Sp_off .. Sp_off + Dh * Dh];
            const has_sp = t > 0;
            // r = S_{t-1} k_t ; d = v_t - a*r
            for (0..Dh) |i| {
                var ri: f32 = 0;
                if (has_sp) { for (0..Dh) |j| ri += Sp[i * Dh + j] * kt[j]; }
                r[i] = ri;
                d[i] = vt[i] - a * ri;
            }
            // G += g_t q_t^T
            for (0..Dh) |i| for (0..Dh) |j| { G[i * Dh + j] += gt[i] * qt[j]; };
            // Gk = G k_t
            for (0..Dh) |i| {
                var s: f32 = 0;
                for (0..Dh) |j| s += G[i * Dh + j] * kt[j];
                Gk[i] = s;
            }
            // dq_t = S_t^T g_t
            for (0..Dh) |i| {
                var s: f32 = 0;
                for (0..Dh) |j| s += St[j * Dh + i] * gt[j];
                dq.data[h * LT + t * Dh + i] = s;
            }
            // dv_t = b * Gk
            for (0..Dh) |i| dv.data[h * LT + t * Dh + i] = b * Gk[i];
            // Gv = G^T v_t ; SprevGk = S_{t-1}^T Gk ; Gr = G^T r
            for (0..Dh) |i| {
                var gv: f32 = 0;
                var gr: f32 = 0;
                for (0..Dh) |j| {
                    gv += G[j * Dh + i] * vt[j];
                    gr += G[j * Dh + i] * r[j];
                }
                Gv[i] = gv;
                Gr[i] = gr;
                var spg: f32 = 0;
                if (has_sp) { for (0..Dh) |j| spg += Sp[j * Dh + i] * Gk[j]; }
                SprevGk[i] = spg;
            }
            // dk_t = b*Gv - a*b*SprevGk - a*b*Gr
            for (0..Dh) |i| dk.data[h * LT + t * Dh + i] = b * Gv[i] - a * b * SprevGk[i] - a * b * Gr[i];
            // dalpha_t = <G, S_{t-1}> - b*(r . Gk) ; dbeta_t = d . Gk
            var gs: f32 = 0;
            if (has_sp) { for (0..Dh * Dh) |i| gs += G[i] * Sp[i]; }
            var rdotgk: f32 = 0;
            for (0..Dh) |i| rdotgk += r[i] * Gk[i];
            da.data[h * L + t] = gs - b * rdotgk;
            var ddotgk: f32 = 0;
            for (0..Dh) |i| ddotgk += d[i] * Gk[i];
            db.data[h * L + t] = ddotgk;
            // G <- a*G - a*b*Gk k_t^T
            for (0..Dh) |i| for (0..Dh) |j| { G[i * Dh + j] = a * G[i * Dh + j] - a * b * Gk[i] * kt[j]; };
        }
    }
    accumulate(self.alloc, &q.grad, dq);
    accumulate(self.alloc, &k.grad, dk);
    accumulate(self.alloc, &v.grad, dv);
    accumulate(self.alloc, &alpha.grad, da);
    accumulate(self.alloc, &beta.grad, db);
}
pub fn vGatedDelta(alloc: mem.Allocator, q: *Var, k: *Var, v: *Var, alpha: *Var, beta: *Var) !*Var {
    const Bh = q.value.shape[0];
    const L = q.value.shape[1];
    const Dh = q.value.shape[2];
    const val = try ten.zeros(alloc, &.{ Bh, L, Dh });
    const S = try ten.zeros(alloc, &.{ Bh, L, Dh, Dh }); // stashed states for backward
    const ST = L * Dh * Dh;
    const LT = L * Dh;
    for (0..Bh) |h| {
        var state: []f32 = alloc.alloc(f32, Dh * Dh) catch return error.OutOfMemory;
        defer alloc.free(state);
        for (0..Dh * Dh) |i| state[i] = 0;
        for (0..L) |t| {
            const a = alpha.value.data[h * L + t];
            const b = beta.value.data[h * L + t];
            const kt = k.value.data[h * LT + t * Dh .. h * LT + (t + 1) * Dh];
            const vt = v.value.data[h * LT + t * Dh .. h * LT + (t + 1) * Dh];
            const qt = q.value.data[h * LT + t * Dh .. h * LT + (t + 1) * Dh];
            // r = S k ; d = v - a*r ; S <- a*S + b * outer(d, k)
            for (0..Dh) |i| {
                var ri: f32 = 0;
                for (0..Dh) |j| ri += state[i * Dh + j] * kt[j];
                const di = vt[i] - a * ri;
                for (0..Dh) |j| state[i * Dh + j] = a * state[i * Dh + j] + b * di * kt[j];
            }
            // stash S_t
            const St = S.data[h * ST + t * Dh * Dh .. h * ST + (t + 1) * Dh * Dh];
            for (0..Dh * Dh) |i| St[i] = state[i];
            // o_t = S q_t
            for (0..Dh) |i| {
                var s: f32 = 0;
                for (0..Dh) |j| s += state[i * Dh + j] * qt[j];
                val.data[h * LT + t * Dh + i] = s;
            }
        }
    }
    const ps = try alloc.alloc(*Var, 5);
    ps[0] = q;
    ps[1] = k;
    ps[2] = v;
    ps[3] = alpha;
    ps[4] = beta;
    const ctx = try alloc.create(GatedDeltaCtx);
    ctx.* = .{ .S = S, .Dh = Dh, .L = L, .Bh = Bh };
    return makeVar(alloc, val, ps, ctx, gatedDeltaBack);
}

// =============================================================================
// Gated Delta Attention — CHUNKED PARALLEL forward (the hero form).
// =============================================================================
// Same recurrence as `vGatedDelta` (S_t = a_t S_{t-1} + b_t(v_t - a_t S_{t-1}
// k_t) k_t^T,  o_t = S_t q_t) but computed chunk-by-chunk with matmuls + a
// triangular solve instead of a Python-style step loop. This is a *tensor*
// function (no autograd here): it is the verified reference for the chunked
// parallel structure. The lesson derives it; the recurrent op carries the
// training backward (BPTT). A fully-parallel analytic backward is the
// capstone exercise.
//
// Derivation (per head, per batch; a=alpha=forget/decay, b=beta=write):
//   1. Absorb the write gate b by scaling: k~ = sqrt(b) k,  v~ = sqrt(b) v.
//      Then S_t = a_t S_{t-1}(I - k~_t k~_t^T) + v~_t k~_t^T.
//   2. Remove the decay by going to the "decay-free" frame S~_t = S_t / G_t
//      where G_t = prod_{m<=t} a_m (G_{-1}=1). Then
//      S~_t = S~_{t-1} + (v~_t/G_t - S~_{t-1} k~_t) k~_t^T,
//      a PLAIN delta rule (no decay) on (k~, v~/G) with state S~.
//   3. Per chunk of size Cc with incoming S~_in: form the chunk Gram and solve
//      T = (I + tril(K~ K~^T, -1))^{-1}   (lower-tri, forward substitution),
//      W = T K~,  U = T (v~/G).  Then (plain-delta parallel identities):
//        S~_out = S~_in + (U - W S~_in^T)^T K~^T
//        o~     = Q S~_in^T + tril(Q K~^T, 0) (U - W S~_in^T)   (incl. diagonal)
//      and the real output is  o_t = G_t * o~_t  (scale back by the decay).
//   4. Carry S~_out and G_end to the next chunk.
// Inputs: q,k,v [Bh, L, Dh]; alpha,beta [Bh, L] (gate VALUES, as in vGatedDelta).
// Output: [Bh, L, Dh]. Requires L % Cc == 0 (the test configs satisfy this).
pub fn gatedDeltaChunkedForward(
    alloc: mem.Allocator,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    beta: Tensor,
    Cc: usize,
) !Tensor {
    const Bh = q.shape[0];
    const L = q.shape[1];
    const Dh = q.shape[2];
    std.debug.assert(L % Cc == 0);
    const nchunks = L / Cc;
    const out = try ten.zeros(alloc, &.{ Bh, L, Dh });
    for (0..Bh) |h| {
        // carried decay-free state S~ [Dh, Dh] and cumulative decay G
        var st = try alloc.alloc(f32, Dh * Dh);
        defer alloc.free(st);
        for (st) |*x| x.* = 0;
        var g_acc: f32 = 1.0; // G_{start-1} for the current chunk
        for (0..nchunks) |ci| {
            const s = ci * Cc; // chunk start (absolute index)
            // local chunk buffers
            const kt = try alloc.alloc(f32, Cc * Dh); // K~  [Cc, Dh]
            defer alloc.free(kt);
            const vt = try alloc.alloc(f32, Cc * Dh); // V~  [Cc, Dh]
            defer alloc.free(vt);
            const qt = try alloc.alloc(f32, Cc * Dh); // Q   [Cc, Dh]
            defer alloc.free(qt);
            // G within chunk: gloc[i] = G_{s+i} = g_acc * prod_{m<=i} a_{s+m}
            const gloc = try alloc.alloc(f32, Cc);
            defer alloc.free(gloc);
            {
                var run: f32 = 1.0;
                for (0..Cc) |i| {
                    const a = alpha.data[h * L + s + i];
                    run *= a;
                    gloc[i] = g_acc * run;
                    const b = beta.data[h * L + s + i];
                    const sb = @sqrt(b);
                    for (0..Dh) |d| {
                        kt[i * Dh + d] = sb * k.data[h * L * Dh + (s + i) * Dh + d];
                        vt[i * Dh + d] = sb * v.data[h * L * Dh + (s + i) * Dh + d] / gloc[i];
                        qt[i * Dh + d] = q.data[h * L * Dh + (s + i) * Dh + d];
                    }
                }
            }
            // Gram K~ K~^T (strict lower for the solve; full lower for output mask)
            const gram = try alloc.alloc(f32, Cc * Cc);
            defer alloc.free(gram);
            for (gram) |*x| x.* = 0;
            for (0..Cc) |i| {
                for (0..Cc) |j| {
                    if (j > i) continue;
                    var s_: f32 = 0;
                    for (0..Dh) |d| s_ += kt[i * Dh + d] * kt[j * Dh + d];
                    gram[i * Cc + j] = s_;
                }
            }
            // T = (I + tril(KK^T,-1))^{-1}: lower-tri unit-diag, forward subst.
            // T = I - N T  (N = strict-lower of gram). T[i][j], j<=i.
            const T = try alloc.alloc(f32, Cc * Cc);
            defer alloc.free(T);
            for (T) |*x| x.* = 0;
            for (0..Cc) |i| {
                T[i * Cc + i] = 1.0;
                for (0..i) |j| {
                    var acc: f32 = 0;
                    var kk: usize = j;
                    while (kk <= i - 1) : (kk += 1) acc += gram[i * Cc + kk] * T[kk * Cc + j];
                    T[i * Cc + j] = -acc;
                }
            }
            // W = T K~  [Cc, Dh] ;  U = T V~  [Cc, Dh]
            const W = try alloc.alloc(f32, Cc * Dh);
            defer alloc.free(W);
            const U = try alloc.alloc(f32, Cc * Dh);
            defer alloc.free(U);
            for (0..Cc) |i| {
                for (0..Dh) |d| {
                    var wv: f32 = 0;
                    var uv: f32 = 0;
                    var kk: usize = 0;
                    while (kk <= i) : (kk += 1) {
                        wv += T[i * Cc + kk] * kt[kk * Dh + d];
                        uv += T[i * Cc + kk] * vt[kk * Dh + d];
                    }
                    W[i * Dh + d] = wv;
                    U[i * Dh + d] = uv;
                }
            }
            // S~_in^T K~^T-style products:  p[i] = W[i] . S~_in  (row [Dh])
            // and qS = Q S~_in^T  -> o~_inter[i] = qt[i] . (S~_in q)... we need
            // (U - W S~_in^T)[i,d] = U[i,d] - (W[i] . S~_in^T-col?) ... S~_in is
            // [Dh,Dh], W[i] is [Dh]. W S~_in^T : row i = W[i] @ S~_in^T = (S~_in W[i])^T,
            // i.e. [i,d] = sum_e W[i,e] * S~_in[d,e]  (= S~_in @ W[i]^T as column).
            const corr = try alloc.alloc(f32, Cc * Dh); // U - W S~_in^T  [Cc, Dh]
            defer alloc.free(corr);
            for (0..Cc) |i| {
                for (0..Dh) |d| {
                    var ws: f32 = 0;
                    for (0..Dh) |e| ws += W[i * Dh + e] * st[d * Dh + e]; // S~_in[d,e]
                    corr[i * Dh + d] = U[i * Dh + d] - ws;
                }
            }
            // o~_inter[i] = Q[i] . (S~_in q?) : o = S~ q  -> (S~_in q)[d] = sum_e S~_in[d,e] q[e]
            // but Q S~_in^T row i = sum_e Q[i,e] S~_in[d,e] -> o_inter[i,d] = sum_e qt[i,e]*st[d,e]
            // o~[i,d] = o_inter[i,d] + sum_{j<=i} (qt[i].kt[j]) * corr[j,d]
            for (0..Cc) |i| {
                for (0..Dh) |d| {
                    var otilde: f32 = 0;
                    for (0..Dh) |e| otilde += qt[i * Dh + e] * st[d * Dh + e]; // Q S~_in^T
                    var j: usize = 0;
                    while (j <= i) : (j += 1) {
                        var qk: f32 = 0;
                        for (0..Dh) |dd| qk += qt[i * Dh + dd] * kt[j * Dh + dd];
                        otilde += qk * corr[j * Dh + d];
                    }
                    // real output = G_t * o~
                    out.data[h * L * Dh + (s + i) * Dh + d] = gloc[i] * otilde;
                }
            }
            // state transfer: S~_out = S~_in + (corr)^T K~^T  [Dh, Dh]
            //   (corr = U - W S~_in^T) ; (corr)^T K~^T : [d,e] = sum_i corr[i,d] * kt[i,e]
            for (0..Dh) |d| {
                for (0..Dh) |e| {
                    var acc: f32 = 0;
                    for (0..Cc) |i| acc += corr[i * Dh + d] * kt[i * Dh + e];
                    st[d * Dh + e] += acc;
                }
            }
            // advance cumulative decay to end of chunk
            g_acc = gloc[Cc - 1];
        }
    }
    return out;
}

test "Gate 10: chunked parallel gated-delta forward == recurrent forward" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(2020);
    const rand = prng.random();
    const Bh: usize = 2;
    const L: usize = 16;
    const Dh: usize = 6;
    const Cc: usize = 4; // L % Cc == 0
    const q = try ten.rand(a, &.{ Bh, L, Dh }, &prng);
    const k = try ten.rand(a, &.{ Bh, L, Dh }, &prng);
    const v = try ten.rand(a, &.{ Bh, L, Dh }, &prng);
    const alpha = try ten.zeros(a, &.{ Bh, L });
    const beta = try ten.zeros(a, &.{ Bh, L });
    for (0..Bh * L) |i| {
        alpha.data[i] = 0.8 + 0.15 * rand.float(f32); // forget ~0.8-0.95
        beta.data[i] = 0.4 + 0.4 * rand.float(f32); // write ~0.4-0.8
    }
    // recurrent reference
    const qv = try Var.parameter(a, q);
    const kv = try Var.parameter(a, k);
    const vv = try Var.parameter(a, v);
    const av = try Var.parameter(a, alpha);
    const bv = try Var.parameter(a, beta);
    const ref = try vGatedDelta(a, qv, kv, vv, av, bv);
    const chk = try gatedDeltaChunkedForward(a, q, k, v, alpha, beta, Cc);
    var max_diff: f32 = 0;
    for (0..Bh * L * Dh) |i| max_diff = @max(max_diff, @abs(ref.value.data[i] - chk.data[i]));
    std.debug.print("  chunked vs recurrent max |diff| = {d:.6} (Bh={d} L={d} Dh={d} Cc={d})\n", .{ max_diff, Bh, L, Dh, Cc });
    try std.testing.expect(max_diff < 1e-3);
}

// ---- RoPE: rotary position embedding on the LAST dim, position = the 2nd axis. ----
// x: [B, L, D] (D even). For position p (the L index) and pair i in [0, D/2),
// using the "half-rotation" (Llama) layout that rotates the pair (i, i + D/2):
//   theta_i = 10000^(-2i/D)
//   ang = p * theta_i
//   x'[i]      = x[i]*cos(ang) - x[i + D/2]*sin(ang)
//   x[i + D/2] = x[i + D/2]*cos(ang) + x[i]*sin(ang)
// The rotation is orthogonal per (position, pair), so the backward applies the
// INVERSE rotation (transpose: cos stays, sin negates) to the gradient. This is
// applied per-head: the caller reshapes Q/K to [B*H, L, Dh] first (position is
// still the L axis). RoPE replaces learned positional embeddings for Q,K.
const RopeCtx = struct { D: usize };
fn ropeBack(self: *const Var) void {
    const c: *const RopeCtx = @ptrCast(@alignCast(self.ctx.?));
    const a = self.parents[0];
    const g = self.grad; // [B, L, D]
    const B = a.value.shape[0];
    const L = a.value.shape[1];
    const D = c.D;
    const half = D / 2;
    const dx = ten.zeros(self.alloc, a.value.shape) catch return;
    for (0..B) |b| {
        for (0..L) |p| {
            const f = p * D;
            for (0..half) |i| {
                const theta_i = std.math.pow(f32, 10000.0, -@as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(D)));
                const ang = @as(f32, @floatFromInt(p)) * theta_i;
                const co = @cos(ang);
                const si = @sin(ang);
                const gx = g.data[b * L * D + f + i];
                const gy = g.data[b * L * D + f + i + half];
                // inverse rotation (sin negated): dx = gx*cos + gy*sin ; dy = gy*cos - gx*sin
                dx.data[b * L * D + f + i] = gx * co + gy * si;
                dx.data[b * L * D + f + i + half] = gy * co - gx * si;
            }
        }
    }
    accumulate(self.alloc, &a.grad, dx);
}
pub fn vRoPE(alloc: mem.Allocator, a: *Var) !*Var {
    const B = a.value.shape[0];
    const L = a.value.shape[1];
    const D = a.value.shape[2];
    const half = D / 2;
    const val = try ten.zeros(alloc, a.value.shape);
    for (0..B) |b| {
        for (0..L) |p| {
            const f = p * D;
            for (0..half) |i| {
                const theta_i = std.math.pow(f32, 10000.0, -@as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(D)));
                const ang = @as(f32, @floatFromInt(p)) * theta_i;
                const co = @cos(ang);
                const si = @sin(ang);
                const x = a.value.data[b * L * D + f + i];
                const y = a.value.data[b * L * D + f + i + half];
                val.data[b * L * D + f + i] = x * co - y * si;
                val.data[b * L * D + f + i + half] = y * co + x * si;
            }
        }
    }
    const ps = try alloc.alloc(*Var, 1);
    ps[0] = a;
    const ctx = try alloc.create(RopeCtx);
    ctx.* = .{ .D = D };
    return makeVar(alloc, val, ps, ctx, ropeBack);
}


// `build` reconstructs the forward graph from the leaves (which live in `ctx`)
// and returns the scalar output Var. We perturb leaf values, rebuild, and
// compare analytic grads (from one backward pass) to numeric (central diff).
//
// Tolerance is the standard combined form (what real frameworks use):
//   |num - ana| <= tol * max(|num|, |ana|) + atol
// `tol` is the RELATIVE tolerance (catches real bugs on well-conditioned grads);
// `atol` is the ABSOLUTE floor for f32 finite-difference roundoff, which dominates
// when a gradient element is tiny (cancellation in outer products, deep-chain
// attenuation, or 1/B averaging). Without the absolute floor, f32 fd roundoff
// (~1e-4) reads as a huge *relative* error on a ~1e-4 gradient and produces false
// failures that have nothing to do with the backward being correct.
pub fn gradCheck(
    comptime Ctx: type,
    alloc: mem.Allocator,
    ctx: *Ctx,
    build: *const fn (mem.Allocator, *Ctx) Error!*Var,
    leaves: []const *Var,
    eps: f32,
    tol: f32,
) Error!void {
    const atol: f32 = 1e-4; // f32 finite-difference roundoff floor
    // analytic pass
    const out0 = try build(alloc, ctx);
    for (leaves) |l| l.zerosGrad();
    out0.backprop();
    // stash analytic grads
    const analytic = try alloc.alloc([]f32, leaves.len);
    defer {
        for (analytic) |a| alloc.free(a);
        alloc.free(analytic);
    }
    for (leaves, 0..) |l, i| {
        analytic[i] = try alloc.alloc(f32, l.value.numel());
        @memcpy(analytic[i], l.grad.data);
    }
    // numeric pass: perturb each element, track the worst scaled error overall.
    // scaled_err = |num-ana| / (tol*max(|num|,|ana|) + atol); a real bug has
    // scaled_err >> 1 on well-conditioned (large) gradients, while f32 fd
    // roundoff only inflates scaled_err on tiny gradients and is bounded by atol.
    var max_scaled: f32 = 0;
    var max_li: usize = 0;
    var max_e: usize = 0;
    var max_num: f32 = 0;
    var max_ana: f32 = 0;
    for (leaves, 0..) |l, li| {
        const n = l.value.numel();
        for (0..n) |e| {
            const orig = l.value.data[e];
            l.value.data[e] = orig + eps;
            const op = try build(alloc, ctx);
            const lp = op.value.data[0];
            l.value.data[e] = orig - eps;
            const om = try build(alloc, ctx);
            const lm = om.value.data[0];
            l.value.data[e] = orig;
            const num = (lp - lm) / (2.0 * eps);
            const ana = analytic[li][e];
            const abs_err = @abs(num - ana);
            const bound = tol * @max(@abs(num), @abs(ana)) + atol;
            const scaled = abs_err / @max(1e-12, bound);
            if (scaled > max_scaled) {
                max_scaled = scaled;
                max_li = li;
                max_e = e;
                max_num = num;
                max_ana = ana;
            }
        }
    }
    std.debug.print("gradCheck max_scaled={d:.4} (leaf {d} elem {d}: num={d:.6} ana={d:.6}) eps={d}\n", .{ max_scaled, max_li, max_e, max_num, max_ana, eps });
    if (max_scaled > 1.0) {
        std.debug.print("gradCheck FAIL: max_scaled {d:.4} > 1.0 (tol={d:.4} atol={d:.4})\n", .{ max_scaled, tol, atol });
        return Error.GradientMismatch;
    }
}

// ===========================================================================
// tests
// ===========================================================================

const testing = std.testing;

test "scalar Value: a*b + a, backward" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const x = Value.leaf(a, 2.0);
    const y = Value.leaf(a, 3.0);
    const z = vAddS(a, vMulS(a, x, y), x); // z = x*y + x = 2*3+2 = 8
    z.backprop();
    try testing.expectApproxEqAbs(@as(f32, 8.0), z.data, 1e-5);
    // dz/dx = y + 1 = 4 ; dz/dy = x = 2
    try testing.expectApproxEqAbs(@as(f32, 4.0), x.grad, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.0), y.grad, 1e-5);
}

const GCTest = struct {
    a: *Var,
    b: *Var,
    fn build(alloc: mem.Allocator, ctx: *GCTest) Error!*Var {
        // z = relu(a @ b)  -> [2,2], then sum to scalar. (shapes must be compatible)
        const mm = try vMatmul(alloc, ctx.a, ctx.b);
        const z = try vRelu(alloc, mm);
        return try vSumAll(alloc, z);
    }
};

test "grad-check: matmul + relu + sumAll" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(1);
    const ta = try ten.rand(a, &.{ 2, 3 }, &prng);
    const tb = try ten.rand(a, &.{ 3, 2 }, &prng);
    const A = try Var.parameter(a, ta);
    const B = try Var.parameter(a, tb);
    var ctx = GCTest{ .a = A, .b = B };
    try gradCheck(GCTest, a, &ctx, GCTest.build, &.{ A, B }, 1e-3, 1e-2);
}

const GCTest2 = struct {
    logits: *Var,
    targets: []const usize,
    fn build(alloc: mem.Allocator, ctx: *GCTest2) Error!*Var {
        return vSoftmaxCE(alloc, ctx.logits, ctx.targets);
    }
};

test "softmax-CE: exact gradient on a hand-computed case" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // 1 sample, 3 classes, logits all 0 -> softmax = [1/3,1/3,1/3], target=0
    // loss = -log(1/3) = ln 3 ; dlogits = softmax - onehot = [-2/3, 1/3, 1/3]
    const L = try Var.parameter(a, try ten.zeros(a, &.{ 1, 3 }));
    const l = try vSoftmaxCE(a, L, &.{0});
    l.backprop();
    try testing.expectApproxEqAbs(@as(f32, @log(3.0)), l.value.data[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, -2.0 / 3.0), L.grad.data[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), L.grad.data[1], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), L.grad.data[2], 1e-4);
}

test "grad-check: softmax cross-entropy" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(7);
    const tl = try ten.rand(a, &.{ 4, 5 }, &prng);
    const L = try Var.parameter(a, tl);
    var ctx = GCTest2{ .logits = L, .targets = &.{ 1, 0, 3, 2 } };
    try gradCheck(GCTest2, a, &ctx, GCTest2.build, &.{L}, 1e-2, 5e-2);
}

const GCTest3 = struct {
    table: *Var,
    idx: []const usize,
    fn build(alloc: mem.Allocator, ctx: *GCTest3) Error!*Var {
        const rows = try vIndexRows(alloc, ctx.table, ctx.idx);
        return try vSumAll(alloc, rows);
    }
};

test "grad-check: indexRows (embedding gather)" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(42);
    const tt = try ten.rand(a, &.{ 6, 4 }, &prng);
    const T = try Var.parameter(a, tt);
    var ctx = GCTest3{ .table = T, .idx = &.{ 0, 3, 3, 1 } };
    try gradCheck(GCTest3, a, &ctx, GCTest3.build, &.{T}, 1e-4, 1e-2);
}

const GCTest4 = struct {
    a: *Var,
    b: *Var,
    fn build(alloc: mem.Allocator, ctx: *GCTest4) Error!*Var {
        // broadcast mul: a:[2,3] b:[1,3]
        const m = try vMul(alloc, ctx.a, ctx.b);
        return try vSumAll(alloc, m);
    }
};

test "grad-check: broadcasted mul" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(99);
    const A = try Var.parameter(a, try ten.rand(a, &.{ 2, 3 }, &prng));
    const B = try Var.parameter(a, try ten.rand(a, &.{ 1, 3 }, &prng));
    var ctx = GCTest4{ .a = A, .b = B };
    try gradCheck(GCTest4, a, &ctx, GCTest4.build, &.{ A, B }, 1e-4, 1e-2);
}

// ===========================================================================
// Gate 8 op grad-checks
// ===========================================================================

// sigmoid + silu: smooth elementwise ops — exact fd, tight tol.
const GCTestSig = struct {
    a: *Var,
    fn build(alloc: mem.Allocator, ctx: *GCTestSig) Error!*Var {
        const s = try vSigmoid(alloc, ctx.a);
        const z = try vSilu(alloc, ctx.a);
        const m = try vMul(alloc, s, z);
        return vSumAll(alloc, m);
    }
};
test "grad-check: sigmoid + silu" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(101);
    const A = try Var.parameter(a, try ten.rand(a, &.{ 3, 4 }, &prng));
    var ctx = GCTestSig{ .a = A };
    try gradCheck(GCTestSig, a, &ctx, GCTestSig.build, &.{A}, 1e-3, 1e-2);
}

// RMSNorm: x:[B,L,D] gamma:[D]. Verifies the per-row normalization backward and
// the gamma gradient (summed over rows). Use a WEIGHTED LINEAR loss
// (sumAll(out * W)) — a squared-sum loss here blows the loss magnitude to O(10+)
// and f32 fd roundoff then swamps the tiny gradient elements that RMSNorm's
// (g - x*c) cancellation produces. A linear loss keeps loss and grads O(1).
const GCTestRms = struct {
    x: *Var,
    gamma: *Var,
    w: *Var,
    fn build(alloc: mem.Allocator, ctx: *GCTestRms) Error!*Var {
        const n = try vRmsNorm(alloc, ctx.x, ctx.gamma);
        const m = try vMul(alloc, n, ctx.w);
        return vSumAll(alloc, m);
    }
};
test "grad-check: RMSNorm" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(202);
    const X = try Var.parameter(a, try ten.rand(a, &.{ 2, 3, 5 }, &prng));
    const G = try Var.parameter(a, try ten.full(a, &.{5}, 1.0));
    const W = try Var.constant(a, try ten.rand(a, &.{ 2, 3, 5 }, &prng));
    var ctx = GCTestRms{ .x = X, .gamma = G, .w = W };
    try gradCheck(GCTestRms, a, &ctx, GCTestRms.build, &.{ X, G }, 1e-3, 1e-2);
}

// permute: 3D [B,L,D] -> [B,D,L] (order 0,2,1). backward inverts.
const GCTestPerm = struct {
    a: *Var,
    fn build(alloc: mem.Allocator, ctx: *GCTestPerm) Error!*Var {
        const p = try vPermute(alloc, ctx.a, &.{ 0, 2, 1 }); // [B,D,L]
        return vSumAll(alloc, p);
    }
};
test "grad-check: permute (transpose 3D)" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(303);
    const A = try Var.parameter(a, try ten.rand(a, &.{ 2, 3, 4 }, &prng));
    var ctx = GCTestPerm{ .a = A };
    try gradCheck(GCTestPerm, a, &ctx, GCTestPerm.build, &.{A}, 1e-4, 1e-2);
}

// causal MHA: q,k,v [B,L,D] -> out [B,L,D]. This is the reference attention op.
// WEIGHTED LINEAR loss (sumAll(out * W)): a squared-sum loss blows the magnitude
// to O(10+) and f32 fd roundoff then swamps the smaller K/V gradient elements
// (attention's softmax + the 1/sqrt(D) scale already shrink some grads). Linear
// loss keeps everything O(1) so fd is accurate; the softmax still exercises the
// nonlinear backward. eps 1e-3 for the softmax's 3rd-order fd error.
const GCTestAttn = struct {
    q: *Var,
    k: *Var,
    v: *Var,
    w: *Var,
    fn build(alloc: mem.Allocator, ctx: *GCTestAttn) Error!*Var {
        const o = try vCausalMHA(alloc, ctx.q, ctx.k, ctx.v); // [B,L,D]
        const m = try vMul(alloc, o, ctx.w);
        return vSumAll(alloc, m);
    }
};
test "grad-check: causal MHA (scaled dot-product attention)" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(404);
    const B: usize = 2;
    const L: usize = 4;
    const D: usize = 6;
    // N(0,1) inputs (NOT *0.5): larger scores -> sharper softmax -> less
    // cancellation in the K/V backward, so every gradient element is O(1e-2+)
    // and f32 fd roundoff (~1e-4 floor) is well below it.
    const Q = try Var.parameter(a, try ten.rand(a, &.{ B, L, D }, &prng));
    const K = try Var.parameter(a, try ten.rand(a, &.{ B, L, D }, &prng));
    const V = try Var.parameter(a, try ten.rand(a, &.{ B, L, D }, &prng));
    const W = try Var.constant(a, try ten.rand(a, &.{ B, L, D }, &prng));
    var ctx = GCTestAttn{ .q = Q, .k = K, .v = V, .w = W };
    try gradCheck(GCTestAttn, a, &ctx, GCTestAttn.build, &.{ Q, K, V }, 1e-3, 2e-2);
}

// RoPE: [B,L,D] -> [B,L,D] (rotate pairs along last dim by position). The
// rotation is orthogonal so a weighted-linear loss keeps grads O(1) and fd exact.
const GCTestRope = struct {
    a: *Var,
    w: *Var,
    fn build(alloc: mem.Allocator, ctx: *GCTestRope) Error!*Var {
        const r = try vRoPE(alloc, ctx.a);
        const m = try vMul(alloc, r, ctx.w);
        return vSumAll(alloc, m);
    }
};
test "grad-check: RoPE (rotary position embedding)" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(505);
    const B: usize = 2;
    const L: usize = 5;
    const D: usize = 8; // even
    const A = try Var.parameter(a, try ten.rand(a, &.{ B, L, D }, &prng));
    const W = try Var.constant(a, try ten.rand(a, &.{ B, L, D }, &prng));
    var ctx = GCTestRope{ .a = A, .w = W };
    try gradCheck(GCTestRope, a, &ctx, GCTestRope.build, &.{A}, 1e-3, 1e-2);
}
// ---- Gate 9: GQA + flash attention grad-checks and numerical equivalence ----
// Both GQA and flash use the WEIGHTED LINEAR loss (sumAll(out * W)) for the same
// reason as the MHA grad-check: keep loss and grads O(1) so f32 finite-difference
// roundoff (~1e-4 floor) stays below the smallest gradient element.

// GQA with group=2: 4 query heads share 2 kv heads. q [Bq,L,Dh], k/v [Bkv,L,Dh],
// Bq = group*Bkv. Exercises the shared-kv accumulation in the backward.
const GCTestGqa = struct {
    q: *Var,
    k: *Var,
    v: *Var,
    w: *Var,
    fn build(alloc: mem.Allocator, ctx: *GCTestGqa) Error!*Var {
        const o = try vCausalGQA(alloc, ctx.q, ctx.k, ctx.v, 2); // [Bq,L,Dh]
        const m = try vMul(alloc, o, ctx.w);
        return vSumAll(alloc, m);
    }
};
test "grad-check: GQA causal attention (group=2, shared kv)" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(606);
    // dims match the MHA grad-check (numel=48): group=2 with Bkv=1 (one kv head
    // shared by both query heads) keeps the loss magnitude ~O(7) and the gradients
    // O(1e-3..1e-1) so f32 fd roundoff (~1e-4) stays below the rel-tolerance bar.
    const Bq: usize = 2; // = group(2) * Bkv(1)
    const Bkv: usize = 1;
    const L: usize = 4;
    const Dh: usize = 6;
    const Q = try Var.parameter(a, try ten.rand(a, &.{ Bq, L, Dh }, &prng));
    const K = try Var.parameter(a, try ten.rand(a, &.{ Bkv, L, Dh }, &prng));
    const V = try Var.parameter(a, try ten.rand(a, &.{ Bkv, L, Dh }, &prng));
    // scale the loss weight by 1/sqrt(numel) so |L| stays O(1): fd roundoff scales
    // with |L| but the absolute tolerance floor is fixed, so this keeps the fd
    // error below the bar without weakening the relative tolerance.
    const ws = 1.0 / @sqrt(@as(f32, @floatFromInt(Bq * L * Dh)));
    const W = try Var.constant(a, try ten.mulScalar(a, try ten.rand(a, &.{ Bq, L, Dh }, &prng), ws));
    var ctx = GCTestGqa{ .q = Q, .k = K, .v = V, .w = W };
    try gradCheck(GCTestGqa, a, &ctx, GCTestGqa.build, &.{ Q, K, V }, 1e-3, 2e-2);
}

// Flash-style fused attention (online softmax, no score matrix) with group=2.
// The backward recomputes the softmax probs from q,k; this checks that recompute
// path matches finite differences.
const GCTestFlash = struct {
    q: *Var,
    k: *Var,
    v: *Var,
    w: *Var,
    fn build(alloc: mem.Allocator, ctx: *GCTestFlash) Error!*Var {
        const o = try vFlashAttn(alloc, ctx.q, ctx.k, ctx.v, 2); // [Bq,L,Dh]
        const m = try vMul(alloc, o, ctx.w);
        return vSumAll(alloc, m);
    }
};
test "grad-check: flash fused attention (online softmax, group=2)" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(707);
    // same dims as the GQA / MHA grad-checks (numel=48) for comparable fd accuracy.
    const Bq: usize = 2;
    const Bkv: usize = 1;
    const L: usize = 4;
    const Dh: usize = 6;
    const Q = try Var.parameter(a, try ten.rand(a, &.{ Bq, L, Dh }, &prng));
    const K = try Var.parameter(a, try ten.rand(a, &.{ Bkv, L, Dh }, &prng));
    const V = try Var.parameter(a, try ten.rand(a, &.{ Bkv, L, Dh }, &prng));
    const ws = 1.0 / @sqrt(@as(f32, @floatFromInt(Bq * L * Dh)));
    const W = try Var.constant(a, try ten.mulScalar(a, try ten.rand(a, &.{ Bq, L, Dh }, &prng), ws));
    var ctx = GCTestFlash{ .q = Q, .k = K, .v = V, .w = W };
    try gradCheck(GCTestFlash, a, &ctx, GCTestFlash.build, &.{ Q, K, V }, 1e-3, 2e-2);
}

// Numerical equivalence: with group=1, flash and GQA must match the reference
// vCausalMHA to within fp tolerance; with group=2, flash must match GQA. This is
// the Gate 9 "max-abs diff < 1e-3" gate.
test "Gate 9: flash & GQA match reference attention (max-abs < 1e-3)" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(808);

    // group=1: q,k,v all [B,L,D]; compare MHA vs GQA(1) vs flash(1)
    {
        const B: usize = 2;
        const L: usize = 5;
        const D: usize = 6;
        const Q = try Var.constant(a, try ten.rand(a, &.{ B, L, D }, &prng));
        const K = try Var.constant(a, try ten.rand(a, &.{ B, L, D }, &prng));
        const V = try Var.constant(a, try ten.rand(a, &.{ B, L, D }, &prng));
        const o_mha = try vCausalMHA(a, Q, K, V);
        const o_gqa = try vCausalGQA(a, Q, K, V, 1);
        const o_fl = try vFlashAttn(a, Q, K, V, 1);
        var d1: f32 = 0;
        var d2: f32 = 0;
        for (o_mha.value.data, o_gqa.value.data, o_fl.value.data) |x, y, z| {
            d1 = @max(d1, @abs(x - y));
            d2 = @max(d2, @abs(x - z));
        }
        try std.testing.expect(d1 < 1e-3);
        try std.testing.expect(d2 < 1e-3);
    }
    // group=2: q [Bq,L,Dh], k/v [Bkv,L,Dh]; compare GQA(2) vs flash(2)
    {
        const Bq: usize = 4;
        const Bkv: usize = 2;
        const L: usize = 5;
        const Dh: usize = 6;
        const Q = try Var.constant(a, try ten.rand(a, &.{ Bq, L, Dh }, &prng));
        const K = try Var.constant(a, try ten.rand(a, &.{ Bkv, L, Dh }, &prng));
        const V = try Var.constant(a, try ten.rand(a, &.{ Bkv, L, Dh }, &prng));
        const o_gqa = try vCausalGQA(a, Q, K, V, 2);
        const o_fl = try vFlashAttn(a, Q, K, V, 2);
        var d: f32 = 0;
        for (o_gqa.value.data, o_fl.value.data) |x, y| d = @max(d, @abs(x - y));
        try std.testing.expect(d < 1e-3);
    }
}

// ---- Gate 10: Gated Delta Attention (recurrent) grad-check ----
// Validates the BPTT backward through the gated delta-rule state recurrence
// against finite differences, for all five inputs (q, k, v, alpha, beta).
// WEIGHTED LINEAR loss scaled by 1/sqrt(numel) (same rationale as the attention
// grad-checks: keep |L|~O(1) so f32 fd roundoff stays below the tolerance bar).
// Gates are kept <1 (decaying) so the short recurrence is numerically stable.
const GTestGatedDelta = struct {
    q: *Var,
    k: *Var,
    v: *Var,
    alpha: *Var,
    beta: *Var,
    w: *Var,
    fn build(alloc: mem.Allocator, ctx: *GTestGatedDelta) Error!*Var {
        const o = try vGatedDelta(alloc, ctx.q, ctx.k, ctx.v, ctx.alpha, ctx.beta); // [Bh,L,Dh]
        const m = try vMul(alloc, o, ctx.w);
        return vSumAll(alloc, m);
    }
};
test "grad-check: gated delta attention (recurrent, 5 inputs)" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(1010);
    const Bh: usize = 2;
    const L: usize = 4;
    const Dh: usize = 4;
    const Q = try Var.parameter(a, try ten.rand(a, &.{ Bh, L, Dh }, &prng));
    const K = try Var.parameter(a, try ten.rand(a, &.{ Bh, L, Dh }, &prng));
    const V = try Var.parameter(a, try ten.rand(a, &.{ Bh, L, Dh }, &prng));
    // alpha ~ 0.8 (forget/decay, <1 for stability); beta ~ 0.5 (write strength)
    const alpha_t = try ten.add(a, try ten.full(a, &.{ Bh, L }, 0.8), try ten.mulScalar(a, try ten.rand(a, &.{ Bh, L }, &prng), 0.1));
    const beta_t = try ten.add(a, try ten.full(a, &.{ Bh, L }, 0.5), try ten.mulScalar(a, try ten.rand(a, &.{ Bh, L }, &prng), 0.15));
    const A = try Var.parameter(a, alpha_t);
    const B = try Var.parameter(a, beta_t);
    const ws = 1.0 / @sqrt(@as(f32, @floatFromInt(Bh * L * Dh)));
    const W = try Var.constant(a, try ten.mulScalar(a, try ten.rand(a, &.{ Bh, L, Dh }, &prng), ws));
    var ctx = GTestGatedDelta{ .q = Q, .k = K, .v = V, .alpha = A, .beta = B, .w = W };
    try gradCheck(GTestGatedDelta, a, &ctx, GTestGatedDelta.build, &.{ Q, K, V, A, B }, 1e-3, 2e-2);
}
