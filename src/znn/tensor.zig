// tensor.zig — a strided, n-dimensional f32 tensor, implemented from scratch.
//
// Design: a Tensor is a *view* over a flat []f32 buffer, described by
// (data, offset, shape[], strides[]). Row-major layout. Strides can be zero
// (broadcasting) or non-unit (transposes/slices) without copying data.
//
// All factories and ops take an explicit allocator; in training loops we feed
// an arena allocator so a whole forward+backward pass is freed in one shot.

const std = @import("std");
const math = std.math;
const mem = std.mem;

pub const Error = error{
    ShapeMismatch,
    NotContiguous,
    InvalidAxis,
    OutOfMemory,
    InvalidMatMul,
};

pub const Tensor = struct {
    alloc: mem.Allocator,
    data: []f32,
    shape: []usize,
    strides: []usize,
    offset: usize = 0,

    pub fn ndim(self: Tensor) usize {
        return self.shape.len;
    }
    pub fn numel(self: Tensor) usize {
        return prodSlice(self.shape);
    }
    pub fn isContiguous(self: Tensor) bool {
        // row-major contiguity check
        var expected: usize = 1;
        var i: usize = self.ndim();
        while (i > 0) {
            i -= 1;
            if (self.shape[i] == 1) continue; // size-1 dims can have any stride
            if (self.strides[i] != expected) return false;
            expected *= self.shape[i];
        }
        return true;
    }
    pub fn dtype(self: Tensor) type {
        _ = self;
        return f32;
    }

    pub fn deinit(self: *Tensor) void {
        self.alloc.free(self.data);
        self.alloc.free(self.shape);
        self.alloc.free(self.strides);
    }

    // Pretty-print up to a small size (for debugging/teaching apps).
    pub fn print(self: Tensor, writer: anytype) !void {
        try writer.print("shape={any} strides={any} data=[", .{ self.shape, self.strides });
        const n = @min(self.numel(), 16);
        for (0..n) |i| {
            if (i != 0) try writer.writeAll(", ");
            try writer.print("{d:.4}", .{self.atFlat(i)});
        }
        if (self.numel() > 16) try writer.writeAll(", ...");
        try writer.writeAll("]");
    }

    // ---- scalar access helpers ----
    pub fn atFlat(self: Tensor, flat: usize) f32 {
        return self.data[self.offset + flat];
    }
    pub fn setFlat(self: *Tensor, flat: usize, v: f32) void {
        self.data[self.offset + flat] = v;
    }

    // element at a multi-index (vararg of usize)
    pub fn at(self: Tensor, idx: []const usize) f32 {
        return self.data[self.offset + flatIndex(self.strides, idx)];
    }
};

// ---------------------------------------------------------------------------
// shape / stride helpers
// ---------------------------------------------------------------------------

pub fn prodSlice(s: []const usize) usize {
    var p: usize = 1;
    for (s) |x| p *= x;
    return p;
}

pub fn rowMajorStrides(alloc: mem.Allocator, shape: []const usize) ![]usize {
    const nd = shape.len;
    const st = try alloc.alloc(usize, nd);
    if (nd == 0) return st;
    st[nd - 1] = 1;
    var i: usize = nd - 1;
    while (i > 0) {
        i -= 1;
        st[i] = st[i + 1] * shape[i + 1];
    }
    return st;
}

fn flatIndex(strides: []const usize, idx: []const usize) usize {
    var off: usize = 0;
    for (idx, strides) |c, s| off += c * s;
    return off;
}

// Map a row-major *logical* index over `t.shape` to the physical data offset.
// Correct for any strides (contiguous, transposed, broadcast-zero-stride),
// so ops can simply iterate 0..numel in logical order and read via this.
pub fn physOffset(t: Tensor, flat: usize) usize {
    var rem = flat;
    var off = t.offset;
    var i: usize = t.ndim();
    while (i > 0) {
        i -= 1;
        const coord = rem % t.shape[i];
        rem /= t.shape[i];
        off += coord * t.strides[i];
    }
    return off;
}

// clone shape/strides into fresh allocations
fn dupShape(alloc: mem.Allocator, shape: []const usize) ![]usize {
    const out = try alloc.alloc(usize, shape.len);
    @memcpy(out, shape);
    return out;
}

// ---------------------------------------------------------------------------
// factories
// ---------------------------------------------------------------------------

pub fn zeros(alloc: mem.Allocator, shape: []const usize) !Tensor {
    const n = prodSlice(shape);
    const data = try alloc.alloc(f32, n);
    @memset(data, 0);
    const sh = try dupShape(alloc, shape);
    const st = try rowMajorStrides(alloc, shape);
    return .{ .alloc = alloc, .data = data, .shape = sh, .strides = st };
}

pub fn ones(alloc: mem.Allocator, shape: []const usize) !Tensor {
    const t = try zeros(alloc, shape);
    @memset(t.data, 1);
    return t;
}

pub fn full(alloc: mem.Allocator, shape: []const usize, val: f32) !Tensor {
    const t = try zeros(alloc, shape);
    @memset(t.data, val);
    return t;
}

pub fn arange(alloc: mem.Allocator, n: usize) !Tensor {
    const t = try zeros(alloc, &.{n});
    for (0..n, t.data) |i, *x| x.* = @floatFromInt(i);
    return t;
}

pub fn fromSlice(alloc: mem.Allocator, shape: []const usize, src: []const f32) !Tensor {
    const n = prodSlice(shape);
    if (src.len != n) return Error.ShapeMismatch;
    const data = try alloc.alloc(f32, n);
    @memcpy(data, src);
    const sh = try dupShape(alloc, shape);
    const st = try rowMajorStrides(alloc, shape);
    return .{ .alloc = alloc, .data = data, .shape = sh, .strides = st };
}

pub fn rand(alloc: mem.Allocator, shape: []const usize, prng: *std.Random.DefaultPrng) !Tensor {
    const t = try zeros(alloc, shape);
    const r = prng.random();
    for (t.data) |*x| x.* = r.floatNorm(f32);
    return t;
}

// like `zeros` but returns a tensor whose data/shape/strides are independent copies
// (used when we already have a buffer and just need a typed view backed by it)
pub fn emptyLike(alloc: mem.Allocator, ref: Tensor) !Tensor {
    return zeros(alloc, ref.shape);
}

// ---------------------------------------------------------------------------
// reshape / transpose / permute (view-ish, but we materialize for simplicity)
// ---------------------------------------------------------------------------

pub fn reshape(alloc: mem.Allocator, t: Tensor, new_shape: []const usize) !Tensor {
    if (!t.isContiguous()) return Error.NotContiguous;
    if (prodSlice(new_shape) != t.numel()) return Error.ShapeMismatch;
    // share data? For safety in autograd we copy so the result is independently owned.
    const data = try alloc.alloc(f32, t.numel());
    @memcpy(data, t.data[t.offset .. t.offset + t.numel()]);
    const sh = try dupShape(alloc, new_shape);
    const st = try rowMajorStrides(alloc, new_shape);
    return .{ .alloc = alloc, .data = data, .shape = sh, .strides = st };
}

// Swap the two trailing dims, keeping any leading batch dims.
// [.., R, C] -> [.., C, R]. Output is a fresh contiguous tensor.
pub fn transpose2D(alloc: mem.Allocator, t: Tensor) !Tensor {
    if (t.ndim() < 2) return Error.ShapeMismatch;
    const nd = t.ndim();
    var out_shape: [8]usize = undefined;
    @memcpy(out_shape[0 .. nd - 2], t.shape[0 .. nd - 2]);
    out_shape[nd - 2] = t.shape[nd - 1];
    out_shape[nd - 1] = t.shape[nd - 2];
    const out = try zeros(alloc, out_shape[0..nd]);
    const total = out.numel();
    var ocoord: [8]usize = .{0} ** 8;
    var scoord: [8]usize = .{0} ** 8;
    for (0..total) |flat| {
        // decompose flat -> output coords (out is contiguous row-major)
        var rem = flat;
        for (0..nd) |d| {
            ocoord[d] = rem / out.strides[d];
            rem %= out.strides[d];
        }
        @memcpy(scoord[0 .. nd - 2], ocoord[0 .. nd - 2]);
        scoord[nd - 2] = ocoord[nd - 1];
        scoord[nd - 1] = ocoord[nd - 2];
        out.data[flat] = t.at(scoord[0..nd]);
    }
    return out;
}

// general permutation: new axis order, e.g. permute(alloc, t, &.{ 0, 2, 1 })
pub fn permute(alloc: mem.Allocator, t: Tensor, order: []const usize) !Tensor {
    const nd = t.ndim();
    if (order.len != nd) return Error.ShapeMismatch;
    var new_shape: [8]usize = undefined;
    for (order, 0..) |ax, i| new_shape[i] = t.shape[ax];
    const sh = try dupShape(alloc, new_shape[0..nd]);
    const st = try rowMajorStrides(alloc, sh); // output is contiguous row-major
    const data = try alloc.alloc(f32, prodSlice(sh));
    // iterate over all output positions (flat is the contiguous row-major index)
    var idx_buf: [8]usize = .{0} ** 8;
    const total = prodSlice(sh);
    for (0..total) |flat| {
        // decompose flat -> output multi-index using output row-major strides
        var rem = flat;
        for (0..nd) |d| {
            idx_buf[d] = rem / st[d];
            rem %= st[d];
        }
        // map output coord back to source coord via order: out[d] uses old axis order[d]
        var src_idx: [8]usize = .{0} ** 8;
        for (order, 0..) |ax, d| src_idx[ax] = idx_buf[d];
        data[flat] = t.at(src_idx[0..nd]);
    }
    return .{ .alloc = alloc, .data = data, .shape = sh, .strides = st };
}

// ---------------------------------------------------------------------------
// broadcasting for binary elementwise ops
// ---------------------------------------------------------------------------

// Compute the broadcast result shape of two shapes (numpy rules).
pub fn broadcastShapes(a: []const usize, b: []const usize, out: []usize) Error!void {
    const na = a.len;
    const nb = b.len;
    const nout = out.len;
    if (nout != @max(na, nb)) return Error.ShapeMismatch;
    var i: usize = 0;
    while (i < nout) : (i += 1) {
        const ai: ?usize = if (i + na >= nout) a[i + na - nout] else null;
        const bi: ?usize = if (i + nb >= nout) b[i + nb - nout] else null;
        const av = ai orelse 1;
        const bv = bi orelse 1;
        if (av != bv and av != 1 and bv != 1) return Error.ShapeMismatch;
        out[i] = @max(av, bv);
    }
}

// Produce a broadcasted stride layout for `in_shape` so it views into a buffer
// of `out_shape`. Dims where in_shape==1 get stride 0.
fn broadcastStrides(in_shape: []const usize, in_strides: []const usize, out_shape: []const usize, out: []usize) Error!void {
    const nin = in_shape.len;
    const nout = out_shape.len;
    if (nin > nout) return Error.ShapeMismatch;
    var i: usize = 0;
    while (i < nout) : (i += 1) {
        const in_i: ?usize = if (i + nin >= nout) in_shape[i + nin - nout] else null;
        const in_s: ?usize = if (i + nin >= nout) in_strides[i + nin - nout] else null;
        if (in_i == null) {
            out[i] = 0; // prepended dim -> stride 0
        } else if (in_i.? == 1 and out_shape[i] != 1) {
            out[i] = 0; // broadcast dim -> stride 0
        } else {
            out[i] = in_s.?;
        }
    }
}

// Generic binary elementwise op with broadcasting. `comptime op` is the element fn.
fn binaryOp(alloc: mem.Allocator, a: Tensor, b: Tensor, comptime op: fn (f32, f32) f32) !Tensor {
    const na = a.ndim();
    const nb = b.ndim();
    const nout = @max(na, nb);
    if (nout > 8) return Error.ShapeMismatch; // MAX_NDIM guard
    var out_shape: [8]usize = undefined;
    try broadcastShapes(a.shape, b.shape, out_shape[0..nout]);
    var a_st: [8]usize = undefined;
    var b_st: [8]usize = undefined;
    try broadcastStrides(a.shape, a.strides, out_shape[0..nout], a_st[0..nout]);
    try broadcastStrides(b.shape, b.strides, out_shape[0..nout], b_st[0..nout]);

    const out = try zeros(alloc, out_shape[0..nout]);
    const total = out.numel();
    var idx: [8]usize = .{0} ** 8;
    for (0..total) |flat| {
        // decompose flat -> coord using out strides (row-major contiguous: strides known)
        var rem = flat;
        for (0..nout) |d| {
            idx[d] = rem / out.strides[d];
            rem %= out.strides[d];
        }
        var oa: usize = a.offset;
        var ob: usize = b.offset;
        for (0..nout) |d| {
            oa += idx[d] * a_st[d];
            ob += idx[d] * b_st[d];
        }
        out.data[flat] = op(a.data[oa], b.data[ob]);
    }
    return out;
}

fn addf(x: f32, y: f32) f32 {
    return x + y;
}
fn subf(x: f32, y: f32) f32 {
    return x - y;
}
fn mulf(x: f32, y: f32) f32 {
    return x * y;
}
fn divf(x: f32, y: f32) f32 {
    return x / y;
}

pub fn add(alloc: mem.Allocator, a: Tensor, b: Tensor) !Tensor {
    return binaryOp(alloc, a, b, addf);
}
pub fn sub(alloc: mem.Allocator, a: Tensor, b: Tensor) !Tensor {
    return binaryOp(alloc, a, b, subf);
}
pub fn mul(alloc: mem.Allocator, a: Tensor, b: Tensor) !Tensor {
    return binaryOp(alloc, a, b, mulf);
}
pub fn divide(alloc: mem.Allocator, a: Tensor, b: Tensor) !Tensor {
    return binaryOp(alloc, a, b, divf);
}

// scalar ops (broadcast a scalar across the tensor)
pub fn addScalar(alloc: mem.Allocator, t: Tensor, s: f32) !Tensor {
    const out = try zeros(alloc, t.shape);
    const total = t.numel();
    for (0..total) |flat| out.data[flat] = t.data[physOffset(t, flat)] + s;
    return out;
}

pub fn mulScalar(alloc: mem.Allocator, t: Tensor, s: f32) !Tensor {
    const out = try zeros(alloc, t.shape);
    const total = t.numel();
    for (0..total) |flat| out.data[flat] = t.data[physOffset(t, flat)] * s;
    return out;
}

// ---------------------------------------------------------------------------
// unary elementwise activations
// ---------------------------------------------------------------------------

fn unary(alloc: mem.Allocator, t: Tensor, comptime f: fn (f32) f32) !Tensor {
    const out = try zeros(alloc, t.shape);
    const total = t.numel();
    for (0..total) |flat| {
        out.data[flat] = f(t.data[physOffset(t, flat)]);
    }
    return out;
}

pub fn relu(alloc: mem.Allocator, t: Tensor) !Tensor {
    return unary(alloc, t, reluf);
}
fn reluf(x: f32) f32 {
    return if (x > 0) x else 0;
}
pub fn sigmoid(alloc: mem.Allocator, t: Tensor) !Tensor {
    return unary(alloc, t, sigmoidf);
}
fn sigmoidf(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}
pub fn tanh(alloc: mem.Allocator, t: Tensor) !Tensor {
    return unary(alloc, t, tanhf);
}
fn tanhf(x: f32) f32 {
    return std.math.tanh(x);
}
pub fn gelu(alloc: mem.Allocator, t: Tensor) !Tensor {
    return unary(alloc, t, geluf);
}
fn geluf(x: f32) f32 {
    // exact gelu: 0.5 x (1 + erf(x/sqrt2))
    const c: f32 = 0.7071067811865475; // 1/sqrt2
    return 0.5 * x * (1.0 + erfFloat(x * c));
}
fn erfFloat(x: f32) f32 {
    // Abramowitz-Stegun 7.1.26 approximation, f32-accurate enough for teaching
    const a1: f32 = 0.254829592;
    const a2: f32 = -0.284496736;
    const a3: f32 = 1.421413741;
    const a4: f32 = -1.453152027;
    const a5: f32 = 1.061405429;
    const p: f32 = 0.3275911;
    const sign: f32 = if (x < 0) -1.0 else 1.0;
    const ax = @abs(x);
    const t = 1.0 / (1.0 + p * ax);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * @exp(-ax * ax);
    return sign * y;
}
pub fn exp(alloc: mem.Allocator, t: Tensor) !Tensor {
    return unary(alloc, t, expf);
}
fn expf(x: f32) f32 {
    return @exp(x);
}
pub fn neg(alloc: mem.Allocator, t: Tensor) !Tensor {
    return unary(alloc, t, negf);
}
fn negf(x: f32) f32 {
    return -x;
}

// ---------------------------------------------------------------------------
// reductions: sum / mean over an axis (or all)
// ---------------------------------------------------------------------------

// sum over the whole tensor -> scalar (shape {})
pub fn sumAll(alloc: mem.Allocator, t: Tensor) !Tensor {
    var s: f32 = 0;
    const total = t.numel();
    for (0..total) |flat| s += t.data[physOffset(t, flat)];
    return fromSlice(alloc, &.{}, &.{s});
}

// sum over a single axis; keeps the dim as size 1 (like keepdims=true)
pub fn sumAxis(alloc: mem.Allocator, t: Tensor, axis: usize) !Tensor {
    if (axis >= t.ndim()) return Error.InvalidAxis;
    var new_shape: [8]usize = undefined;
    for (t.shape, 0..) |s, i| new_shape[i] = if (i == axis) 1 else s;
    const out = try zeros(alloc, new_shape[0..t.ndim()]);
    const total = t.numel();
    var coord: [8]usize = .{0} ** 8;
    for (0..total) |flat| {
        // decompose flat -> logical coords over t.shape (row-major)
        var rem = flat;
        var i: usize = t.ndim();
        while (i > 0) {
            i -= 1;
            coord[i] = rem % t.shape[i];
            rem /= t.shape[i];
        }
        const src_off = physOffset(t, flat);
        // collapsed output index: same coords but axis->0
        var out_off: usize = 0;
        for (0..t.ndim()) |d| out_off += (if (d == axis) 0 else coord[d]) * out.strides[d];
        out.data[out_off] += t.data[src_off];
    }
    return out;
}

pub fn meanAll(alloc: mem.Allocator, t: Tensor) !Tensor {
    const s = try sumAll(alloc, t);
    const n: f32 = @floatFromInt(t.numel());
    s.data[0] /= n;
    return s;
}

// ---------------------------------------------------------------------------
// matmul: batched, with broadcasting batch dims.
//   a: [..., M, K]   b: [..., K, N]  ->  [..., M, N]
// ---------------------------------------------------------------------------

pub fn matmul(alloc: mem.Allocator, a: Tensor, b: Tensor) !Tensor {
    if (a.ndim() < 2 or b.ndim() < 2) return Error.InvalidMatMul;
    const M = a.shape[a.ndim() - 2];
    const K = a.shape[a.ndim() - 1];
    const K2 = b.shape[b.ndim() - 2];
    const N = b.shape[b.ndim() - 1];
    if (K != K2) return Error.InvalidMatMul;

    // batch dims
    const batch_a = a.shape[0 .. a.ndim() - 2];
    const batch_b = b.shape[0 .. b.ndim() - 2];
    const nbatch = @max(batch_a.len, batch_b.len);
    if (nbatch > 6) return Error.ShapeMismatch; // 6 batch + 2 mat = 8 max
    var batch_shape: [6]usize = undefined;
    try broadcastShapes(batch_a, batch_b, batch_shape[0..nbatch]);
    const B = prodSlice(batch_shape[0..nbatch]);

    var full_shape_buf: [8]usize = undefined;
    @memcpy(full_shape_buf[0..nbatch], batch_shape[0..nbatch]);
    full_shape_buf[nbatch] = M;
    full_shape_buf[nbatch + 1] = N;
    const out = try zeros(alloc, full_shape_buf[0 .. nbatch + 2]);

    // broadcast strides for batch dims
    var a_bst: [6]usize = undefined;
    var b_bst: [6]usize = undefined;
    try broadcastStrides(batch_a, a.strides[0 .. a.ndim() - 2], batch_shape[0..nbatch], a_bst[0..nbatch]);
    try broadcastStrides(batch_b, b.strides[0 .. b.ndim() - 2], batch_shape[0..nbatch], b_bst[0..nbatch]);

    const a_str_M = a.strides[a.ndim() - 2];
    const a_str_K = a.strides[a.ndim() - 1];
    const b_str_K = b.strides[b.ndim() - 2];
    const b_str_N = b.strides[b.ndim() - 1];
    const out_str_M = out.strides[nbatch];
    const out_str_N = out.strides[nbatch + 1];

    var bidx: [6]usize = .{0} ** 6;
    for (0..B) |bi| {
        // decompose bi -> batch coords (row-major: divide by product of later dims)
        var rem = bi;
        for (0..nbatch) |d| {
            var block: usize = 1;
            for (d + 1..nbatch) |e| block *= batch_shape[e];
            bidx[d] = rem / block;
            rem %= block;
        }
        var a_base = a.offset;
        var b_base = b.offset;
        for (0..nbatch) |d| {
            a_base += bidx[d] * a_bst[d];
            b_base += bidx[d] * b_bst[d];
        }
        const out_base = bi * (M * N);
        // classic GEMM
        for (0..M) |i| {
            for (0..N) |j| {
                var acc: f32 = 0;
                for (0..K) |k| {
                    acc += a.data[a_base + i * a_str_M + k * a_str_K] * b.data[b_base + k * b_str_K + j * b_str_N];
                }
                out.data[out_base + i * out_str_M + j * out_str_N] = acc;
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// softmax along last axis, numerically stable (subtract max)
// ---------------------------------------------------------------------------

pub fn softmax(alloc: mem.Allocator, t: Tensor) !Tensor {
    if (t.ndim() == 0) return Error.ShapeMismatch;
    const out = try zeros(alloc, t.shape);
    const last = t.ndim() - 1;
    const D = t.shape[last];
    const outer = if (t.ndim() > 1) prodSlice(t.shape[0..last]) else 1;
    // out is contiguous row-major, so output row o lives at o*D .. o*D+D.
    for (0..outer) |o| {
        // logical flat index of element (leadingCoords..., k) is o*D + k
        var m: f32 = -std.math.inf(f32);
        for (0..D) |k| {
            m = @max(m, t.data[physOffset(t, o * D + k)]);
        }
        var s: f32 = 0;
        for (0..D) |k| {
            const e = @exp(t.data[physOffset(t, o * D + k)] - m);
            out.data[o * D + k] = e;
            s += e;
        }
        const inv = 1.0 / s;
        for (0..D) |k| out.data[o * D + k] *= inv;
    }
    return out;
}

// ---------------------------------------------------------------------------
// equality (exact) for tests
// ---------------------------------------------------------------------------

pub fn approxEq(a: Tensor, b: Tensor, tol: f32) bool {
    if (a.ndim() != b.ndim()) return false;
    for (a.shape, b.shape) |x, y| if (x != y) return false;
    const total = a.numel();
    if (b.numel() != total) return false;
    var idx: [8]usize = .{0} ** 8;
    for (0..total) |flat| {
        var rem = flat;
        for (0..a.ndim()) |d| {
            idx[d] = rem / a.strides[d];
            rem %= a.strides[d];
        }
        var oa = a.offset;
        var ob = b.offset;
        for (0..a.ndim()) |d| {
            oa += idx[d] * a.strides[d];
            ob += idx[d] * b.strides[d];
        }
        if (@abs(a.data[oa] - b.data[ob]) > tol) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "zeros/ones/shape/numel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const t = try ones(a, &.{ 2, 3 });
    try testing.expectEqual(@as(usize, 2), t.shape[0]);
    try testing.expectEqual(@as(usize, 3), t.shape[1]);
    try testing.expectEqual(@as(usize, 6), t.numel());
    try testing.expect(t.isContiguous());
}

test "matmul 2x3 @ 3x2" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const A = try fromSlice(a, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    const B = try fromSlice(a, &.{ 3, 2 }, &.{ 7, 8, 9, 10, 11, 12 });
    const C = try matmul(a, A, B);
    // expected: [[58,64],[139,154]]
    try testing.expectEqual(@as(usize, 2), C.shape[0]);
    try testing.expectEqual(@as(usize, 2), C.shape[1]);
    try testing.expectApproxEqAbs(@as(f32, 58), C.data[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 64), C.data[1], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 139), C.data[2], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 154), C.data[3], 1e-4);
}

test "matmul batched with broadcast" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // A: [2, 2, 3] two stacked matrices; B: [3, 2] broadcast across batch
    const A = try fromSlice(a, &.{ 2, 2, 3 }, &.{ 1, 2, 3, 4, 5, 6, 1, 1, 1, 2, 2, 2 });
    const B = try fromSlice(a, &.{ 3, 2 }, &.{ 1, 0, 0, 1, 1, 1 });
    const C = try matmul(a, A, B);
    try testing.expectEqualSlices(usize, &.{ 2, 2, 2 }, C.shape);
    // batch 0: [[1,2,3],[4,5,6]] @ [[1,0],[0,1],[1,1]] = [[4,5],[9,11]]
    try testing.expectApproxEqAbs(@as(f32, 4), C.data[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 5), C.data[1], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 10), C.data[2], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 11), C.data[3], 1e-4);
    // batch 1: [[1,1,1],[2,2,2]] @ same = [[2,2],[4,4]]
    try testing.expectApproxEqAbs(@as(f32, 2), C.data[4], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 4), C.data[6], 1e-4);
}

test "broadcasting add [2,1]+[1,3]" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const x = try fromSlice(a, &.{ 2, 1 }, &.{ 1, 10 });
    const y = try fromSlice(a, &.{ 1, 3 }, &.{ 1, 2, 3 });
    const z = try add(a, x, y);
    try testing.expectEqualSlices(usize, &.{ 2, 3 }, z.shape);
    // row0: 2,3,4 ; row1: 11,12,13
    try testing.expectApproxEqAbs(@as(f32, 2), z.data[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 3), z.data[1], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 4), z.data[2], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 11), z.data[3], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 12), z.data[4], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 13), z.data[5], 1e-4);
}

test "transpose2D and permute" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const x = try fromSlice(a, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    const xt = try transpose2D(a, x);
    // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
    try testing.expectEqualSlices(usize, &.{ 3, 2 }, xt.shape);
    try testing.expectApproxEqAbs(@as(f32, 1), xt.data[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 4), xt.data[1], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 2), xt.data[2], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 5), xt.data[3], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 3), xt.data[4], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 6), xt.data[5], 1e-4);

    // permute [0,2,1] on a [1,2,3] tensor -> [1,3,2]
    // y=[[[1,2,3],[4,5,6]]]; yp[a][b][c] = y[a][c][b] -> data=[1,4,2,5,3,6]
    const y = try fromSlice(a, &.{ 1, 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    const yp = try permute(a, y, &.{ 0, 2, 1 });
    try testing.expectEqualSlices(usize, &.{ 1, 3, 2 }, yp.shape);
    try testing.expectApproxEqAbs(@as(f32, 1), yp.data[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 4), yp.data[1], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 2), yp.data[2], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 5), yp.data[3], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 3), yp.data[4], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 6), yp.data[5], 1e-4);
}

test "sumAxis keeps dim" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const x = try fromSlice(a, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    const s0 = try sumAxis(a, x, 0); // -> [1,3] = [[5,7,9]]
    try testing.expectEqualSlices(usize, &.{ 1, 3 }, s0.shape);
    try testing.expectApproxEqAbs(@as(f32, 5), s0.data[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 7), s0.data[1], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 9), s0.data[2], 1e-4);
    const s1 = try sumAxis(a, x, 1); // -> [2,1] = [[6],[15]]
    try testing.expectEqualSlices(usize, &.{ 2, 1 }, s1.shape);
    try testing.expectApproxEqAbs(@as(f32, 6), s1.data[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 15), s1.data[1], 1e-4);
}

test "softmax sums to 1 and is correct" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // row0 = [1000,0,0] -> one-hot at 0 ; row1 = [0,0,0] -> uniform 1/3
    const x = try fromSlice(a, &.{ 2, 3 }, &.{ 1000, 0, 0, 0, 0, 0 });
    const p = try softmax(a, x);
    try testing.expectApproxEqAbs(@as(f32, 1.0), p.data[0] + p.data[1] + p.data[2], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 1.0), p.data[3] + p.data[4] + p.data[5], 1e-4);
    // row0 one-hot
    try testing.expectApproxEqAbs(@as(f32, 1.0), p.data[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.0), p.data[1], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.0), p.data[2], 1e-4);
    // row1 uniform 1/3
    try testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), p.data[3], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), p.data[4], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), p.data[5], 1e-4);
}

test "gelu and sigmoid sanity" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const x = try fromSlice(a, &.{3}, &.{ 0, 1, -1 });
    const g = try gelu(a, x);
    try testing.expectApproxEqAbs(@as(f32, 0), g.data[0], 1e-4); // gelu(0)=0
    const s = try sigmoid(a, x);
    try testing.expectApproxEqAbs(@as(f32, 0.5), s.data[0], 1e-4);
}