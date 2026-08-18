// optim.zig — optimizers. Each owns per-parameter state and exposes `step()`.
//
// Gate 3 ships SGD with momentum. Adam and Muon (Newton-Schulz orthogonalization)
// are added in Gate 7 on the same `Optimizer` interface.

const std = @import("std");
const mem = std.mem;
const ten = @import("tensor.zig");
const Var = @import("autograd.zig").Var;

pub const Optimizer = union(enum) {
    sgd: SGD,
    adam: Adam,
    muon: Muon,
    hybrid: Hybrid,

    pub fn step(self: *Optimizer, params: []const *Var) void {
        switch (self.*) {
            .sgd => |*o| o.step(params),
            .adam => |*o| o.step(params),
            .muon => |*o| o.step(params),
            .hybrid => |*o| o.step(),
        }
    }

    pub fn setLr(self: *Optimizer, lr: f32) void {
        switch (self.*) {
            .sgd => |*o| o.lr = lr,
            .adam => |*o| o.lr = lr,
            .muon => |*o| o.lr = lr,
            .hybrid => |*o| o.setLr(lr),
        }
    }

    pub fn zeroGrads(self: *Optimizer, params: []const *Var) void {
        _ = self;
        for (params) |p| p.zerosGrad();
    }
};

// ---------------------------------------------------------------------------
// SGD with (optional) momentum and weight decay
//   v = mu * v + grad (+ wd*param)
//   param -= lr * v
// ---------------------------------------------------------------------------

pub const SGD = struct {
    alloc: mem.Allocator,
    lr: f32,
    momentum: f32 = 0.0,
    weight_decay: f32 = 0.0,
    velocities: std.AutoHashMap(usize, []f32),

    pub fn init(alloc: mem.Allocator, lr: f32, momentum: f32, weight_decay: f32) SGD {
        return .{
            .alloc = alloc,
            .lr = lr,
            .momentum = momentum,
            .weight_decay = weight_decay,
            .velocities = std.AutoHashMap(usize, []f32).init(alloc),
        };
    }

    pub fn deinit(self: *SGD) void {
        var it = self.velocities.iterator();
        while (it.next()) |e| self.alloc.free(e.value_ptr.*);
        self.velocities.deinit();
    }

    pub fn step(self: *SGD, params: []const *Var) void {
        for (params) |p| {
            const key = @intFromPtr(p);
            const n = p.value.numel();
            if (self.velocities.getPtr(key)) |vp| {
                const v = vp.*;
                for (0..n) |i| {
                    var g = p.grad.data[i];
                    if (self.weight_decay != 0) g += self.weight_decay * p.value.data[i];
                    v[i] = self.momentum * v[i] + g;
                    p.value.data[i] -= self.lr * v[i];
                }
            } else {
                const v = self.alloc.alloc(f32, n) catch continue;
                @memset(v, 0);
                for (0..n) |i| {
                    var g = p.grad.data[i];
                    if (self.weight_decay != 0) g += self.weight_decay * p.value.data[i];
                    v[i] = self.momentum * v[i] + g;
                    p.value.data[i] -= self.lr * v[i];
                }
                self.velocities.put(key, v) catch {};
            }
        }
    }
};
// ---------------------------------------------------------------------------
// Adam (bias-corrected first/second moments) + optional weight decay
//   m = b1*m + (1-b1)*g
//   v = b2*v + (1-b2)*g^2
//   mhat = m / (1 - b1^t) ; vhat = v / (1 - b2^t)
//   param -= lr * mhat / (sqrt(vhat) + eps)
// ---------------------------------------------------------------------------

pub const Adam = struct {
    alloc: mem.Allocator,
    lr: f32,
    b1: f32 = 0.9,
    b2: f32 = 0.999,
    eps: f32 = 1e-8,
    weight_decay: f32 = 0.0,
    t: usize = 0,
    m: std.AutoHashMap(usize, []f32),
    v: std.AutoHashMap(usize, []f32),

    pub fn init(alloc: mem.Allocator, lr: f32, weight_decay: f32) Adam {
        return .{
            .alloc = alloc,
            .lr = lr,
            .weight_decay = weight_decay,
            .m = std.AutoHashMap(usize, []f32).init(alloc),
            .v = std.AutoHashMap(usize, []f32).init(alloc),
        };
    }

    pub fn deinit(self: *Adam) void {
        var mit = self.m.iterator();
        while (mit.next()) |e| self.alloc.free(e.value_ptr.*);
        var vit = self.v.iterator();
        while (vit.next()) |e| self.alloc.free(e.value_ptr.*);
        self.m.deinit();
        self.v.deinit();
    }

    pub fn step(self: *Adam, params: []const *Var) void {
        self.t += 1;
        const t_f: f32 = @floatFromInt(self.t);
        const b1t = std.math.pow(f32, self.b1, t_f);
        const b2t = std.math.pow(f32, self.b2, t_f);
        for (params) |p| {
            const key = @intFromPtr(p);
            const n = p.value.numel();
            const m = self.getOrCreate(&self.m, key, n);
            const v = self.getOrCreate(&self.v, key, n);
            for (0..n) |i| {
                var g = p.grad.data[i];
                if (self.weight_decay != 0) g += self.weight_decay * p.value.data[i];
                m[i] = self.b1 * m[i] + (1.0 - self.b1) * g;
                v[i] = self.b2 * v[i] + (1.0 - self.b2) * g * g;
                const mhat = m[i] / (1.0 - b1t);
                const vhat = v[i] / (1.0 - b2t);
                p.value.data[i] -= self.lr * mhat / (@sqrt(vhat) + self.eps);
            }
        }
    }

    fn getOrCreate(self: *Adam, map: *std.AutoHashMap(usize, []f32), key: usize, n: usize) []f32 {
        if (map.getPtr(key)) |vp| return vp.*;
        const buf = self.alloc.alloc(f32, n) catch return &.{};
        @memset(buf, 0);
        map.put(key, buf) catch {};
        return buf;
    }

    pub fn setLr(self: *Adam, lr: f32) void {
        self.lr = lr;
    }
};

// ---------------------------------------------------------------------------
// Newton-Schulz orthogonalization (the heart of Muon).
//
// Given a 2D gradient matrix G (shape [m, n]), produce an *approximately
// orthogonal* matrix O of the same shape: O = U @ V^T where G = U @ S @ V^T is
// the SVD and each singular value is replaced by its sign (s -> sgn(s)). This is
// the "orthogonal sign factor" — it removes the magnitude structure of the
// gradient and keeps only its direction, which is what Muon updates along.
//
// We do NOT compute an SVD (expensive). Instead we run the 5-step Newton-Schulz
// iteration from the Muon paper: a cubic matrix polynomial
//     X_{k+1} = a*X_k + b*(X X^T) X_k + c*(X X^T)^2 X_k
// with tuned coefficients (a,b,c) = (3.44420, -4.77557, 2.03120). Initialized at
// X_0 = G / ||G||_F (so all singular values are in (0,1)), five iterations
// converge to sign(S) on the spectrum. To keep the iteration well-conditioned we
// always operate on the "short and fat" orientation (transpose if the matrix is
// taller than wide so the Gram matrix X@X^T is the smaller [M,M]).
//
// This routine works on flat row-major f32 buffers and allocates scratch. It is
// deliberately a free function (no Var) so it can be unit-tested in isolation:
// the orthogonality test (O@O^T ≈ I) is the Gate 7 correctness gate.
// ---------------------------------------------------------------------------

pub const NSCoeffs = struct { a: f32, b: f32, c: f32 };
pub const NS5: NSCoeffs = .{ .a = 3.44420, .b = -4.77557, .c = 2.03120 };

pub fn newtonSchulz5(
    alloc: mem.Allocator,
    g: []const f32, // flat row-major [m*n]
    m: usize,
    n: usize,
    out: []f32, // flat row-major [m*n], caller-owned, written in place
    steps: usize,
) !void {
    std.debug.assert(m > 0 and n > 0);
    std.debug.assert(out.len == m * n);
    // operate on the [M, N] orientation with M <= N (transpose if taller than wide)
    const transpose = m > n;
    const M: usize = if (transpose) n else m;
    const N: usize = if (transpose) m else n;
    const X = try alloc.alloc(f32, M * N);
    defer alloc.free(X);
    // copy G into X (transposing if needed)
    if (transpose) {
        for (0..m) |i| {
            for (0..n) |j| X[j * M + i] = g[i * n + j]; // [M=j, N=i] = G[i,j]
        }
    } else {
        @memcpy(X, g);
    }
    // normalize: X /= (||X||_F + eps) so singular values land in (0,1)
    var fnorm_sq: f32 = 0;
    for (X) |x| fnorm_sq += x * x;
    const inv_norm = 1.0 / (@sqrt(fnorm_sq) + 1e-7);
    for (X) |*x| x.* *= inv_norm;

    const A = try alloc.alloc(f32, M * M); // Gram matrix X@X^T
    defer alloc.free(A);
    const AA = try alloc.alloc(f32, M * M); // (X@X^T)^2
    defer alloc.free(AA);
    const B = try alloc.alloc(f32, M * M); // b*A + c*AA
    defer alloc.free(B);
    const BX = try alloc.alloc(f32, M * N); // B @ X
    defer alloc.free(BX);

    const coeffs = NS5;
    for (0..steps) |_| {
        // A = X @ X^T  (A[i,j] = sum_k X[i,k]*X[j,k])  -> [M, M]
        for (0..M) |i| {
            for (0..M) |j| {
                var s: f32 = 0;
                for (0..N) |k| s += X[i * N + k] * X[j * N + k];
                A[i * M + j] = s;
            }
        }
        // AA = A @ A
        for (0..M) |i| {
            for (0..M) |j| {
                var s: f32 = 0;
                for (0..M) |k| s += A[i * M + k] * A[k * M + j];
                AA[i * M + j] = s;
            }
        }
        // B = b*A + c*AA ; then BX = B @ X ; then X = a*X + BX
        for (0..M * M) |idx| B[idx] = coeffs.b * A[idx] + coeffs.c * AA[idx];
        for (0..M) |i| {
            for (0..N) |k| {
                var s: f32 = 0;
                for (0..M) |j| s += B[i * M + j] * X[j * N + k];
                BX[i * N + k] = s;
            }
        }
        for (0..M * N) |idx| X[idx] = coeffs.a * X[idx] + BX[idx];
    }

    // copy X back into out (undo transpose)
    if (transpose) {
        for (0..m) |i| {
            for (0..n) |j| out[i * n + j] = X[j * M + i];
        }
    } else {
        @memcpy(out, X);
    }
}

// ---------------------------------------------------------------------------
// Muon: momentum + Newton-Schulz orthogonalization for 2D parameters.
//
//   buf = momentum * buf + (g * sqrt(max(1, m/n)))   # scaled gradient, heavy-ball
//   u   = newtonSchulz5(buf)                         # orthogonal sign factor
//   param -= lr * u
//
// Apply ONLY to 2D matrix parameters. 1D params (biases, norms) and embedding
// tables (2D but updated sparsely, where the orthogonal factor is ill-defined)
// must use Adam instead — see `Hybrid` below for the practical router.
// ---------------------------------------------------------------------------

pub const Muon = struct {
    alloc: mem.Allocator,
    lr: f32,
    momentum: f32 = 0.95,
    nesterov: bool = true,
    ns_steps: usize = 5,
    weight_decay: f32 = 0.0,
    buf: std.AutoHashMap(usize, []f32), // momentum buffer per param

    pub fn init(alloc: mem.Allocator, lr: f32, momentum: f32, weight_decay: f32) Muon {
        return .{
            .alloc = alloc,
            .lr = lr,
            .momentum = momentum,
            .weight_decay = weight_decay,
            .buf = std.AutoHashMap(usize, []f32).init(alloc),
        };
    }

    pub fn deinit(self: *Muon) void {
        var it = self.buf.iterator();
        while (it.next()) |e| self.alloc.free(e.value_ptr.*);
        self.buf.deinit();
    }

    pub fn setLr(self: *Muon, lr: f32) void {
        self.lr = lr;
    }

    pub fn step(self: *Muon, params: []const *Var) void {
        for (params) |p| {
            // Muon is only defined for 2D matrix parameters.
            if (p.value.ndim() != 2) continue;
            const m = p.value.shape[0];
            const n = p.value.shape[1];
            const nn = p.value.numel();
            const key = @intFromPtr(p);
            const b = blk: {
                if (self.buf.getPtr(key)) |vp| break :blk vp.*;
                const nb = self.alloc.alloc(f32, nn) catch continue;
                @memset(nb, 0);
                self.buf.put(key, nb) catch {};
                break :blk nb;
            };
            // gradient scaling: balance wide vs tall matrices (Muon paper).
            const grad_scale = @sqrt(@as(f32, @floatFromInt(@max(m, n))) / @as(f32, @floatFromInt(@min(m, n))));
            // keep the scaled gradient `g` (with weight decay folded in) so we can
            // form the Nesterov lookahead g_eff = g + momentum*buf below.
            const g = self.alloc.alloc(f32, nn) catch continue;
            defer self.alloc.free(g);
            for (0..nn) |i| {
                var gi = p.grad.data[i];
                if (self.weight_decay != 0) gi += self.weight_decay * p.value.data[i];
                gi *= grad_scale;
                g[i] = gi;
            }
            // heavy-ball momentum: buf = momentum*buf + g
            for (0..nn) |i| b[i] = self.momentum * b[i] + g[i];
            // Nesterov: orthogonalize g_eff = g + momentum*buf ; else g_eff = buf
            const g_eff = self.alloc.alloc(f32, nn) catch continue;
            defer self.alloc.free(g_eff);
            if (self.nesterov) {
                for (0..nn) |i| g_eff[i] = g[i] + self.momentum * b[i];
            } else {
                @memcpy(g_eff, b);
            }
            const u = self.alloc.alloc(f32, nn) catch continue;
            defer self.alloc.free(u);
            newtonSchulz5(self.alloc, g_eff, m, n, u, self.ns_steps) catch continue;
            for (0..nn) |i| p.value.data[i] -= self.lr * u[i];
        }
    }
};

// ---------------------------------------------------------------------------
// Hybrid optimizer: the practical "Muon for matrices, Adam for everything else"
// router. Partition the parameter list ONCE at init time:
//   - 2D matrix params NOT in the Adam-override set  -> Muon
//   - 1D params (biases, norms) + overridden 2D params (embeddings, output proj)
//     -> Adam
// Embedding tables and the output projection are 2D but updated sparsely / tied,
// so the orthogonal factor is ill-defined there — route them to Adam explicitly.
// ---------------------------------------------------------------------------

pub const Hybrid = struct {
    alloc: mem.Allocator,
    muon: Muon,
    adam: Adam,
    muon_params: []*Var,
    adam_params: []*Var,

    /// `all_params`: every trainable parameter.
    /// `adam_override`: the subset to send to Adam despite being 2D
    ///   (typically: token/pos embedding tables, the final LM head).
    pub fn init(
        alloc: mem.Allocator,
        all_params: []const *Var,
        adam_override: []const *Var,
        lr_muon: f32,
        momentum: f32,
        lr_adam: f32,
        weight_decay: f32,
    ) !Hybrid {
        // mark the override params by pointer identity
        var override_keys = std.AutoHashMap(usize, void).init(alloc);
        defer override_keys.deinit();
        for (adam_override) |p| override_keys.put(@intFromPtr(p), {}) catch {};
        var muon_list: std.ArrayList(*Var) = .empty;
        var adam_list: std.ArrayList(*Var) = .empty;
        for (all_params) |p| {
            const is2d = p.value.ndim() == 2;
            const overridden = override_keys.contains(@intFromPtr(p));
            if (is2d and !overridden) {
                try muon_list.append(alloc, p);
            } else {
                try adam_list.append(alloc, p);
            }
        }
        return .{
            .alloc = alloc,
            .muon = Muon.init(alloc, lr_muon, momentum, weight_decay),
            .adam = Adam.init(alloc, lr_adam, weight_decay),
            .muon_params = try muon_list.toOwnedSlice(alloc),
            .adam_params = try adam_list.toOwnedSlice(alloc),
        };
    }

    pub fn deinit(self: *Hybrid) void {
        self.muon.deinit();
        self.adam.deinit();
        self.alloc.free(self.muon_params);
        self.alloc.free(self.adam_params);
    }

    pub fn setLr(self: *Hybrid, lr: f32) void {
        // scale both groups together by the scheduler factor
        self.muon.lr = lr;
        self.adam.lr = lr;
    }

    pub fn step(self: *Hybrid) void {
        self.muon.step(self.muon_params);
        self.adam.step(self.adam_params);
    }
};

// ---------------------------------------------------------------------------
// Global-norm gradient clipping (used by both Adam and Muon pipelines).
// Rescales every param's grad so the global L2 norm <= max_norm. Returns the
// pre-clip global norm (useful for logging gradient explosion). If the norm is
// already <= max_norm this is a no-op.
// ---------------------------------------------------------------------------

pub fn clipGradNorm(params: []const *Var, max_norm: f32) f32 {
    var total_sq: f32 = 0;
    for (params) |p| {
        for (p.grad.data) |g| total_sq += g * g;
    }
    const total = @sqrt(total_sq);
    if (total > max_norm and total > 0) {
        const scale = max_norm / total;
        for (params) |p| {
            for (p.grad.data) |*g| g.* *= scale;
        }
    }
    return total;
}

// ---------------------------------------------------------------------------
// LR scheduler: linear warmup -> cosine decay to min_factor.
//   factor(step) = 1.0                          if step < warmup  (ramp)
//   factor(step) = warmup/step (linear)        handled by the ramp branch below
//   factor(step) = min + 0.5*(1-min)*(1+cos(pi*progress))  after warmup
// The training loop multiplies the base lr by this factor each step.
// ---------------------------------------------------------------------------

pub const Scheduler = struct {
    warmup: usize, // steps of linear warmup
    total: usize, // total steps (for cosine period)
    min_factor: f32 = 0.1, // floor the cosine decays to (0 = full decay)

    pub fn factor(self: Scheduler, step: usize) f32 {
        if (step < self.warmup) {
            // linear ramp from 0 -> 1 across the warmup window
            return @as(f32, @floatFromInt(step + 1)) / @as(f32, @floatFromInt(@max(self.warmup, 1)));
        }
        const progress = @as(f32, @floatFromInt(step - self.warmup)) /
            @as(f32, @floatFromInt(@max(self.total -| self.warmup, 1)));
        const clamped = if (progress > 1.0) 1.0 else progress;
        return self.min_factor + 0.5 * (1.0 - self.min_factor) * (1.0 + @cos(std.math.pi * clamped));
    }
};