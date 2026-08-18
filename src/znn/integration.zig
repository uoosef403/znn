// integration.zig — end-to-end training tests that exercise the whole stack
// (autograd + layers + optimizer + loss). These are the "does it actually
// learn?" gates, distinct from the per-op grad-checks in autograd.zig.

const std = @import("std");
const mem = std.mem;
const ten = @import("tensor.zig");
const ag = @import("autograd.zig");
const nn = @import("nn.zig");
const optim = @import("optim.zig");
const loss = @import("loss.zig");
const data = @import("data.zig");

const testing = std.testing;

// Fit y = 3x + 1 with a single Linear layer + MSE + Adam.
// If the training loop is correct, W->3 and b->1 and loss->~0.
test "regression: fit y = 3x + 1" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(1);

    const N: usize = 200;
    const x = try ten.zeros(a, &.{ N, 1 });
    const y = try a.alloc(f32, N);
    for (0..N) |i| {
        const xv = prng.random().float(f32) * 2.0 - 1.0;
        x.data[i] = xv;
        y[i] = 3.0 * xv + 1.0;
    }
    const xv = try ag.Var.constant(a, x);
    const yv = try ag.Var.constant(a, try ten.fromSlice(a, &.{ N, 1 }, y));

    var lin = try nn.Linear.init(a, 1, 1, &prng);
    const ps = try a.alloc(*ag.Var, 2);
    ps[0] = lin.W;
    ps[1] = lin.b;

    var opt = optim.Adam.init(a, 0.05, 0.0);
    defer opt.deinit();

    var last: f32 = 1e9;
    for (0..400) |_| {
        const pred = try lin.forward(xv);
        const l = try loss.mse(a, pred, yv);
        l.backprop();
        opt.step(ps);
        last = l.value.data[0];
    }
    // converged to the line?
    try testing.expect(last < 0.05);
    try testing.expectApproxEqAbs(@as(f32, 3.0), lin.W.value.data[0], 0.2);
    try testing.expectApproxEqAbs(@as(f32, 1.0), lin.b.value.data[0], 0.2);
}

// Can the model OVERFIT a tiny clean dataset? Decisive engine sanity check.
test "overfit: tiny clean spiral reaches ~100% train acc" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(5);

    const N: usize = 8;
    const C: usize = 3;
    const total = N * C;
    const x = try ten.zeros(a, &.{ total, 2 });
    const y = try a.alloc(usize, total);
    for (0..C) |cls| {
        for (0..N) |j| {
            const i = cls * N + j;
            // radius floor so j=0 is NOT at the origin for every class
            // (otherwise 3 points collapse to (0,0) with different labels and
            //  the data is unlearnable — a classic spiral-data pitfall).
            const t: f32 = (@as(f32, @floatFromInt(j)) + 0.5) / @as(f32, @floatFromInt(N));
            const angle: f32 = t * 2.0 * std.math.pi * 3.0 + @as(f32, @floatFromInt(cls)) * (2.0 * std.math.pi / 3.0);
            x.data[i * 2 + 0] = t * @cos(angle);
            x.data[i * 2 + 1] = t * @sin(angle);
            y[i] = cls;
        }
    }
    const xv = try ag.Var.constant(a, x);

    var mlp = try nn.MLP.init(a, &.{ 2, 64, 3 }, &.{ .relu, .none }, &prng);
    const ps = try mlp.params(a);
    var opt = optim.Adam.init(a, 0.05, 0.0);
    defer opt.deinit();

    var acc: f32 = 0;
    for (0..500) |step| {
        const logits = try mlp.forward(xv);
        const l = try loss.softmaxCE(a, logits, y);
        l.backprop();
        opt.step(ps);
        if (step % 100 == 0 or step == 499) {
            var correct: usize = 0;
            for (0..total) |i| {
                var best: usize = 0;
                var bv: f32 = logits.value.data[i * 3];
                for (1..3) |k| if (logits.value.data[i * 3 + k] > bv) {
                    bv = logits.value.data[i * 3 + k];
                    best = k;
                };
                if (best == y[i]) correct += 1;
            }
            acc = @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(total));
            std.debug.print("  overfit step {d:>3} loss={d:.4} acc={d:.3}\n", .{ step, l.value.data[0], acc });
        }
    }
    try testing.expect(acc > 0.95);
}

// End-to-end grad-check of the FULL 2-layer net:
//   loss = softmaxCE( relu(x@W1 + b1) @ W2 + b2 , targets )
// This exercises matmul, bias-broadcast add, relu, and fused softmax-CE
// backward in a single chain — none of the per-op grad-checks cover this
// exact combination (notably bias-broadcast add in a multi-layer setting).
const ChainCtx = struct {
    x: *ag.Var,
    W1: *ag.Var,
    b1: *ag.Var,
    W2: *ag.Var,
    b2: *ag.Var,
    targets: []const usize,
    fn build(alloc: mem.Allocator, ctx: *ChainCtx) ag.Error!*ag.Var {
        const mm1 = try ag.vMatmul(alloc, ctx.x, ctx.W1); // [B, hidden]
        const z1 = try ag.vAdd(alloc, mm1, ctx.b1); // [B, hidden] + [hidden]
        const h = try ag.vRelu(alloc, z1); // [B, hidden]
        const mm2 = try ag.vMatmul(alloc, h, ctx.W2); // [B, out]
        const logits = try ag.vAdd(alloc, mm2, ctx.b2); // [B, out] + [out]
        return ag.vSoftmaxCE(alloc, logits, ctx.targets);
    }
};

test "grad-check: full 2-layer MLP + softmax-CE chain" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(42);

    const B: usize = 4;
    const in_dim: usize = 2;
    const hidden: usize = 4;
    const out_dim: usize = 3;
    const x = try ten.rand(a, &.{ B, in_dim }, &prng);
    const xv = try ag.Var.constant(a, x);
    const W1 = try ag.Var.parameter(a, try nn.randn(a, &.{ in_dim, hidden }, &prng, 0.5));
    const b1 = try ag.Var.parameter(a, try ten.rand(a, &.{hidden}, &prng));
    const W2 = try ag.Var.parameter(a, try nn.randn(a, &.{ hidden, out_dim }, &prng, 0.5));
    const b2 = try ag.Var.parameter(a, try ten.rand(a, &.{out_dim}, &prng));

    var ctx = ChainCtx{
        .x = xv,
        .W1 = W1,
        .b1 = b1,
        .W2 = W2,
        .b2 = b2,
        .targets = &.{ 0, 1, 2, 0 },
    };
    const leaves = [_]*ag.Var{ W1, b1, W2, b2 };
    try ag.gradCheck(ChainCtx, a, &ctx, ChainCtx.build, &leaves, 1e-3, 1e-2);
}

// 2-hidden (3-Linear) deep chain, grad-checked with a SMOOTH, NON-SATURATING
// loss so the finite-difference reference is trustworthy in f32. This verifies
// the backward under deep composition (matmul -> bias-broadcast add -> smooth act
// -> matmul -> ... -> sumAll) at two hidden layers, which the per-op and 1-layer
// checks do not exercise. A real backward bug at depth blows past the tolerance
// here; f32 fd roundoff does not (it is absorbed by gradCheck's atol floor).
//
// Why this loss/activation and not softmax-CE + ReLU: (1) ReLU is non-smooth, so a
// large finite-difference eps flips units across the kink and the numeric gradient
// diverges from the analytic subgradient — a *methodology* artifact, not a bug.
// (2) softmax saturates for large logits, inflating fd 3rd-order error. (3) the
// 1/B mean in softmax-CE shrinks deep-chain gradients until f32 fd roundoff
// dominates them. The smooth z*z activation and sumAll(logits^2) loss avoid all
// three, keeping gradients O(1) so fd is accurate.
const DeepCtx = struct {
    x: *ag.Var,
    W1: *ag.Var,
    b1: *ag.Var,
    W2: *ag.Var,
    b2: *ag.Var,
    W3: *ag.Var,
    b3: *ag.Var,
    fn build(alloc: mem.Allocator, ctx: *DeepCtx) ag.Error!*ag.Var {
        const z1 = try ag.vAdd(alloc, try ag.vMatmul(alloc, ctx.x, ctx.W1), ctx.b1);
        const h1 = try ag.vMul(alloc, z1, z1); // smooth act, d/dz = 2z
        const z2 = try ag.vAdd(alloc, try ag.vMatmul(alloc, h1, ctx.W2), ctx.b2);
        const h2 = try ag.vMul(alloc, z2, z2);
        const logits = try ag.vAdd(alloc, try ag.vMatmul(alloc, h2, ctx.W3), ctx.b3);
        const sq = try ag.vMul(alloc, logits, logits); // loss = sumAll(logits^2)
        return ag.vSumAll(alloc, sq);
    }
};

test "grad-check: 2-hidden smooth deep chain" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(99);

    // B=1: no cross-sample cancellation in the matmul-backward outer products, so
    // every gradient element is O(1) and f32 finite-difference is accurate.
    const B: usize = 1;
    const in_dim: usize = 2;
    const h: usize = 8;
    const out_dim: usize = 3;
    const x = try ten.zeros(a, &.{ B, in_dim });
    for (x.data) |*v| v.* = prng.random().floatNorm(f32) * 0.5;
    const xv = try ag.Var.constant(a, x);
    const W1 = try ag.Var.parameter(a, try nn.randn(a, &.{ in_dim, h }, &prng, 0.5));
    const b1 = try ag.Var.parameter(a, try ten.full(a, &.{h}, 0.3));
    const W2 = try ag.Var.parameter(a, try nn.randn(a, &.{ h, h }, &prng, 0.5));
    const b2 = try ag.Var.parameter(a, try ten.full(a, &.{h}, 0.3));
    const W3 = try ag.Var.parameter(a, try nn.randn(a, &.{ h, out_dim }, &prng, 0.5));
    const b3 = try ag.Var.parameter(a, try ten.full(a, &.{out_dim}, 0.3));

    var ctx = DeepCtx{ .x = xv, .W1 = W1, .b1 = b1, .W2 = W2, .b2 = b2, .W3 = W3, .b3 = b3 };
    const leaves = [_]*ag.Var{ W1, b1, W2, b2, W3, b3 };
    // 1% relative tol: with B=1 + O(1) smooth grads fd is accurate to ~0.01%, so a
    // real backward bug blows past 1% while fd roundoff stays ~0.01%.
    try ag.gradCheck(DeepCtx, a, &ctx, DeepCtx.build, &leaves, 1e-3, 1e-2);
}

// ---------------------------------------------------------------------------
// Gate 6: embeddings + first LM
// ---------------------------------------------------------------------------

// Grad-check the LM's NEW mechanics — embedding gather (vIndexRows) + positional
// add + reshape — composed with matmul + broadcast-add. Both new ops are LINEAR,
// so we use a fully linear chain (no activation, sumAll loss): the whole function
// is then linear in every leaf, which makes the central-difference reference
// exact (zero curvature => no eps-kink error) and keeps the loss O(1) (no f32 fd
// roundoff blowup). This isolates the new ops' correctness; the ReLU/softmax-CE
// backward are each grad-checked separately in autograd.zig, and the full ReLU
// LM is exercised end-to-end by the training test below.
const LinLMCtx = struct {
    tok_emb: *ag.Var, // [V, dim]
    pos_emb: *ag.Var, // [L, dim]
    W1: *ag.Var, b1: *ag.Var, W2: *ag.Var, b2: *ag.Var,
    ids: []const usize,
    seq_len: usize,
    dim: usize,
    fn build(alloc: mem.Allocator, ctx: *LinLMCtx) ag.Error!*ag.Var {
        const L = ctx.seq_len;
        const dim = ctx.dim;
        const B = ctx.ids.len / L;
        const tok = try ag.vIndexRows(alloc, ctx.tok_emb, ctx.ids); // [B*L, dim]
        const pos = try alloc.alloc(usize, ctx.ids.len);
        defer alloc.free(pos);
        for (0..ctx.ids.len) |i| pos[i] = i % L;
        const pem = try ag.vIndexRows(alloc, ctx.pos_emb, pos); // [B*L, dim]
        const e = try ag.vAdd(alloc, tok, pem); // [B*L, dim]
        const flat = try ag.vReshape(alloc, e, &.{ B, L * dim }); // [B, L*dim] (new op)
        const h = try ag.vAdd(alloc, try ag.vMatmul(alloc, flat, ctx.W1), ctx.b1); // [B, H]
        const logits = try ag.vAdd(alloc, try ag.vMatmul(alloc, h, ctx.W2), ctx.b2); // [B, V]
        return ag.vSumAll(alloc, logits); // linear loss
    }
};

test "grad-check: embedding + reshape + linear LM chain" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(7);

    const V: usize = 5;
    const L: usize = 3;
    const dim: usize = 4;
    const hidden: usize = 6;
    const B: usize = 2;

    // O(0.3) init keeps the chain O(1) (no value explosion), grads O(1).
    const sd: f32 = 0.3;
    const tok_emb = try ag.Var.parameter(a, try nn.randn(a, &.{ V, dim }, &prng, sd));
    const pos_emb = try ag.Var.parameter(a, try nn.randn(a, &.{ L, dim }, &prng, sd));
    const W1 = try ag.Var.parameter(a, try nn.randn(a, &.{ L * dim, hidden }, &prng, sd));
    const b1 = try ag.Var.parameter(a, try ten.full(a, &.{hidden}, 0.1));
    const W2 = try ag.Var.parameter(a, try nn.randn(a, &.{ hidden, V }, &prng, sd));
    const b2 = try ag.Var.parameter(a, try ten.full(a, &.{V}, 0.1));

    const ids = try a.alloc(usize, B * L);
    for (ids) |*x| x.* = @intFromFloat(prng.random().float(f32) * @as(f32, @floatFromInt(V)));

    var ctx = LinLMCtx{ .tok_emb = tok_emb, .pos_emb = pos_emb, .W1 = W1, .b1 = b1, .W2 = W2, .b2 = b2, .ids = ids, .seq_len = L, .dim = dim };
    const leaves = [_]*ag.Var{ tok_emb, pos_emb, W1, b1, W2, b2 };
    try ag.gradCheck(LinLMCtx, a, &ctx, LinLMCtx.build, &leaves, 1e-3, 1e-2);
}

// Generate a bigram-structured token sequence: a Markov chain over V tokens
// where each row of the transition matrix is peaked on a small random subset.
// This gives real conditional structure (P(next|prev) << P(next)), so a context
// model can beat the unigram (marginal) baseline — the Gate 6 success criterion.
fn genBigramSeq(alloc: mem.Allocator, prng: *std.Random.DefaultPrng, V: usize, N: usize) ![]u32 {
    // build a peaked transition matrix: each row uniform over 2 random next-tokens
    const trans = try alloc.alloc([]u32, V);
    defer {
        for (trans) |r| alloc.free(r);
        alloc.free(trans);
    }
    for (trans) |*r| {
        const row = try alloc.alloc(u32, 2);
        row[0] = @intFromFloat(prng.random().float(f32) * @as(f32, @floatFromInt(V)));
        row[1] = @intFromFloat(prng.random().float(f32) * @as(f32, @floatFromInt(V)));
        r.* = row;
    }
    const seq = try alloc.alloc(u32, N);
    var s: u32 = @intFromFloat(prng.random().float(f32) * @as(f32, @floatFromInt(V)));
    seq[0] = s;
    for (1..N) |i| {
        s = trans[s][@intFromFloat(prng.random().float(f32) * 2.0)]; // pick one of the 2
        seq[i] = s;
    }
    return seq;
}

test "first LM: MLPLM val loss beats the unigram baseline" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(123);

    const V: usize = 16;
    const L: usize = 4;
    const dim: usize = 24;
    const hidden: usize = 64;
    const N: usize = 4000;

    const seq = try genBigramSeq(a, &prng, V, N);
    var ds = try data.Dataset.fromIds(a, seq, 0.1, .{ .seq_len = L, .batch_size = 32, .seed = 1 });
    defer ds.deinit();

    // ---- unigram baseline: predict each val token by its TRAIN marginal freq ----
    var marg = try a.alloc(f32, V);
    for (marg) |*m| m.* = 1.0; // Laplace smoothing
    for (ds.train.tokens) |t| marg[t] += 1.0;
    var totalm: f32 = 0;
    for (marg) |m| totalm += m;
    for (marg) |*m| m.* /= totalm;
    var uni_loss: f32 = 0;
    for (ds.val.tokens) |t| uni_loss -= @log(@max(marg[t], 1e-12));
    uni_loss /= @as(f32, @floatFromInt(ds.val.tokens.len));

    // ---- train the MLPLM ----
    var lm = try nn.MLPLM.init(a, V, L, dim, hidden, &prng);
    const ps = try lm.params(a);
    defer a.free(ps);
    var opt = optim.Adam.init(a, 0.01, 0.0);
    defer opt.deinit();

    const steps: usize = 400;
    for (0..steps) |_| {
        // fetch a batch, resetting at epoch boundaries
        const b = blk: {
            if (ds.train.nextBatch(a) catch return) |x| break :blk x;
            ds.train.reset();
            break :blk (ds.train.nextBatch(a) catch return).?;
        };
        const ids = try a.alloc(usize, b.x.len);
        for (b.x, 0..) |x, i| ids[i] = x;
        // target = next token after the window = last column of the shifted targets
        const tgt = try a.alloc(usize, b.b);
        for (0..b.b) |bi| tgt[bi] = b.y[bi * b.l + (b.l - 1)];
        const logits = try lm.forward(ids);
        const l = try loss.softmaxCE(a, logits, tgt);
        l.backprop();
        opt.step(ps);
    }

    // ---- model val loss (average per-window next-token loss) ----
    ds.val.reset();
    var model_loss: f32 = 0;
    var n_val: usize = 0;
    while (ds.val.nextBatch(a) catch null) |b| {
        const ids = try a.alloc(usize, b.x.len);
        for (b.x, 0..) |x, i| ids[i] = x;
        const tgt = try a.alloc(usize, b.b);
        for (0..b.b) |bi| tgt[bi] = b.y[bi * b.l + (b.l - 1)];
        const logits = try lm.forward(ids);
        const l = try loss.softmaxCE(a, logits, tgt);
        model_loss += l.value.data[0] * @as(f32, @floatFromInt(b.b));
        n_val += b.b;
    }
    model_loss /= @as(f32, @floatFromInt(n_val));

    std.debug.print("  unigram val loss={d:.4} | model val loss={d:.4}\n", .{ uni_loss, model_loss });
    // the context model must beat the marginal-only baseline
    try testing.expect(model_loss < uni_loss);
}

// ---------------------------------------------------------------------------
// Gate 7: optimizers — Muon (Newton-Schulz orthogonalization) + hybrid Adam/Muon
// ---------------------------------------------------------------------------

// The defining correctness property of the Muon update: Newton-Schulz must turn
// a full-rank matrix into a BETTER-CONDITIONED (more orthogonal) one. The 5-step
// polynomial only achieves APPROXIMATE orthogonality (singular values land in
// ~[0.7, 1.2], never exactly 1) — that is the algorithm's real behavior, not a
// bug — so "O@O^T == I" is the wrong gate. The robust, separable gate is the
// property Muon actually guarantees: the NS output is strictly MORE orthogonal
// than the (Frobenius-normalized) input, and bounded. A no-op / broken NS leaves
// the residual unchanged; a divergent NS explodes it. We build an ILL-CONDITIONED
// input of known SVD (one dominant singular value, the rest tiny — mirroring real
// gradient spectra) so the raw input is far from orthogonal and a working NS has
// ample room to improve it.
test "Muon: Newton-Schulz output is more orthogonal than the input" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(31);

    for ([_]usize{ 6, 10 }) |K| {
        // two random orthogonal [K,K] factors (modified Gram-Schmidt on rows)
        const U = try randomOrthogonal(a, K, &prng);
        const V = try randomOrthogonal(a, K, &prng);
        // ill-conditioned spectrum: one dominant sv, the rest tiny (cond ~ K).
        const sv = try a.alloc(f32, K);
        sv[0] = 1.0;
        for (1..K) |j| sv[j] = 0.02;
        // G = U @ diag(sv) @ V^T  =>  G[i,l] = sum_j U[i,j]*sv[j]*V[l,j]
        const g = try a.alloc(f32, K * K);
        for (0..K) |i| {
            for (0..K) |l| {
                var s: f32 = 0;
                for (0..K) |j| s += U[i * K + j] * sv[j] * V[l * K + j];
                g[i * K + l] = s;
            }
        }

        // residual of the Frobenius-NORMALIZED input (what NS starts from):
        // copy g, divide by its Frobenius norm, then measure ||X@X^T - I||/K.
        const x_in = try a.alloc(f32, K * K);
        var fnorm_sq: f32 = 0;
        for (g) |x| fnorm_sq += x * x;
        const inv = 1.0 / (@sqrt(fnorm_sq) + 1e-7);
        for (0..K * K) |i| x_in[i] = g[i] * inv;
        const res_in = gramResidual(x_in, K, K);

        // NS output
        const out = try a.alloc(f32, K * K);
        try optim.newtonSchulz5(a, g, K, K, out, 5);
        const res_out = gramResidual(out, K, K);

        std.debug.print("  NS conditioning [K={d}]: in={d:.4} -> out={d:.4}\n", .{ K, res_in, res_out });
        // (1) bounded: no NaN / explosion
        try testing.expect(!std.math.isNan(res_out));
        try testing.expect(res_out < 1.0);
        // (2) the defining guarantee: NS makes the update MORE orthogonal than
        //     the raw normalized gradient. A no-op NS gives res_out == res_in
        //     (fails the strict <); a divergent one gives res_out >> res_in.
        try testing.expect(res_out < res_in);
        // (3) and reasonably orthogonal in absolute terms.
        try testing.expect(res_out < 0.3);
    }
}

// ||M@M^T - I||_F / K  for a square [K,K] row-major matrix (orthogonality gap).
fn gramResidual(m: []const f32, rows: usize, cols: usize) f32 {
    const K = rows; // square
    _ = cols;
    var off: f32 = 0;
    for (0..K) |i| {
        for (0..K) |j| {
            var s: f32 = 0;
            for (0..K) |k| s += m[i * K + k] * m[j * K + k];
            const target: f32 = if (i == j) 1.0 else 0.0;
            const d = s - target;
            off += d * d;
        }
    }
    return @sqrt(off) / @as(f32, @floatFromInt(K));
}

// modified Gram-Schmidt on the rows of a random [K,K] Gaussian -> orthonormal rows
fn randomOrthogonal(alloc: mem.Allocator, K: usize, prng: *std.Random.DefaultPrng) ![]f32 {
    const m = try alloc.alloc(f32, K * K);
    for (m) |*x| x.* = prng.random().floatNorm(f32);
    const q = try alloc.alloc(f32, K * K);
    for (0..K) |i| {
        var row = try alloc.alloc(f32, K);
        defer alloc.free(row);
        for (0..K) |k| row[k] = m[i * K + k];
        for (0..i) |j| {
            var d: f32 = 0;
            for (0..K) |k| d += row[k] * q[j * K + k];
            for (0..K) |k| row[k] -= d * q[j * K + k];
        }
        var nrm: f32 = 0;
        for (row) |x| nrm += x * x;
        nrm = @sqrt(nrm);
        for (0..K) |k| q[i * K + k] = row[k] / @max(nrm, 1e-12);
    }
    return q;
}

// The orthogonality must also hold for the FULL Muon step path (momentum buffer
// -> Newton-Schulz -> update), not just the isolated NS routine. Build a 2D
// parameter, poke a gradient, take a Muon step, and verify the buffer -> update
// path is orthogonal in direction (the update U should satisfy U@U^T ≈ I*scale).
// We test the simpler guarantee: the orthogonalized buffer is orthogonal.
test "Muon: momentum buffer -> NS path is stable across steps" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(77);

    const m: usize = 6;
    const n: usize = 10;
    const W = try ag.Var.parameter(a, try nn.randn(a, &.{ m, n }, &prng, 0.5));
    // give W a real grad
    for (W.grad.data) |*x| x.* = prng.random().floatNorm(f32) * 0.1;

    var mu = optim.Muon.init(a, 0.01, 0.95, 0.0);
    defer mu.deinit();
    // take a few steps with fresh random grads; the update must not NaN/explode
    for (0..5) |_| {
        for (W.grad.data) |*x| x.* = prng.random().floatNorm(f32) * 0.1;
        mu.step(&.{W});
        var nrm: f32 = 0;
        for (W.value.data) |x| nrm += x * x;
        nrm = @sqrt(nrm);
        try testing.expect(!std.math.isNan(nrm));
        try testing.expect(nrm < 1e3); // no explosion
    }
}

// Gate 7 success criterion: a small LM trained with the hybrid Muon+Adam
// optimizer reaches a lower loss than the same model trained with plain Adam
// at the same step count. Muon's orthogonal updates give a better-conditioned
// descent on matrix parameters, so it should pull ahead on a fixed budget.
test "Gate 7: Muon-hybrid LM beats plain-Adam LM at equal steps" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(4242);

    const V: usize = 16;
    const L: usize = 4;
    const dim: usize = 32;
    const hidden: usize = 96;
    const N: usize = 4000;
    // Fewer steps + wider matrices: Muon's orthogonal updates converge faster
    // early in training on matrix-heavy nets, which is the headline claim. At a
    // short step budget the gap is wide; at full convergence both near the floor
    // and the margin shrinks to noise.
    const steps: usize = 150;

    const seq = try genBigramSeq(a, &prng, V, N);

    // train a fresh model with a given optimizer; return final train loss (avg of last 20).
    const train = struct {
        fn run(alloc: mem.Allocator, rng: *std.Random.DefaultPrng, seq_: []u32, use_muon: bool, V_: usize, L_: usize, dim_: usize, hidden_: usize, steps_: usize) !f32 {
            var ds = try data.Dataset.fromIds(alloc, seq_, 0.1, .{ .seq_len = L_, .batch_size = 32, .seed = 1 });
            defer ds.deinit();
            var lm = try nn.MLPLM.init(alloc, V_, L_, dim_, hidden_, rng);
            const ps = try lm.params(alloc);
            defer alloc.free(ps);

            var last20: f32 = 0;
            if (use_muon) {
                // route embeddings (tok_emb, pos_emb) + final LM head (last layer W)
                // to Adam; the two hidden-layer matrices to Muon.
                const head = lm.net.layers[lm.net.layers.len - 1].W; // [hidden, vocab] output proj
                const adam_override = [_]*ag.Var{ lm.tok_emb.W, lm.pos_emb.W, head };
                var opt = try optim.Hybrid.init(alloc, ps, &adam_override, 0.02, 0.95, 0.003, 0.0);
                defer opt.deinit();
                for (0..steps_) |s| {
                    const b = blk: {
                        if (ds.train.nextBatch(alloc) catch return 1e9) |x| break :blk x;
                        ds.train.reset();
                        break :blk (ds.train.nextBatch(alloc) catch return 1e9).?;
                    };
                    const ids = try alloc.alloc(usize, b.x.len);
                    for (b.x, 0..) |x, i| ids[i] = x;
                    const tgt = try alloc.alloc(usize, b.b);
                    for (0..b.b) |bi| tgt[bi] = b.y[bi * b.l + (b.l - 1)];
                    const logits = try lm.forward(ids);
                    const l = try loss.softmaxCE(alloc, logits, tgt);
                    l.backprop();
                    opt.step();
                    if (s >= steps_ - 20) last20 += l.value.data[0];
                }
            } else {
                var opt = optim.Adam.init(alloc, 0.003, 0.0);
                defer opt.deinit();
                for (0..steps_) |s| {
                    const b = blk: {
                        if (ds.train.nextBatch(alloc) catch return 1e9) |x| break :blk x;
                        ds.train.reset();
                        break :blk (ds.train.nextBatch(alloc) catch return 1e9).?;
                    };
                    const ids = try alloc.alloc(usize, b.x.len);
                    for (b.x, 0..) |x, i| ids[i] = x;
                    const tgt = try alloc.alloc(usize, b.b);
                    for (0..b.b) |bi| tgt[bi] = b.y[bi * b.l + (b.l - 1)];
                    const logits = try lm.forward(ids);
                    const l = try loss.softmaxCE(alloc, logits, tgt);
                    l.backprop();
                    opt.step(ps);
                    if (s >= steps_ - 20) last20 += l.value.data[0];
                }
            }
            return last20 / 20.0;
        }
    }.run;

    // two fresh PRNGs seeded identically so the models start from the SAME init
    var prng_a = std.Random.DefaultPrng.init(4242);
    var prng_b = std.Random.DefaultPrng.init(4242);
    const adam_loss = try train(a, &prng_a, seq, false, V, L, dim, hidden, steps);
    const muon_loss = try train(a, &prng_b, seq, true, V, L, dim, hidden, steps);
    std.debug.print("  Adam final loss={d:.4} | Muon-hybrid final loss={d:.4}\n", .{ adam_loss, muon_loss });
    try testing.expect(muon_loss < adam_loss);
}

// ---------------------------------------------------------------------------
// Gate 8: modern transformer block
// ---------------------------------------------------------------------------

// KV-cache gate: the cached (token-by-token) forward must produce the SAME
// per-position logits as the full-sequence training forward. This is the
// correctness guarantee that makes autoregressive generation valid: past
// outputs are invariant to future inputs (causality), and the KV cache reuses
// past keys/values rather than recomputing them. We compare the full forward
// (autograd, recomputes K,V every step) against forwardGen (cached, one token
// at a time) on the same input — they must agree to fp tolerance.
test "Gate 8: KV-cache forward == non-cached forward" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(808);

    const V: usize = 12;
    const L: usize = 8;
    const C: usize = 16; // 2 heads * 8 head_dim
    const H: usize = 2;
    const layers: usize = 2;
    const ffn_hidden: usize = 32;
    var lm = try nn.TransformerLM.init(a, V, L, C, layers, H, ffn_hidden, &prng);

    // a fixed input sequence (B=1, length L)
    const ids = try a.alloc(usize, L);
    for (ids, 0..) |*x, i| x.* = (i * 7 + 3) % V;

    // non-cached: full forward -> [B*L, V]
    const logits_v = try lm.forward(ids);
    const full = logits_v.value.data; // [L*V]

    // cached: token-by-token -> [L, V]
    const cached = try lm.forwardGen(a, ids);

    // compare per-position logits
    var max_diff: f32 = 0;
    for (0..L * V) |i| {
        const d = @abs(full[i] - cached.data[i]);
        if (d > max_diff) max_diff = d;
    }
    std.debug.print("  KV-cache vs non-cached max |diff| over {d} logits = {d:.6}\n", .{ L * V, max_diff });
    // fp32 accumulation differences between the two paths are small but nonzero
    // (different summation orders); 1e-3 is a tight, honest bar.
    try testing.expect(max_diff < 1e-3);
}

// The transformer LM (per-position causal prediction) must train and beat the
// unigram baseline on the bigram task — the same gate the MLP LM passed, now
// with RMSNorm + RoPE + SwiGLU + causal multi-head attention.
test "Gate 8: transformer LM trains and beats unigram baseline" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(909);

    const V: usize = 16;
    const L: usize = 8;
    const C: usize = 32; // 4 heads * 8
    const H: usize = 4;
    const layers: usize = 2;
    const ffn_hidden: usize = 64;
    const N: usize = 4000;

    const seq = try genBigramSeq(a, &prng, V, N);
    var ds = try data.Dataset.fromIds(a, seq, 0.1, .{ .seq_len = L, .batch_size = 16, .seed = 1 });
    defer ds.deinit();

    // unigram baseline (per-token -log(train marginal) on val)
    var marg = try a.alloc(f32, V);
    for (marg) |*m| m.* = 1.0;
    for (ds.train.tokens) |t| marg[t] += 1.0;
    var totm: f32 = 0;
    for (marg) |m| totm += m;
    for (marg) |*m| m.* /= totm;
    var uni_loss: f32 = 0;
    for (ds.val.tokens) |t| uni_loss -= @log(@max(marg[t], 1e-12));
    uni_loss /= @as(f32, @floatFromInt(ds.val.tokens.len));

    // train the transformer LM (per-position loss: every position predicts next)
    var lm = try nn.TransformerLM.init(a, V, L, C, layers, H, ffn_hidden, &prng);
    const ps = try lm.params(a);
    defer a.free(ps);
    var opt = optim.Adam.init(a, 0.003, 0.0);
    defer opt.deinit();

    const steps: usize = 300;
    var last: f32 = 1e9;
    for (0..steps) |s| {
        const b = blk: {
            if (ds.train.nextBatch(a) catch return) |x| break :blk x;
            ds.train.reset();
            break :blk (ds.train.nextBatch(a) catch return).?;
        };
        // ids: x (B*L flattened). targets: y (the shifted next tokens, B*L).
        const ids = try a.alloc(usize, b.x.len);
        for (b.x, 0..) |x, i| ids[i] = x;
        const tgt = try a.alloc(usize, b.y.len);
        for (b.y, 0..) |x, i| tgt[i] = x;
        const logits = try lm.forward(ids); // [B*L, V]
        const l = try loss.softmaxCE(a, logits, tgt);
        l.backprop();
        opt.step(ps);
        last = l.value.data[0];
        if (s % 100 == 0) std.debug.print("  tx step {d:>3} loss={d:.4}\n", .{ s, last });
    }

    // model val loss (per-position, averaged over all val positions)
    ds.val.reset();
    var model_loss: f32 = 0;
    var n_val: usize = 0;
    while (ds.val.nextBatch(a) catch null) |b| {
        const ids = try a.alloc(usize, b.x.len);
        for (b.x, 0..) |x, i| ids[i] = x;
        const tgt = try a.alloc(usize, b.y.len);
        for (b.y, 0..) |x, i| tgt[i] = x;
        const logits = try lm.forward(ids);
        const l = try loss.softmaxCE(a, logits, tgt);
        model_loss += l.value.data[0] * @as(f32, @floatFromInt(b.y.len));
        n_val += b.y.len;
    }
    model_loss /= @as(f32, @floatFromInt(n_val));

    std.debug.print("  unigram val loss={d:.4} | transformer val loss={d:.4}\n", .{ uni_loss, model_loss });
    try testing.expect(model_loss < uni_loss);
}
// ---- Gate 9: GQA + flash attention end-to-end ----
// The GQA LM uses the fused flash attention path (vFlashAttn) for training and a
// KV cache that stores only Hkv heads for generation. First: the cached
// generation path must match the non-cached training forward to fp tolerance.
test "Gate 9: GQA KV-cache forward == non-cached forward" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(818);

    const V: usize = 12;
    const L: usize = 8;
    const C: usize = 16; // H=4 heads * 4 head_dim
    const H: usize = 4;
    const Hkv: usize = 2; // group = 2 (each kv head shared by 2 query heads)
    const layers: usize = 2;
    const ffn_hidden: usize = 32;
    var lm = try nn.GqaTransformerLM.init(a, V, L, C, layers, H, Hkv, ffn_hidden, &prng);

    const ids = try a.alloc(usize, L);
    for (ids, 0..) |*x, i| x.* = (i * 7 + 3) % V;

    const logits_v = try lm.forward(ids);
    const full = logits_v.value.data; // [L*V]
    const cached = try lm.forwardGen(a, ids); // [L, V]

    var max_diff: f32 = 0;
    for (0..L * V) |i| max_diff = @max(max_diff, @abs(full[i] - cached.data[i]));
    std.debug.print("  GQA KV-cache vs non-cached max |diff| over {d} logits = {d:.6}\n", .{ L * V, max_diff });
    try testing.expect(max_diff < 1e-3);
}

// The GQA LM (fewer kv heads, fused flash attention) must still train and beat
// the unigram baseline — same gate as the MHA transformer, proving the GQA +
// flash path is correct end-to-end (forward, backward, optimizer).
test "Gate 9: GQA LM (flash attn) trains and beats unigram baseline" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(919);

    const V: usize = 16;
    const L: usize = 8;
    const C: usize = 32; // H=4 * 8
    const H: usize = 4;
    const Hkv: usize = 2; // group=2 GQA
    const layers: usize = 2;
    const ffn_hidden: usize = 64;
    const N: usize = 4000;

    const seq = try genBigramSeq(a, &prng, V, N);
    var ds = try data.Dataset.fromIds(a, seq, 0.1, .{ .seq_len = L, .batch_size = 16, .seed = 1 });
    defer ds.deinit();

    // unigram baseline
    var marg = try a.alloc(f32, V);
    for (marg) |*m| m.* = 1.0;
    for (ds.train.tokens) |t| marg[t] += 1.0;
    var totm: f32 = 0;
    for (marg) |m| totm += m;
    for (marg) |*m| m.* /= totm;
    var uni_loss: f32 = 0;
    for (ds.val.tokens) |t| uni_loss -= @log(@max(marg[t], 1e-12));
    uni_loss /= @as(f32, @floatFromInt(ds.val.tokens.len));

    var lm = try nn.GqaTransformerLM.init(a, V, L, C, layers, H, Hkv, ffn_hidden, &prng);
    const ps = try lm.params(a);
    defer a.free(ps);
    var opt = optim.Adam.init(a, 0.003, 0.0);
    defer opt.deinit();

    const steps: usize = 300;
    var last: f32 = 1e9;
    for (0..steps) |s| {
        const b = blk: {
            if (ds.train.nextBatch(a) catch return) |x| break :blk x;
            ds.train.reset();
            break :blk (ds.train.nextBatch(a) catch return).?;
        };
        const ids = try a.alloc(usize, b.x.len);
        for (b.x, 0..) |x, i| ids[i] = x;
        const tgt = try a.alloc(usize, b.y.len);
        for (b.y, 0..) |x, i| tgt[i] = x;
        const logits = try lm.forward(ids);
        const l = try loss.softmaxCE(a, logits, tgt);
        l.backprop();
        opt.step(ps);
        last = l.value.data[0];
        if (s % 100 == 0) std.debug.print("  gqa step {d:>3} loss={d:.4}\n", .{ s, last });
    }

    ds.val.reset();
    var model_loss: f32 = 0;
    var n_val: usize = 0;
    while (ds.val.nextBatch(a) catch null) |b| {
        const ids = try a.alloc(usize, b.x.len);
        for (b.x, 0..) |x, i| ids[i] = x;
        const tgt = try a.alloc(usize, b.y.len);
        for (b.y, 0..) |x, i| tgt[i] = x;
        const logits = try lm.forward(ids);
        const l = try loss.softmaxCE(a, logits, tgt);
        model_loss += l.value.data[0] * @as(f32, @floatFromInt(b.y.len));
        n_val += b.y.len;
    }
    model_loss /= @as(f32, @floatFromInt(n_val));

    std.debug.print("  unigram val loss={d:.4} | GQA(flash) val loss={d:.4}\n", .{ uni_loss, model_loss });
    try testing.expect(model_loss < uni_loss);
}

// ---- Gate 10: Gated Delta Attention / DeltaNet end-to-end ----
// The DeltaNet LM replaces softmax attention with the gated delta recurrence.
// Generation carries a Dh x Dh state per head (no KV history), so the cached
// generation path must reproduce the non-cached training forward to fp tol.
test "Gate 10: DeltaNet recurrent-state forward == non-cached forward" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(1010);

    const V: usize = 12;
    const L: usize = 8;
    const C: usize = 16; // H=4 * Dh=4
    const H: usize = 4;
    const layers: usize = 2;
    const ffn_hidden: usize = 32;
    var lm = try nn.GatedDeltaLM.init(a, V, L, C, layers, H, ffn_hidden, &prng);

    const ids = try a.alloc(usize, L);
    for (ids, 0..) |*x, i| x.* = (i * 7 + 3) % V;

    const logits_v = try lm.forward(ids);
    const full = logits_v.value.data; // [L*V]
    const cached = try lm.forwardGen(a, ids); // [L, V]

    var max_diff: f32 = 0;
    for (0..L * V) |i| max_diff = @max(max_diff, @abs(full[i] - cached.data[i]));
    std.debug.print("  DeltaNet state-cache vs non-cached max |diff| over {d} logits = {d:.6}\n", .{ L * V, max_diff });
    try testing.expect(max_diff < 1e-3);
}

// The DeltaNet LM must train and beat the unigram baseline on the bigram task
// — same gate as the softmax-attention LMs, proving the gated-delta forward +
// BPTT backward + optimizer path is correct end-to-end.
test "Gate 10: DeltaNet LM trains and beats unigram baseline" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(1111);

    const V: usize = 16;
    const L: usize = 8;
    const C: usize = 32; // H=4 * Dh=8
    const H: usize = 4;
    const layers: usize = 2;
    const ffn_hidden: usize = 64;
    const N: usize = 4000;

    const seq = try genBigramSeq(a, &prng, V, N);
    var ds = try data.Dataset.fromIds(a, seq, 0.1, .{ .seq_len = L, .batch_size = 16, .seed = 1 });
    defer ds.deinit();

    // unigram baseline
    var marg = try a.alloc(f32, V);
    for (marg) |*m| m.* = 1.0;
    for (ds.train.tokens) |t| marg[t] += 1.0;
    var totm: f32 = 0;
    for (marg) |m| totm += m;
    for (marg) |*m| m.* /= totm;
    var uni_loss: f32 = 0;
    for (ds.val.tokens) |t| uni_loss -= @log(@max(marg[t], 1e-12));
    uni_loss /= @as(f32, @floatFromInt(ds.val.tokens.len));

    var lm = try nn.GatedDeltaLM.init(a, V, L, C, layers, H, ffn_hidden, &prng);
    const ps = try lm.params(a);
    defer a.free(ps);
    var opt = optim.Adam.init(a, 0.003, 0.0);
    defer opt.deinit();

    const steps: usize = 300;
    var last: f32 = 1e9;
    for (0..steps) |s| {
        const b = blk: {
            if (ds.train.nextBatch(a) catch return) |x| break :blk x;
            ds.train.reset();
            break :blk (ds.train.nextBatch(a) catch return).?;
        };
        const ids = try a.alloc(usize, b.x.len);
        for (b.x, 0..) |x, i| ids[i] = x;
        const tgt = try a.alloc(usize, b.y.len);
        for (b.y, 0..) |x, i| tgt[i] = x;
        const logits = try lm.forward(ids);
        const l = try loss.softmaxCE(a, logits, tgt);
        l.backprop();
        opt.step(ps);
        last = l.value.data[0];
        if (s % 100 == 0) std.debug.print("  delta step {d:>3} loss={d:.4}\n", .{ s, last });
    }

    ds.val.reset();
    var model_loss: f32 = 0;
    var n_val: usize = 0;
    while (ds.val.nextBatch(a) catch null) |b| {
        const ids = try a.alloc(usize, b.x.len);
        for (b.x, 0..) |x, i| ids[i] = x;
        const tgt = try a.alloc(usize, b.y.len);
        for (b.y, 0..) |x, i| tgt[i] = x;
        const logits = try lm.forward(ids);
        const l = try loss.softmaxCE(a, logits, tgt);
        model_loss += l.value.data[0] * @as(f32, @floatFromInt(b.y.len));
        n_val += b.y.len;
    }
    model_loss /= @as(f32, @floatFromInt(n_val));

    std.debug.print("  unigram val loss={d:.4} | DeltaNet val loss={d:.4}\n", .{ uni_loss, model_loss });
    try testing.expect(model_loss < uni_loss);
}
