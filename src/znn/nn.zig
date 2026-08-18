// nn.zig — neural-network layers built on the autograd Var engine.
//
// A `Layer` owns its parameters (as Vars) and a forward function. A `Module`
// is just a layer that may hold sub-layers; we flatten all parameters via
// `params()` so the optimizer can update them in one call.

const std = @import("std");
const mem = std.mem;
const ten = @import("tensor.zig");
const ag = @import("autograd.zig");
const Var = ag.Var;
const Tensor = ten.Tensor;

pub const Error = ag.Error;

// ---------------------------------------------------------------------------
// parameter init
// ---------------------------------------------------------------------------

pub fn randn(alloc: mem.Allocator, shape: []const usize, prng: *std.Random.DefaultPrng, std_dev: f32) !Tensor {
    const t = try ten.zeros(alloc, shape);
    const r = prng.random();
    for (t.data) |*x| x.* = r.floatNorm(f32) * std_dev;
    return t;
}

// Kaiming/He init for a weight of shape [fan_in, fan_out] (deep ReLU nets): std = sqrt(2/fan_in).
// Appropriate for *deep* nets where you must preserve activation variance across
// many layers. For a 1-2 hidden-layer net, prefer `initLinear` (uniform) below —
// Kaiming's std approaches 1 for small fan_in (e.g. fan_in=2 -> std=1.0), which is
// large relative to O(1) inputs and makes shallow-net optimization ill-conditioned.
pub fn initKaiming(alloc: mem.Allocator, fan_in: usize, fan_out: usize, prng: *std.Random.DefaultPrng) !Tensor {
    const sd = @sqrt(2.0 / @as(f32, @floatFromInt(fan_in)));
    return randn(alloc, &.{ fan_in, fan_out }, prng, sd);
}

// Xavier/Glorot init (tanh/linear nets): std = sqrt(1/fan_in)
pub fn initXavier(alloc: mem.Allocator, fan_in: usize, fan_out: usize, prng: *std.Random.DefaultPrng) !Tensor {
    const sd = @sqrt(1.0 / @as(f32, @floatFromInt(fan_in)));
    return randn(alloc, &.{ fan_in, fan_out }, prng, sd);
}

// PyTorch-default Linear init: uniform U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
// std = 1/sqrt(3*fan_in). Well-conditioned for shallow nets and the usual baseline;
// Kaiming/Xavier are kept above for the cases that need them.
pub fn initLinear(alloc: mem.Allocator, fan_in: usize, fan_out: usize, prng: *std.Random.DefaultPrng) !Tensor {
    const bound = 1.0 / @sqrt(@as(f32, @floatFromInt(fan_in)));
    const t = try ten.zeros(alloc, &.{ fan_in, fan_out });
    const r = prng.random();
    for (t.data) |*x| x.* = (r.float(f32) * 2.0 - 1.0) * bound;
    return t;
}

// ---------------------------------------------------------------------------
// Linear: y = x @ W + b
//   x: [B, in]   W: [in, out]   b: [out]   ->  y: [B, out]
// ---------------------------------------------------------------------------

pub const Linear = struct {
    alloc: mem.Allocator,
    W: *Var,
    b: *Var,

    pub fn init(alloc: mem.Allocator, in_dim: usize, out_dim: usize, prng: *std.Random.DefaultPrng) !Linear {
        const w = try initLinear(alloc, in_dim, out_dim, prng);
        const b = try ten.zeros(alloc, &.{out_dim});
        return .{
            .alloc = alloc,
            .W = try Var.parameter(alloc, w),
            .b = try Var.parameter(alloc, b),
        };
    }

    pub fn forward(self: *Linear, x: *Var) Error!*Var {
        const mm = try ag.vMatmul(self.alloc, x, self.W); // [B,out]
        return ag.vAdd(self.alloc, mm, self.b); // broadcast b over batch
    }

    pub fn params(self: *Linear) [2]*Var {
        return .{ self.W, self.b };
    }
};

// ---------------------------------------------------------------------------
// MLP: stack of Linear layers with activations between them.
// ---------------------------------------------------------------------------

pub const Act = enum { relu, tanh, none };

pub const MLP = struct {
    alloc: mem.Allocator,
    layers: []Linear,
    acts: []const Act,

    pub fn init(alloc: mem.Allocator, dims: []const usize, acts: []const Act, prng: *std.Random.DefaultPrng) !MLP {
        std.debug.assert(dims.len >= 2);
        std.debug.assert(acts.len == dims.len - 1);
        const layers = try alloc.alloc(Linear, dims.len - 1);
        for (0..dims.len - 1) |i| {
            layers[i] = try Linear.init(alloc, dims[i], dims[i + 1], prng);
        }
        return .{ .alloc = alloc, .layers = layers, .acts = acts };
    }

    pub fn forward(self: *MLP, x: *Var) Error!*Var {
        var h: *Var = x;
        for (self.layers, self.acts) |*lin, act| {
            const z = try lin.forward(h);
            h = switch (act) {
                .relu => try ag.vRelu(self.alloc, z),
                .tanh => blk: {
                    const t = try ten.tanh(self.alloc, z.value);
                    break :blk try ag.Var.constant(self.alloc, t);
                },
                .none => z,
            };
        }
        return h;
    }

    // collect every parameter pointer into a fresh slice (caller frees the slice)
    pub fn params(self: *MLP, alloc: mem.Allocator) ![]*Var {
        var list: std.ArrayList(*Var) = .empty;
        defer list.deinit(alloc);
        for (self.layers) |*lin| {
            try list.append(alloc, lin.W);
            try list.append(alloc, lin.b);
        }
        return list.toOwnedSlice(alloc);
    }
};

// ---------------------------------------------------------------------------
// Embedding: gather rows of a [vocab, dim] table by index.
//   forward(idx: [N]) -> [N, dim] where out[n] = W[idx[n]]
// Implemented with vIndexRows (scatter-add backward), so the gradient flows
// correctly to the table rows — this IS the embedding lookup, grad-checkable,
// no one-hot trick needed.
// ---------------------------------------------------------------------------

pub fn initEmbedding(alloc: mem.Allocator, vocab: usize, dim: usize, prng: *std.Random.DefaultPrng) !Tensor {
    // GPT-style small init: N(0, 0.02). Keeps initial embedding norms modest so
    // downstream activations stay well-conditioned early in training.
    return randn(alloc, &.{ vocab, dim }, prng, 0.02);
}

pub const Embedding = struct {
    alloc: mem.Allocator,
    W: *Var,

    pub fn init(alloc: mem.Allocator, vocab: usize, dim: usize, prng: *std.Random.DefaultPrng) !Embedding {
        return .{ .alloc = alloc, .W = try Var.parameter(alloc, try initEmbedding(alloc, vocab, dim, prng)) };
    }

    pub fn forward(self: *Embedding, idx: []const usize) Error!*Var {
        return ag.vIndexRows(self.alloc, self.W, idx);
    }

    pub fn params(self: *Embedding) [1]*Var {
        return .{self.W};
    }
};

// ---------------------------------------------------------------------------
// MLPLM — the "first LM": a makemore-style MLP that predicts the NEXT token
// from a flattened context window.
//
//   ids: [B*L] (row-major: row b, position j -> ids[b*L + j])
//   tok_emb = Embedding(vocab, dim)(ids)            -> [B*L, dim]
//   pos_emb = Embedding(L,     dim)(positions)     -> [B*L, dim]   (pos = j)
//   h       = tok_emb + pos_emb                    -> [B*L, dim]
//   flat    = reshape(h, [B, L*dim])               -> [B, L*dim]  (pure view)
//   logits  = MLP([L*dim, H, H, vocab])(flat)      -> [B, vocab]
//
// This conditions each prediction on the FULL preceding window (causal: the
// target is the token AFTER the window, so no future leaks), so it beats a
// unigram baseline. It predicts one token per window; the transformer (Gate 8)
// upgrades this to per-position causal prediction. Sampling is autoregressive:
// predict, append, slide the window.
// ---------------------------------------------------------------------------

pub const MLPLM = struct {
    alloc: mem.Allocator,
    vocab: usize,
    seq_len: usize,
    dim: usize,
    tok_emb: Embedding,
    pos_emb: Embedding,
    net: MLP,

    pub fn init(
        alloc: mem.Allocator,
        vocab: usize,
        seq_len: usize,
        dim: usize,
        hidden: usize,
        prng: *std.Random.DefaultPrng,
    ) !MLPLM {
        return .{
            .alloc = alloc,
            .vocab = vocab,
            .seq_len = seq_len,
            .dim = dim,
            .tok_emb = try Embedding.init(alloc, vocab, dim, prng),
            .pos_emb = try Embedding.init(alloc, seq_len, dim, prng),
            .net = try MLP.init(alloc, &.{ seq_len * dim, hidden, hidden, vocab }, &.{ .relu, .relu, .none }, prng),
        };
    }

    // ids: [B*seq_len] flattened token ids. Returns logits [B, vocab].
    pub fn forward(self: *MLPLM, ids: []const usize) Error!*Var {
        const B = ids.len / self.seq_len;
        const tok = try self.tok_emb.forward(ids); // [B*L, dim]

        // positions: 0..L-1 repeated per row
        const pos = try self.alloc.alloc(usize, ids.len);
        defer self.alloc.free(pos);
        for (0..ids.len) |i| pos[i] = i % self.seq_len;
        const pem = try self.pos_emb.forward(pos); // [B*L, dim]

        const e = try ag.vAdd(self.alloc, tok, pem); // [B*L, dim]
        const flat = try ag.vReshape(self.alloc, e, &.{ B, self.seq_len * self.dim }); // [B, L*dim]
        return self.net.forward(flat); // [B, vocab]
    }

    // Run a single context window [L] and return the full vocab logits row
    // [vocab] (for sampling / greedy next-token).
    pub fn forwardOne(self: *MLPLM, ctx: []const usize) Error!*Var {
        const logits = try self.forward(ctx); // [1, vocab]
        return ag.vReshape(self.alloc, logits, &.{self.vocab}); // [vocab]
    }

    pub fn params(self: *MLPLM, alloc: mem.Allocator) ![]*Var {
        var list: std.ArrayList(*Var) = .empty;
        defer list.deinit(alloc);
        try list.append(alloc, self.tok_emb.W);
        try list.append(alloc, self.pos_emb.W);
        for (self.net.layers) |*lin| {
            try list.append(alloc, lin.W);
            try list.append(alloc, lin.b);
        }
        return list.toOwnedSlice(alloc);
    }
};

// ===========================================================================
// Gate 8: modern transformer block — RMSNorm, RoPE, SwiGLU, causal MHA, KV cache
// ===========================================================================

// RMSNorm: out = x / sqrt(mean(x^2)+eps) * gamma, normalized over the last dim.
// Owns gamma [dim] (init 1). Stateless otherwise.
pub const RMSNorm = struct {
    alloc: mem.Allocator,
    gamma: *Var,

    pub fn init(alloc: mem.Allocator, dim: usize) !RMSNorm {
        return .{ .alloc = alloc, .gamma = try Var.parameter(alloc, try ten.full(alloc, &.{dim}, 1.0)) };
    }

    pub fn forward(self: *RMSNorm, x: *Var) Error!*Var {
        return ag.vRmsNorm(self.alloc, x, self.gamma);
    }

    pub fn params(self: *RMSNorm) [1]*Var {
        return .{self.gamma};
    }
};

// SwiGLU FFN: out = (silu(x @ W_gate) * (x @ W_up)) @ W_down
//   x: [B, L, C] -> gate/up: [B, L, H] -> down: [B, L, C]
// 3x expansion factor folded into H = C * ffn_mult (caller picks H).
pub const SwiGLU = struct {
    alloc: mem.Allocator,
    W_gate: *Var,
    W_up: *Var,
    W_down: *Var,

    pub fn init(alloc: mem.Allocator, c: usize, hidden: usize, prng: *std.Random.DefaultPrng) !SwiGLU {
        return .{
            .alloc = alloc,
            .W_gate = try Var.parameter(alloc, try initLinear(alloc, c, hidden, prng)),
            .W_up = try Var.parameter(alloc, try initLinear(alloc, c, hidden, prng)),
            .W_down = try Var.parameter(alloc, try initLinear(alloc, hidden, c, prng)),
        };
    }

    pub fn forward(self: *SwiGLU, x: *Var) Error!*Var {
        const g = try ag.vMatmul(self.alloc, x, self.W_gate); // [B,L,H]
        const u = try ag.vMatmul(self.alloc, x, self.W_up); // [B,L,H]
        const sg = try ag.vSilu(self.alloc, g); // silu(gate)
        const h = try ag.vMul(self.alloc, sg, u); // [B,L,H]
        return ag.vMatmul(self.alloc, h, self.W_down); // [B,L,C]
    }

    pub fn params(self: *SwiGLU) [3]*Var {
        return .{ self.W_gate, self.W_up, self.W_down };
    }
};

// Causal multi-head self-attention with RoPE.
//   x: [B, L, C]; C = n_heads * head_dim.
//   Q,K,V = x @ Wq/Wk/Wv  -> [B, L, C]
//   reshape -> [B, L, H, Dh] -> permute -> [B, H, L, Dh] -> reshape -> [B*H, L, Dh]
//   RoPE on Q, K (position = L index)
//   attn = causal_sdpa(Q, K, V) -> [B*H, L, Dh]
//   reshape back -> [B, H, L, Dh] -> permute -> [B, L, H, Dh] -> reshape -> [B, L, C]
//   out = attn @ Wo -> [B, L, C]
pub const CausalSelfAttention = struct {
    alloc: mem.Allocator,
    n_heads: usize,
    head_dim: usize,
    c: usize,
    Wq: *Var,
    Wk: *Var,
    Wv: *Var,
    Wo: *Var,
    cap: usize = 0, // KV-cache capacity (set when generation is initialized)

    pub fn init(alloc: mem.Allocator, c: usize, n_heads: usize, prng: *std.Random.DefaultPrng) !CausalSelfAttention {
        std.debug.assert(c % n_heads == 0);
        return .{
            .alloc = alloc,
            .n_heads = n_heads,
            .head_dim = c / n_heads,
            .c = c,
            .Wq = try Var.parameter(alloc, try initLinear(alloc, c, c, prng)),
            .Wk = try Var.parameter(alloc, try initLinear(alloc, c, c, prng)),
            .Wv = try Var.parameter(alloc, try initLinear(alloc, c, c, prng)),
            .Wo = try Var.parameter(alloc, try initLinear(alloc, c, c, prng)),
        };
    }

    // full training forward (autograd). x: [B, L, C] -> [B, L, C]
    pub fn forward(self: *CausalSelfAttention, x: *Var) Error!*Var {
        const B = x.value.shape[0];
        const L = x.value.shape[1];
        const C = self.c;
        const H = self.n_heads;
        const Dh = self.head_dim;
        // projections [B, L, C]
        const q = try ag.vMatmul(self.alloc, x, self.Wq);
        const k = try ag.vMatmul(self.alloc, x, self.Wk);
        const v = try ag.vMatmul(self.alloc, x, self.Wv);
        // split heads: [B,L,C] -> [B,L,H,Dh] -> [B,H,L,Dh] -> [B*H, L, Dh]
        const qh = try ag.vReshape(self.alloc, q, &.{ B, L, H, Dh });
        const kh = try ag.vReshape(self.alloc, k, &.{ B, L, H, Dh });
        const vh = try ag.vReshape(self.alloc, v, &.{ B, L, H, Dh });
        const qp = try ag.vPermute(self.alloc, qh, &.{ 0, 2, 1, 3 });
        const kp = try ag.vPermute(self.alloc, kh, &.{ 0, 2, 1, 3 });
        const vp = try ag.vPermute(self.alloc, vh, &.{ 0, 2, 1, 3 });
        const qf = try ag.vReshape(self.alloc, qp, &.{ B * H, L, Dh });
        const kf = try ag.vReshape(self.alloc, kp, &.{ B * H, L, Dh });
        const vf = try ag.vReshape(self.alloc, vp, &.{ B * H, L, Dh });
        // RoPE on Q, K (rotary by position = L index)
        const qr = try ag.vRoPE(self.alloc, qf);
        const kr = try ag.vRoPE(self.alloc, kf);
        // causal scaled-dot-product attention
        const ao = try ag.vCausalMHA(self.alloc, qr, kr, vf); // [B*H, L, Dh]
        // merge heads back: [B*H,L,Dh] -> [B,H,L,Dh] -> [B,L,H,Dh] -> [B,L,C]
        const aoh = try ag.vReshape(self.alloc, ao, &.{ B, H, L, Dh });
        const aop = try ag.vPermute(self.alloc, aoh, &.{ 0, 2, 1, 3 });
        const aof = try ag.vReshape(self.alloc, aop, &.{ B, L, C });
        return ag.vMatmul(self.alloc, aof, self.Wo); // [B, L, C]
    }

    pub fn params(self: *CausalSelfAttention) [4]*Var {
        return .{ self.Wq, self.Wk, self.Wv, self.Wo };
    }

    // ---- KV cache for incremental generation (forward-only, no autograd) ----
    // During autoregressive generation we process ONE token at a time and reuse
    // the keys/values of all past tokens instead of recomputing them. The cache
    // holds K,V for positions 0..len-1 as [B*H, cap, Dh] (preallocated to `cap`).
    pub const Cache = struct {
        k: ?Tensor = null,
        v: ?Tensor = null,
        len: usize = 0,
    };

    // x: [B, 1, C] (one new token), pos = its absolute position (for RoPE).
    // Returns [B, 1, C]. Advances the cache by one.
    pub fn forwardIncr(self: *CausalSelfAttention, alloc: mem.Allocator, x: Tensor, pos: usize, cache: *Cache) !Tensor {
        const B = x.shape[0];
        const H = self.n_heads;
        const Dh = self.head_dim;
        // projections (raw matmul, no autograd)
        const q = try ten.matmul(alloc, x, self.Wq.value); // [B,1,C]
        const k = try ten.matmul(alloc, x, self.Wk.value);
        const v = try ten.matmul(alloc, x, self.Wv.value);
        // split heads -> [B*H, 1, Dh]
        const qh = try splitHeads(alloc, q, B, 1, H, Dh);
        const kh = try splitHeads(alloc, k, B, 1, H, Dh);
        const vh = try splitHeads(alloc, v, B, 1, H, Dh);
        // RoPE on the single new q,k at position `pos`
        const qr = try ropeTensor(alloc, qh, pos);
        const kr = try ropeTensor(alloc, kh, pos);
        // append k,v into the cache buffer
        if (cache.k == null) {
            cache.k = try ten.zeros(alloc, &.{ B * H, self.cap, Dh });
            cache.v = try ten.zeros(alloc, &.{ B * H, self.cap, Dh });
        }
        const kc = cache.k.?;
        const vc = cache.v.?;
        for (0..B * H) |h| {
            for (0..Dh) |d| {
                kc.data[h * (self.cap * Dh) + cache.len * Dh + d] = kr.data[h * Dh + d];
                vc.data[h * (self.cap * Dh) + cache.len * Dh + d] = vh.data[h * Dh + d];
            }
        }
        cache.len += 1;
        const n = cache.len; // keys available: positions 0..n-1
        // attention: scores [B*H, 1, n] = Q @ Kcache^T / sqrt(Dh); softmax; @ Vcache
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(Dh)));
        const out = try ten.zeros(alloc, &.{ B * H, 1, Dh });
        for (0..B * H) |h| {
            // scores for the single query against n cached keys
            const sc = try alloc.alloc(f32, n);
            defer alloc.free(sc);
            var mx: f32 = -1e30;
            for (0..n) |j| {
                var s: f32 = 0;
                for (0..Dh) |d| s += qr.data[h * Dh + d] * kc.data[h * (self.cap * Dh) + j * Dh + d];
                s *= scale;
                sc[j] = s;
                if (s > mx) mx = s;
            }
            var z: f32 = 0;
            for (0..n) |j| {
                sc[j] = @exp(sc[j] - mx);
                z += sc[j];
            }
            const invz = if (z > 0) 1.0 / z else 0.0;
            for (0..n) |j| {
                const a = sc[j] * invz;
                for (0..Dh) |d| out.data[h * Dh + d] += a * vc.data[h * (self.cap * Dh) + j * Dh + d];
            }
        }
        // merge heads -> [B, 1, C] then @ Wo
        const merged = try mergeHeads(alloc, out, B, 1, H, Dh);
        return ten.matmul(alloc, merged, self.Wo.value);
    }
};

// split [B, L, C] -> [B*H, L, Dh] via reshape+permute (raw tensors)
fn splitHeads(alloc: mem.Allocator, t: Tensor, B: usize, L: usize, H: usize, Dh: usize) !Tensor {
    const r = try ten.reshape(alloc, t, &.{ B, L, H, Dh });
    const p = try ten.permute(alloc, r, &.{ 0, 2, 1, 3 }); // [B,H,L,Dh]
    return ten.reshape(alloc, p, &.{ B * H, L, Dh });
}
fn mergeHeads(alloc: mem.Allocator, t: Tensor, B: usize, L: usize, H: usize, Dh: usize) !Tensor {
    const r = try ten.reshape(alloc, t, &.{ B, H, L, Dh });
    const p = try ten.permute(alloc, r, &.{ 0, 2, 1, 3 }); // [B,L,H,Dh]
    return ten.reshape(alloc, p, &.{ B, L, H * Dh });
}

// RoPE on a raw [B*H, L=1, Dh] tensor at a single position `pos` (forward-only).
fn ropeTensor(alloc: mem.Allocator, t: Tensor, pos: usize) !Tensor {
    const n = t.data.len;
    const out = try ten.zeros(alloc, t.shape);
    const Dh = t.shape[t.ndim() - 1];
    const rows = n / Dh;
    const half = Dh / 2;
    for (0..rows) |r| {
        for (0..half) |i| {
            const theta_i = std.math.pow(f32, 10000.0, -@as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(Dh)));
            const ang = @as(f32, @floatFromInt(pos)) * theta_i;
            const co = @cos(ang);
            const si = @sin(ang);
            const x = t.data[r * Dh + i];
            const y = t.data[r * Dh + i + half];
            out.data[r * Dh + i] = x * co - y * si;
            out.data[r * Dh + i + half] = y * co + x * si;
        }
    }
    return out;
}

// raw RMSNorm (forward-only): x [...,D] / sqrt(mean(x^2)+eps) * gamma
fn rmsNormTensor(alloc: mem.Allocator, x: Tensor, gamma: Tensor) !Tensor {
    const D = x.shape[x.ndim() - 1];
    const rows = x.numel() / D;
    const out = try ten.zeros(alloc, x.shape);
    for (0..rows) |r| {
        var ms: f32 = 0;
        for (0..D) |i| {
            const xv = x.data[r * D + i];
            ms += xv * xv;
        }
        ms /= @as(f32, @floatFromInt(D));
        const rr = 1.0 / @sqrt(ms + 1e-5);
        for (0..D) |i| out.data[r * D + i] = x.data[r * D + i] * rr * gamma.data[i];
    }
    return out;
}

// raw SwiGLU (forward-only)
fn swigluTensor(alloc: mem.Allocator, x: Tensor, wg: Tensor, wu: Tensor, wd: Tensor) !Tensor {
    const g = try ten.matmul(alloc, x, wg);
    const u = try ten.matmul(alloc, x, wu);
    const H = g.shape[g.ndim() - 1];
    const sg = try ten.zeros(alloc, g.shape);
    for (0..g.numel()) |i| {
        const xv = g.data[i];
        sg.data[i] = xv / (1.0 + @exp(-xv)); // silu = x*sigmoid(x)
    }
    const m = try ten.mul(alloc, sg, u);
    _ = H;
    return ten.matmul(alloc, m, wd);
}

// A decoder transformer block:
//   h = x + attn(rmsnorm(x))
//   out = h + ffn(rmsnorm(h))
pub const TransformerBlock = struct {
    alloc: mem.Allocator,
    norm1: RMSNorm,
    attn: CausalSelfAttention,
    norm2: RMSNorm,
    ffn: SwiGLU,

    pub fn init(alloc: mem.Allocator, c: usize, n_heads: usize, ffn_hidden: usize, prng: *std.Random.DefaultPrng) !TransformerBlock {
        return .{
            .alloc = alloc,
            .norm1 = try RMSNorm.init(alloc, c),
            .attn = try CausalSelfAttention.init(alloc, c, n_heads, prng),
            .norm2 = try RMSNorm.init(alloc, c),
            .ffn = try SwiGLU.init(alloc, c, ffn_hidden, prng),
        };
    }

    pub fn forward(self: *TransformerBlock, x: *Var) Error!*Var {
        const n1 = try self.norm1.forward(x);
        const a = try self.attn.forward(n1);
        const h = try ag.vAdd(self.alloc, x, a); // residual
        const n2 = try self.norm2.forward(h);
        const f = try self.ffn.forward(n2);
        return ag.vAdd(self.alloc, h, f); // residual
    }

    // incremental (cached) forward for one token. x: [B,1,C] Tensor -> [B,1,C].
    pub fn forwardIncr(self: *TransformerBlock, alloc: mem.Allocator, x: Tensor, pos: usize, cache: *CausalSelfAttention.Cache) !Tensor {
        const n1 = try rmsNormTensor(alloc, x, self.norm1.gamma.value);
        const a = try self.attn.forwardIncr(alloc, n1, pos, cache);
        const h = try ten.add(alloc, x, a); // residual
        const n2 = try rmsNormTensor(alloc, h, self.norm2.gamma.value);
        const f = try swigluTensor(alloc, n2, self.ffn.W_gate.value, self.ffn.W_up.value, self.ffn.W_down.value);
        return ten.add(alloc, h, f); // residual
    }

    pub fn params(self: *TransformerBlock, alloc: mem.Allocator) ![]*Var {
        var list: std.ArrayList(*Var) = .empty;
        defer list.deinit(alloc);
        try list.append(alloc, self.norm1.gamma);
        for (self.attn.params()) |p| try list.append(alloc, p);
        try list.append(alloc, self.norm2.gamma);
        for (self.ffn.params()) |p| try list.append(alloc, p);
        return list.toOwnedSlice(alloc);
    }
};

// Decoder LM: token embedding -> N x TransformerBlock -> final RMSNorm -> lm head.
// Per-position causal prediction (every position predicts its NEXT token).
// Position is injected via RoPE inside attention (no learned positional embedding).
//
//   ids: [B*L] flattened token ids (row-major: row b, pos j -> ids[b*L + j])
//   x = Embedding(vocab, C)(ids)           -> [B*L, C] -> reshape [B, L, C]
//   for block: x = block(x)                 -> [B, L, C]
//   x = rmsnorm_final(x)
//   logits = x @ W_head                     -> [B, L, V]  (one logit row per position)
//
// Loss: softmax-CE over all B*L positions vs the shifted targets (next tokens).
pub const TransformerLM = struct {
    alloc: mem.Allocator,
    vocab: usize,
    seq_len: usize,
    dim: usize, // C
    tok_emb: Embedding,
    blocks: []TransformerBlock,
    norm_f: RMSNorm,
    W_head: *Var, // [C, V] (untied head)

    pub fn init(
        alloc: mem.Allocator,
        vocab: usize,
        seq_len: usize,
        dim: usize,
        n_layers: usize,
        n_heads: usize,
        ffn_hidden: usize,
        prng: *std.Random.DefaultPrng,
    ) !TransformerLM {
        std.debug.assert(dim % n_heads == 0);
        const blocks = try alloc.alloc(TransformerBlock, n_layers);
        for (0..n_layers) |i| blocks[i] = try TransformerBlock.init(alloc, dim, n_heads, ffn_hidden, prng);
        return .{
            .alloc = alloc,
            .vocab = vocab,
            .seq_len = seq_len,
            .dim = dim,
            .tok_emb = try Embedding.init(alloc, vocab, dim, prng),
            .blocks = blocks,
            .norm_f = try RMSNorm.init(alloc, dim),
            .W_head = try Var.parameter(alloc, try initLinear(alloc, dim, vocab, prng)),
        };
    }

    // ids: [B*L] -> logits: [B*L, V] (one row per position; row t predicts token t+1).
    pub fn forward(self: *TransformerLM, ids: []const usize) Error!*Var {
        const B = ids.len / self.seq_len;
        const L = self.seq_len;
        const C = self.dim;
        const emb = try self.tok_emb.forward(ids); // [B*L, C]
        var x = try ag.vReshape(self.alloc, emb, &.{ B, L, C });
        for (self.blocks) |*blk| x = try blk.forward(x);
        x = try self.norm_f.forward(x); // [B, L, C]
        const logits = try ag.vMatmul(self.alloc, x, self.W_head); // [B, L, V]
        return ag.vReshape(self.alloc, logits, &.{ B * L, self.vocab }); // [B*L, V]
    }

    pub fn params(self: *TransformerLM, alloc: mem.Allocator) ![]*Var {
        var list: std.ArrayList(*Var) = .empty;
        defer list.deinit(alloc);
        try list.append(alloc, self.tok_emb.W);
        for (self.blocks) |*blk| {
            const bp = try blk.params(alloc);
            defer alloc.free(bp);
            for (bp) |p| try list.append(alloc, p);
        }
        try list.append(alloc, self.norm_f.gamma);
        try list.append(alloc, self.W_head);
        return list.toOwnedSlice(alloc);
    }

    // ---- cached generation (forward-only) ----
    // Process `ids` one token at a time with per-block KV caches, returning the
    // per-position logits [ids.len, V]. This is the inference path; it must match
    // the training `forward` to within fp tolerance (the Gate 8 cache gate).
    pub fn forwardGen(self: *TransformerLM, alloc: mem.Allocator, ids: []const usize) !Tensor {
        const L = ids.len;
        const B: usize = 1;
        const C = self.dim;
        const V = self.vocab;
        // set cache capacities on every attention layer
        for (self.blocks) |*blk| blk.attn.cap = L;
        var caches = try alloc.alloc(CausalSelfAttention.Cache, self.blocks.len);
        defer alloc.free(caches);
        for (caches) |*c| c.* = .{};
        const out = try ten.zeros(alloc, &.{ L, V });
        for (0..L) |pos| {
            // embed the single token -> [1, C] -> [1, 1, C]
            const id = ids[pos];
            const row = try ag.vIndexRows(alloc, self.tok_emb.W, &.{id}); // [1, C]
            var x = row.value; // [1, C]
            x = try ten.reshape(alloc, x, &.{ B, 1, C });
            for (self.blocks, 0..) |*blk, bi| {
                x = try blk.forwardIncr(alloc, x, pos, &caches[bi]);
            }
            // final norm + head
            x = try rmsNormTensor(alloc, x, self.norm_f.gamma.value); // [1,1,C]
            const logits1 = try ten.matmul(alloc, x, self.W_head.value); // [1,1,V]
            // copy row into out[pos]
            for (0..V) |v| out.data[pos * V + v] = logits1.data[v];
        }
        return out;
    }
};
// ============================================================================
// Gate 9 — Grouped-Query Attention + flash-style fused attention.
// GqaSelfAttention is the MHA layer with a separate (smaller) K/V head count:
// H query heads share Hkv key/value heads (group = H/Hkv). The attention itself
// uses the memory-fused vFlashAttn op (online softmax, no [Bq,L,L] score matrix)
// so the training forward is also the efficient path. With Hkv == H this is
// ordinary MHA; the win is the KV cache for generation stores Hkv heads (fewer)
// and the attention forward never materializes the L*L score tile.
// ============================================================================
pub const GqaSelfAttention = struct {
    alloc: mem.Allocator,
    n_heads: usize, // H  (query heads)
    n_kv_heads: usize, // Hkv (shared key/value heads)
    head_dim: usize, // Dh
    c: usize, // model dim = H*Dh
    c_kv: usize, // = Hkv*Dh
    group: usize, // H / Hkv
    Wq: *Var,
    Wk: *Var,
    Wv: *Var,
    Wo: *Var,
    cap: usize = 0, // KV-cache capacity (set when generation starts)

    pub fn init(alloc: mem.Allocator, c: usize, n_heads: usize, n_kv_heads: usize, prng: *std.Random.DefaultPrng) !GqaSelfAttention {
        std.debug.assert(c % n_heads == 0);
        std.debug.assert(n_heads % n_kv_heads == 0);
        const Dh = c / n_heads;
        const c_kv = n_kv_heads * Dh;
        return .{
            .alloc = alloc,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = Dh,
            .c = c,
            .c_kv = c_kv,
            .group = n_heads / n_kv_heads,
            .Wq = try Var.parameter(alloc, try initLinear(alloc, c, c, prng)),
            .Wk = try Var.parameter(alloc, try initLinear(alloc, c, c_kv, prng)),
            .Wv = try Var.parameter(alloc, try initLinear(alloc, c, c_kv, prng)),
            .Wo = try Var.parameter(alloc, try initLinear(alloc, c, c, prng)),
        };
    }

    // full training forward (autograd). x: [B, L, C] -> [B, L, C]
    pub fn forward(self: *GqaSelfAttention, x: *Var) Error!*Var {
        const B = x.value.shape[0];
        const L = x.value.shape[1];
        const C = self.c;
        const H = self.n_heads;
        const Hkv = self.n_kv_heads;
        const Dh = self.head_dim;
        const group = self.group;
        // projections
        const q = try ag.vMatmul(self.alloc, x, self.Wq); // [B,L,C]
        const k = try ag.vMatmul(self.alloc, x, self.Wk); // [B,L,Ckv]
        const v = try ag.vMatmul(self.alloc, x, self.Wv); // [B,L,Ckv]
        // split query heads: [B,L,C] -> [B*H, L, Dh]
        const qh = try ag.vReshape(self.alloc, q, &.{ B, L, H, Dh });
        const qp = try ag.vPermute(self.alloc, qh, &.{ 0, 2, 1, 3 });
        const qf = try ag.vReshape(self.alloc, qp, &.{ B * H, L, Dh });
        // split kv heads: [B,L,Ckv] -> [B*Hkv, L, Dh]
        const kh = try ag.vReshape(self.alloc, k, &.{ B, L, Hkv, Dh });
        const kp = try ag.vPermute(self.alloc, kh, &.{ 0, 2, 1, 3 });
        const kf = try ag.vReshape(self.alloc, kp, &.{ B * Hkv, L, Dh });
        const vh = try ag.vReshape(self.alloc, v, &.{ B, L, Hkv, Dh });
        const vp = try ag.vPermute(self.alloc, vh, &.{ 0, 2, 1, 3 });
        const vf = try ag.vReshape(self.alloc, vp, &.{ B * Hkv, L, Dh });
        // RoPE on Q, K (position = L axis)
        const qr = try ag.vRoPE(self.alloc, qf);
        const kr = try ag.vRoPE(self.alloc, kf);
        // fused GQA attention (online softmax; group query heads share kv heads)
        const ao = try ag.vFlashAttn(self.alloc, qr, kr, vf, group); // [B*H, L, Dh]
        // merge query heads back: [B*H,L,Dh] -> [B, L, C]
        const aoh = try ag.vReshape(self.alloc, ao, &.{ B, H, L, Dh });
        const aop = try ag.vPermute(self.alloc, aoh, &.{ 0, 2, 1, 3 });
        const aof = try ag.vReshape(self.alloc, aop, &.{ B, L, C });
        return ag.vMatmul(self.alloc, aof, self.Wo); // [B, L, C]
    }

    pub fn params(self: *GqaSelfAttention) [4]*Var {
        return .{ self.Wq, self.Wk, self.Wv, self.Wo };
    }

    // ---- KV cache for incremental generation (stores Hkv heads, not H) ----
    pub const Cache = struct {
        k: ?Tensor = null, // [B*Hkv, cap, Dh]
        v: ?Tensor = null,
        len: usize = 0,
    };

    // x: [B, 1, C] (one new token), pos = absolute position. Returns [B, 1, C].
    pub fn forwardIncr(self: *GqaSelfAttention, alloc: mem.Allocator, x: Tensor, pos: usize, cache: *Cache) !Tensor {
        const B = x.shape[0];
        const H = self.n_heads;
        const Hkv = self.n_kv_heads;
        const Dh = self.head_dim;
        const group = self.group;
        // projections (raw)
        const q = try ten.matmul(alloc, x, self.Wq.value); // [B,1,C]
        const k = try ten.matmul(alloc, x, self.Wk.value); // [B,1,Ckv]
        const v = try ten.matmul(alloc, x, self.Wv.value); // [B,1,Ckv]
        const qh = try splitHeads(alloc, q, B, 1, H, Dh); // [B*H,1,Dh]
        const kh = try splitHeads(alloc, k, B, 1, Hkv, Dh); // [B*Hkv,1,Dh]
        const vh = try splitHeads(alloc, v, B, 1, Hkv, Dh); // [B*Hkv,1,Dh]
        const qr = try ropeTensor(alloc, qh, pos); // [B*H,1,Dh]
        const kr = try ropeTensor(alloc, kh, pos); // [B*Hkv,1,Dh]
        // append into the (Hkv-head) cache
        if (cache.k == null) {
            cache.k = try ten.zeros(alloc, &.{ B * Hkv, self.cap, Dh });
            cache.v = try ten.zeros(alloc, &.{ B * Hkv, self.cap, Dh });
        }
        const kc = cache.k.?;
        const vc = cache.v.?;
        for (0..B * Hkv) |h| {
            for (0..Dh) |d| {
                kc.data[h * (self.cap * Dh) + cache.len * Dh + d] = kr.data[h * Dh + d];
                vc.data[h * (self.cap * Dh) + cache.len * Dh + d] = vh.data[h * Dh + d];
            }
        }
        cache.len += 1;
        const n = cache.len;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(Dh)));
        const out = try ten.zeros(alloc, &.{ B * H, 1, Dh });
        const sc = try alloc.alloc(f32, n);
        defer alloc.free(sc);
        for (0..B * H) |h| {
            const kv = h / group; // shared kv head for query head h
            // scores for the single query against n cached keys
            var mx: f32 = -1e30;
            for (0..n) |j| {
                var s: f32 = 0;
                for (0..Dh) |d| s += qr.data[h * Dh + d] * kc.data[kv * (self.cap * Dh) + j * Dh + d];
                s *= scale;
                sc[j] = s;
                if (s > mx) mx = s;
            }
            var z: f32 = 0;
            for (0..n) |j| {
                sc[j] = @exp(sc[j] - mx);
                z += sc[j];
            }
            const invz = if (z > 0) 1.0 / z else 0.0;
            for (0..n) |j| {
                const a = sc[j] * invz;
                for (0..Dh) |d| out.data[h * Dh + d] += a * vc.data[kv * (self.cap * Dh) + j * Dh + d];
            }
        }
        const merged = try mergeHeads(alloc, out, B, 1, H, Dh); // [B,1,C]
        return ten.matmul(alloc, merged, self.Wo.value);
    }
};

// GQA decoder block (mirrors TransformerBlock; isolated so the MHA path stays
// the verified Gate-8 reference). h = x + gqa_attn(rmsnorm(x)); out = h + ffn(rmsnorm(h))
pub const GqaTransformerBlock = struct {
    alloc: mem.Allocator,
    norm1: RMSNorm,
    attn: GqaSelfAttention,
    norm2: RMSNorm,
    ffn: SwiGLU,

    pub fn init(alloc: mem.Allocator, c: usize, n_heads: usize, n_kv_heads: usize, ffn_hidden: usize, prng: *std.Random.DefaultPrng) !GqaTransformerBlock {
        return .{
            .alloc = alloc,
            .norm1 = try RMSNorm.init(alloc, c),
            .attn = try GqaSelfAttention.init(alloc, c, n_heads, n_kv_heads, prng),
            .norm2 = try RMSNorm.init(alloc, c),
            .ffn = try SwiGLU.init(alloc, c, ffn_hidden, prng),
        };
    }

    pub fn forward(self: *GqaTransformerBlock, x: *Var) Error!*Var {
        const n1 = try self.norm1.forward(x);
        const a = try self.attn.forward(n1);
        const h = try ag.vAdd(self.alloc, x, a);
        const n2 = try self.norm2.forward(h);
        const f = try self.ffn.forward(n2);
        return ag.vAdd(self.alloc, h, f);
    }

    pub fn forwardIncr(self: *GqaTransformerBlock, alloc: mem.Allocator, x: Tensor, pos: usize, cache: *GqaSelfAttention.Cache) !Tensor {
        const n1 = try rmsNormTensor(alloc, x, self.norm1.gamma.value);
        const a = try self.attn.forwardIncr(alloc, n1, pos, cache);
        const h = try ten.add(alloc, x, a);
        const n2 = try rmsNormTensor(alloc, h, self.norm2.gamma.value);
        const f = try swigluTensor(alloc, n2, self.ffn.W_gate.value, self.ffn.W_up.value, self.ffn.W_down.value);
        return ten.add(alloc, h, f);
    }

    pub fn params(self: *GqaTransformerBlock, alloc: mem.Allocator) ![]*Var {
        var list: std.ArrayList(*Var) = .empty;
        defer list.deinit(alloc);
        try list.append(alloc, self.norm1.gamma);
        for (self.attn.params()) |p| try list.append(alloc, p);
        try list.append(alloc, self.norm2.gamma);
        for (self.ffn.params()) |p| try list.append(alloc, p);
        return list.toOwnedSlice(alloc);
    }
};

// Decoder LM using GQA blocks (fused flash attention). Same shape contract as
// TransformerLM: ids [B*L] -> logits [B*L, V]; forwardGen [L] -> [L, V] via cache.
pub const GqaTransformerLM = struct {
    alloc: mem.Allocator,
    vocab: usize,
    seq_len: usize,
    dim: usize,
    tok_emb: Embedding,
    blocks: []GqaTransformerBlock,
    norm_f: RMSNorm,
    W_head: *Var,

    pub fn init(alloc: mem.Allocator, vocab: usize, seq_len: usize, dim: usize, n_layers: usize, n_heads: usize, n_kv_heads: usize, ffn_hidden: usize, prng: *std.Random.DefaultPrng) !GqaTransformerLM {
        std.debug.assert(dim % n_heads == 0);
        std.debug.assert(n_heads % n_kv_heads == 0);
        const blocks = try alloc.alloc(GqaTransformerBlock, n_layers);
        for (0..n_layers) |i| blocks[i] = try GqaTransformerBlock.init(alloc, dim, n_heads, n_kv_heads, ffn_hidden, prng);
        return .{
            .alloc = alloc,
            .vocab = vocab,
            .seq_len = seq_len,
            .dim = dim,
            .tok_emb = try Embedding.init(alloc, vocab, dim, prng),
            .blocks = blocks,
            .norm_f = try RMSNorm.init(alloc, dim),
            .W_head = try Var.parameter(alloc, try initLinear(alloc, dim, vocab, prng)),
        };
    }

    pub fn forward(self: *GqaTransformerLM, ids: []const usize) Error!*Var {
        const B = ids.len / self.seq_len;
        const L = self.seq_len;
        const C = self.dim;
        const emb = try self.tok_emb.forward(ids); // [B*L, C]
        var x = try ag.vReshape(self.alloc, emb, &.{ B, L, C });
        for (self.blocks) |*blk| x = try blk.forward(x);
        x = try self.norm_f.forward(x);
        const logits = try ag.vMatmul(self.alloc, x, self.W_head); // [B, L, V]
        return ag.vReshape(self.alloc, logits, &.{ B * L, self.vocab });
    }

    pub fn params(self: *GqaTransformerLM, alloc: mem.Allocator) ![]*Var {
        var list: std.ArrayList(*Var) = .empty;
        defer list.deinit(alloc);
        try list.append(alloc, self.tok_emb.W);
        for (self.blocks) |*blk| {
            const bp = try blk.params(alloc);
            defer alloc.free(bp);
            for (bp) |p| try list.append(alloc, p);
        }
        try list.append(alloc, self.norm_f.gamma);
        try list.append(alloc, self.W_head);
        return list.toOwnedSlice(alloc);
    }

    pub fn forwardGen(self: *GqaTransformerLM, alloc: mem.Allocator, ids: []const usize) !Tensor {
        const L = ids.len;
        const B: usize = 1;
        const C = self.dim;
        const V = self.vocab;
        for (self.blocks) |*blk| blk.attn.cap = L;
        var caches = try alloc.alloc(GqaSelfAttention.Cache, self.blocks.len);
        defer alloc.free(caches);
        for (caches) |*c| c.* = .{};
        const out = try ten.zeros(alloc, &.{ L, V });
        for (0..L) |pos| {
            const id = ids[pos];
            const row = try ag.vIndexRows(alloc, self.tok_emb.W, &.{id}); // [1, C]
            var x = row.value;
            x = try ten.reshape(alloc, x, &.{ B, 1, C });
            for (self.blocks, 0..) |*blk, bi| {
                x = try blk.forwardIncr(alloc, x, pos, &caches[bi]);
            }
            x = try rmsNormTensor(alloc, x, self.norm_f.gamma.value);
            const logits1 = try ten.matmul(alloc, x, self.W_head.value); // [1,1,V]
            for (0..V) |v| out.data[pos * V + v] = logits1.data[v];
        }
        return out;
    }
};

// ===========================================================================
// Gate 10 — Gated Delta Attention / DeltaNet
// ===========================================================================
// Replaces softmax (QK^T -> softmax -> @V) attention with an associative
// recurrence: a per-head state S in R^{Dh x Dh} updated every step by the
// *delta rule* with a forget gate (alpha = exp(g), decays old state) and a
// write gate (beta = sigmoid(b), gates the new write):
//   r_t   = S_{t-1} k_t
//   d_t   = v_t - alpha_t * r_t
//   S_t   = alpha_t * S_{t-1} + beta_t * outer(d_t, k_t)
//   o_t   = S_t q_t
// This is a *linear* attention with a learnable correction (the delta term),
// so it is O(L * Dh^2) in time and O(Dh^2) in memory per head — no L x L
// score matrix. Position information comes from the recurrence + gates, so
// (unlike softmax attention) we do NOT apply RoPE here; that is the standard
// DeltaNet design. The recurrent op `ag.vGatedDelta` (grad-checked, Gate 10)
// implements the forward + BPTT backward for this exact recurrence.
//
// Why two gates?  alpha (forget) controls how much of the past state survives
// (exp so it stays positive and gives clean cumulative-decay ratios in the
// chunked-parallel form); beta (write) controls how strongly the new (v - Sk)
// correction is written (sigmoid, bounded). Together they let the layer learn
// to both remember long context (alpha ~ 1) and selectively overwrite (beta).
pub const GatedDeltaAttn = struct {
    alloc: mem.Allocator,
    n_heads: usize, // H
    head_dim: usize, // Dh
    c: usize, // model dim = H*Dh
    Wq: *Var,
    Wk: *Var,
    Wv: *Var,
    Wo: *Var,
    W_alpha: *Var, // gate logits -> exp (forget)   [C, H]
    W_beta: *Var, // gate logits -> sigmoid (write) [C, H]

    pub fn init(alloc: mem.Allocator, c: usize, n_heads: usize, prng: *std.Random.DefaultPrng) !GatedDeltaAttn {
        std.debug.assert(c % n_heads == 0);
        const Dh = c / n_heads;
        return .{
            .alloc = alloc,
            .n_heads = n_heads,
            .head_dim = Dh,
            .c = c,
            .Wq = try Var.parameter(alloc, try initLinear(alloc, c, c, prng)),
            .Wk = try Var.parameter(alloc, try initLinear(alloc, c, c, prng)),
            .Wv = try Var.parameter(alloc, try initLinear(alloc, c, c, prng)),
            .Wo = try Var.parameter(alloc, try initLinear(alloc, c, c, prng)),
            // small init so exp(g)~1 (forget ~1 at start) and sigmoid(b)~0.5
            .W_alpha = try Var.parameter(alloc, try initLinear(alloc, c, n_heads, prng)),
            .W_beta = try Var.parameter(alloc, try initLinear(alloc, c, n_heads, prng)),
        };
    }

    // full training forward (autograd). x: [B, L, C] -> [B, L, C]
    pub fn forward(self: *GatedDeltaAttn, x: *Var) Error!*Var {
        const B = x.value.shape[0];
        const L = x.value.shape[1];
        const C = self.c;
        const H = self.n_heads;
        const Dh = self.head_dim;
        // projections -> [B, L, C]
        const q = try ag.vMatmul(self.alloc, x, self.Wq);
        const k = try ag.vMatmul(self.alloc, x, self.Wk);
        const v = try ag.vMatmul(self.alloc, x, self.Wv);
        // scale k by 1/sqrt(Dh) for numerical stability of the delta outer product
        const k_s = try ag.vMulScalar(self.alloc, k, 1.0 / @sqrt(@as(f32, @floatFromInt(Dh))));
        // split heads: [B,L,C] -> [B*H, L, Dh]
        const qh = try ag.vReshape(self.alloc, q, &.{ B, L, H, Dh });
        const qp = try ag.vPermute(self.alloc, qh, &.{ 0, 2, 1, 3 });
        const qf = try ag.vReshape(self.alloc, qp, &.{ B * H, L, Dh });
        const kh = try ag.vReshape(self.alloc, k_s, &.{ B, L, H, Dh });
        const kp = try ag.vPermute(self.alloc, kh, &.{ 0, 2, 1, 3 });
        const kf = try ag.vReshape(self.alloc, kp, &.{ B * H, L, Dh });
        const vh = try ag.vReshape(self.alloc, v, &.{ B, L, H, Dh });
        const vp = try ag.vPermute(self.alloc, vh, &.{ 0, 2, 1, 3 });
        const vf = try ag.vReshape(self.alloc, vp, &.{ B * H, L, Dh });
        // gates: x @ W_gate -> [B,L,H] -> [B*H, L]; exp / sigmoid
        const glogits = try ag.vMatmul(self.alloc, x, self.W_alpha); // [B,L,H]
        const gp = try ag.vPermute(self.alloc, try ag.vReshape(self.alloc, glogits, &.{ B, L, H }), &.{ 0, 2, 1 });
        const gv = try ag.vReshape(self.alloc, gp, &.{ B * H, L });
        const alpha = try ag.vExp(self.alloc, gv); // forget gate (decay)
        const blogits = try ag.vMatmul(self.alloc, x, self.W_beta); // [B,L,H]
        const bp = try ag.vPermute(self.alloc, try ag.vReshape(self.alloc, blogits, &.{ B, L, H }), &.{ 0, 2, 1 });
        const bv = try ag.vReshape(self.alloc, bp, &.{ B * H, L });
        const beta = try ag.vSigmoid(self.alloc, bv); // write gate
        // recurrent gated delta attention -> [B*H, L, Dh]
        const ao = try ag.vGatedDelta(self.alloc, qf, kf, vf, alpha, beta);
        // merge heads back: [B*H,L,Dh] -> [B, L, C]
        const aoh = try ag.vReshape(self.alloc, ao, &.{ B, H, L, Dh });
        const aop = try ag.vPermute(self.alloc, aoh, &.{ 0, 2, 1, 3 });
        const aof = try ag.vReshape(self.alloc, aop, &.{ B, L, C });
        return ag.vMatmul(self.alloc, aof, self.Wo); // [B, L, C]
    }

    pub fn params(self: *GatedDeltaAttn) [6]*Var {
        return .{ self.Wq, self.Wk, self.Wv, self.Wo, self.W_alpha, self.W_beta };
    }

    // ---- recurrent state cache for incremental generation ----
    // The delta rule is inherently stateful & sequential, so generation just
    // carries the Dh x Dh state per (batch,head) and advances one step at a
    // time. No key/value history is stored (memory is O(H * Dh^2), independent
    // of sequence length — the whole point of a linear-attention recurrence).
    pub const Cache = struct {
        state: ?Tensor = null, // [B*H, Dh, Dh]
    };

    // x: [B, 1, C] (one new token), pos = absolute position (unused: position
    // is implicit in the carried state). Returns [B, 1, C].
    pub fn forwardIncr(self: *GatedDeltaAttn, alloc: mem.Allocator, x: Tensor, pos: usize, cache: *Cache) !Tensor {
        _ = pos;
        const B = x.shape[0];
        const H = self.n_heads;
        const Dh = self.head_dim;
        const C = self.c;
        _ = C;
        const q = try ten.matmul(alloc, x, self.Wq.value); // [B,1,C]
        const k = try ten.matmul(alloc, x, self.Wk.value);
        const v = try ten.matmul(alloc, x, self.Wv.value);
        const glog = try ten.matmul(alloc, x, self.W_alpha.value); // [B,1,H]
        const blog = try ten.matmul(alloc, x, self.W_beta.value); // [B,1,H]
        const qh = try splitHeads(alloc, q, B, 1, H, Dh); // [B*H,1,Dh]
        const kh_raw = try splitHeads(alloc, k, B, 1, H, Dh);
        const vh = try splitHeads(alloc, v, B, 1, H, Dh);
        const kscale = 1.0 / @sqrt(@as(f32, @floatFromInt(Dh)));
        // one delta step per (batch,head) against the carried state
        if (cache.state == null) cache.state = try ten.zeros(alloc, &.{ B * H, Dh, Dh });
        const S = cache.state.?;
        const out = try ten.zeros(alloc, &.{ B * H, 1, Dh });
        for (0..B * H) |h| {
            const a = @exp(glog.data[h]); // forget gate ([B,1,H] -> [B*H] interleaved)
            const b = 1.0 / (1.0 + @exp(-blog.data[h])); // write gate (sigmoid)
            const kt = kh_raw.data[h * Dh .. h * Dh + Dh];
            const vt = vh.data[h * Dh .. h * Dh + Dh];
            const qt = qh.data[h * Dh .. h * Dh + Dh];
            const st = S.data[h * Dh * Dh .. h * Dh * Dh + Dh * Dh];
            var ks: [64]f32 = undefined;
            std.debug.assert(Dh <= 64);
            for (0..Dh) |d| ks[d] = kt[d] * kscale;
            // r = S k ; d = v - a*r ; S <- a*S + b*outer(d,k)
            for (0..Dh) |i| {
                var ri: f32 = 0;
                for (0..Dh) |j| ri += st[i * Dh + j] * ks[j];
                const di = vt[i] - a * ri;
                for (0..Dh) |j| st[i * Dh + j] = a * st[i * Dh + j] + b * di * ks[j];
            }
            // o = S q
            for (0..Dh) |i| {
                var s: f32 = 0;
                for (0..Dh) |j| s += st[i * Dh + j] * qt[j];
                out.data[h * Dh + i] = s;
            }
        }
        const merged = try mergeHeads(alloc, out, B, 1, H, Dh); // [B,1,C]
        return ten.matmul(alloc, merged, self.Wo.value);
    }
};

// DeltaNet decoder block (mirrors the GQA block; only the attention swaps).
// h = x + delta_attn(rmsnorm(x)); out = h + ffn(rmsnorm(h))
pub const GatedDeltaBlock = struct {
    alloc: mem.Allocator,
    norm1: RMSNorm,
    attn: GatedDeltaAttn,
    norm2: RMSNorm,
    ffn: SwiGLU,

    pub fn init(alloc: mem.Allocator, c: usize, n_heads: usize, ffn_hidden: usize, prng: *std.Random.DefaultPrng) !GatedDeltaBlock {
        return .{
            .alloc = alloc,
            .norm1 = try RMSNorm.init(alloc, c),
            .attn = try GatedDeltaAttn.init(alloc, c, n_heads, prng),
            .norm2 = try RMSNorm.init(alloc, c),
            .ffn = try SwiGLU.init(alloc, c, ffn_hidden, prng),
        };
    }

    pub fn forward(self: *GatedDeltaBlock, x: *Var) Error!*Var {
        const n1 = try self.norm1.forward(x);
        const a = try self.attn.forward(n1);
        const h = try ag.vAdd(self.alloc, x, a);
        const n2 = try self.norm2.forward(h);
        const f = try self.ffn.forward(n2);
        return ag.vAdd(self.alloc, h, f);
    }

    pub fn forwardIncr(self: *GatedDeltaBlock, alloc: mem.Allocator, x: Tensor, pos: usize, cache: *GatedDeltaAttn.Cache) !Tensor {
        const n1 = try rmsNormTensor(alloc, x, self.norm1.gamma.value);
        const a = try self.attn.forwardIncr(alloc, n1, pos, cache);
        const h = try ten.add(alloc, x, a);
        const n2 = try rmsNormTensor(alloc, h, self.norm2.gamma.value);
        const f = try swigluTensor(alloc, n2, self.ffn.W_gate.value, self.ffn.W_up.value, self.ffn.W_down.value);
        return ten.add(alloc, h, f);
    }

    pub fn params(self: *GatedDeltaBlock, alloc: mem.Allocator) ![]*Var {
        var list: std.ArrayList(*Var) = .empty;
        defer list.deinit(alloc);
        try list.append(alloc, self.norm1.gamma);
        for (self.attn.params()) |p| try list.append(alloc, p);
        try list.append(alloc, self.norm2.gamma);
        for (self.ffn.params()) |p| try list.append(alloc, p);
        return list.toOwnedSlice(alloc);
    }
};

// Decoder LM using gated-delta (DeltaNet) blocks. Same shape contract as
// TransformerLM / GqaTransformerLM: ids [B*L] -> logits [B*L, V]; forwardGen
// [L] -> [L, V] via the recurrent state cache.
pub const GatedDeltaLM = struct {
    alloc: mem.Allocator,
    vocab: usize,
    seq_len: usize,
    dim: usize,
    tok_emb: Embedding,
    blocks: []GatedDeltaBlock,
    norm_f: RMSNorm,
    W_head: *Var,

    pub fn init(alloc: mem.Allocator, vocab: usize, seq_len: usize, dim: usize, n_layers: usize, n_heads: usize, ffn_hidden: usize, prng: *std.Random.DefaultPrng) !GatedDeltaLM {
        std.debug.assert(dim % n_heads == 0);
        const blocks = try alloc.alloc(GatedDeltaBlock, n_layers);
        for (0..n_layers) |i| blocks[i] = try GatedDeltaBlock.init(alloc, dim, n_heads, ffn_hidden, prng);
        return .{
            .alloc = alloc,
            .vocab = vocab,
            .seq_len = seq_len,
            .dim = dim,
            .tok_emb = try Embedding.init(alloc, vocab, dim, prng),
            .blocks = blocks,
            .norm_f = try RMSNorm.init(alloc, dim),
            .W_head = try Var.parameter(alloc, try initLinear(alloc, dim, vocab, prng)),
        };
    }

    pub fn forward(self: *GatedDeltaLM, ids: []const usize) Error!*Var {
        const B = ids.len / self.seq_len;
        const L = self.seq_len;
        const C = self.dim;
        const emb = try self.tok_emb.forward(ids); // [B*L, C]
        var x = try ag.vReshape(self.alloc, emb, &.{ B, L, C });
        for (self.blocks) |*blk| x = try blk.forward(x);
        x = try self.norm_f.forward(x);
        const logits = try ag.vMatmul(self.alloc, x, self.W_head); // [B, L, V]
        return ag.vReshape(self.alloc, logits, &.{ B * L, self.vocab });
    }

    pub fn params(self: *GatedDeltaLM, alloc: mem.Allocator) ![]*Var {
        var list: std.ArrayList(*Var) = .empty;
        defer list.deinit(alloc);
        try list.append(alloc, self.tok_emb.W);
        for (self.blocks) |*blk| {
            const bp = try blk.params(alloc);
            defer alloc.free(bp);
            for (bp) |p| try list.append(alloc, p);
        }
        try list.append(alloc, self.norm_f.gamma);
        try list.append(alloc, self.W_head);
        return list.toOwnedSlice(alloc);
    }

    pub fn forwardGen(self: *GatedDeltaLM, alloc: mem.Allocator, ids: []const usize) !Tensor {
        const L = ids.len;
        const B: usize = 1;
        const C = self.dim;
        const V = self.vocab;
        var caches = try alloc.alloc(GatedDeltaAttn.Cache, self.blocks.len);
        defer alloc.free(caches);
        for (caches) |*c| c.* = .{};
        const out = try ten.zeros(alloc, &.{ L, V });
        for (0..L) |pos| {
            const id = ids[pos];
            const row = try ag.vIndexRows(alloc, self.tok_emb.W, &.{id}); // [1, C]
            var x = row.value;
            x = try ten.reshape(alloc, x, &.{ B, 1, C });
            for (self.blocks, 0..) |*blk, bi| {
                x = try blk.forwardIncr(alloc, x, pos, &caches[bi]);
            }
            x = try rmsNormTensor(alloc, x, self.norm_f.gamma.value);
            const logits1 = try ten.matmul(alloc, x, self.W_head.value); // [1,1,V]
            for (0..V) |v| out.data[pos * V + v] = logits1.data[v];
        }
        return out;
    }
};
