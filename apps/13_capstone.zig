// 13-capstone — the capstone: a configurable, checkpointable, sampleable
// language model trained on FineWeb from scratch.
//
// This ties together every gate of the curriculum into one runnable artifact:
//   * data pipeline   (Gate 5)  — FineWeb corpus + a persisted BPE tokenizer
//   * embeddings/LM   (Gate 6)  — token embeddings + a decoder LM
//   * optimizers      (Gate 7)  — Hybrid Muon(matrices)+Adam(embeds/head) with
//                                warmup→cosine LR schedule and global grad clip
//   * transformer     (Gates 8/9) — RMSNorm + RoPE + SwiGLU + GQA + KV cache
//   * sampling        — autoregressive generation with temperature / top-k /
//                       top-p (nucleus) over the KV-cache forward path
//   * checkpointing   — serialize every parameter to data/ckpt.bin and resume
//
// The GPU backend (Gate 11) is NOT wired into the training loop here: the
// CPU autograd graph is the reference path. The GPU path (cuBLAS matmul +
// the gated-delta kernel) is exercised by `zig build bench -Dgpu`. Porting
// the whole autograd graph to GPU is the natural next step (see lesson 12).
//
// USAGE
//   zig build run-13-capstone                 # smoke: tiny config, ~50 steps
//   zig build run-13-capstone -- train        # the REAL config below
//   zig build run-13-capstone -- resume       # load data/ckpt.bin + keep training
//   zig build run-13-capstone -- sample       # load ckpt + generate text
// (Add -Doptimize=ReleaseFast for any non-smoke run.)
//
// The SMOKE config is sized to finish in seconds on CPU so the gate
// (loss drops + sampling works) is verifiable locally. The REAL config is a
// starting point for a RunPod run — see scripts/runpod_capstone.sh.

const std = @import("std");
const znn = @import("znn");
const BPE = znn.tokenizer.BPE;
const data = znn.data;
const nn = znn.nn;
const optim = znn.optim;
const loss = znn.loss;
const ten = znn.tensor;

const EMBEDDED =
    \\the quick brown fox jumps over the lazy dog
    \\a journey of a thousand miles begins with a single step
    \\to be or not to be that is the question
    \\the only thing we have to fear is fear itself
    \\ask not what your country can do for you ask what you can do for your country
    \\knowledge is power education is the premise of progress
    \\science is the poetry of reality we are made of star stuff
    \\the universe is under no obligation to make sense to you
    \\over the hill the quick brown fox jumps the lazy dog again and again
    \\dogs sleep foxes jump and the brown hill watches them all day long
    \\the quick fox and the lazy dog play over the hill every day
    \\a single step begins a journey of a thousand miles over the hill
;

// --------------------------------------------------------------------------
// Config — every hyperparameter in one place. Edit REAL for a real run.
// --------------------------------------------------------------------------
const Config = struct {
    mode: Mode = .smoke,
    // architecture
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize, // GQA: n_kv_heads < n_heads shrinks the KV cache
    ffn_hidden: usize,
    seq_len: usize,
    // data
    batch_size: usize,
    vocab: usize, // BPE target vocab
    corpus_slice: usize, // bytes of FineWeb to train the tokenizer on
    val_frac: f32,
    // optimization
    steps: usize,
    warmup: usize,
    lr_muon: f32,
    lr_adam: f32,
    weight_decay: f32,
    grad_clip: f32,
    min_lr_factor: f32,
    // sampling
    sample_len: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed_text: []const u8,
    // io
    ckpt_path: []const u8,
    bpe_path: []const u8,
    corpus_path: []const u8,
    log_every: usize,
    ckpt_every: usize,
};

const Mode = enum { smoke, train, cont, sample };

const SMOKE = Config{
    .mode = .smoke,
    .dim = 64, .n_layers = 2, .n_heads = 4, .n_kv_heads = 2, .ffn_hidden = 128,
    .seq_len = 16, .batch_size = 8, .vocab = 512,
    .corpus_slice = 1 << 16, .val_frac = 0.1,
    .steps = 50, .warmup = 8, .lr_muon = 0.02, .lr_adam = 0.003,
    .weight_decay = 0.0, .grad_clip = 1.0, .min_lr_factor = 0.1,
    .sample_len = 48, .temperature = 0.8, .top_k = 40, .top_p = 0.9,
    .seed_text = "the quick brown ",
    .ckpt_path = "data/ckpt_smoke.bin", .bpe_path = "data/bpe_smoke.bin",
    .corpus_path = "data/text.bin", .log_every = 10, .ckpt_every = 25,
};

// A starting point for a "real" RunPod run (A100/H100, this codebase CPU-bound
// would be slow — see the RunPod script which uses the GPU bench-verified path).
// Bigger model, longer context, more steps. NOT run locally here.
const REAL = Config{
    .mode = .train,
    .dim = 384, .n_layers = 8, .n_heads = 6, .n_kv_heads = 2, .ffn_hidden = 1024,
    .seq_len = 256, .batch_size = 32, .vocab = 4096,
    .corpus_slice = 1 << 26, .val_frac = 0.05,
    .steps = 4000, .warmup = 200, .lr_muon = 0.02, .lr_adam = 0.003,
    .weight_decay = 0.01, .grad_clip = 1.0, .min_lr_factor = 0.05,
    .sample_len = 200, .temperature = 0.7, .top_k = 80, .top_p = 0.9,
    .seed_text = "In a world where",
    .ckpt_path = "data/ckpt.bin", .bpe_path = "data/bpe.bin",
    .corpus_path = "data/text.bin", .log_every = 50, .ckpt_every = 500,
};

fn io() std.Io {
    return std.Options.debug_io;
}

// --------------------------------------------------------------------------
// Checkpointing is provided by znn.checkpoint (shared with the test suite):
//   znn.checkpoint.save(a, params, path)
//   znn.checkpoint.load(a, params, path)   (error.BadCheckpoint if missing)
// The format and the round-trip test live in src/znn/checkpoint.zig.
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// Sampling: temperature → top-k → top-p → softmax → categorical draw.
// --------------------------------------------------------------------------
fn sampleNext(a: std.mem.Allocator, logits: []const f32, prng: *std.Random.DefaultPrng, temp: f32, top_k: usize, top_p: f32) usize {
    const V = logits.len;
    // greedy / temperature == 0 -> argmax
    if (temp <= 0.0 or top_k == 1) {
        var best: f32 = logits[0];
        var bi: usize = 0;
        for (1..V) |i| if (logits[i] > best) {
            best = logits[i];
            bi = i;
        };
        return bi;
    }
    var l = a.alloc(f32, V) catch return 0;
    defer a.free(l);
    for (0..V) |i| l[i] = logits[i] / temp;

    // index array sorted by logit descending
    var idx = a.alloc(usize, V) catch return 0;
    defer a.free(idx);
    for (0..V) |i| idx[i] = i;
    std.sort.pdq(usize, idx, l, struct {
        fn cmp(ctx: []const f32, lhs: usize, rhs: usize) bool {
            return ctx[lhs] > ctx[rhs];
        }
    }.cmp);

    // top-k mask: zero probability of everything past rank k
    const k = if (top_k > 0 and top_k < V) top_k else V;
    for (k..V) |r| l[idx[r]] = -std.math.inf(f32);

    // softmax over the survivors (max-subtract for stability)
    var maxv: f32 = -std.math.inf(f32);
    for (0..V) |i| maxv = @max(maxv, l[i]);
    var probs = a.alloc(f32, V) catch return 0;
    defer a.free(probs);
    var zsum: f32 = 0;
    for (0..V) |i| {
        const e = if (l[i] == -std.math.inf(f32)) 0.0 else std.math.exp(l[i] - maxv);
        probs[i] = e;
        zsum += e;
    }
    if (zsum <= 0) return idx[0];
    for (0..V) |i| probs[i] /= zsum;

    // top-p (nucleus): walk the sorted-by-prob order, keep the smallest set
    // whose cumulative probability >= top_p; renormalize over the nucleus.
    if (top_p < 1.0) {
        // idx is already sorted by logit desc ~ by prob desc; re-sort by prob
        std.sort.pdq(usize, idx, probs, struct {
            fn cmp(ctx: []const f32, lhs: usize, rhs: usize) bool {
                return ctx[lhs] > ctx[rhs];
            }
        }.cmp);
        var cum: f32 = 0;
        var cutoff: usize = V;
        for (0..V) |r| {
            cum += probs[idx[r]];
            if (cum >= top_p) {
                cutoff = r + 1;
                break;
            }
        }
        // build a fresh normalized distribution over the top `cutoff` tokens
        var nz: f32 = 0;
        for (0..V) |i| probs[i] = 0;
        for (0..cutoff) |r| {
            const e = std.math.exp(l[idx[r]] - maxv);
            probs[idx[r]] = e;
            nz += e;
        }
        if (nz > 0) for (0..V) |i| {
            if (probs[i] > 0) probs[i] /= nz;
        };
    }

    // categorical draw via inverse-CDF
    const r = prng.random().float(f32);
    var acc: f32 = 0;
    for (0..V) |i| {
        acc += probs[i];
        if (r < acc) return i;
    }
    return V - 1;
}

// --------------------------------------------------------------------------
// Training loop.
// --------------------------------------------------------------------------
fn train(a: std.mem.Allocator, lm: *nn.GqaTransformerLM, ds: *data.Dataset, cfg: Config, tk: *BPE, do_resume: bool) void {
    const ps = lm.params(a) catch return;
    // Hybrid: Muon for 2D matrix weights, Adam for embeddings + LM head + 1D.
    // The token-embedding table and the output head are 2D but updated
    // sparsely/as a projection, where the Newton-Schulz factor is ill-defined —
    // route them to Adam explicitly.
    const adam_override = a.alloc(*znn.Var, 2) catch return;
    defer a.free(adam_override);
    adam_override[0] = lm.tok_emb.W;
    adam_override[1] = lm.W_head;
    var opt = optim.Hybrid.init(a, ps, adam_override, cfg.lr_muon, 0.95, cfg.lr_adam, cfg.weight_decay) catch return;
    defer opt.deinit();
    const sched = optim.Scheduler{ .warmup = cfg.warmup, .total = cfg.steps, .min_factor = cfg.min_lr_factor };

    if (do_resume) {
        znn.checkpoint.load(a, ps, cfg.ckpt_path) catch {
            std.debug.print("(starting fresh)\n", .{});
        };
    }

    var best_val: f32 = 1e9;
    const start_step: usize = if (do_resume) 0 else 0; // real resume would seek; kept simple
    var s: usize = start_step;
    while (s < cfg.steps) : (s += 1) {
        const f = sched.factor(s);
        opt.muon.lr = cfg.lr_muon * f;
        opt.adam.lr = cfg.lr_adam * f;

        const b = blk: {
            if (ds.train.nextBatch(a) catch return) |x| break :blk x;
            ds.train.reset();
            break :blk (ds.train.nextBatch(a) catch return) orelse return;
        };
        const ids = a.alloc(usize, b.x.len) catch return;
        for (b.x, 0..) |x, i| ids[i] = x;
        const tgt = a.alloc(usize, b.y.len) catch return;
        for (b.y, 0..) |x, i| tgt[i] = x;
        const logits = lm.forward(ids) catch return;
        const l = loss.softmaxCE(a, logits, tgt) catch return;
        l.backprop();
        _ = optim.clipGradNorm(ps, cfg.grad_clip);
        opt.step();

        if (s % cfg.log_every == 0 or s == cfg.steps - 1) {
            // quick val over a few batches
            ds.val.reset();
            var vl: f32 = 0;
            var nv: usize = 0;
            var vi: usize = 0;
            while (vi < @min(ds.val.numBatches(), 4)) : (vi += 1) {
                const vb = (ds.val.nextBatch(a) catch null) orelse break;
                const vids = a.alloc(usize, vb.x.len) catch return;
                for (vb.x, 0..) |x, i| vids[i] = x;
                const vtgt = a.alloc(usize, vb.y.len) catch return;
                for (vb.y, 0..) |x, i| vtgt[i] = x;
                const vl_logits = lm.forward(vids) catch return;
                const vl_l = loss.softmaxCE(a, vl_logits, vtgt) catch return;
                vl += vl_l.value.data[0] * @as(f32, @floatFromInt(vb.y.len));
                nv += vb.y.len;
            }
            const val = if (nv > 0) vl / @as(f32, @floatFromInt(nv)) else 0;
            if (val < best_val) {
                best_val = val;
                znn.checkpoint.save(a, ps, cfg.ckpt_path) catch {};
            }
            std.debug.print("  step {d:>5}/{d}  lr_factor={d:.3}  train={d:.4}  val={d:.4}  best={d:.4}\n", .{
                s, cfg.steps, f, l.value.data[0], val, best_val,
            });
            ds.train.reset();
        }
        if (cfg.ckpt_every > 0 and s > 0 and s % cfg.ckpt_every == 0) {
            znn.checkpoint.save(a, ps, cfg.ckpt_path) catch {};
        }
    }
    znn.checkpoint.save(a, ps, cfg.ckpt_path) catch {};
    std.debug.print("training done. best val = {d:.4}  (checkpoint: {s})\n", .{ best_val, cfg.ckpt_path });
    _ = tk;
}

// --------------------------------------------------------------------------
// Generation: autoregressive sampling over the KV-cache forward path.
// --------------------------------------------------------------------------
fn generate(a: std.mem.Allocator, lm: *nn.GqaTransformerLM, ds: *data.Dataset, V: usize, cfg: Config, tk: *BPE) void {
    const ps = lm.params(a) catch return;
    znn.checkpoint.load(a, ps, cfg.ckpt_path) catch {
        std.debug.print("no checkpoint to sample from ({s})\n", .{cfg.ckpt_path});
        return;
    };
    var prng = std.Random.DefaultPrng.init(2027);
    const L = ds.train.seq_len;
    const seed_ids = tk.encode(cfg.seed_text) catch return;
    var ctx = a.alloc(usize, L) catch return;
    if (seed_ids.len >= L) {
        for (0..L) |i| ctx[i] = seed_ids[seed_ids.len - L + i];
    } else {
        for (0..L) |i| ctx[i] = if (i < L - seed_ids.len) seed_ids[0] else seed_ids[i - (L - seed_ids.len)];
    }
    std.debug.print("sample ({d:.2}/{d}/{d:.2}): {s}", .{ cfg.temperature, cfg.top_k, cfg.top_p, cfg.seed_text });
    var n: usize = cfg.sample_len;
    while (n > 0) : (n -= 1) {
        const gen = lm.forwardGen(a, ctx) catch return;
        const last_logits = gen.data[(L - 1) * V .. L * V];
        const next_id = sampleNext(a, last_logits, &prng, cfg.temperature, cfg.top_k, cfg.top_p);
        const piece = tk.decode(&[_]u32{@intCast(next_id)}) catch return;
        std.debug.print("{s}", .{piece});
        var i: usize = 0;
        while (i + 1 < L) : (i += 1) ctx[i] = ctx[i + 1];
        ctx[L - 1] = next_id;
    }
    std.debug.print("\n", .{});
}

// --------------------------------------------------------------------------
// main
// --------------------------------------------------------------------------
pub fn main(init: std.process.Init.Minimal) void {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // ---- parse mode from argv ----
    // The config (SMOKE by default) selects the model size/data/checkpoint
    // paths; the mode arg only selects the *behavior* (train / resume /
    // sample) within that config. `real` swaps to the REAL config (for a
    // RunPod run) and defaults to training. This keeps `sample` usable on the
    // smoke checkpoint you just trained, instead of needing the real one.
    var cfg = SMOKE;
    var args_it = std.process.Args.Iterator.initAllocator(init.args, std.heap.smp_allocator) catch return;
    defer args_it.deinit();
    _ = args_it.next(); // exe name
    if (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "train")) {
            cfg.mode = .train;
        } else if (std.mem.eql(u8, arg, "resume")) {
            cfg.mode = .cont;
        } else if (std.mem.eql(u8, arg, "sample")) {
            cfg.mode = .sample;
        } else if (std.mem.eql(u8, arg, "smoke")) {
            cfg.mode = .smoke;
        } else if (std.mem.eql(u8, arg, "real")) {
            cfg = REAL;
        } else {
            std.debug.print("unknown mode '{s}'; using smoke\n", .{arg});
        }
    }
    std.debug.print("=== capstone [{s}] ({s} config) ===\n", .{ @tagName(cfg.mode), if (cfg.vocab == SMOKE.vocab) "smoke" else "real" });

    // ---- corpus ----
    var text: []const u8 = "";
    if (std.Io.Dir.cwd().readFileAlloc(io(), cfg.corpus_path, a, .limited(1 << 30))) |buf| {
        text = buf;
        std.debug.print("loaded corpus: {d} bytes\n", .{text.len});
    } else |_| {
        var t: std.ArrayList(u8) = .empty;
        var k: usize = 0;
        const reps: usize = if (cfg.mode == .smoke) 1500 else 200000;
        while (k < reps) : (k += 1) {
            t.appendSlice(a, EMBEDDED) catch return;
            t.append(a, '\n') catch return;
        }
        text = t.items;
        std.debug.print("corpus not found; using embedded prose ({d} bytes).\n", .{text.len});
    }

    // ---- tokenizer (load if present, else train + save) ----
    var tk = BPE.init(a);
    defer tk.deinit();
    tk.load(cfg.bpe_path) catch {
        tk.train(text[0..@min(text.len, cfg.corpus_slice)], cfg.vocab) catch {
            std.debug.print("tokenizer train failed\n", .{});
            return;
        };
        tk.save(cfg.bpe_path) catch {};
        std.debug.print("tokenizer trained: vocab {d} -> {s}\n", .{ tk.vocabSize(), cfg.bpe_path });
    };
    if (tk.vocabSize() == 0) {
        // loaded but empty? retrain
        tk.train(text[0..@min(text.len, cfg.corpus_slice)], cfg.vocab) catch return;
    }
    std.debug.print("tokenizer vocab: {d}\n", .{tk.vocabSize()});
    const V = tk.vocabSize();

    // ---- dataset ----
    var ds = data.Dataset.fromText(a, &tk, text, cfg.val_frac, .{
        .seq_len = cfg.seq_len, .batch_size = cfg.batch_size, .seed = 1337,
    }) catch {
        std.debug.print("dataset build failed\n", .{});
        return;
    };
    defer ds.deinit();
    std.debug.print("dataset: train {d} tokens/{d} batches | val {d} tokens\n", .{
        ds.train.tokens.len, ds.train.numBatches(), ds.val.tokens.len,
    });

    // ---- model ----
    var prng = std.Random.DefaultPrng.init(2026);
    var lm = nn.GqaTransformerLM.init(a, V, cfg.seq_len, cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.ffn_hidden, &prng) catch {
        std.debug.print("model init failed\n", .{});
        return;
    };
    // rough parameter count
    const ps0 = lm.params(a) catch return;
    var total: usize = 0;
    for (ps0) |p| total += p.value.numel();
    std.debug.print("model: {d} layers, dim {d}, heads {d}/{d} (GQA), {d} params\n", .{
        cfg.n_layers, cfg.dim, cfg.n_heads, cfg.n_kv_heads, total,
    });

    switch (cfg.mode) {
        .smoke, .train => train(a, &lm, &ds, cfg, &tk, false),
        .cont => train(a, &lm, &ds, cfg, &tk, true),
        .sample => generate(a, &lm, &ds, V, cfg, &tk),
    }
}