// 09-transformer — milestone: the modern transformer LM.
// A decoder-only transformer (RMSNorm + RoPE + SwiGLU + causal multi-head
// attention with KV cache) trained on the tokenized corpus, then sampled
// autoregressively via the cached generation path.
// Run with:  zig build run-09-transformer   (slow in Debug; use -Doptimize=ReleaseFast)
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

pub fn main() void {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.Options.debug_io;

    // 1) corpus
    var text: []const u8 = "";
    if (std.Io.Dir.cwd().readFileAlloc(io, "data/text.bin", a, .limited(1 << 30))) |buf| {
        text = buf;
        std.debug.print("loaded corpus: {d} bytes\n", .{text.len});
    } else |_| {
        var t: std.ArrayList(u8) = .empty;
        var k: usize = 0;
        while (k < 3000) : (k += 1) {
            t.appendSlice(a, EMBEDDED) catch return;
            t.append(a, '\n') catch return;
        }
        text = t.items;
        std.debug.print("data/text.bin not found; using embedded prose ({d} bytes).\n", .{text.len});
    }

    // 2) tokenizer (slice-trained)
    const train_slice_len = @min(text.len, 1 << 18);
    var tk = BPE.init(a);
    defer tk.deinit();
    tk.train(text[0..train_slice_len], 512) catch {
        std.debug.print("tokenizer train failed\n", .{});
        return;
    };
    std.debug.print("tokenizer: vocab {d}\n", .{tk.vocabSize()});

    // 3) dataset (per-position causal windows)
    const seq_len: usize = 16;
    const batch_size: usize = 16;
    var ds = data.Dataset.fromText(a, &tk, text, 0.1, .{ .seq_len = seq_len, .batch_size = batch_size, .seed = 1337 }) catch {
        std.debug.print("dataset build failed\n", .{});
        return;
    };
    defer ds.deinit();
    const V = tk.vocabSize();
    std.debug.print("dataset: train {d} tokens / {d} batches | val {d} tokens\n\n", .{
        ds.train.tokens.len, ds.train.numBatches(), ds.val.tokens.len,
    });

    // 4) build the transformer LM
    const C: usize = 64; // 4 heads x 16 head_dim
    const H: usize = 4;
    const layers: usize = 2;
    const ffn_hidden: usize = 128;
    var prng = std.Random.DefaultPrng.init(2025);
    var lm = nn.TransformerLM.init(a, V, seq_len, C, layers, H, ffn_hidden, &prng) catch {
        std.debug.print("LM init failed\n", .{});
        return;
    };
    const ps = lm.params(a) catch return;
    // route embeddings + lm head to Adam, hidden matrices to Muon (Gate 7 hybrid)
    const adam_override = a.alloc(*znn.Var, 2) catch return;
    adam_override[0] = lm.tok_emb.W;
    adam_override[1] = lm.W_head;
    var opt = optim.Hybrid.init(a, ps, adam_override, 0.01, 0.95, 0.003, 0.0) catch {
        var opt2 = optim.Adam.init(a, 0.003, 0.0);
        defer opt2.deinit();
        return;
    };
    defer opt.deinit();

    // 5) train
    const steps: usize = 600;
    var best_val: f32 = 1e9;
    for (0..steps) |s| {
        const b = blk: {
            if (ds.train.nextBatch(a) catch return) |x| break :blk x;
            ds.train.reset();
            break :blk (ds.train.nextBatch(a) catch return).?;
        };
        const ids = a.alloc(usize, b.x.len) catch return;
        for (b.x, 0..) |x, i| ids[i] = x;
        const tgt = a.alloc(usize, b.y.len) catch return;
        for (b.y, 0..) |x, i| tgt[i] = x;
        const logits = lm.forward(ids) catch return;
        const l = loss.softmaxCE(a, logits, tgt) catch return;
        l.backprop();
        opt.step();
        if (s % 100 == 0 or s == steps - 1) {
            ds.val.reset();
            var vl: f32 = 0;
            var nv: usize = 0;
            var vi: usize = 0;
            while (vi < @min(ds.val.numBatches(), 6)) : (vi += 1) {
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
            if (val < best_val) best_val = val;
            std.debug.print("step {d:>4}  train_loss={d:.4}  val_loss={d:.4}\n", .{ s, l.value.data[0], val });
            ds.train.reset();
        }
    }
    std.debug.print("\nbest val loss = {d:.4}\n", .{best_val});

    // 6) sample with the KV cache (autoregressive generation)
    const seed_text = "the quick brown ";
    const seed_ids = tk.encode(seed_text) catch return;
    var ctx = a.alloc(usize, seq_len) catch return;
    if (seed_ids.len >= seq_len) {
        for (0..seq_len) |i| ctx[i] = seed_ids[seed_ids.len - seq_len + i];
    } else {
        for (0..seq_len) |i| ctx[i] = if (i < seq_len - seed_ids.len) seed_ids[0] else seed_ids[i - (seq_len - seed_ids.len)];
    }

    std.debug.print("\nseed: \"{s}\"\nsample (greedy, KV-cache):\n  {s}", .{ seed_text, seed_text });
    // prime the cache with the seed context, then generate
    var gen = lm.forwardGen(a, ctx) catch return; // logits for the last position
    var last_logits = gen.data[(seq_len - 1) * V .. seq_len * V];
    var sample_len: usize = 48;
    while (sample_len > 0) : (sample_len -= 1) {
        // greedy pick from the last position's logits
        var best: f32 = last_logits[0];
        var next_id: usize = 0;
        for (1..V) |k| if (last_logits[k] > best) {
            best = last_logits[k];
            next_id = k;
        };
        const piece = tk.decode(&[_]u32{@intCast(next_id)}) catch return;
        std.debug.print("{s}", .{piece});
        // feed the new token through the cached generator (extends the cache)
        const ext = lm.forwardGen(a, &[_]usize{next_id}) catch return;
        last_logits = ext.data[0..V];
        _ = &last_logits;
    }
    std.debug.print("\n", .{});
}