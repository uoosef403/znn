// 11-gated-delta — milestone: Gated Delta Attention / DeltaNet.
// Trains a DeltaNet LM (gated delta recurrence, no softmax, O(L*Dh^2)) on the
// tokenized corpus, samples autoregressively, and compares quality against a
// matched softmax-attention (GQA) LM at the same width/depth.
//
// This is the "hero" architecture of the curriculum: a modern linear-attention
// model with a learnable delta update rule and forget/write gates, verified
// op-by-op against finite differences (the recurrent + chunked-parallel forms
// match to <1e-3) and here end-to-end on real text.
//
// Run with:  zig build run-11-gated-delta   (use -Doptimize=ReleaseFast)
const std = @import("std");
const znn = @import("znn");
const BPE = znn.tokenizer.BPE;
const data = znn.data;
const nn = znn.nn;
const optim = znn.optim;
const loss = znn.loss;
const ten = znn.tensor;
const Var = znn.Var;

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

const Config = struct { steps: usize, lr: f32 };

// Generic train + best-val + sample. Works for any LM with
// .forward/.params/.forwardGen/.seq_len/.vocab (GatedDeltaLM, GqaTransformerLM).
// The caller constructs `lm` (architectures have different init arities) and
// hands us the parameter list.
fn run(comptime LM: type, a: std.mem.Allocator, lm: *LM, ds: *data.Dataset, V: usize, cfg: Config, tk: *BPE, label: []const u8) f32 {
    const ps = lm.params(a) catch return 1e9;
    var opt = optim.Adam.init(a, cfg.lr, 0.0);
    defer opt.deinit();

    var best_val: f32 = 1e9;
    for (0..cfg.steps) |s| {
        const b = blk: {
            if (ds.train.nextBatch(a) catch return 1e9) |x| break :blk x;
            ds.train.reset();
            break :blk (ds.train.nextBatch(a) catch return 1e9).?;
        };
        const ids = a.alloc(usize, b.x.len) catch return 1e9;
        for (b.x, 0..) |x, i| ids[i] = x;
        const tgt = a.alloc(usize, b.y.len) catch return 1e9;
        for (b.y, 0..) |x, i| tgt[i] = x;
        const logits = lm.forward(ids) catch return 1e9;
        const l = loss.softmaxCE(a, logits, tgt) catch return 1e9;
        l.backprop();
        opt.step(ps);
        if (s % 100 == 0 or s == cfg.steps - 1) {
            ds.val.reset();
            var vl: f32 = 0;
            var nv: usize = 0;
            var vi: usize = 0;
            while (vi < @min(ds.val.numBatches(), 6)) : (vi += 1) {
                const vb = (ds.val.nextBatch(a) catch null) orelse break;
                const vids = a.alloc(usize, vb.x.len) catch return 1e9;
                for (vb.x, 0..) |x, i| vids[i] = x;
                const vtgt = a.alloc(usize, vb.y.len) catch return 1e9;
                for (vb.y, 0..) |x, i| vtgt[i] = x;
                const vl_logits = lm.forward(vids) catch return 1e9;
                const vl_l = loss.softmaxCE(a, vl_logits, vtgt) catch return 1e9;
                vl += vl_l.value.data[0] * @as(f32, @floatFromInt(vb.y.len));
                nv += vb.y.len;
            }
            const val = if (nv > 0) vl / @as(f32, @floatFromInt(nv)) else 0;
            if (val < best_val) best_val = val;
            std.debug.print("  [{s}] step {d:>4}  train={d:.4}  val={d:.4}\n", .{ label, s, l.value.data[0], val });
            ds.train.reset();
        }
    }
    std.debug.print("  [{s}] best val = {d:.4}\n", .{ label, best_val });

    // sample: rolling seq_len window, rebuild state each step (matches training).
    const seed_text = "the quick brown ";
    const seed_ids = tk.encode(seed_text) catch return best_val;
    const L = ds.train.seq_len;
    var ctx = a.alloc(usize, L) catch return best_val;
    if (seed_ids.len >= L) {
        for (0..L) |i| ctx[i] = seed_ids[seed_ids.len - L + i];
    } else {
        for (0..L) |i| ctx[i] = if (i < L - seed_ids.len) seed_ids[0] else seed_ids[i - (L - seed_ids.len)];
    }
    std.debug.print("  [{s}] sample: {s}", .{ label, seed_text });
    var sample_len: usize = 40;
    while (sample_len > 0) : (sample_len -= 1) {
        const gen = lm.forwardGen(a, ctx) catch return best_val;
        const last_logits = gen.data[(L - 1) * V .. L * V];
        var best: f32 = last_logits[0];
        var next_id: usize = 0;
        for (1..V) |k| if (last_logits[k] > best) {
            best = last_logits[k];
            next_id = k;
        };
        const piece = tk.decode(&[_]u32{@intCast(next_id)}) catch return best_val;
        std.debug.print("{s}", .{piece});
        var i: usize = 0;
        while (i + 1 < L) : (i += 1) ctx[i] = ctx[i + 1];
        ctx[L - 1] = next_id;
    }
    std.debug.print("\n", .{});
    return best_val;
}

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

    // 2) tokenizer
    const train_slice_len = @min(text.len, 1 << 18);
    var tk = BPE.init(a);
    defer tk.deinit();
    tk.train(text[0..train_slice_len], 512) catch {
        std.debug.print("tokenizer train failed\n", .{});
        return;
    };
    std.debug.print("tokenizer: vocab {d}\n", .{tk.vocabSize()});

    // 3) dataset
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

    // matched config: same width/depth/FFN for both architectures. DeltaNet
    // adds 2 tiny gate matrices [C,H] per layer (~negligible), so the two
    // models are within ~1% in parameter count — a fair, equal-capacity
    // comparison. Same optimizer (Adam), lr, steps, data, and seed.
    const C: usize = 64;
    const H: usize = 4;
    const layers: usize = 2;
    const ffn: usize = 128;
    const cfg = Config{ .steps = 400, .lr = 0.003 };

    std.debug.print("=== DeltaNet (gated delta attention) ===\n", .{});
    var prng1 = std.Random.DefaultPrng.init(2026);
    ds.train.reset();
    var delta_lm = nn.GatedDeltaLM.init(a, V, seq_len, C, layers, H, ffn, &prng1) catch return;
    const delta_val = run(nn.GatedDeltaLM, a, &delta_lm, &ds, V, cfg, &tk, "delta");

    std.debug.print("\n=== Softmax attention (GQA, matched) ===\n", .{});
    var prng2 = std.Random.DefaultPrng.init(2026); // same seed -> fair init-scale comparison
    ds.train.reset();
    var gqa_lm = nn.GqaTransformerLM.init(a, V, seq_len, C, layers, H, H, ffn, &prng2) catch return; // Hkv=H (MHA, full kv)
    const gqa_val = run(nn.GqaTransformerLM, a, &gqa_lm, &ds, V, cfg, &tk, "gqa");

    std.debug.print("\n=== comparison (equal params, same data/optimizer/steps/seed) ===\n", .{});
    std.debug.print("  DeltaNet  best val = {d:.4}\n", .{delta_val});
    std.debug.print("  GQA       best val = {d:.4}\n", .{gqa_val});
    if (delta_val < gqa_val) {
        std.debug.print("  -> DeltaNet wins by {d:.4}\n", .{gqa_val - delta_val});
    } else {
        std.debug.print("  -> GQA wins by {d:.4}\n", .{delta_val - gqa_val});
    }
}