// 07-first-lm — milestone: the first language model.
// Tokenize a corpus, train a makemore-style MLP context LM to predict the NEXT
// token, watch the loss fall below the unigram baseline, then sample
// autoregressively (predict -> append -> slide the window).
// Run with:  zig build run-07-first-lm
//           (add -- --release for a bigger/longer run via program args)
const std = @import("std");
const znn = @import("znn");
const BPE = znn.tokenizer.BPE;
const data = znn.data;
const nn = znn.nn;
const optim = znn.optim;
const loss = znn.loss;
const ten = znn.tensor;

// A small embedded prose corpus so the milestone is always smoke-testable
// offline. For the real run: python scripts/fetch_fineweb.py --bytes 100_000_000
// writes data/text.bin, which this app prefers.
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

    // 1) corpus: prefer data/text.bin, else repeat the embedded prose
    var text: []const u8 = "";
    if (std.Io.Dir.cwd().readFileAlloc(io, "data/text.bin", a, .limited(1 << 30))) |buf| {
        text = buf;
        std.debug.print("loaded corpus: {d} bytes\n", .{text.len});
    } else |_| {
        var t: std.ArrayList(u8) = .empty;
        var k: usize = 0;
        while (k < 4000) : (k += 1) {
            t.appendSlice(a, EMBEDDED) catch return;
            t.append(a, '\n') catch return;
        }
        text = t.items;
        std.debug.print("data/text.bin not found; using embedded prose ({d} bytes).\n", .{text.len});
    }

    // 2) train byte-level BPE on a slice (O(V*N)/merge; slice keeps it fast)
    const train_slice_len = @min(text.len, 1 << 18);
    var tk = BPE.init(a);
    defer tk.deinit();
    tk.train(text[0..train_slice_len], 512) catch {
        std.debug.print("tokenizer train failed\n", .{});
        return;
    };
    std.debug.print("tokenizer: vocab {d} (merges {d})\n", .{ tk.vocabSize(), tk.merges.items.len });

    // 3) build the causal-window dataset
    const seq_len: usize = 16;
    const batch_size: usize = 32;
    var ds = data.Dataset.fromText(a, &tk, text, 0.1, .{
        .seq_len = seq_len,
        .batch_size = batch_size,
        .seed = 1337,
    }) catch {
        std.debug.print("dataset build failed\n", .{});
        return;
    };
    defer ds.deinit();
    const V = tk.vocabSize();
    std.debug.print("dataset: train {d} tokens / {d} batches | val {d} tokens\n\n", .{
        ds.train.tokens.len, ds.train.numBatches(), ds.val.tokens.len,
    });

    // 4) ---- unigram baseline: per-token -log(train marginal) on val ----
    var marg = a.alloc(f32, V) catch return;
    for (marg) |*m| m.* = 1.0; // Laplace smoothing
    for (ds.train.tokens) |t| marg[t] += 1.0;
    var totm: f32 = 0;
    for (marg) |m| totm += m;
    for (marg) |*m| m.* /= totm;
    var uni_loss: f32 = 0;
    for (ds.val.tokens) |t| uni_loss -= @log(@max(marg[t], 1e-12));
    uni_loss /= @as(f32, @floatFromInt(ds.val.tokens.len));
    std.debug.print("baseline unigram val loss = {d:.4}\n\n", .{uni_loss});

    // 5) ---- train the MLPLM ----
    const dim: usize = 64;
    const hidden: usize = 128;
    var prng = std.Random.DefaultPrng.init(2024);
    var lm = nn.MLPLM.init(a, V, seq_len, dim, hidden, &prng) catch {
        std.debug.print("LM init failed\n", .{});
        return;
    };
    const ps = lm.params(a) catch return;
    var opt = optim.Adam.init(a, 0.003, 0.0);
    defer opt.deinit();

    const steps: usize = 800;
    var prev_loss: f32 = 1e9;
    var best_val: f32 = 1e9;
    for (0..steps) |step| {
        const b = blk: {
            if (ds.train.nextBatch(a) catch return) |x| break :blk x;
            ds.train.reset();
            break :blk (ds.train.nextBatch(a) catch return).?;
        };
        const ids = a.alloc(usize, b.x.len) catch return;
        for (b.x, 0..) |x, i| ids[i] = x;
        // target = the token AFTER the window = last column of the shifted targets
        const tgt = a.alloc(usize, b.b) catch return;
        for (0..b.b) |bi| tgt[bi] = b.y[bi * b.l + (b.l - 1)];

        const logits = lm.forward(ids) catch return;
        const l = loss.softmaxCE(a, logits, tgt) catch return;
        l.backprop();
        opt.step(ps);

        prev_loss = l.value.data[0];
        if (step % 100 == 0 or step == steps - 1) {
            // quick val estimate on a few val batches
            ds.val.reset();
            var vl: f32 = 0;
            var nv: usize = 0;
            var vi: usize = 0;
            while (vi < @min(ds.val.numBatches(), 8)) : (vi += 1) {
                const vb = (ds.val.nextBatch(a) catch null) orelse break;
                const vids = a.alloc(usize, vb.x.len) catch return;
                for (vb.x, 0..) |x, i| vids[i] = x;
                const vtgt = a.alloc(usize, vb.b) catch return;
                for (0..vb.b) |bi| vtgt[bi] = vb.y[bi * vb.l + (vb.l - 1)];
                const vl_logits = lm.forward(vids) catch return;
                const vl_l = loss.softmaxCE(a, vl_logits, vtgt) catch return;
                vl += vl_l.value.data[0] * @as(f32, @floatFromInt(vb.b));
                nv += vb.b;
            }
            const val = if (nv > 0) vl / @as(f32, @floatFromInt(nv)) else 0;
            if (val < best_val) best_val = val;
            std.debug.print("step {d:>4}  train_loss={d:.4}  val_loss={d:.4}\n", .{ step, prev_loss, val });
            ds.train.reset(); // keep train cursor from drifting mid-epoch on small data
        }
    }

    std.debug.print("\nbest val loss = {d:.4}  (unigram baseline = {d:.4})\n", .{ best_val, uni_loss });
    if (best_val < uni_loss) {
        std.debug.print("=> the context model BEATS the unigram baseline\n", .{});
    } else {
        std.debug.print("=> did not beat baseline (try more steps / a bigger corpus)\n", .{});
    }

    // 6) ---- sample autoregressively from a seed window ----
    // encode a seed, then predict next token, append, slide the window.
    const seed_text = "the quick brown ";
    const seed_ids_raw = tk.encode(seed_text) catch return;
    var ctx = a.alloc(usize, seq_len) catch return;
    // left-pad / fill the context window with the seed (repeat if too short)
    if (seed_ids_raw.len >= seq_len) {
        for (0..seq_len) |i| ctx[i] = seed_ids_raw[seed_ids_raw.len - seq_len + i];
    } else {
        for (0..seq_len) |i| ctx[i] = if (i < seq_len - seed_ids_raw.len) seed_ids_raw[0] else seed_ids_raw[i - (seq_len - seed_ids_raw.len)];
    }

    std.debug.print("\nseed: \"{s}\"\nsample (greedy, temp=0):\n  ", .{seed_text});
    var sample_len: usize = 48;
    const temperature: f32 = 0.0; // 0 = greedy
    while (sample_len > 0) : (sample_len -= 1) {
        const logits_v = lm.forwardOne(ctx) catch return; // [V]
        const logits = logits_v.value.data;
        var next_id: usize = 0;
        if (temperature == 0.0) {
            var best: f32 = logits[0];
            for (1..V) |k| if (logits[k] > best) {
                best = logits[k];
                next_id = k;
            };
        } else {
            // softmax with temperature, multinomial sample
            const probs = a.alloc(f32, V) catch return;
            var mx: f32 = logits[0];
            for (logits) |l| if (l > mx) { mx = l; };
            var z: f32 = 0;
            for (0..V) |k| {
                probs[k] = std.math.exp((logits[k] - mx) / temperature);
                z += probs[k];
            }
            for (probs) |*p| p.* /= z;
            const r = prng.random().float(f32);
            var cdf: f32 = 0;
            for (0..V) |k| {
                cdf += probs[k];
                if (r <= cdf) {
                    next_id = k;
                    break;
                }
            }
        }
        // emit the decoded byte(s) for the chosen token
        const piece = tk.decode(&[_]u32{@intCast(next_id)}) catch return;
        std.debug.print("{s}", .{piece});
        // slide the window
        for (0..seq_len - 1) |i| ctx[i] = ctx[i + 1];
        ctx[seq_len - 1] = next_id;
    }
    std.debug.print("\n", .{});
}