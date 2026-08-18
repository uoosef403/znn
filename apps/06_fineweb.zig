// 06-fineweb — milestone: the data pipeline. Tokenize a corpus, build a
// next-token-prediction dataset, and stream deterministic batches.
// Run with:  zig build run-06-fineweb
//
// Real path:  python scripts/fetch_fineweb.py --bytes 100_000_000
//             (writes data/text.bin), then this app reads it.
// Fallback:   if data/text.bin is absent, trains on an embedded prose slice so
//             the pipeline is always smoke-testable offline.
const std = @import("std");
const znn = @import("znn");
const BPE = znn.tokenizer.BPE;
const data = znn.data;

const EMBEDDED =
    \\The quick brown fox jumps over the lazy dog.
    \\A journey of a thousand miles begins with a single step.
    \\To be, or not to be, that is the question.
    \\The only thing we have to fear is fear itself.
    \\Ask not what your country can do for you; ask what you can do for your country.
    \\Knowledge is power; education is the premise of progress.
    \\Science is the poetry of reality; we are made of star-stuff.
    \\The universe is under no obligation to make sense to you.
    \\Over the hill the quick brown fox jumps the lazy dog again and again.
    \\Dogs sleep, foxes jump, and the brown hill watches them all day long.
;

pub fn main() void {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // 1) get the corpus: prefer data/text.bin, fall back to embedded prose
    var text: []const u8 = "";
    const io = std.Options.debug_io;
    const path = "data/text.bin";
    if (std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 30))) |buf| {
        text = buf;
        std.debug.print("loaded corpus from {s}: {d} bytes\n", .{ path, text.len });
    } else |_| {
        var t: std.ArrayList(u8) = .empty;
        var k: usize = 0;
        while (k < 2000) : (k += 1) {
            t.appendSlice(a, EMBEDDED) catch return;
            t.append(a, '\n') catch return;
        }
        text = t.items;
        std.debug.print("data/text.bin not found; using embedded prose ({d} bytes).\n", .{text.len});
        std.debug.print("  for the real corpus: python scripts/fetch_fineweb.py --bytes 100_000_000\n", .{});
    }

    // 2) train a byte-level BPE on a SLICE of the corpus (training is O(V*N) per
    //    merge, so we cap the training text; the merges still cover the full
    //    corpus because BPE always falls back to single bytes for unseen pairs).
    //    Use -Doptimize=ReleaseFast to train on a bigger slice / vocab quickly.
    const train_slice_len = @min(text.len, 1 << 20); // 1 MiB
    var tk = BPE.init(a);
    defer tk.deinit();
    tk.train(text[0..train_slice_len], 512) catch {
        std.debug.print("tokenizer train failed\n", .{});
        return;
    };
    std.debug.print("tokenizer: vocab {d} (merges {d}) trained on {d}-byte slice\n", .{ tk.vocabSize(), tk.merges.items.len, train_slice_len });

    // 3) build the dataset (tokenize the whole corpus, 10% val, causal windows)
    const cfg = data.Config{ .seq_len = 32, .batch_size = 16, .seed = 1337 };
    var ds = data.Dataset.fromText(a, &tk, text, 0.1, cfg) catch {
        std.debug.print("dataset build failed\n", .{});
        return;
    };
    defer ds.deinit();
    std.debug.print("dataset: train {d} tokens / {d} windows / {d} batches | val {d} tokens / {d} windows\n\n", .{
        ds.train.tokens.len, ds.train.numSeqs(), ds.train.numBatches(),
        ds.val.tokens.len, ds.val.numSeqs(),
    });

    // 4) stream a few batches and time the iteration (in-memory => no IO stalls)
    const nb: usize = @min(ds.train.numBatches(), 20);
    const t0 = std.Io.Clock.now(.awake, io);
    var total_examples: usize = 0;
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const b = (ds.train.nextBatch(a) catch return) orelse break;
        defer b.deinit(a);
        total_examples += b.b;
    }
    const t1 = std.Io.Clock.now(.awake, io);
    const elapsed_ns: u64 = @intCast(t1.nanoseconds - t0.nanoseconds);
    std.debug.print("streamed {d} batches ({d} examples) in {d} us\n", .{ nb, total_examples, elapsed_ns / 1000 });
    if (elapsed_ns > 0)
        std.debug.print("  throughput: {d} examples/s (pure in-memory iteration)\n", .{
            @as(u64, total_examples) * std.time.ns_per_s / elapsed_ns,
        });

    // 5) peek at the first batch's first row to confirm x->y shift
    ds.train.reset();
    if (ds.train.nextBatch(a) catch null) |b| {
        const L = b.l;
        std.debug.print("\nfirst batch, row 0:\n  x:", .{});
        for (b.x[0..L]) |id| std.debug.print(" {d}", .{id});
        std.debug.print("\n  y:", .{});
        for (b.y[0..L]) |id| std.debug.print(" {d}", .{id});
        std.debug.print("\n  (y[i] == x[i+1]: {})\n", .{std.mem.eql(u32, b.y[0 .. L - 1], b.x[1..L])});
        b.deinit(a);
    }

    // 6) determinism: same seed -> identical start order
    var ds2 = data.Dataset.fromText(a, &tk, text, 0.1, cfg) catch return;
    defer ds2.deinit();
    std.debug.print("\ndeterminism (same seed -> same start order): {}\n", .{
        std.mem.eql(usize, ds.train.starts, ds2.train.starts),
    });
}