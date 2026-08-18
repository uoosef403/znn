// data.zig — turn raw token ids into next-token-prediction batches.
//
// Pipeline role: a corpus (bytes) -> BPE -> []u32 token ids -> this module.
// We split the flat token stream into train/val by position (the standard,
// leakage-free split: the val block is a contiguous tail of the stream, not a
// random scatter), then turn each split into fixed-length context windows.
//
// A window starting at index t uses `seq_len` input tokens tokens[t..t+L] and
// `seq_len` targets tokens[t+1..t+L+1] (shifted by one) — the causal LM objective.
// So a split of N tokens yields N-L windows (start t in [0, N-L-1]).
//
// Batching is deterministic given the seed: we Fisher-Yates shuffle the window
// start indices with a seeded PRNG, so `zig build` runs are reproducible. The
// whole corpus lives in memory once loaded, so iteration is pure pointer math
// — no IO stalls. (For a corpus that doesn't fit in RAM, memory-map the token
// file instead; the start-index scheme is identical.)

const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const testing = std.testing;
const BPE = @import("tokenizer.zig").BPE;

pub const Config = struct {
    seq_len: usize,
    batch_size: usize,
    seed: u64 = 0,
};

// One minibatch: flat row-major input + target token ids, each length B*L.
// Kept as []u32 (integers), not f32 tensors — token ids are indices, not
// numbers; the embedding layer (Gate 6) will lift them into the f32 graph.
pub const Batch = struct {
    x: []u32,
    y: []u32,
    b: usize,
    l: usize,

    pub fn deinit(self: Batch, alloc: Allocator) void {
        alloc.free(self.x);
        alloc.free(self.y);
    }
};

pub const Split = struct {
    tokens: []const u32,
    seq_len: usize,
    batch_size: usize,
    starts: []usize, // shuffled window-start indices
    pos: usize = 0,

    pub fn numSeqs(self: Split) usize {
        if (self.tokens.len <= self.seq_len) return 0;
        return self.tokens.len - self.seq_len;
    }

    pub fn numBatches(self: Split) usize {
        return self.numSeqs() / self.batch_size;
    }

    pub fn reset(self: *Split) void {
        self.pos = 0;
    }

    // Returns the next batch, or null at end of epoch.
    pub fn nextBatch(self: *Split, alloc: Allocator) !?Batch {
        if (self.pos + self.batch_size > self.starts.len) return null;
        const B = self.batch_size;
        const L = self.seq_len;
        const x = try alloc.alloc(u32, B * L);
        const y = try alloc.alloc(u32, B * L);
        for (0..B) |bi| {
            const t = self.starts[self.pos + bi];
            for (0..L) |j| {
                x[bi * L + j] = self.tokens[t + j];
                y[bi * L + j] = self.tokens[t + j + 1];
            }
        }
        self.pos += B;
        return Batch{ .x = x, .y = y, .b = B, .l = L };
    }
};

pub const Dataset = struct {
    alloc: Allocator,
    train: Split,
    val: Split,
    // we keep the full id buffer + own the shuffled index arrays
    _ids: []u32,
    _train_starts: []usize,
    _val_starts: []usize,

    pub fn deinit(self: *Dataset) void {
        self.alloc.free(self._ids);
        self.alloc.free(self._train_starts);
        self.alloc.free(self._val_starts);
    }

    // Build a dataset directly from pre-tokenized ids (used by tests and by
    // callers that already have a token stream).
    pub fn fromIds(alloc: Allocator, ids: []const u32, val_frac: f32, cfg: Config) !Dataset {
        const n = ids.len;
        const val_n = @as(usize, @intFromFloat(@as(f32, @floatFromInt(n)) * val_frac));
        const train_n = n - val_n;
        const train_ids = ids[0..train_n];
        const val_ids = ids[train_n..];

        const ids_copy = try alloc.alloc(u32, n);
        @memcpy(ids_copy, ids);

        const train_starts = try makeStarts(alloc, train_ids.len, cfg.seq_len, cfg.seed);
        // different seed for val so its order differs from train's
        const val_starts = try makeStarts(alloc, val_ids.len, cfg.seq_len, cfg.seed ^ 0x9e3779b97f4a7c15);

        return .{
            .alloc = alloc,
            .train = .{ .tokens = ids_copy[0..train_n], .seq_len = cfg.seq_len, .batch_size = cfg.batch_size, .starts = train_starts },
            .val = .{ .tokens = ids_copy[train_n..], .seq_len = cfg.seq_len, .batch_size = cfg.batch_size, .starts = val_starts },
            ._ids = ids_copy,
            ._train_starts = train_starts,
            ._val_starts = val_starts,
        };
    }

    // Convenience: tokenize raw text with a trained tokenizer, then build.
    pub fn fromText(alloc: Allocator, tk: *BPE, text: []const u8, val_frac: f32, cfg: Config) !Dataset {
        const ids = try tk.encode(text);
        defer alloc.free(ids);
        return fromIds(alloc, ids, val_frac, cfg);
    }

    // Load raw text from a file (raw bytes), tokenize, build. Uses the new
    // std.Io file API (see memory: zig016-file-io-api).
    pub fn fromTokenFile(alloc: Allocator, tk: *BPE, path: []const u8, val_frac: f32, cfg: Config) !Dataset {
        const io = std.Options.debug_io;
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(1 << 30));
        defer alloc.free(text);
        return fromText(alloc, tk, text, val_frac, cfg);
    }
};

// Build the shuffled list of window start indices [0, n_seq) for a token count.
// Fisher-Yates with a seeded PRNG => deterministic across runs.
fn makeStarts(alloc: Allocator, token_count: usize, seq_len: usize, seed: u64) ![]usize {
    var n_seq: usize = 0;
    if (token_count > seq_len) n_seq = token_count - seq_len;
    var starts = try alloc.alloc(usize, n_seq);
    for (0..n_seq) |i| starts[i] = i;
    var prng = std.Random.DefaultPrng.init(seed);
    var r = prng.random();
    // Fisher-Yates
    var i: usize = n_seq;
    while (i > 1) {
        i -= 1;
        const j = r.uintLessThan(usize, i + 1);
        const tmp = starts[i];
        starts[i] = starts[j];
        starts[j] = tmp;
    }
    return starts;
}

// ===========================================================================
// tests
// ===========================================================================

test "Split: counts and no out-of-range starts" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const ids = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    var ds = try Dataset.fromIds(a, &ids, 0.25, .{ .seq_len = 4, .batch_size = 2, .seed = 1 });
    defer ds.deinit();

    // train = first 9 tokens (0..8), val = last 3 (9..11)
    try testing.expectEqual(@as(usize, 9), ds.train.tokens.len);
    try testing.expectEqual(@as(usize, 3), ds.val.tokens.len);
    // train windows: 9 - 4 = 5 ; val windows: 3 <= 4 -> 0
    try testing.expectEqual(@as(usize, 5), ds.train.numSeqs());
    try testing.expectEqual(@as(usize, 0), ds.val.numSeqs());
    for (ds.train.starts) |s| try testing.expect(s + 4 + 1 <= ds.train.tokens.len);
}

test "Dataset: batches are the shifted causal windows (x[t..t+L], y[t+1..t+L+1])" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // 20 tokens, seq_len 4, batch 3, no val
    var ids_list: [20]u32 = undefined;
    for (0..20) |i| ids_list[i] = @intCast(i + 100); // 100..119
    var ds = try Dataset.fromIds(a, &ids_list, 0.0, .{ .seq_len = 4, .batch_size = 3, .seed = 7 });
    defer ds.deinit();

    // with val_frac 0, val is empty (0 tokens) and train is all 20
    try testing.expectEqual(@as(usize, 20), ds.train.tokens.len);
    try testing.expectEqual(@as(usize, 16), ds.train.numSeqs()); // 20-4
    try testing.expectEqual(@as(usize, 5), ds.train.numBatches()); // 16/3 = 5

    var n_batches: usize = 0;
    while (try ds.train.nextBatch(a)) |b| {
        n_batches += 1;
        try testing.expectEqual(@as(usize, 3), b.b);
        try testing.expectEqual(@as(usize, 4), b.l);
        // for each row, y must equal x shifted by one token in the ORIGINAL stream
        for (0..b.b) |bi| {
            for (0..b.l) |j| {
                try testing.expectEqual(b.x[bi * b.l + j] + 1, b.y[bi * b.l + j]);
            }
        }
        b.deinit(a);
    }
    try testing.expectEqual(@as(usize, 5), n_batches);
}

test "Dataset: deterministic order (same seed -> same batches)" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const ids = [_]u32{ 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    var ds1 = try Dataset.fromIds(a, &ids, 0.0, .{ .seq_len = 4, .batch_size = 2, .seed = 42 });
    defer ds1.deinit();
    var ds2 = try Dataset.fromIds(a, &ids, 0.0, .{ .seq_len = 4, .batch_size = 2, .seed = 42 });
    defer ds2.deinit();

    // start-index orders must match
    try testing.expectEqualSlices(usize, ds1.train.starts, ds2.train.starts);

    // and the first batches must match
    const b1 = (try ds1.train.nextBatch(a)).?;
    defer b1.deinit(a);
    const b2 = (try ds2.train.nextBatch(a)).?;
    defer b2.deinit(a);
    try testing.expectEqualSlices(u32, b1.x, b2.x);
    try testing.expectEqualSlices(u32, b1.y, b2.y);
}

test "Dataset: train and val do not share windows (leakage-free split)" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // 1000 tokens, 10% val, seq_len 8
    var ids = try a.alloc(u32, 1000);
    defer a.free(ids);
    for (0..1000) |i| ids[i] = @intCast(i);
    var ds = try Dataset.fromIds(a, ids, 0.1, .{ .seq_len = 8, .batch_size = 16, .seed = 3 });
    defer ds.deinit();

    try testing.expectEqual(@as(usize, 900), ds.train.tokens.len);
    try testing.expectEqual(@as(usize, 100), ds.val.tokens.len);

    // every train window must lie entirely within [0, 900) and every val window
    // within [900, 1000). A window covers [t, t+seq_len]; since val starts at 900
    // and train ends at 900, the only possible cross-boundary window would start
    // in train and reach into val. Verify none do.
    const train_end: usize = 900;
    for (ds.train.starts) |s| {
        try testing.expect(s + 8 + 1 <= train_end); // x and y both inside train
    }
    for (ds.val.starts) |s| {
        try testing.expect(s + 8 + 1 <= ds.val.tokens.len);
    }
    // val tokens begin exactly where train ends
    try testing.expectEqual(@as(u32, 900), ds.val.tokens[0]);
}

test "Dataset: full epoch returns numBatches batches then null; reset re-epochs" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const ids = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8 };
    var ds = try Dataset.fromIds(a, &ids, 0.0, .{ .seq_len = 2, .batch_size = 2, .seed = 0 });
    defer ds.deinit();
    // 9 tokens, seq_len 2 -> numSeqs = 9-2 = 7 -> batches = 3
    try testing.expectEqual(@as(usize, 7), ds.train.numSeqs());
    try testing.expectEqual(@as(usize, 3), ds.train.numBatches());

    var count: usize = 0;
    while (try ds.train.nextBatch(a)) |b| : (count += 1) b.deinit(a);
    try testing.expectEqual(@as(usize, 3), count);
    try testing.expectEqual(@as(?Batch, null), try ds.train.nextBatch(a));

    ds.train.reset();
    const again = (try ds.train.nextBatch(a)).?;
    again.deinit(a);
    try testing.expect(again.b == 2);
}