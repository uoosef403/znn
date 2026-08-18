// tokenizer.zig — byte-level Byte-Pair-Encoding (BPE), GPT-2 style.
//
// This is the real GPT-2 tokenizer machinery, built from scratch:
//   - base vocab = the 256 bytes (token id == byte value, 0..255)
//   - GPT-2-style pre-tokenization splits raw text into chunks; each chunk is
//     BPE-encoded by repeatedly applying the highest-priority (lowest-rank)
//     learned merge until none remain.
//   - training counts adjacent token pairs across chunks and merges the most
//     frequent, repeating until the vocab reaches the target size.
//   - decode concatenates each token's raw bytes -> exact roundtrip (ANY
//     pre-tokenization gives a perfect roundtrip, because the token byte
//     sequences partition the input bytes).
//
// One deliberate simplification, clearly noted: the GPT-2 pre-tokenization
// regex uses \p{L}/\p{N}/\s (full unicode categories). Zig's std has no unicode
// category tables, so we approximate: ASCII letters/digits/whitespace are exact;
// codepoints >= 0xC0 are treated as letters (covers accented Latin + all
// multilingual scripts), and everything else is "other". This keeps roundtrip
// exact and compression good; only the exact boundary placement on rare
// high-codepoint punctuation (smart quotes, em-dash) differs from OpenAI's
// regex build. Making this byte-exact would require shipping a unicode
// category table — noted as the path to a drop-in tiktoken match.

const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const testing = std.testing;

const BASE_VOCAB: u32 = 256; // token ids 0..255 are the raw bytes

// ---------------------------------------------------------------------------
// codepoint helpers (operate on raw UTF-8 bytes; robust to invalid UTF-8)
// ---------------------------------------------------------------------------

const Cp = struct { cp: u21, len: u8 };

// decode one codepoint at byte offset i; invalid/truncated bytes decode to
// themselves as a single-byte "codepoint" (so byte-level coverage is total).
fn nextCp(text: []const u8, i: usize) ?Cp {
    if (i >= text.len) return null;
    const b = text[i];
    if (b < 0x80) return .{ .cp = b, .len = 1 };
    const seq: u8 = if (b >= 0xF0) 4 else if (b >= 0xE0) 3 else if (b >= 0xC0) 2 else 1;
    if (seq == 1) return .{ .cp = b, .len = 1 }; // stray continuation byte
    if (i + seq > text.len) return .{ .cp = b, .len = 1 }; // truncated
    return .{ .cp = std.unicode.utf8Decode(text[i .. i + seq]) catch return .{ .cp = b, .len = 1 }, .len = seq };
}

// \s : common whitespace. ASCII (tab..cr, space) + NBSP(0xA0) + NEL(0x85) +
// line/para separator (0x2028/0x2029). Good enough for natural text.
fn isSpaceCp(cp: u21) bool {
    return cp == 0x20 or (cp >= 0x09 and cp <= 0x0D) or cp == 0xA0 or cp == 0x85 or cp == 0x2028 or cp == 0x2029;
}

// \p{N} : ASCII digits only (the overwhelmingly common case; superscripts etc.
// are rare and roundtrip is unaffected). Extend if you need full \p{N}.
fn isDigitCp(cp: u21) bool {
    return cp >= 0x30 and cp <= 0x39;
}

// \p{L} : ASCII letters exact; codepoints >= 0xC0 treated as letters (accented
// Latin + all multilingual scripts). Codepoints 0x80..0xBF (C1 controls / stray
// bytes) and 0xA1..0xBF (Latin-1 symbols) fall to "other". See file header note.
fn isLetterCp(cp: u21) bool {
    return (cp >= 0x41 and cp <= 0x5A) or (cp >= 0x61 and cp <= 0x7A) or cp >= 0xC0;
}

fn isOtherCp(cp: u21) bool {
    return !isSpaceCp(cp) and !isLetterCp(cp) and !isDigitCp(cp);
}

// ---------------------------------------------------------------------------
// GPT-2-style pre-tokenization (manual scanner replicating the regex)
//   's|'t|'re|'ve|'m|'ll|'d | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+
//   | \s+(?!\S) | \s+
// Returns slices INTO `text` (no copies of the bytes).
// ---------------------------------------------------------------------------

fn matchContraction(text: []const u8, i: usize) usize {
    if (text[i] != '\'') return 0; // ASCII apostrophe only
    if (i + 2 <= text.len) {
        const c = text[i + 1];
        switch (c) {
            's', 't', 'm', 'd' => return 2,
            else => {},
        }
    }
    if (i + 3 <= text.len) {
        if (text[i + 1] == 'r' and text[i + 2] == 'e') return 3;
        if (text[i + 1] == 'v' and text[i + 2] == 'e') return 3;
        if (text[i + 1] == 'l' and text[i + 2] == 'l') return 3;
    }
    return 0;
}

// ` ?\p{L}+` etc.: optional single leading space, then 1+ of the class.
// Returns match length (0 = no match). `cls` is the class predicate.
fn matchSpaceClass(text: []const u8, i: usize, cls: *const fn (u21) bool) usize {
    var j = i;
    var had_space = false;
    if (j < text.len and text[j] == 0x20) { // optional single ASCII space
        const c = nextCp(text, j + 1);
        if (c) |nc| {
            if (cls(nc.cp)) { j += 1; had_space = true; }
        }
    }
    // require 1+ of the class
    var first = true;
    while (true) {
        const c = nextCp(text, j) orelse break;
        if (!cls(c.cp)) break;
        if (first and had_space and isSpaceCp(c.cp)) break; // space already consumed; class must be non-space
        first = false;
        j += c.len;
    }
    if (first) return 0; // needed at least one class char
    return j - i;
}

// `\s+(?!\S) | \s+` : a whitespace run. If the run is followed by a non-space,
// leave the LAST whitespace char for the next token's leading ` ?`. Returns
// the length to consume as a whitespace token (0 = consume nothing; the
// trailing single space is left for the next chunk).
fn matchWhitespaceRun(text: []const u8, i: usize) usize {
    var j = i;
    while (true) {
        const c = nextCp(text, j) orelse break;
        if (!isSpaceCp(c.cp)) break;
        j += c.len;
    }
    if (j == i) return 0;
    if (j < text.len) {
        // followed by non-space: back up one whitespace codepoint so the last
        // whitespace attaches to the following chunk's leading ` ?`.
        const back = prevCpLen(text, j);
        const run = j - i;
        if (run > back) return run - back;
        return 0; // a single whitespace char followed by non-space -> let next chunk take it
    }
    return j - i; // run at end of text -> consume all
}

// length of the final codepoint ending at byte offset `end`
fn prevCpLen(text: []const u8, end: usize) usize {
    var k: usize = end;
    var steps: usize = 0;
    while (k > 0 and steps < 4) {
        k -= 1;
        steps += 1;
        if ((text[k] & 0xC0) != 0x80) { // not a continuation byte -> start found
            return end - k;
        }
    }
    return end - k;
}

pub fn preTokenize(alloc: Allocator, text: []const u8) ![][]const u8 {
    var chunks: std.ArrayList([]const u8) = .empty;
    errdefer chunks.deinit(alloc);

    var i: usize = 0;
    while (i < text.len) {
        // 1) contractions
        const c = matchContraction(text, i);
        if (c > 0) {
            try chunks.append(alloc, text[i .. i + c]);
            i += c;
            continue;
        }
        // 2) ` ?\p{L}+`
        var m = matchSpaceClass(text, i, &isLetterCp);
        if (m > 0) {
            try chunks.append(alloc, text[i .. i + m]);
            i += m;
            continue;
        }
        // 3) ` ?\p{N}+`
        m = matchSpaceClass(text, i, &isDigitCp);
        if (m > 0) {
            try chunks.append(alloc, text[i .. i + m]);
            i += m;
            continue;
        }
        // 4) ` ?[^\s\p{L}\p{N}]+`
        m = matchSpaceClass(text, i, &isOtherCp);
        if (m > 0) {
            try chunks.append(alloc, text[i .. i + m]);
            i += m;
            continue;
        }
        // 5) whitespace run
        const w = matchWhitespaceRun(text, i);
        if (w > 0) {
            try chunks.append(alloc, text[i .. i + w]);
            i += w;
            continue;
        }
        // safety: force-advance one byte so we never loop forever
        try chunks.append(alloc, text[i .. i + 1]);
        i += 1;
    }
    return chunks.toOwnedSlice(alloc);
}

// ---------------------------------------------------------------------------
// BPE
// ---------------------------------------------------------------------------

pub const BPE = struct {
    alloc: Allocator,
    // token id -> the raw bytes it represents (owned)
    token_bytes: std.ArrayList([]u8),
    // ordered merges; merges[k] = {a, b} produced token id (BASE_VOCAB + k)
    merges: std.ArrayList([2]u32),
    // (a<<32 | b) -> merge rank (lower rank = applied first). lookup for encode.
    pair_rank: std.AutoHashMap(u64, u32),

    pub fn init(alloc: Allocator) BPE {
        return .{
            .alloc = alloc,
            .token_bytes = .empty,
            .merges = .empty,
            .pair_rank = std.AutoHashMap(u64, u32).init(alloc),
        };
    }

    pub fn deinit(self: *BPE) void {
        for (self.token_bytes.items) |b| self.alloc.free(b);
        self.token_bytes.deinit(self.alloc);
        self.merges.deinit(self.alloc);
        self.pair_rank.deinit();
    }

    fn baseInit(self: *BPE) !void {
        // 256 single-byte tokens
        try self.token_bytes.ensureTotalCapacity(self.alloc, BASE_VOCAB);
        var b: u32 = 0;
        while (b < BASE_VOCAB) : (b += 1) {
            const buf = try self.alloc.alloc(u8, 1);
            buf[0] = @intCast(b);
            try self.token_bytes.append(self.alloc, buf);
        }
    }

    fn pairKey(a: u32, b: u32) u64 {
        return (@as(u64, a) << 32) | @as(u64, b);
    }

    // Train on raw text until the vocab reaches `target_vocab_size` (or no pair
    // occurs >= 2 times). Recomputes pair stats each merge (minbpe style) —
    // O(V * N); clear and correct, fine for a corpus slice. The slice size is
    // the knob for training-time vs. coverage.
    pub fn train(self: *BPE, text: []const u8, target_vocab_size: usize) !void {
        try self.baseInit();

        // pre-tokenize once; each chunk becomes a mutable list of token ids
        const chunks = try preTokenize(self.alloc, text);
        defer self.alloc.free(chunks);

        var words: std.ArrayList(std.ArrayList(u32)) = .empty;
        defer {
            for (words.items) |*w| w.deinit(self.alloc);
            words.deinit(self.alloc);
        }
        for (chunks) |ch| {
            var w: std.ArrayList(u32) = .empty;
            for (ch) |b| try w.append(self.alloc, @as(u32, b));
            try words.append(self.alloc, w);
        }

        while (self.token_bytes.items.len < target_vocab_size) {
            // count adjacent pairs across all words
            var counts = std.AutoHashMap(u64, u32).init(self.alloc);
            defer counts.deinit();
            for (words.items) |w| {
                var k: usize = 0;
                while (k + 1 < w.items.len) : (k += 1) {
                    const key = pairKey(w.items[k], w.items[k + 1]);
                    const e = try counts.getOrPut(key);
                    if (!e.found_existing) e.value_ptr.* = 0;
                    e.value_ptr.* += 1;
                }
            }
            // most frequent pair
            var best_key: u64 = 0;
            var best_count: u32 = 0;
            var it = counts.iterator();
            while (it.next()) |e| {
                if (e.value_ptr.* > best_count) {
                    best_count = e.value_ptr.*;
                    best_key = e.key_ptr.*;
                }
            }
            if (best_count < 2) break; // nothing left worth merging

            const a: u32 = @intCast(best_key >> 32);
            const b: u32 = @intCast(best_key & 0xFFFFFFFF);
            const new_id: u32 = @intCast(self.token_bytes.items.len);

            // record merge + new token's bytes
            try self.merges.append(self.alloc, .{ a, b });
            try self.pair_rank.put(best_key, @intCast(self.merges.items.len - 1));
            const ab = try self.alloc.alloc(u8, self.token_bytes.items[a].len + self.token_bytes.items[b].len);
            @memcpy(ab[0..self.token_bytes.items[a].len], self.token_bytes.items[a]);
            @memcpy(ab[self.token_bytes.items[a].len..], self.token_bytes.items[b]);
            try self.token_bytes.append(self.alloc, ab);

            // apply the merge to every word
            for (words.items) |*w| {
                var dst: std.ArrayList(u32) = .empty;
                var k: usize = 0;
                while (k < w.items.len) {
                    if (k + 1 < w.items.len and w.items[k] == a and w.items[k + 1] == b) {
                        try dst.append(self.alloc, new_id);
                        k += 2;
                    } else {
                        try dst.append(self.alloc, w.items[k]);
                        k += 1;
                    }
                }
                w.deinit(self.alloc);
                w.* = dst;
            }
        }
    }

    // BPE-encode a single chunk (raw bytes) into token ids by repeatedly
    // applying the lowest-rank (highest-priority) merge present.
    fn encodeChunk(self: *BPE, bytes: []const u8) ![]u32 {
        var word: std.ArrayList(u32) = .empty;
        for (bytes) |b| try word.append(self.alloc, @as(u32, b));

        while (word.items.len >= 2) {
            var best_rank: ?u32 = null;
            var best_a: u32 = 0;
            var best_b: u32 = 0;
            var k: usize = 0;
            while (k + 1 < word.items.len) : (k += 1) {
                const rank = self.pair_rank.get(pairKey(word.items[k], word.items[k + 1]));
                if (rank != null and (best_rank == null or rank.? < best_rank.?)) {
                    best_rank = rank;
                    best_a = word.items[k];
                    best_b = word.items[k + 1];
                }
            }
            if (best_rank == null) break;

            const new_id: u32 = BASE_VOCAB + best_rank.?;
            var dst: std.ArrayList(u32) = .empty;
            var j: usize = 0;
            while (j < word.items.len) {
                if (j + 1 < word.items.len and word.items[j] == best_a and word.items[j + 1] == best_b) {
                    try dst.append(self.alloc, new_id);
                    j += 2;
                } else {
                    try dst.append(self.alloc, word.items[j]);
                    j += 1;
                }
            }
            word.deinit(self.alloc);
            word = dst;
        }
        return word.toOwnedSlice(self.alloc);
    }

    // Encode raw text into a flat list of token ids.
    pub fn encode(self: *BPE, text: []const u8) ![]u32 {
        const chunks = try preTokenize(self.alloc, text);
        defer self.alloc.free(chunks);
        var ids: std.ArrayList(u32) = .empty;
        errdefer ids.deinit(self.alloc);
        for (chunks) |ch| {
            const cids = try self.encodeChunk(ch);
            defer self.alloc.free(cids);
            try ids.appendSlice(self.alloc, cids);
        }
        return ids.toOwnedSlice(self.alloc);
    }

    // Decode token ids back to the original bytes (exact roundtrip).
    pub fn decode(self: *BPE, ids: []const u32) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(self.alloc);
        for (ids) |id| {
            std.debug.assert(id < self.token_bytes.items.len);
            try out.appendSlice(self.alloc, self.token_bytes.items[id]);
        }
        return out.toOwnedSlice(self.alloc);
    }

    pub fn vocabSize(self: *BPE) usize {
        return self.token_bytes.items.len;
    }

    // ---- persistence: binary format ----
    //   u32 num_merges | for each merge: u32 a, u32 b
    //   u32 merged_token_count | for each merged token (id 256..): u32 byte_len, bytes
    // The base 256 single-byte tokens are implied (not re-saved); the file
    // stores the merge list and the merged tokens' raw bytes (ids 256..).
    //
    // File IO in Zig 0.16 goes through the new `std.Io` interface: every Dir
    // /File op takes an `io: std.Io` handle. We use `std.Options.debug_io`, the
    // process-default threaded Io singleton (the same one std.debug.print uses),
    // so callers don't have to thread an Io through every tokenizer call.

    fn putU32(buf: *std.ArrayList(u8), alloc: Allocator, v: u32) !void {
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, v, .little);
        try buf.appendSlice(alloc, &b);
    }

    pub fn save(self: *BPE, path: []const u8) !void {
        const io = std.Options.debug_io;
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(self.alloc);
        try putU32(&buf, self.alloc, @intCast(self.merges.items.len));
        for (self.merges.items) |m| {
            try putU32(&buf, self.alloc, m[0]);
            try putU32(&buf, self.alloc, m[1]);
        }
        const merged_count = self.token_bytes.items.len - BASE_VOCAB;
        try putU32(&buf, self.alloc, @intCast(merged_count));
        var i: usize = BASE_VOCAB;
        while (i < self.token_bytes.items.len) : (i += 1) {
            const tb = self.token_bytes.items[i];
            try putU32(&buf, self.alloc, @intCast(tb.len));
            try buf.appendSlice(self.alloc, tb);
        }
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = buf.items });
    }

    pub fn load(self: *BPE, path: []const u8) !void {
        const io = std.Options.debug_io;
        try self.baseInit();
        const data = try std.Io.Dir.cwd().readFileAlloc(io, path, self.alloc, .limited(1 << 28));
        defer self.alloc.free(data);
        var off: usize = 0;
        const num_merges = std.mem.readInt(u32, data[off..][0..4], .little);
        off += 4;
        var mi: u32 = 0;
        while (mi < num_merges) : (mi += 1) {
            const a = std.mem.readInt(u32, data[off..][0..4], .little);
            off += 4;
            const b = std.mem.readInt(u32, data[off..][0..4], .little);
            off += 4;
            try self.merges.append(self.alloc, .{ a, b });
            try self.pair_rank.put(pairKey(a, b), mi);
        }
        const merged_count = std.mem.readInt(u32, data[off..][0..4], .little);
        off += 4;
        var ti: u32 = 0;
        while (ti < merged_count) : (ti += 1) {
            const blen = std.mem.readInt(u32, data[off..][0..4], .little);
            off += 4;
            const tb = try self.alloc.alloc(u8, blen);
            @memcpy(tb, data[off .. off + blen]);
            off += blen;
            try self.token_bytes.append(self.alloc, tb);
        }
    }
};

// ===========================================================================
// tests
// ===========================================================================

test "preTokenize: splits words, keeps leading space, handles punctuation" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const chunks = try preTokenize(a, "Hello, world! 123");
    // "Hello" , "," , " world" , "!" , " 123"  (leading space attaches)
    var s = std.ArrayList(u8).empty;
    for (chunks) |c| {
        try s.append(a, '|');
        try s.appendSlice(a, c);
    }
    try testing.expectEqualStrings("|Hello|,| world|!| 123", s.items);
}

test "BPE: roundtrip on plain ASCII" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const corpus = "the quick brown fox jumps over the lazy dog. " ** 20;
    var tk = BPE.init(a);
    defer tk.deinit();
    try tk.train(corpus, 320);

    const text = "the quick brown fox";
    const ids = try tk.encode(text);
    defer a.free(ids);
    const out = try tk.decode(ids);
    defer a.free(out);
    try testing.expectEqualStrings(text, out);
}

test "BPE: roundtrip on unicode + punctuation" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const corpus = "Café résumé naïve — don't you think? " ** 30;
    var tk = BPE.init(a);
    defer tk.deinit();
    try tk.train(corpus, 400);

    const text = "Café — don't!";
    const ids = try tk.encode(text);
    defer a.free(ids);
    const out = try tk.decode(ids);
    defer a.free(out);
    try testing.expectEqualStrings(text, out);
}

test "BPE: learns the obvious merge on a repetitive corpus" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // "ab" repeated many times -> the pair (a,b)=('a','b') must be the first merge.
    const corpus = "ab" ** 200;
    var tk = BPE.init(a);
    defer tk.deinit();
    try tk.train(corpus, 260);
    try testing.expect(tk.merges.items.len >= 1);
    try testing.expectEqual(@as(u32, 'a'), tk.merges.items[0][0]);
    try testing.expectEqual(@as(u32, 'b'), tk.merges.items[0][1]);
    // encoding "ab" should collapse to a single token
    const ids = try tk.encode("ab");
    defer a.free(ids);
    try testing.expectEqual(@as(usize, 1), ids.len);
    try testing.expectEqual(@as(u32, 256), ids[0]); // first merge -> id 256
}

test "BPE: training is deterministic (same corpus -> same merges)" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const corpus = "the cat sat on the mat. the dog sat on the log. " ** 50;
    var t1 = BPE.init(a);
    defer t1.deinit();
    try t1.train(corpus, 350);
    var t2 = BPE.init(a);
    defer t2.deinit();
    try t2.train(corpus, 350);

    try testing.expectEqual(t1.merges.items.len, t2.merges.items.len);
    for (t1.merges.items, t2.merges.items) |m1, m2| {
        try testing.expectEqual(m1[0], m2[0]);
        try testing.expectEqual(m1[1], m2[1]);
    }
}

test "BPE: save/load preserves merges and roundtrip" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const corpus = "low lower lowest low lower lowest " ** 40;
    var tk = BPE.init(a);
    defer tk.deinit();
    try tk.train(corpus, 330);

    const path = "znn_test_tok.bin";
    try tk.save(path);
    defer std.Io.Dir.cwd().deleteFile(std.Options.debug_io, path) catch {};

    var tk2 = BPE.init(a);
    defer tk2.deinit();
    try tk2.load(path);

    try testing.expectEqual(tk.merges.items.len, tk2.merges.items.len);
    const text = "lowest lower";
    const ids1 = try tk.encode(text);
    defer a.free(ids1);
    const ids2 = try tk2.encode(text);
    defer a.free(ids2);
    try testing.expectEqualSlices(u32, ids1, ids2);
    const out = try tk2.decode(ids2);
    defer a.free(out);
    try testing.expectEqualStrings(text, out);
}