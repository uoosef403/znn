// 05-bpe — milestone: train a byte-level BPE tokenizer on a text slice and
// demonstrate encode/decode + merge discovery.
// Run with:  zig build run-05-bpe
//
// For a real FineWeb tokenizer, see Gate 5 (fetch_fineweb.py) — there we train
// on a slice of the corpus and persist tokenizer.bin for the LM. Here we use
// a self-contained prose slice so the app is runnable standalone.
const std = @import("std");
const znn = @import("znn");
const BPE = znn.tokenizer.BPE;

const CORPUS =
    \\The quick brown fox jumps over the lazy dog.
    \\The quick brown fox runs past the sleeping dog.
    \\A quick brown dog is not a lazy dog.
    \\Dogs and foxes run quickly over the brown hill.
    \\The lazy dog sleeps while the quick fox jumps.
    \\Quick brown foxes and lazy dogs live on the hill.
    \\The brown hill is home to quick foxes and lazy dogs.
    \\Over the hill, the quick brown fox jumps the lazy dog again.
    \\Dogs sleep, foxes jump, and the brown hill watches them all.
    \\The quick and the lazy share the brown hill day after day.
;

// render a token's raw bytes into a readable string (escape control/space/punct
// bytes as <hex>); allocated from `alloc` so the caller can print it safely.
fn showBytes(alloc: std.mem.Allocator, b: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    for (b) |c| {
        if (c >= 0x20 and c < 0x7F and c != '|') {
            try out.append(alloc, c);
        } else {
            try out.appendSlice(alloc, "<");
            var hex: [3]u8 = undefined;
            _ = try std.fmt.bufPrint(&hex, "{x}", .{c});
            try out.appendSlice(alloc, &hex);
            try out.append(alloc, '>');
        }
    }
    return out.toOwnedSlice(alloc);
}

pub fn main() void {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // repeat the prose to give BPE enough signal to find multi-byte merges
    var buf: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < 300) : (i += 1) {
        buf.appendSlice(a, CORPUS) catch return;
        buf.append(a, '\n') catch return;
    }
    const corpus = buf.items;

    var tk = BPE.init(a);
    defer tk.deinit();
    tk.train(corpus, 400) catch {
        std.debug.print("train failed\n", .{});
        return;
    };
    std.debug.print("trained vocab size: {d}  (merges: {d})\n\n", .{ tk.vocabSize(), tk.merges.items.len });

    // show the first ~20 learned merges
    const show = @min(tk.merges.items.len, 20);
    std.debug.print("first {d} merges (id -> bytes):\n", .{show});
    for (0..show) |k| {
        const m = tk.merges.items[k];
        const id: u32 = 256 + @as(u32, @intCast(k));
        const tb = tk.token_bytes.items[id];
        const s = showBytes(a, tb) catch return;
        std.debug.print("  [{d}] {d}+{d} -> '{s}'\n", .{ id, m[0], m[1], s });
    }
    std.debug.print("\n", .{});

    // encode + decode roundtrip demo
    const text = "the quick brown fox jumps";
    const ids = tk.encode(text) catch return;
    defer a.free(ids);
    std.debug.print("text:   '{s}'\n", .{text});
    std.debug.print("tokens ({d}):", .{ids.len});
    for (ids) |id| std.debug.print(" {d}", .{id});
    std.debug.print("\n", .{});
    var pretty: std.ArrayList(u8) = .empty;
    for (ids) |id| {
        pretty.appendSlice(a, "|") catch return;
        const s = showBytes(a, tk.token_bytes.items[id]) catch return;
        pretty.appendSlice(a, s) catch return;
    }
    std.debug.print("segmented: {s}\n", .{pretty.items});

    const out = tk.decode(ids) catch return;
    defer a.free(out);
    std.debug.print("roundtrip: '{s}'\n", .{out});
    std.debug.print("roundtrip ok: {}\n", .{std.mem.eql(u8, text, out)});
}