// checkpoint.zig — serialize / restore a model's parameters (Gate 12).
//
// A checkpoint is a flat dump of every parameter tensor in the order returned
// by `model.params()`, plus its shape so the loader can verify a structural
// match before copying. The format is intentionally trivial:
//
//   b"ZNN1"                      4-byte magic
//   u32  param_count             little-endian
//   per parameter:
//     u8   ndim
//     u64  shape[ndim]            little-endian
//     f32  data[numel]            native-endian (= little-endian on x86)
//
// Save and load use the SAME model config, so `params()` returns the parameters
// in the same order on both sides — there is no per-parameter name table. The
// f32 data is written in native byte order, so a checkpoint is portable across
// same-endianness machines but not a cross-arch interchange format (good enough
// for training/resume on one cluster; for that, add explicit endian handling).
//
// What this deliberately does NOT store: optimizer state (Adam moments, Muon
// buffers), the RNG state, and the data-stream position. Resuming training
// therefore restarts the optimizer moments (acceptable for a from-scratch
// framework; a production trainer would persist them too). The model weights —
// the only thing that is expensive to recompute — are fully restored.

const std = @import("std");
const Var = @import("autograd.zig").Var;

pub const MAGIC = "ZNN1";
pub const Error = error{ BadCheckpoint, ShapeMismatch, ParamCountMismatch, EndOfBuffer };

pub fn save(alloc: std.mem.Allocator, params: []const *Var, path: []const u8) !void {
    const io = std.Options.debug_io;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(alloc);
    try buf.appendSlice(alloc, MAGIC);
    var hdr: [4]u8 = undefined;
    std.mem.writeInt(u32, &hdr, @intCast(params.len), .little);
    try buf.appendSlice(alloc, &hdr);
    var dim: [8]u8 = undefined;
    for (params) |p| {
        const sh = p.value.shape;
        try buf.append(alloc, @intCast(sh.len));
        for (sh) |d| {
            std.mem.writeInt(u64, &dim, d, .little);
            try buf.appendSlice(alloc, &dim);
        }
        const n = p.value.numel();
        const bytes = @as([*]const u8, @ptrCast(p.value.data.ptr))[0 .. n * @sizeOf(f32)];
        try buf.appendSlice(alloc, bytes);
    }
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = buf.items });
}

pub fn load(alloc: std.mem.Allocator, params: []const *Var, path: []const u8) Error!void {
    _ = alloc;
    const io = std.Options.debug_io;
    const file = std.Io.Dir.cwd().readFileAlloc(io, path, std.heap.smp_allocator, .limited(1 << 30)) catch return Error.BadCheckpoint;
    defer std.heap.smp_allocator.free(file);
    if (file.len < MAGIC.len + 4 or !std.mem.eql(u8, file[0..MAGIC.len], MAGIC)) return Error.BadCheckpoint;
    var off: usize = MAGIC.len;
    const count = std.mem.readInt(u32, file[off..][0..4], .little);
    off += 4;
    if (count != params.len) return Error.ParamCountMismatch;
    for (params) |p| {
        if (off >= file.len) return Error.EndOfBuffer;
        const ndim = file[off];
        off += 1;
        const sh = p.value.shape;
        if (ndim != sh.len) return Error.ShapeMismatch;
        for (sh) |d| {
            if (off + 8 > file.len) return Error.EndOfBuffer;
            const ck = std.mem.readInt(u64, file[off..][0..8], .little);
            off += 8;
            if (ck != d) return Error.ShapeMismatch;
        }
        const n = p.value.numel();
        const nb = n * @sizeOf(f32);
        if (off + nb > file.len) return Error.EndOfBuffer;
        @memcpy(@as([*]u8, @ptrCast(p.value.data.ptr))[0..nb], file[off .. off + nb]);
        off += nb;
    }
}

// ---------------------------------------------------------------------------
// Test: a checkpoint round-trip reproduces the model's forward output exactly.
// Train model A one step (so its weights leave the init distribution), save,
// load into a freshly-initialized model B (different weights), and assert the
// two produce identical logits on the same input. This is the property the
// capstone's resume/sample modes rely on.
// ---------------------------------------------------------------------------
const testing = std.testing;
const nn = @import("nn.zig");
const optim = @import("optim.zig");
const loss = @import("loss.zig");

test "Gate 12: checkpoint round-trip reproduces forward exactly" {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const V: usize = 24;
    const L: usize = 8;
    const C: usize = 32;
    const layers: usize = 1;
    const H: usize = 4;
    const Hkv: usize = 2;
    const ffn: usize = 64;

    // Model A: train one step so weights are non-trivial.
    var prng_a = std.Random.DefaultPrng.init(11);
    var lm_a = try nn.GqaTransformerLM.init(a, V, L, C, layers, H, Hkv, ffn, &prng_a);
    const ps_a = try lm_a.params(a);

    // Model B: different seed -> different init weights (must be overwritten).
    var prng_b = std.Random.DefaultPrng.init(222);
    var lm_b = try nn.GqaTransformerLM.init(a, V, L, C, layers, H, Hkv, ffn, &prng_b);
    const ps_b = try lm_b.params(a);

    // one training step on A
    var iprng = std.Random.DefaultPrng.init(5);
    const ids = try a.alloc(usize, L * 2); // batch 2
    for (ids) |*id| id.* = iprng.random().uintLessThan(usize, V);
    const tgt = try a.alloc(usize, L * 2);
    for (0..L * 2) |i| tgt[i] = iprng.random().uintLessThan(usize, V);
    var opt = optim.Adam.init(a, 0.01, 0.0);
    defer opt.deinit();
    const logits_a = try lm_a.forward(ids);
    const l = try loss.softmaxCE(a, logits_a, tgt);
    l.backprop();
    opt.step(ps_a);

    // forward A again (post-update) -> reference output
    const ref = try lm_a.forward(ids);

    // save A, load into B, forward B -> must match ref
    try save(a, ps_a, "data/_ckpt_roundtrip_test.bin");
    try load(a, ps_b, "data/_ckpt_roundtrip_test.bin");
    const out_b = try lm_b.forward(ids);

    // B's weights before load differ from A; confirm load actually changed B
    // (sanity: param 0 differs from A's pre-save only if load wrote it). Just
    // compare outputs.
    var max_diff: f32 = 0;
    for (0..ref.value.numel()) |i| {
        max_diff = @max(max_diff, @abs(ref.value.data[i] - out_b.value.data[i]));
    }
    std.debug.print("  ckpt round-trip max|diff| = {d:.6}\n", .{max_diff});
    try testing.expect(max_diff < 1e-5);
}