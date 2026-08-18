// bench — CPU vs GPU matmul throughput + correctness (Gate 11).
//   zig build bench            # CPU-only baseline
//   zig build bench -Dgpu       # CPU vs cuBLAS + gated-delta kernel checks
//
// The GPU path links cuBLAS (for matmul) and the nvcc-compiled CUDA kernels
// (gated-delta recurrent forward) via src/znn/backend_cuda.zig. Each GPU op
// is checked against the verified CPU reference (max|diff| < 1e-4 matmul,
// < 1e-3 gated-delta) and timed for throughput.
const std = @import("std");
const znn = @import("znn");
const opts = @import("znn_opts");
const GPU = opts.gpu;

const io = std.Options.debug_io;

fn nowNs() i128 {
    return std.Io.Clock.now(.awake, io).nanoseconds;
}
fn msBetween(t0: i128, t1: i128) f64 {
    return @as(f64, @floatFromInt(t1 - t0)) / 1e6;
}

pub fn main() void {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // ---------------- CPU matmul baseline ----------------
    const M: usize = 512;
    const K: usize = 512;
    const N: usize = 512;
    var prng = std.Random.DefaultPrng.init(7);
    const A = znn.tensor.rand(a, &.{ M, K }, &prng) catch return;
    const B = znn.tensor.rand(a, &.{ K, N }, &prng) catch return;

    // warm + time CPU matmul (a few repeats for a stable number)
    var C: znn.tensor.Tensor = undefined;
    const tc0 = nowNs();
    const reps: usize = 3;
    for (0..reps) |_| C = znn.tensor.matmul(a, A, B) catch return;
    const tc1 = nowNs();
    const flops = 2.0 * @as(f64, @floatFromInt(M)) * @as(f64, @floatFromInt(K)) * @as(f64, @floatFromInt(N));
    const cpu_ms = msBetween(tc0, tc1) / @as(f64, @floatFromInt(reps));
    std.debug.print("[CPU] matmul {d}x{d}x{d}: {d:.2} ms  ({d:.1} MFLOP/s)\n", .{ M, K, N, cpu_ms, flops / (cpu_ms * 1e3) });
    std.debug.print("[CPU] C[0,0]={d:.4}  C[1,1]={d:.4}\n", .{ C.data[0], C.data[(N + 1)] });

    if (!GPU) {
        std.debug.print("\n(build without -Dgpu: GPU path skipped)\n", .{});
        return;
    }

    // ---------------- GPU path ----------------
    const cuda = @import("backend_cuda");
    var dev = cuda.CudaBackend.init() catch {
        std.debug.print("[GPU] CudaBackend init failed\n", .{});
        return;
    };
    defer dev.deinit();

    // ---- cuBLAS matmul correctness + throughput ----
    const dA = dev.alloc(M * K * 4) catch return;
    const dB = dev.alloc(K * N * 4) catch return;
    const dC = dev.alloc(M * N * 4) catch return;
    dev.h2d(dA, A.data);
    dev.h2d(dB, B.data);
    // warm
    dev.matmul(dA, dB, dC, M, N, K) catch return;
    dev.sync();
    const tg0 = nowNs();
    const greps: usize = 10;
    for (0..greps) |_| dev.matmul(dA, dB, dC, M, N, K) catch return;
    dev.sync();
    const tg1 = nowNs();
    const gpu_ms = msBetween(tg0, tg1) / @as(f64, @floatFromInt(greps));
    const Cg = a.alloc(f32, M * N) catch return;
    dev.d2h(Cg, dC);

    var max_diff: f32 = 0;
    for (0..M * N) |i| max_diff = @max(max_diff, @abs(C.data[i] - Cg[i]));
    std.debug.print("[GPU] cuBLAS matmul {d}x{d}x{d}: {d:.3} ms  ({d:.1} MFLOP/s)  max|diff|vs CPU = {d:.6}\n", .{ M, K, N, gpu_ms, flops / (gpu_ms * 1e3), max_diff });
    std.debug.print("      speedup = {d:.1}x\n", .{cpu_ms / gpu_ms});
    if (max_diff > 1e-3) std.debug.print("      !! matmul mismatch > 1e-3\n", .{});

    // ---- gated-delta recurrent kernel vs CPU recurrent ----
    const Bh: usize = 4;
    const L: usize = 16;
    const Dh: usize = 8;
    const Q = znn.tensor.rand(a, &.{ Bh, L, Dh }, &prng) catch return;
    const Kk = znn.tensor.rand(a, &.{ Bh, L, Dh }, &prng) catch return;
    const V = znn.tensor.rand(a, &.{ Bh, L, Dh }, &prng) catch return;
    const alphat = znn.tensor.mulScalar(a, znn.tensor.rand(a, &.{ Bh, L }, &prng) catch return, 0.15) catch return;
    const alpha = znn.tensor.add(a, znn.tensor.full(a, &.{ Bh, L }, 0.8) catch return, alphat) catch return;
    const betat = znn.tensor.mulScalar(a, znn.tensor.rand(a, &.{ Bh, L }, &prng) catch return, 0.4) catch return;
    const beta = znn.tensor.add(a, znn.tensor.full(a, &.{ Bh, L }, 0.4) catch return, betat) catch return;

    // CPU recurrent reference (autograd op's forward)
    const Qv = znn.Var.parameter(a, Q) catch return;
    const Kv = znn.Var.parameter(a, Kk) catch return;
    const Vv = znn.Var.parameter(a, V) catch return;
    const Av = znn.Var.parameter(a, alpha) catch return;
    const Bv = znn.Var.parameter(a, beta) catch return;
    const ref = znn.autograd.vGatedDelta(a, Qv, Kv, Vv, Av, Bv) catch return;

    const dq = dev.alloc(Bh * L * Dh * 4) catch return;
    const dk = dev.alloc(Bh * L * Dh * 4) catch return;
    const dv = dev.alloc(Bh * L * Dh * 4) catch return;
    const da = dev.alloc(Bh * L * 4) catch return;
    const db = dev.alloc(Bh * L * 4) catch return;
    const dout = dev.alloc(Bh * L * Dh * 4) catch return;
    dev.h2d(dq, Q.data);
    dev.h2d(dk, Kk.data);
    dev.h2d(dv, V.data);
    dev.h2d(da, alpha.data);
    dev.h2d(db, beta.data);
    dev.gatedDeltaFwd(dq, dk, dv, da, db, dout, Bh, L, Dh) catch return;
    dev.sync();
    const gout = a.alloc(f32, Bh * L * Dh) catch return;
    dev.d2h(gout, dout);

    var gd_diff: f32 = 0;
    for (0..Bh * L * Dh) |i| gd_diff = @max(gd_diff, @abs(ref.value.data[i] - gout[i]));
    std.debug.print("[GPU] gated-delta kernel vs CPU recurrent: max|diff| = {d:.6} (Bh={d} L={d} Dh={d})\n", .{ gd_diff, Bh, L, Dh });
    if (gd_diff > 1e-3) std.debug.print("      !! gated-delta kernel mismatch > 1e-3\n", .{});

    dev.free(dA);
    dev.free(dB);
    dev.free(dC);
    dev.free(dq);
    dev.free(dk);
    dev.free(dv);
    dev.free(da);
    dev.free(db);
    dev.free(dout);
}