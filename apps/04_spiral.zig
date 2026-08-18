// 04_spiral — milestone: train an MLP on the classic 2D 3-class spiral.
// Watch a 2-layer network go from random to >95% accuracy.
// Run with:  zig build run-04-spiral
const std = @import("std");
const znn = @import("znn");
const ten = znn.tensor;
const ag = znn.autograd;
const nn = @import("znn").nn;
const optim = @import("znn").optim;
const loss = @import("znn").loss;

const N: usize = 100; // points per class
const C: usize = 3;

// build the spiral dataset: 3 classes, 2 features
fn makeSpiral(alloc: std.mem.Allocator, prng: *std.Random.DefaultPrng) !struct { x: ten.Tensor, y: []usize } {
    const total = N * C;
    const x = try ten.zeros(alloc, &.{ total, 2 });
    const y = try alloc.alloc(usize, total);
    for (0..C) |cls| {
        for (0..N) |j| {
            const i = cls * N + j;
            // radius floor so the innermost point of each class is distinct,
            // not all collapsed at the origin with different labels.
            const t: f32 = (@as(f32, @floatFromInt(j)) + 0.5) / @as(f32, @floatFromInt(N));
            const angle: f32 = t * 2.0 * std.math.pi * 3.0 + @as(f32, @floatFromInt(cls)) * (2.0 * std.math.pi / 3.0);
            const noise = prng.random().floatNorm(f32) * 0.05;
            x.data[i * 2 + 0] = t * @cos(angle) + noise;
            x.data[i * 2 + 1] = t * @sin(angle) + noise;
            y[i] = cls;
        }
    }
    return .{ .x = x, .y = y };
}

pub fn main() void {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0);

    const ds = makeSpiral(a, &prng) catch return;
    const xv = ag.Var.constant(a, ds.x) catch return;

    // MLP: 2 -> 64 (relu) -> 64 (relu) -> 3
    // Two hidden layers: the 3-turn spiral collapses toward the origin, so the
    // inner arms are points ~0.005 apart in input space — too fine for a single
    // hidden layer to resolve with O(1) weights. A second hidden layer gives the
    // depth to compose sharp inner boundaries (the canonical spiral architecture).
    var mlp = nn.MLP.init(a, &.{ 2, 64, 64, 3 }, &.{ .relu, .relu, .none }, &prng) catch return;
    const ps = mlp.params(a) catch return;

    const total_steps: usize = 6000;
    const lr0: f32 = 0.05; // peak lr (stable for this 2-hidden stack; no warmup needed here)
    const lr_min: f32 = 0.001; // floor so Adam keeps making *tiny* steps at the end
    var opt = optim.Adam.init(a, lr0, 0.0);
    defer opt.deinit();

    var step: usize = 0;
    while (step < total_steps) : (step += 1) {
        // cosine lr decay: shrink the step size over training so Adam can settle
        // into a tight minimum instead of oscillating around it forever (the
        // failure mode of constant-lr Adam). Full scheduler machinery arrives in
        // Gate 7; this is the one-line version that makes the demo converge.
        const p: f32 = @as(f32, @floatFromInt(step)) / @as(f32, @floatFromInt(total_steps));
        opt.lr = lr_min + 0.5 * (lr0 - lr_min) * (1.0 + @cos(std.math.pi * p));

        // fresh graph each step (arena holds old nodes; we just stop referencing them)
        const logits = mlp.forward(xv) catch return;
        const l = loss.softmaxCE(a, logits, ds.y) catch return;
        l.backprop();
        opt.step(ps);
        if (step % 500 == 0 or step == total_steps - 1) {
            // accuracy
            var correct: usize = 0;
            for (0..ds.y.len) |i| {
                var best: usize = 0;
                var bestv: f32 = logits.value.data[i * 3 + 0];
                for (1..3) |k| {
                    if (logits.value.data[i * 3 + k] > bestv) {
                        bestv = logits.value.data[i * 3 + k];
                        best = k;
                    }
                }
                if (best == ds.y[i]) correct += 1;
            }
            const acc: f32 = @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(ds.y.len));
            std.debug.print("step {d:>3}  loss={d:.4}  acc={d:.3}\n", .{ step, l.value.data[0], acc });
        }
    }
}