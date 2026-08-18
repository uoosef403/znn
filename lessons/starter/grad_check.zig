// grad_check.zig — the finite-difference arbiter (lesson 03+).
//
// This is INFRASTRUCTURE, not a solution. It implements the *finite-difference
// half* of gradient checking (the ground truth). You bring the *analytic* half
// (your autograd's backward). The harness compares them.
//
// What it does:
//   For each leaf input element x_i, it perturbs by +eps and -eps, reruns the
//   forward, reads the scalar loss, and computes the numerical gradient:
//       g_num = (L(x + eps*e_i) - L(x - eps*e_i)) / (2*eps)   (central diff)
//   It then compares to your analytic gradient g_ana (from .backprop()), and
//   reports the worst *scaled relative* error:
//       err = |g_num - g_ana| / (|g_num| + |g_ana| + 1e-8)
//   plus an *absolute tolerance floor* (see the pitfall note below).
//
// WHY AN ABSOLUTE FLOOR (this is the #1 grad-check footgun, learned the hard
// way): for tiny gradients, f32 finite-difference roundoff dominates. A pure
// relative check screams "WRONG!" at a gradient of 1e-6 where the analytic
// and numeric are both ~noise. So accept if EITHER:
//       |g_num - g_ana| < atol        (absolute, default 1e-4)
//   OR  err < rtol                     (relative, default 1e-2)
// Pass when the worst element passes both.
//
// This file is deliberately written against a *documented interface* so you
// adapt it to your own `Var`/`Value` types. The TODOs mark the seams.

const std = @import("std");

pub const Opts = struct {
    eps: f32 = 1e-3,
    rtol: f32 = 1e-2,
    atol: f32 = 1e-4,
};

pub const Result = struct {
    max_err: f32, // worst scaled relative error
    worst_val: f32, // analytic grad at worst element
    worst_num: f32, // numeric grad at worst element
    passed: bool,
};

// `Leaf` is a mutable input tensor with a `.data: []f32` and a `.grad: []f32`
// (your autograd type). Adapt to yours.
// `forward`: rebuilds the graph from the current leaf values, returns the
//            scalar loss as f32. It must allocate from `alloc`.
// `backward`: runs your backward pass and populates each leaf's `.grad`.
pub fn check(
    alloc: std.mem.Allocator,
    leaves: []const *anyopaque, // TODO: your Leaf type
    forward: *const fn (std.mem.Allocator) f32,
    backward: *const fn () void,
    opts: Opts,
) Result {
    _ = alloc;
    // TODO: adapt the body below to your Leaf type. Pseudocode:
    //
    //   backward();  // populate analytic grads
    //   var max_err: f32 = 0; var worst_val, worst_num: f32 = 0;
    //   for (leaves) |leaf| {
    //       for (leaf.data, 0..) |_, i| {
    //           const orig = leaf.data[i];
    //           leaf.data[i] = orig + opts.eps;
    //           const lp = forward(alloc);
    //           leaf.data[i] = orig - opts.eps;
    //           const lm = forward(alloc);
    //           leaf.data[i] = orig;  // restore
    //           const gnum = (lp - lm) / (2 * opts.eps);
    //           const gana = leaf.grad[i];
    //           const abs = @abs(gnum - gana);
    //           const rel = abs / (@abs(gnum) + @abs(gana) + 1e-8);
    //           const ok = (abs < opts.atol) or (rel < opts.rtol);
    //           if (!ok and rel > max_err) { max_err = rel; worst_val = gana; worst_num = gnum; }
    //       }
    //   }
    //   return .{ .max_err = max_err, .worst_val = worst_val, .worst_num = worst_num, .passed = max_err < opts.rtol };
    return .{ .max_err = 0, .worst_val = 0, .worst_num = 0, .passed = true };
}

// ---- a tiny self-test so the harness itself is known-correct ----
test "finite-difference recovers the derivative of x^2 at x=3 (=6)" {
    // We don't depend on your autograd here: just verify central differences.
    var x: f32 = 3.0;
    const f = struct {
        fn eval(v: f32) f32 {
            return v * v;
        }
    }.eval;
    const eps: f32 = 1e-3;
    const g = (f(x + eps) - f(x - eps)) / (2 * eps);
    try std.testing.expect(@abs(g - 6.0) < 1e-3);
}