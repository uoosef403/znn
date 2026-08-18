// loss.zig — loss functions built from autograd primitives (no new backwards).
//
// MSE = mean( (pred - target)^2 )               — uses vSub, vMul, vSumAll, scale.
// Cross-entropy for classification is `vSoftmaxCE` from autograd.zig (fused).

const std = @import("std");
const mem = std.mem;
const ag = @import("autograd.zig");
const Var = ag.Var;

pub const Error = ag.Error;

// mean squared error, averaged over the batch (first dim)
pub fn mse(alloc: mem.Allocator, pred: *Var, target: *Var) Error!*Var {
    const d = try ag.vSub(alloc, pred, target); // [B, ...]
    const sq = try ag.vMul(alloc, d, d);
    const s = try ag.vSumAll(alloc, sq); // scalar
    const n: f32 = @floatFromInt(pred.value.numel());
    return ag.vMulScalar(alloc, s, 1.0 / n);
}

// Re-export the fused softmax cross-entropy for convenience.
pub const softmaxCE = ag.vSoftmaxCE;