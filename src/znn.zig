// znn — Zig Neural Network, from scratch.
// Root module: re-exports the public surface so consumers do `@import("znn")`.
//
// Build order mirrors the curriculum: arrays (tensor) -> graph (autograd) ->
// losses/optim -> data/tokenizer -> modern layers -> attention/delta -> GPU.

pub const tensor = @import("znn/tensor.zig");
pub const Tensor = tensor.Tensor;
pub const autograd = @import("znn/autograd.zig");
pub const Var = autograd.Var;
pub const Value = autograd.Value;
pub const nn = @import("znn/nn.zig");
pub const optim = @import("znn/optim.zig");
pub const loss = @import("znn/loss.zig");
pub const tokenizer = @import("znn/tokenizer.zig");
pub const BPE = tokenizer.BPE;
pub const data = @import("znn/data.zig");
pub const checkpoint = @import("znn/checkpoint.zig");

// re-export the free-function tensor ops for ergonomic call sites: znn.matmul(...)
pub const zeros = tensor.zeros;
pub const ones = tensor.ones;
pub const full = tensor.full;
pub const arange = tensor.arange;
pub const fromSlice = tensor.fromSlice;
pub const rand = tensor.rand;
pub const emptyLike = tensor.emptyLike;
pub const reshape = tensor.reshape;
pub const transpose2D = tensor.transpose2D;
pub const permute = tensor.permute;
pub const add = tensor.add;
pub const sub = tensor.sub;
pub const mul = tensor.mul;
pub const divide = tensor.divide;
pub const addScalar = tensor.addScalar;
pub const mulScalar = tensor.mulScalar;
pub const relu = tensor.relu;
pub const sigmoid = tensor.sigmoid;
pub const tanh = tensor.tanh;
pub const gelu = tensor.gelu;
pub const exp = tensor.exp;
pub const neg = tensor.neg;
pub const sumAll = tensor.sumAll;
pub const sumAxis = tensor.sumAxis;
pub const meanAll = tensor.meanAll;
pub const matmul = tensor.matmul;
pub const softmax = tensor.softmax;
pub const approxEq = tensor.approxEq;
pub const prodSlice = tensor.prodSlice;
pub const broadcastShapes = tensor.broadcastShapes;