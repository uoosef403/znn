// 02_tensor_lab — milestone app: exercise the tensor core.
// Run with:  zig build run-02-tensor-lab
const std = @import("std");
const znn = @import("znn");

fn show(name: []const u8, t: znn.Tensor) void {
    std.debug.print("{s} shape={any} =", .{ name, t.shape });
    const n = @min(t.numel(), 24);
    for (0..n) |i| {
        std.debug.print(" {d:.2}", .{t.atFlat(i)});
    }
    if (t.numel() > n) std.debug.print(" ...", .{});
    std.debug.print("\n", .{});
}

pub fn main() void {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const A = znn.fromSlice(a, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 }) catch return;
    const B = znn.fromSlice(a, &.{ 3, 2 }, &.{ 7, 8, 9, 10, 11, 12 }) catch return;
    const C = znn.matmul(a, A, B) catch return;
    show("A@B", C);

    const col = znn.fromSlice(a, &.{ 2, 1 }, &.{ 0, 10 }) catch return;
    const row = znn.fromSlice(a, &.{ 1, 3 }, &.{ 1, 2, 3 }) catch return;
    const sum = znn.add(a, col, row) catch return;
    show("col+row (broadcast)", sum);

    const x = znn.fromSlice(a, &.{ 2, 3 }, &.{ 1000, 0, 0, 0, 0, 0 }) catch return;
    const p = znn.softmax(a, x) catch return;
    show("softmax", p);
}