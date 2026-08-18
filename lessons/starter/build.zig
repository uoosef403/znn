// build.zig — starter template for your from-scratch znn (lessons/).
//
// Copy this into a fresh folder (e.g. ~/myznn/build.zig) next to a `src/`
// you will create lesson by lesson. It gives you:
//   - a `znn` module rooted at src/znn.zig (you create that file in lesson 02)
//   - a `test` step that runs every test in src/znn/*.zig
//   - a `run-NN` step per app in apps/ (you add apps as you go)
//
// This is scaffolding only — it contains NO neural-network logic. The whole
// point of the curriculum is that *you* write that, lesson by lesson.

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Your library module. Create src/znn.zig and re-export your modules.
    const znn_mod = b.addModule("znn", .{
        .root_source_file = b.path("src/znn.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Test step: test each module file directly (Zig 0.16 does NOT collect
    // `test` blocks from @import-ed files when the root only re-exports them,
    // so test each file as its own root and aggregate here).
    const test_step = b.step("test", "Run all znn tests");
    const test_files = [_][]const u8{
        "src/tensor.zig",
        "src/autograd.zig",
        "src/integration.zig",
        // add more as you create them: "src/tokenizer.zig", "src/data.zig", ...
    };
    for (test_files) |file| {
        const t = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path(file),
                .target = target,
                .optimize = optimize,
            }),
        });
        const run_t = b.addRunArtifact(t);
        test_step.dependOn(&run_t.step);
    }

    // Milestone apps: `zig build run-NN-<name>`. Add entries as you build them.
    const apps = [_]struct { name: []const u8, file: []const u8 }{
        // .{ .name = "04-spiral", .file = "apps/04_spiral.zig" },
        // .{ .name = "13-capstone", .file = "apps/13_capstone.zig" },
    };
    for (apps) |app| {
        const exe = b.addExecutable(.{
            .name = app.name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(app.file),
                .target = target,
                .optimize = optimize,
                .imports = &.{.{ .name = "znn", .module = znn_mod }},
            }),
        });
        b.installArtifact(exe);
        const run_cmd = b.addRunArtifact(exe);
        const run_step = b.step(b.fmt("run-{s}", .{app.name}), b.fmt("Run {s}", .{app.name}));
        run_step.dependOn(&run_cmd.step);
        if (b.args) |args| run_cmd.addArgs(args);
    }
}