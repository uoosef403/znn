const std = @import("std");

// znn — a from-scratch neural network framework in Zig 0.16.0.
// Build graph: build.zig orchestrates compilation of the znn library + per-milestone apps.

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // GPU backend is opt-in: `zig build bench -Dgpu` links cuBLAS + the nvcc-
    // compiled CUDA kernels and runs the CPU-vs-GPU benchmark + correctness
    // checks. Without -Dgpu everything stays pure-CPU and CUDA-header-free.
    const gpu = b.option(bool, "gpu", "Link the CUDA backend (cuBLAS + kernels)") orelse false;

    // ---- znn library module (importable as "znn") ----
    const znn_mod = b.addModule("znn", .{
        .root_source_file = b.path("src/znn.zig"),
        .target = target,
        .optimize = optimize,
    });

    // ---- test step ----
    // NOTE: Zig 0.16 does NOT collect `test` blocks from @import-ed files when
    // the root only re-exports them, so testing znn.zig runs 0 tests. Instead we
    // test each module file directly as its own root and aggregate them here.
    const test_step = b.step("test", "Run all znn unit + grad-check tests");
    const test_files = [_][]const u8{
        "src/znn/tensor.zig",
        "src/znn/autograd.zig",
        "src/znn/integration.zig",
        "src/znn/tokenizer.zig",
        "src/znn/data.zig",
        "src/znn/checkpoint.zig",
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

    // ---- milestone apps: `zig build run-NN` ----
    // Each app is a tiny main that exercises one arc of the curriculum.
    const apps = [_]App{
        .{ .name = "02-tensor-lab", .file = "apps/02_tensor_lab.zig" },
        .{ .name = "03-autograd-lab", .file = "apps/03_autograd_lab.zig" },
        .{ .name = "04-spiral", .file = "apps/04_spiral.zig" },
        .{ .name = "05-bpe", .file = "apps/05_bpe.zig" },
        .{ .name = "06-fineweb", .file = "apps/06_fineweb.zig" },
        .{ .name = "07-first-lm", .file = "apps/07_first_lm.zig" },
        .{ .name = "09-transformer", .file = "apps/09_transformer.zig" },
        .{ .name = "11-gated-delta", .file = "apps/11_gated_delta.zig" },
        .{ .name = "13-capstone", .file = "apps/13_capstone.zig" },
    };
    for (apps) |app| {
        const exe = b.addExecutable(.{
            .name = app.name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(app.file),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "znn", .module = znn_mod },
                },
            }),
        });
        b.installArtifact(exe);
        const run_cmd = b.addRunArtifact(exe);
        const run_step = b.step(b.fmt("run-{s}", .{app.name}), b.fmt("Run {s}", .{app.name}));
        run_step.dependOn(&run_cmd.step);
        if (b.args) |args| run_cmd.addArgs(args);
    }

    // ---- bench step: CPU vs (with -Dgpu) GPU matmul throughput ----
    // Build bench's imports conditionally: always znn; add backend_cuda only
    // when -Dgpu (so CPU builds never touch CUDA headers).
    var bench_imports = std.ArrayList(std.Build.Module.Import).empty;
    defer bench_imports.deinit(b.allocator);
    bench_imports.append(b.allocator, .{ .name = "znn", .module = znn_mod }) catch @panic("OOM");
    if (gpu) {
        const cuda_mod = b.createModule(.{
            .root_source_file = b.path("src/znn/backend_cuda.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        });
        bench_imports.append(b.allocator, .{ .name = "backend_cuda", .module = cuda_mod }) catch @panic("OOM");
    }
    const bench_mod = b.createModule(.{
        .root_source_file = b.path("apps/bench.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .imports = bench_imports.items,
    });
    // pass the gpu flag into bench.zig at comptime
    const opts = b.addOptions();
    opts.addOption(bool, "gpu", gpu);
    bench_mod.addOptions("znn_opts", opts);
    var nvcc_obj_step: ?*std.Build.Step = null;
    if (gpu) {
        // compile the CUDA kernels with nvcc -> .o, link cuBLAS + cudart + libc
        const nvcc = b.addSystemCommand(&.{ "nvcc", "-c", "-O2", "-arch=sm_75" });
        nvcc.addFileArg(b.path("src/kernels/znn_kernels.cu"));
        nvcc.addArg("-o"); // nvcc needs `-o` and the path as SEPARATE argv
        const obj = nvcc.addOutputFileArg("znn_kernels.o");
        bench_mod.addObjectFile(obj);
        bench_mod.linkSystemLibrary("cublas", .{});
        bench_mod.linkSystemLibrary("cudart", .{});
        bench_mod.linkSystemLibrary("c", .{}); // atexit for cuda static registration
        nvcc_obj_step = &nvcc.step;
    }
    const bench = b.addExecutable(.{
        .name = "bench",
        .root_module = bench_mod,
    });
    if (nvcc_obj_step) |ns| bench.step.dependOn(ns);
    b.installArtifact(bench);
    const bench_cmd = b.addRunArtifact(bench);
    const bench_step = b.step("bench", "Run matmul benchmark (CPU; add -Dgpu for CPU-vs-GPU)");
    bench_step.dependOn(&bench_cmd.step);

    // `zig build gpu` builds/links the CUDA backend as a smoke build (run the
    // benchmark itself with `zig build bench -Dgpu`).
    const gpu_step = b.step("gpu", "Build/link the CUDA backend (smoke build; run `zig build bench -Dgpu`)");
    gpu_step.dependOn(&bench.step);
}

const App = struct {
    name: []const u8,
    file: []const u8,
};