// backend_cuda.zig — the znn GPU backend (Gate 11).
//
// This file @cImports the CUDA runtime + cuBLAS headers, so it MUST only be
// built when the `-Dgpu` build option is set (otherwise a CPU-only machine
// would fail to find the headers). It is imported ONLY by the GPU build path
// (bench / gpu apps). The CPU framework (src/znn/*) never touches CUDA.
//
// Design: a tiny `CudaBackend` holding one cuBLAS handle, exposing
//   - device memory: alloc / free / h2d / d2h
//   - matmul: row-major C[M,N] = A[M,K] @ B[K,N] via cuBLAS sgemm
//   - batched matmul: loop of sgemm over a leading batch dim (correctness-first;
//     cublasSgemmStridedBatched is the obvious optimization, left as an exercise)
//   - the gated-delta recurrent kernel (the hero), launched via an extern host
//     launcher compiled from src/kernels/znn_kernels.cu by nvcc.
//
// cuBLAS is column-major. The row-major trick: A[M,K] row-major == A^T col-major
// (ld=K); B[K,N] row-major == B^T col-major (ld=N). We ask cuBLAS for
//   C_col[N,M] = B_col[N,K] @ A_col[K,M]   (OP_N, OP_N), ldc=N,
// which in memory is C^T col-major == C row-major [M,N]. So the call is
//   cublasSgemm(handle, OP_N, OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N).

const std = @import("std");
const c = @cImport({
    @cInclude("cuda_runtime.h");
    @cInclude("cublas_v2.h");
});

pub const Error = error{
    CudaSetDevice,
    CudaMalloc,
    CublasCreate,
    CubasSgemm,
    CudaLaunch,
};

pub const CudaBackend = struct {
    handle: c.cublasHandle_t,

    pub fn init() Error!CudaBackend {
        if (c.cudaSetDevice(0) != c.cudaSuccess) return Error.CudaSetDevice;
        var handle: c.cublasHandle_t = null;
        if (c.cublasCreate(&handle) != c.CUBLAS_STATUS_SUCCESS) return Error.CublasCreate;
        return .{ .handle = handle };
    }

    pub fn deinit(self: *CudaBackend) void {
        _ = c.cublasDestroy(self.handle);
    }

    pub fn sync(self: *CudaBackend) void {
        _ = self;
        _ = c.cudaDeviceSynchronize();
    }

    // ---- device memory ----
    pub fn alloc(self: *CudaBackend, bytes: usize) Error!?*anyopaque {
        _ = self;
        var p: ?*anyopaque = null;
        if (c.cudaMalloc(&p, bytes) != c.cudaSuccess) return Error.CudaMalloc;
        return p;
    }
    pub fn free(self: *CudaBackend, p: ?*anyopaque) void {
        _ = self;
        _ = c.cudaFree(p);
    }
    pub fn h2d(self: *CudaBackend, dst: ?*anyopaque, src: []const f32) void {
        _ = self;
        _ = c.cudaMemcpy(dst, src.ptr, src.len * @sizeOf(f32), c.cudaMemcpyHostToDevice);
    }
    pub fn d2h(self: *CudaBackend, dst: []f32, src: ?*anyopaque) void {
        _ = self;
        _ = c.cudaMemcpy(dst.ptr, src, dst.len * @sizeOf(f32), c.cudaMemcpyDeviceToHost);
    }

    // ---- row-major matmul: C[M,N] = A[M,K] @ B[K,N] ----
    pub fn matmul(self: *CudaBackend, dA: ?*anyopaque, dB: ?*anyopaque, dC: ?*anyopaque, M: usize, N: usize, K: usize) Error!void {
        const alpha: f32 = 1.0;
        const beta: f32 = 0.0;
        const stat = c.cublasSgemm(
            self.handle,
            c.CUBLAS_OP_N,
            c.CUBLAS_OP_N,
            @intCast(N),
            @intCast(M),
            @intCast(K),
            &alpha,
            @ptrCast(@alignCast(dB)),
            @intCast(N),
            @ptrCast(@alignCast(dA)),
            @intCast(K),
            &beta,
            @ptrCast(@alignCast(dC)),
            @intCast(N),
        );
        if (stat != c.CUBLAS_STATUS_SUCCESS) return Error.CubasSgemm;
    }

    // ---- batched row-major matmul: C[B,M,N] = A[B,M,K] @ B[B,K,N] ----
    // (loop of sgemm; strided-batched is the optimization). Batch is the
    // leading dim with stride M*K / K*N / M*N.
    pub fn matmulBatched(self: *CudaBackend, dA: ?*anyopaque, dB: ?*anyopaque, dC: ?*anyopaque, B: usize, M: usize, N: usize, K: usize) Error!void {
        const a_stride = M * K * @sizeOf(f32);
        const b_stride = K * N * @sizeOf(f32);
        const c_stride = M * N * @sizeOf(f32);
        const alpha: f32 = 1.0;
        const beta: f32 = 0.0;
        for (0..B) |i| {
            const ai = @intFromPtr(dA) + i * a_stride;
            const bi = @intFromPtr(dB) + i * b_stride;
            const ci = @intFromPtr(dC) + i * c_stride;
            const ap: ?*anyopaque = @ptrFromInt(ai);
            const bp: ?*anyopaque = @ptrFromInt(bi);
            const cp: ?*anyopaque = @ptrFromInt(ci);
            const stat = c.cublasSgemm(
                self.handle,
                c.CUBLAS_OP_N,
                c.CUBLAS_OP_N,
                @intCast(N),
                @intCast(M),
                @intCast(K),
                &alpha,
                @ptrCast(@alignCast(bp)),
                @intCast(N),
                @ptrCast(@alignCast(ap)),
                @intCast(K),
                &beta,
                @ptrCast(@alignCast(cp)),
                @intCast(N),
            );
            if (stat != c.CUBLAS_STATUS_SUCCESS) return Error.CubasSgemm;
        }
    }

    // ---- gated delta attention recurrent forward (the hero kernel) ----
    // q,k,v:[Bh,L,Dh]; alpha,beta:[Bh,L]; out:[Bh,L,Dh]. Exact mirror of
    // autograd.vGatedDelta's forward. Requires Dh <= 32 (MAXD in the kernel).
    pub fn gatedDeltaFwd(self: *CudaBackend, dq: ?*anyopaque, dk: ?*anyopaque, dv: ?*anyopaque, dalpha: ?*anyopaque, dbeta: ?*anyopaque, dout: ?*anyopaque, Bh: usize, L: usize, Dh: usize) Error!void {
        _ = self;
        std.debug.assert(Dh <= 32);
        launch_gated_delta_fwd(
            @ptrCast(@alignCast(dq)),
            @ptrCast(@alignCast(dk)),
            @ptrCast(@alignCast(dv)),
            @ptrCast(@alignCast(dalpha)),
            @ptrCast(@alignCast(dbeta)),
            @ptrCast(@alignCast(dout)),
            @intCast(Bh),
            @intCast(L),
            @intCast(Dh),
        );
        if (c.cudaGetLastError() != c.cudaSuccess) return Error.CudaLaunch;
    }
};

extern fn launch_gated_delta_fwd(q: *const f32, k: *const f32, v: *const f32, alpha: *const f32, beta: *const f32, out: *f32, Bh: c_int, L: c_int, Dh: c_int) void;