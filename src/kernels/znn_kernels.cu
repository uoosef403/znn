// znn_kernels.cu — hand-written CUDA kernels for the znn GPU backend (Gate 11).
// Compiled with:  nvcc -c -O2 -arch=sm_75 znn_kernels.cu -o znn_kernels.o
// and linked into the Zig exe (see build.zig, -Dgpu). Host launchers are
// `extern "C"` so Zig can call them; kernels are `__global__`.
//
// Target: GTX 1660 Ti, compute 7.5 (Turing). No tensor cores -> FP32 paths.
//
// Convention: every kernel is the *GPU mirror* of a verified CPU op in
// src/znn/autograd.zig. The bench/gpu tests assert max|diff| < 1e-3 vs CPU.

#define MAXD 32 // max head_dim supported by the gated-delta state-buffer kernel

// ---- Gated Delta Attention — recurrent forward (the hero kernel) ----
// Exact mirror of `vGatedDelta` in autograd.zig:
//   r_t = S_{t-1} k_t ; d_t = v_t - alpha_t r_t
//   S_t = alpha_t S_{t-1} + beta_t outer(d_t, k_t) ;  o_t = S_t q_t
// One thread per (batch,head) carries a Dh x Dh state in local memory;
// the L axis is sequential (the recurrence is inherently sequential — this
// is the *recurrent* form; the chunked-parallel kernel is the next step).
// Launch: <<<ceil(Bh/128), 128>>>. Inputs q,k,v:[Bh,L,Dh], alpha,beta:[Bh,L].
extern "C" __global__ void gated_delta_fwd_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ alpha,
    const float* __restrict__ beta, float* __restrict__ out,
    int Bh, int L, int Dh)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= Bh) return;
    float state[MAXD * MAXD];
    for (int i = 0; i < Dh * Dh; ++i) state[i] = 0.0f;
    const int LT = L * Dh;
    for (int t = 0; t < L; ++t) {
        float a = alpha[h * L + t];
        float b = beta[h * L + t];
        const float* kt = k + h * LT + t * Dh;
        const float* vt = v + h * LT + t * Dh;
        const float* qt = q + h * LT + t * Dh;
        float* ot = out + h * LT + t * Dh;
        float d[MAXD];
        // r = S k ; d = v - a*r
        for (int i = 0; i < Dh; ++i) {
            float ri = 0.0f;
            for (int j = 0; j < Dh; ++j) ri += state[i * Dh + j] * kt[j];
            d[i] = vt[i] - a * ri;
        }
        // S <- a*S + b*outer(d,k)
        for (int i = 0; i < Dh; ++i)
            for (int j = 0; j < Dh; ++j)
                state[i * Dh + j] = a * state[i * Dh + j] + b * d[i] * kt[j];
        // o = S q
        for (int i = 0; i < Dh; ++i) {
            float s = 0.0f;
            for (int j = 0; j < Dh; ++j) s += state[i * Dh + j] * qt[j];
            ot[i] = s;
        }
    }
}

extern "C" void launch_gated_delta_fwd(
    const float* q, const float* k, const float* v,
    const float* alpha, const float* beta, float* out,
    int Bh, int L, int Dh)
{
    int blk = 128;
    int grid = (Bh + blk - 1) / blk;
    gated_delta_fwd_kernel<<<grid, blk>>>(q, k, v, alpha, beta, out, Bh, L, Dh);
}