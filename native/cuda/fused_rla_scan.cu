// Fused RLA/RDN (Residual Linear Attention) Scan Kernel
//
// Dual-state recurrence with base state S and residual state R.
// Two variants controlled by mode flag:
//
// Variant 0 (RLA — moving average):
//   retrieval = S @ k
//   r_error = clip(v - retrieval, -c, c)
//   S_t = alpha * S_{t-1} + beta * outer(v, k)
//   R_t = alpha * R_{t-1} + gamma * outer(r_error, k)
//   o_t = (S_t + R_t) @ q
//
// Variant 1 (RDN — delta rule):
//   retrieval_s = S @ k
//   retrieval_r = R @ k
//   r_error = clip(v - retrieval_s, -c, c)
//   S_t = alpha * S_{t-1} + beta * outer(v - retrieval_s, k)
//   R_t = alpha * R_{t-1} + gamma * outer(r_error - retrieval_r, k)
//   o_t = (S_t + R_t) @ q
//
// Thread layout: one thread BLOCK per (batch, head), head_dim threads.
// Each thread owns row i of both S[d][d] and R[d][d].
//
// Inputs (all pre-computed on Elixir/XLA side):
//   q:     [B, T, H, d] — query vectors
//   k:     [B, T, H, d] — key vectors
//   v:     [B, T, H, d] — value vectors
//   alpha: [B, T, H]    — decay gate (per-head scalar)
//   beta:  [B, T, H]    — base update rate (per-head scalar)
//   gamma: [B, T, H]    — residual update rate (per-head scalar)
//
// Output:
//   output: [B, T, H, d] — retrieval outputs
//
// Shared memory budget (head_dim=64):
//   S matrix: 64*64*4 = 16KB
//   R matrix: 64*64*4 = 16KB
//   k_shared: 64*4 = 256 bytes
//   q_shared: 64*4 = 256 bytes
//   v_shared: 64*4 = 256 bytes
//   Total: ~33KB — within 48KB limit

#include <cuda_runtime.h>
#include "precision.cuh"

// ============================================================================
// Kernel
// ============================================================================

__global__ void fused_rla_scan_kernel(
    const io_type* __restrict__ q,       // [B, T, H, d]
    const io_type* __restrict__ k,       // [B, T, H, d]
    const io_type* __restrict__ v,       // [B, T, H, d]
    const io_type* __restrict__ alpha,   // [B, T, H]
    const io_type* __restrict__ beta,    // [B, T, H]
    const io_type* __restrict__ gamma,   // [B, T, H]
    io_type* __restrict__ output,        // [B, T, H, d]
    int seq_len,
    int num_heads,
    int head_dim,
    int variant,              // 0 = RLA, 1 = RDN
    float clip_threshold
) {
    int b = blockIdx.x;   // batch index
    int h = blockIdx.y;   // head index
    int i = threadIdx.x;  // row index (0..head_dim-1)

    if (i >= head_dim) return;

    // Shared memory layout:
    //   S[d][d] — base state matrix
    //   R[d][d] — residual state matrix
    //   k_shared[d], q_shared[d], v_shared[d]
    extern __shared__ float smem[];
    float* S = smem;                                          // [d][d]
    float* R = S + head_dim * head_dim;                       // [d][d]
    float* k_shared = R + head_dim * head_dim;                // [d]
    float* q_shared = k_shared + head_dim;                    // [d]
    float* v_shared = q_shared + head_dim;                    // [d]

    // Initialize both state matrices to zero
    for (int j = 0; j < head_dim; j++) {
        S[i * head_dim + j] = 0.0f;
        R[i * head_dim + j] = 0.0f;
    }
    __syncthreads();

    // Strides for [B, T, H, d] tensors
    int BHd = seq_len * num_heads * head_dim;
    int THd = num_heads * head_dim;
    int Hd  = head_dim;

    // Strides for [B, T, H] gate tensors
    int gate_BH = seq_len * num_heads;
    int gate_TH = num_heads;

    int base_bh = b * BHd + h * Hd;
    int gate_base_b = b * gate_BH;

    for (int t = 0; t < seq_len; t++) {
        int offset = base_bh + t * THd;
        int gate_offset = gate_base_b + t * gate_TH + h;

        // Load k and v into shared memory
        k_shared[i] = IO_LOAD(k, offset + i);
        v_shared[i] = IO_LOAD(v, offset + i);
        __syncthreads();

        // Load per-head scalar gates
        float alpha_val = IO_LOAD(alpha, gate_offset);
        float beta_val  = IO_LOAD(beta, gate_offset);
        float gamma_val = IO_LOAD(gamma, gate_offset);

        // Compute retrieval from S: retrieval_s[i] = sum_j(S[i][j] * k[j])
        float retrieval_s = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            retrieval_s += S[i * head_dim + j] * k_shared[j];
        }

        // Residual error: clip(v - retrieval_s)
        float v_i = v_shared[i];
        float raw_error = v_i - retrieval_s;
        float r_error = fminf(fmaxf(raw_error, -clip_threshold), clip_threshold);

        if (variant == 0) {
            // RLA: S = alpha*S + beta * outer(v, k)
            //      R = alpha*R + gamma * outer(r_error, k)
            float beta_v_i = beta_val * v_i;
            float gamma_r_i = gamma_val * r_error;

            for (int j = 0; j < head_dim; j++) {
                S[i * head_dim + j] = alpha_val * S[i * head_dim + j]
                                    + beta_v_i * k_shared[j];
                R[i * head_dim + j] = alpha_val * R[i * head_dim + j]
                                    + gamma_r_i * k_shared[j];
            }
        } else {
            // RDN: S = alpha*S + beta * outer(v - S@k, k)
            //      R = alpha*R + gamma * outer(r_error - R@k, k)

            // Compute retrieval from R: retrieval_r[i] = sum_j(R[i][j] * k[j])
            float retrieval_r = 0.0f;
            for (int j = 0; j < head_dim; j++) {
                retrieval_r += R[i * head_dim + j] * k_shared[j];
            }

            float delta_s_i = v_i - retrieval_s;  // same as raw_error
            float delta_r_i = r_error - retrieval_r;
            float beta_ds_i = beta_val * delta_s_i;
            float gamma_dr_i = gamma_val * delta_r_i;

            for (int j = 0; j < head_dim; j++) {
                S[i * head_dim + j] = alpha_val * S[i * head_dim + j]
                                    + beta_ds_i * k_shared[j];
                R[i * head_dim + j] = alpha_val * R[i * head_dim + j]
                                    + gamma_dr_i * k_shared[j];
            }
        }
        __syncthreads();

        // Load q into shared memory (reuse v_shared slot since v is consumed)
        q_shared[i] = IO_LOAD(q, offset + i);
        __syncthreads();

        // Output: o[i] = (S[i] + R[i]) @ q
        float out_i = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            out_i += (S[i * head_dim + j] + R[i * head_dim + j]) * q_shared[j];
        }

        IO_STORE(output, offset + i, out_i);
        __syncthreads();
    }
}

// ============================================================================
// Standalone launch wrapper (C-linkage for NIF / dlopen)
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

int fused_rla_scan_launch(
    cudaStream_t stream,
    const io_type* q, const io_type* k, const io_type* v,
    const io_type* alpha, const io_type* beta, const io_type* gamma,
    io_type* output,
    int batch, int seq_len, int num_heads, int head_dim,
    int variant, float clip_threshold
) {
    dim3 grid(batch, num_heads);
    dim3 block(head_dim);

    // Shared memory: S[d][d] + R[d][d] + k_shared[d] + q_shared[d] + v_shared[d]
    size_t smem_bytes = 2 * (size_t)head_dim * head_dim * sizeof(float)
                      + 3 * head_dim * sizeof(float);

    fused_rla_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        q, k, v, alpha, beta, gamma, output,
        seq_len, num_heads, head_dim,
        variant, clip_threshold
    );

    return (int)cudaGetLastError();
}

}  // extern "C"

#endif  // !EXLA_FFI

// ============================================================================
// XLA FFI integration (for EXLA fork)
// ============================================================================

#ifdef EXLA_FFI

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// Note: variant and clip_threshold passed as attributes from EXLA side
ffi::Error fused_rla_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> q,       // [B, T, H, d]
    ffi::Buffer<FFI_IO_TYPE> k,       // [B, T, H, d]
    ffi::Buffer<FFI_IO_TYPE> v,       // [B, T, H, d]
    ffi::Buffer<FFI_IO_TYPE> alpha,   // [B, T, H]
    ffi::Buffer<FFI_IO_TYPE> beta,    // [B, T, H]
    ffi::Buffer<FFI_IO_TYPE> gamma,   // [B, T, H]
    ffi::ResultBuffer<FFI_IO_TYPE> output,  // [B, T, H, d]
    int32_t variant,
    float clip_threshold
) {
    auto dims = q.dimensions();
    int batch     = static_cast<int>(dims[0]);
    int seq_len   = static_cast<int>(dims[1]);
    int num_heads = static_cast<int>(dims[2]);
    int head_dim  = static_cast<int>(dims[3]);

    dim3 grid(batch, num_heads);
    dim3 block(head_dim);
    size_t smem_bytes = 2 * (size_t)head_dim * head_dim * sizeof(float)
                      + 3 * head_dim * sizeof(float);

    fused_rla_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const io_type*>(q.untyped_data()),
        reinterpret_cast<const io_type*>(k.untyped_data()),
        reinterpret_cast<const io_type*>(v.untyped_data()),
        reinterpret_cast<const io_type*>(alpha.untyped_data()),
        reinterpret_cast<const io_type*>(beta.untyped_data()),
        reinterpret_cast<const io_type*>(gamma.untyped_data()),
        reinterpret_cast<io_type*>(output->untyped_data()),
        seq_len, num_heads, head_dim,
        static_cast<int>(variant), clip_threshold
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_rla_scan, fused_rla_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // q
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // k
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // v
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // alpha
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // beta
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // gamma
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // output
        .Attr<int32_t>("variant")
        .Attr<float>("clip_threshold")
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_rla_scan_" PRECISION_SUFFIX, "CUDA", fused_rla_scan);

#endif  // EXLA_FFI
