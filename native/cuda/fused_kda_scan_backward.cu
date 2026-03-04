// Fused KDA (Kimi Delta Attention) Scan Backward Kernel
//
// Computes gradients for the KDA channel-wise decay delta rule recurrence.
// KDA differs from GatedDeltaNet: alpha is per-channel [B,T,H,d] (not scalar
// per head), and beta is scalar per head [B,T,H] (not per-dimension).
//
// Forward recurrence (from fused_kda_scan.cu):
//   decay_i   = exp(alpha_t[i])               -- channel-wise decay
//   S_decayed = diag(decay) * S_{t-1}         -- each row i scaled by decay_i
//   retrieval = S_decayed @ k_t
//   error     = v_t - retrieval
//   S_t       = S_decayed + beta_t * outer(error, k_t)
//   o_t       = S_t @ q_t
//
// Backward derivation (per timestep, reverse order):
//
//   From o = S_t @ q:
//     dS[i][j] += do[i] * q[j]
//     dq[j]     = sum_i(S_t[i][j] * do[i])         -- cross-thread reduction
//
//   From S_t = S_decayed + beta * outer(error, k):
//     Let dS_k[i] = sum_j(dS[i][j] * k[j])         -- thread-local dot product
//     d_error[i]  = beta * dS_k[i]
//     dbeta      += sum_i(error[i] * dS_k[i])       -- cross-thread reduction
//     dk[j]      += sum_i(dS[i][j] * beta * error[i])  -- cross-thread reduction
//
//   From error = v - retrieval:
//     dv[i]          += d_error[i]
//     d_retrieval[i]  = -d_error[i]
//
//   From retrieval = S_decayed @ k:
//     dS_decayed[i][j] = dS[i][j] + d_retrieval[i] * k[j]
//     dk[j]           += sum_i(S_decayed[i][j] * d_retrieval[i])  -- cross-thread
//
//   From S_decayed[i][j] = exp(alpha[i]) * S_prev[i][j]:
//     dS_prev[i][j]  = exp(alpha[i]) * dS_decayed[i][j]
//     d_alpha[i]     = exp(alpha[i]) * sum_j(dS_decayed[i][j] * S_prev[i][j])
//
// Two-pass approach:
//   Pass 1 (forward): recompute S at each timestep, store retrieval per thread.
//   Pass 2 (backward): reverse iterate with dS accumulator in registers.
//     At each t, undo the update to recover S_decayed, then undo decay to
//     recover S_prev. Use S_prev and S_decayed for gradient computation.
//
// Thread layout: one block per (batch, head), head_dim threads.
// Each thread i owns row i of S[d][d] and dS[d][d].
//
// Inputs:
//   q:           [B, T, H, d]
//   k:           [B, T, H, d]
//   v:           [B, T, H, d]
//   alpha:       [B, T, H, d]  -- per-channel decay (log-space)
//   beta:        [B, T, H]     -- scalar update gate per head
//   forward_out: [B, T, H, d]  -- forward pass outputs (unused, kept for API consistency)
//   grad_output: [B, T, H, d]  -- upstream gradient dL/do
//
// Outputs:
//   grad_q:     [B, T, H, d]
//   grad_k:     [B, T, H, d]
//   grad_v:     [B, T, H, d]
//   grad_alpha: [B, T, H, d]
//   grad_beta:  [B, T, H]

#include <cuda_runtime.h>
#include "precision.cuh"

#define MAX_SEQ_LEN 1024

// ============================================================================
// Kernel
// ============================================================================

#ifdef EXLA_FFI
namespace {  // anonymous namespace — internal linkage per compilation unit
#endif

__global__ void fused_kda_scan_backward_kernel(
    const io_type* __restrict__ q,            // [B, T, H, d]
    const io_type* __restrict__ k,            // [B, T, H, d]
    const io_type* __restrict__ v,            // [B, T, H, d]
    const io_type* __restrict__ alpha,        // [B, T, H, d]
    const io_type* __restrict__ beta,         // [B, T, H]
    const io_type* __restrict__ forward_out,  // [B, T, H, d]
    const io_type* __restrict__ grad_output,  // [B, T, H, d]
    io_type* __restrict__ grad_q,             // [B, T, H, d]
    io_type* __restrict__ grad_k,             // [B, T, H, d]
    io_type* __restrict__ grad_v,             // [B, T, H, d]
    io_type* __restrict__ grad_alpha,         // [B, T, H, d]
    io_type* __restrict__ grad_beta,          // [B, T, H]
    int seq_len,
    int num_heads,
    int head_dim
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = threadIdx.x;

    if (i >= head_dim) return;

    // Shared memory layout:
    //   S[d][d]        -- state matrix (each thread owns row i)
    //   k_shared[d]    -- current k vector
    //   q_shared[d]    -- current q vector / reused for reductions
    //   temp_shared[d] -- cross-thread dk accumulator
    //   reduce_shared[1] -- for beta gradient reduction (scalar per head)
    extern __shared__ float smem[];
    float* S = smem;                                      // [d][d]
    float* k_shared = smem + head_dim * head_dim;         // [d]
    float* q_shared = k_shared + head_dim;                // [d]
    float* temp_shared = q_shared + head_dim;             // [d]
    float* reduce_shared = temp_shared + head_dim;        // [1]

    // Strides for [B, T, H, d] tensors
    int THd = seq_len * num_heads * head_dim;  // batch stride
    int Hd = num_heads * head_dim;             // time stride
    int d = head_dim;                          // head stride
    int base_bh = b * THd + h * d;

    // Strides for beta: [B, T, H]
    int beta_TH = seq_len * num_heads;
    int beta_base = b * beta_TH + h;

    // ========================================================================
    // Pass 1: Forward -- recompute S at each timestep, store retrieval values
    // ========================================================================
    for (int j = 0; j < head_dim; j++) {
        S[i * head_dim + j] = 0.0f;
    }
    __syncthreads();

    // Per-thread storage for retrieval values at each timestep
    float local_retrieval[MAX_SEQ_LEN];

    for (int t = 0; t < seq_len; t++) {
        int offset = base_bh + t * Hd;

        // Load k_t into shared memory
        k_shared[i] = IO_LOAD(k, offset + i);
        __syncthreads();

        // Channel-wise decay: S[i][j] *= exp(alpha_t[i])
        float alpha_i = IO_LOAD(alpha, offset + i);
        float decay_i = expf(alpha_i);
        for (int j = 0; j < head_dim; j++) {
            S[i * head_dim + j] *= decay_i;
        }
        __syncthreads();

        // Compute retrieval[i] = sum_j(S[i][j] * k_shared[j])
        float retrieval = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            retrieval += S[i * head_dim + j] * k_shared[j];
        }
        local_retrieval[t] = retrieval;

        // Delta rule update: S += beta * outer(error, k)
        float v_i = IO_LOAD(v, offset + i);
        float beta_val = IO_LOAD(beta, beta_base + t * num_heads);
        float error_i = v_i - retrieval;
        float scaled_error_i = beta_val * error_i;

        for (int j = 0; j < head_dim; j++) {
            S[i * head_dim + j] += scaled_error_i * k_shared[j];
        }
        __syncthreads();
    }

    // ========================================================================
    // Pass 2: Backward -- reverse iterate accumulating dS
    // ========================================================================
    // dS accumulator: thread i owns row i (in registers)
    float dS_row[128];  // head_dim <= 128
    for (int j = 0; j < head_dim; j++) {
        dS_row[j] = 0.0f;
    }

    for (int t = seq_len - 1; t >= 0; t--) {
        int offset = base_bh + t * Hd;

        // Load values for this timestep
        float q_i = IO_LOAD(q, offset + i);
        float k_i = IO_LOAD(k, offset + i);
        float v_i = IO_LOAD(v, offset + i);
        float alpha_i = IO_LOAD(alpha, offset + i);
        float decay_i = expf(alpha_i);
        float beta_val = IO_LOAD(beta, beta_base + t * num_heads);
        float do_i = IO_LOAD(grad_output, offset + i);

        // Reconstruct error from forward pass
        float retrieval_i = local_retrieval[t];
        float error_i = v_i - retrieval_i;
        float scaled_err_i = beta_val * error_i;

        // ---- grad_q: dq[j] = sum_i(S_t[i][j] * do[i]) ----
        // S_t is current S in shared memory.
        // dq[j] = column j of S dotted with do vector.
        // Thread i contributes S_t[i][j] * do_i for each j.
        // Use shared mem for do vector, then each thread reads S[j][i] (column i).
        q_shared[i] = do_i;
        __syncthreads();

        float dq_i = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            dq_i += S[j * head_dim + i] * q_shared[j];
        }
        IO_STORE(grad_q, offset + i, dq_i);

        // ---- dS from output: o = S @ q, so dS[i][j] += do[i] * q[j] ----
        k_shared[i] = q_i;  // reuse k_shared to broadcast q
        __syncthreads();

        for (int j = 0; j < head_dim; j++) {
            dS_row[j] += do_i * k_shared[j];
        }

        // ---- Gradients from update: S_t = S_decayed + beta * outer(error, k) ----
        // Load k into shared memory
        k_shared[i] = k_i;
        __syncthreads();

        // dS_k[i] = sum_j(dS[i][j] * k[j]) -- thread-local dot product
        float dS_k_i = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            dS_k_i += dS_row[j] * k_shared[j];
        }

        // d_error[i] = beta * dS_k[i]
        float d_error_i = beta_val * dS_k_i;

        // dv[i] = d_error[i]  (from error = v - retrieval)
        float dv_i = d_error_i;

        // d_retrieval[i] = -d_error[i]
        float d_retrieval_i = -d_error_i;

        // ---- dbeta: sum_i(error[i] * dS_k[i]) -- cross-thread scalar reduction ----
        // Reduce across threads into shared memory, then thread 0 writes.
        // No global atomicAdd needed: each (b,t,h) is written by exactly one block.
        if (i == 0) reduce_shared[0] = 0.0f;
        __syncthreads();
        atomicAdd(reduce_shared, error_i * dS_k_i);
        __syncthreads();

        if (i == 0) {
            IO_STORE(grad_beta, beta_base + t * num_heads, reduce_shared[0]);
        }

        // ---- dk from update: dk[j] += sum_i(dS[i][j] * beta * error[i]) ----
        // Each thread i contributes dS_row[j] * beta * error[i] for each j.
        if (i == 0) {
            for (int j = 0; j < head_dim; j++) {
                temp_shared[j] = 0.0f;
            }
        }
        __syncthreads();

        for (int j = 0; j < head_dim; j++) {
            atomicAdd(&temp_shared[j], dS_row[j] * scaled_err_i);
        }
        __syncthreads();

        // ---- dS_decayed from retrieval gradient ----
        // dS_decayed[i][j] = dS[i][j] + d_retrieval[i] * k[j]
        // (dS already contains dS_new contribution; add retrieval gradient)
        for (int j = 0; j < head_dim; j++) {
            dS_row[j] += d_retrieval_i * k_shared[j];
        }

        // ---- Undo update to recover S_decayed ----
        // S_decayed = S_t - beta * outer(error, k)
        for (int j = 0; j < head_dim; j++) {
            S[i * head_dim + j] -= scaled_err_i * k_shared[j];
        }
        __syncthreads();
        // Now S holds S_decayed for this timestep.

        // ---- dk from retrieval: dk[j] += sum_i(S_decayed[i][j] * d_retrieval[i]) ----
        // Cross-thread reduction into q_shared (reused as accumulator)
        if (i == 0) {
            for (int j = 0; j < head_dim; j++) {
                q_shared[j] = 0.0f;
            }
        }
        __syncthreads();

        for (int j = 0; j < head_dim; j++) {
            atomicAdd(&q_shared[j], S[i * head_dim + j] * d_retrieval_i);
        }
        __syncthreads();

        // Total dk = dk from update (temp_shared) + dk from retrieval (q_shared)
        float dk_i = temp_shared[i] + q_shared[i];

        IO_STORE(grad_k, offset + i, dk_i);
        IO_STORE(grad_v, offset + i, dv_i);

        // ---- Gradient through channel-wise decay ----
        // S_decayed[i][j] = exp(alpha[i]) * S_prev[i][j]
        // S_prev[i][j] = S_decayed[i][j] / exp(alpha[i])
        //
        // d_alpha[i] = exp(alpha[i]) * sum_j(dS_decayed[i][j] * S_prev[i][j])
        //            = sum_j(dS_decayed[i][j] * S_decayed[i][j])
        //   (because exp(alpha[i]) * S_prev[i][j] = S_decayed[i][j])
        //
        // dS_prev[i][j] = exp(alpha[i]) * dS_decayed[i][j]

        float d_alpha_i = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            d_alpha_i += dS_row[j] * S[i * head_dim + j];
        }
        IO_STORE(grad_alpha, offset + i, d_alpha_i);

        // Propagate dS through decay: dS_prev[i][j] = decay_i * dS_decayed[i][j]
        for (int j = 0; j < head_dim; j++) {
            dS_row[j] *= decay_i;
        }

        // Undo decay to recover S_prev: S_prev = S_decayed / exp(alpha)
        if (decay_i != 0.0f) {
            float inv_decay = 1.0f / decay_i;
            for (int j = 0; j < head_dim; j++) {
                S[i * head_dim + j] *= inv_decay;
            }
        }
        __syncthreads();
        // Now S holds S_{t-1}, ready for the next reverse iteration.
    }
}

// ============================================================================
// Standalone launch wrapper (C-linkage for NIF / dlopen)
// ============================================================================

#ifdef EXLA_FFI
}  // anonymous namespace
#endif

#ifndef EXLA_FFI

extern "C" {

// Output: concat [grad_q(B*T*H*d) | grad_k(B*T*H*d) | grad_v(B*T*H*d)
//                | grad_alpha(B*T*H*d) | grad_beta(B*T*H)]
int fused_kda_scan_backward_launch(
    cudaStream_t stream,
    const io_type* q, const io_type* k, const io_type* v,
    const io_type* alpha, const io_type* beta,
    const io_type* forward_out, const io_type* grad_output,
    io_type* output_concat,
    int batch, int seq_len, int num_heads, int head_dim
) {
    int total_4d = batch * seq_len * num_heads * head_dim;
    io_type* grad_q     = output_concat;
    io_type* grad_k     = output_concat + total_4d;
    io_type* grad_v     = output_concat + 2 * total_4d;
    io_type* grad_alpha = output_concat + 3 * total_4d;
    io_type* grad_beta  = output_concat + 4 * total_4d;

    dim3 grid(batch, num_heads);
    dim3 block(head_dim);

    // Shared memory: S[d][d] + k_shared[d] + q_shared[d] + temp_shared[d] + reduce[1]
    size_t smem_bytes = (size_t)head_dim * head_dim * sizeof(float)
                      + 3 * head_dim * sizeof(float)
                      + sizeof(float);

    fused_kda_scan_backward_kernel<<<grid, block, smem_bytes, stream>>>(
        q, k, v, alpha, beta, forward_out, grad_output,
        grad_q, grad_k, grad_v, grad_alpha, grad_beta,
        seq_len, num_heads, head_dim
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

namespace {  // anonymous namespace — prevents symbol collision between f32/bf16

ffi::Error fused_kda_scan_backward_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> q,            // [B, T, H, d]
    ffi::Buffer<FFI_IO_TYPE> k,            // [B, T, H, d]
    ffi::Buffer<FFI_IO_TYPE> v,            // [B, T, H, d]
    ffi::Buffer<FFI_IO_TYPE> alpha,        // [B, T, H, d]
    ffi::Buffer<FFI_IO_TYPE> beta,         // [B, T, H]
    ffi::Buffer<FFI_IO_TYPE> forward_out,  // [B, T, H, d]
    ffi::Buffer<FFI_IO_TYPE> grad_output,  // [B, T, H, d]
    ffi::ResultBuffer<FFI_IO_TYPE> grad_q,      // [B, T, H, d]
    ffi::ResultBuffer<FFI_IO_TYPE> grad_k,      // [B, T, H, d]
    ffi::ResultBuffer<FFI_IO_TYPE> grad_v,      // [B, T, H, d]
    ffi::ResultBuffer<FFI_IO_TYPE> grad_alpha,  // [B, T, H, d]
    ffi::ResultBuffer<FFI_IO_TYPE> grad_beta    // [B, T, H]
) {
    auto dims = q.dimensions();
    int batch     = static_cast<int>(dims[0]);
    int seq_len   = static_cast<int>(dims[1]);
    int num_heads = static_cast<int>(dims[2]);
    int head_dim  = static_cast<int>(dims[3]);

    dim3 grid(batch, num_heads);
    dim3 block(head_dim);
    size_t smem_bytes = (size_t)head_dim * head_dim * sizeof(float)
                      + 3 * head_dim * sizeof(float)
                      + sizeof(float);

    fused_kda_scan_backward_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const io_type*>(q.untyped_data()),
        reinterpret_cast<const io_type*>(k.untyped_data()),
        reinterpret_cast<const io_type*>(v.untyped_data()),
        reinterpret_cast<const io_type*>(alpha.untyped_data()),
        reinterpret_cast<const io_type*>(beta.untyped_data()),
        reinterpret_cast<const io_type*>(forward_out.untyped_data()),
        reinterpret_cast<const io_type*>(grad_output.untyped_data()),
        reinterpret_cast<io_type*>(grad_q->untyped_data()),
        reinterpret_cast<io_type*>(grad_k->untyped_data()),
        reinterpret_cast<io_type*>(grad_v->untyped_data()),
        reinterpret_cast<io_type*>(grad_alpha->untyped_data()),
        reinterpret_cast<io_type*>(grad_beta->untyped_data()),
        seq_len, num_heads, head_dim
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

}  // anonymous namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    HANDLER_SYMBOL(fused_kda_scan_backward), fused_kda_scan_backward_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // q
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // k
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // v
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // alpha
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // beta
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // forward_out
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // grad_output
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_q
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_k
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_v
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_alpha
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_beta
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_kda_scan_backward_" PRECISION_SUFFIX, "CUDA",
    HANDLER_SYMBOL(fused_kda_scan_backward));

#endif  // EXLA_FFI
