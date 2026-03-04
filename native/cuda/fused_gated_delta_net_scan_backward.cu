// Fused GatedDeltaNet Scan Backward Kernel
//
// Extends DeltaNet backward with alpha (scalar forget gate per head).
// Forward recurrence:
//   S_gated = alpha_t * S_{t-1}
//   retrieval = S_gated @ k_t
//   error = v_t - retrieval
//   S_t = S_gated + beta_t * outer(error, k_t)
//   o_t = S_t @ q_t
//
// Reverse pass adds gradient through alpha:
//   dS_gated += dS_t (passed through from update)
//   d_alpha_t = sum_{i,j}(dS_gated[i][j] * S_{t-1}[i][j])  — scalar per (b,h) pair
//   dS_{t-1} += alpha_t * dS_gated
//
// Same thread layout as DeltaNet backward: one block per (batch, head).
//
// Inputs:
//   q:           [B, T, H, d]
//   k:           [B, T, H, d]
//   v:           [B, T, H, d]
//   beta:        [B, T, H, d]
//   alpha:       [B, T, H]     — per-head scalar forget gate
//   forward_out: [B, T, H, d]
//   grad_output: [B, T, H, d]
//
// Outputs:
//   grad_q:     [B, T, H, d]
//   grad_k:     [B, T, H, d]
//   grad_v:     [B, T, H, d]
//   grad_beta:  [B, T, H, d]
//   grad_alpha: [B, T, H]

#include <cuda_runtime.h>
#include "precision.cuh"

#define MAX_SEQ_LEN 1024

// ============================================================================
// Kernel
// ============================================================================

#ifdef EXLA_FFI
namespace {  // anonymous namespace — internal linkage per compilation unit
#endif

__global__ void fused_gated_delta_net_scan_backward_kernel(
    const io_type* __restrict__ q,            // [B, T, H, d]
    const io_type* __restrict__ k,            // [B, T, H, d]
    const io_type* __restrict__ v,            // [B, T, H, d]
    const io_type* __restrict__ beta,         // [B, T, H, d]
    const io_type* __restrict__ alpha,        // [B, T, H]
    const io_type* __restrict__ forward_out,  // [B, T, H, d]
    const io_type* __restrict__ grad_output,  // [B, T, H, d]
    io_type* __restrict__ grad_q,             // [B, T, H, d]
    io_type* __restrict__ grad_k,             // [B, T, H, d]
    io_type* __restrict__ grad_v,             // [B, T, H, d]
    io_type* __restrict__ grad_beta,          // [B, T, H, d]
    io_type* __restrict__ grad_alpha,         // [B, T, H]
    int seq_len,
    int num_heads,
    int head_dim
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = threadIdx.x;

    if (i >= head_dim) return;

    extern __shared__ float smem[];
    float* S = smem;                                 // [d][d]
    // smem[d*d..2*d*d-1] reserved for layout alignment
    float* k_shared = smem + 2 * head_dim * head_dim; // [d]
    float* q_shared = k_shared + head_dim;            // [d]
    float* temp_shared = q_shared + head_dim;         // [d]
    float* reduce_shared = temp_shared + head_dim;    // [1] for alpha gradient reduction

    // Strides
    int THd = seq_len * num_heads * head_dim;
    int Hd = num_heads * head_dim;
    int d = head_dim;
    int base_bh = b * THd + h * d;

    // Alpha strides: [B, T, H]
    int alpha_TH = seq_len * num_heads;
    int alpha_base = b * alpha_TH + h;

    // ========================================
    // Pass 1: Forward — recompute S states, store retrieval values
    // ========================================
    for (int j = 0; j < head_dim; j++) {
        S[i * head_dim + j] = 0.0f;
    }
    __syncthreads();

    float local_retrieval[MAX_SEQ_LEN];

    for (int t = 0; t < seq_len; t++) {
        int offset = base_bh + t * Hd;
        float alpha_val = IO_LOAD(alpha, alpha_base + t * num_heads);

        // Apply alpha decay
        for (int j = 0; j < head_dim; j++) {
            S[i * head_dim + j] *= alpha_val;
        }
        __syncthreads();

        k_shared[i] = IO_LOAD(k, offset + i);
        __syncthreads();

        float retrieval = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            retrieval += S[i * head_dim + j] * k_shared[j];
        }
        local_retrieval[t] = retrieval;

        float v_i = IO_LOAD(v, offset + i);
        float beta_i = IO_LOAD(beta, offset + i);
        float error_i = v_i - retrieval;
        float scaled_error_i = beta_i * error_i;

        for (int j = 0; j < head_dim; j++) {
            S[i * head_dim + j] += scaled_error_i * k_shared[j];
        }
        __syncthreads();
    }

    // ========================================
    // Pass 2: Backward
    // ========================================
    float dS_row[128];
    for (int j = 0; j < head_dim; j++) {
        dS_row[j] = 0.0f;
    }

    for (int t = seq_len - 1; t >= 0; t--) {
        int offset = base_bh + t * Hd;

        float q_i = IO_LOAD(q, offset + i);
        float k_i = IO_LOAD(k, offset + i);
        float v_i = IO_LOAD(v, offset + i);
        float beta_i = IO_LOAD(beta, offset + i);
        float do_i = IO_LOAD(grad_output, offset + i);
        float alpha_val = IO_LOAD(alpha, alpha_base + t * num_heads);

        float retrieval_i = local_retrieval[t];
        float error_i = v_i - retrieval_i;
        float scaled_err_i = beta_i * error_i;

        // ---- grad_q: dq = S_t^T @ do ----
        q_shared[i] = do_i;
        __syncthreads();

        float dq_i = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            dq_i += S[j * head_dim + i] * q_shared[j];
        }
        IO_STORE(grad_q, offset + i, dq_i);

        // ---- dS from output: dS += outer(do, q) ----
        k_shared[i] = q_i;
        __syncthreads();

        for (int j = 0; j < head_dim; j++) {
            dS_row[j] += do_i * k_shared[j];
        }

        // ---- Gradients from update S_t = S_gated + beta*outer(error, k) ----
        k_shared[i] = k_i;
        __syncthreads();

        float d_beta_error_i = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            d_beta_error_i += dS_row[j] * k_shared[j];
        }

        // dk from outer product
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

        float d_error_i = d_beta_error_i * beta_i;
        float dv_i = d_error_i;
        float d_retrieval_i = -d_error_i;

        for (int j = 0; j < head_dim; j++) {
            dS_row[j] += d_retrieval_i * k_shared[j];
        }

        // Undo update to get S_gated
        for (int j = 0; j < head_dim; j++) {
            S[i * head_dim + j] -= scaled_err_i * k_shared[j];
        }
        __syncthreads();

        // dk from retrieval (S_gated @ k)
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

        float dk_i = temp_shared[i] + q_shared[i];

        float d_beta_i = d_beta_error_i * error_i;

        IO_STORE(grad_k, offset + i, dk_i);
        IO_STORE(grad_v, offset + i, dv_i);
        IO_STORE(grad_beta, offset + i, d_beta_i);

        // ---- Gradient through alpha decay: S_gated = alpha * S_{t-1} ----
        // dS_{t-1}[i][j] = alpha * dS_gated[i][j]
        // d_alpha = sum_{i,j}(dS_gated[i][j] * S_{t-1}[i][j])
        //
        // Current S holds S_gated. S_{t-1} = S_gated / alpha.
        // d_alpha partial sum from thread i:
        float d_alpha_partial = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            // S_{t-1}[i][j] = S_gated[i][j] / alpha
            float s_prev_ij = (alpha_val != 0.0f) ? S[i * head_dim + j] / alpha_val : 0.0f;
            d_alpha_partial += dS_row[j] * s_prev_ij;
        }

        // Reduce d_alpha across threads
        if (i == 0) reduce_shared[0] = 0.0f;
        __syncthreads();
        atomicAdd(reduce_shared, d_alpha_partial);
        __syncthreads();

        // Only thread 0 writes grad_alpha
        if (i == 0) {
            IO_STORE(grad_alpha, alpha_base + t * num_heads, reduce_shared[0]);
        }

        // dS propagated through alpha: dS_{t-1} += alpha * dS_gated
        // (dS_row already contains dS_gated for the next iteration)
        // We need to scale it by alpha for the contribution through alpha decay
        for (int j = 0; j < head_dim; j++) {
            dS_row[j] *= alpha_val;
        }

        // Undo alpha decay to get S_{t-1}
        if (alpha_val != 0.0f) {
            float inv_alpha = 1.0f / alpha_val;
            for (int j = 0; j < head_dim; j++) {
                S[i * head_dim + j] *= inv_alpha;
            }
        }
        __syncthreads();
    }
}

// ============================================================================
// Standalone launch wrapper
// ============================================================================

#ifdef EXLA_FFI
}  // anonymous namespace
#endif

#ifndef EXLA_FFI

extern "C" {

// Output: concat [grad_q | grad_k | grad_v | grad_beta (each B*T*H*d) | grad_alpha (B*T*H)]
int fused_gated_delta_net_scan_backward_launch(
    cudaStream_t stream,
    const io_type* q, const io_type* k, const io_type* v, const io_type* beta,
    const io_type* alpha, const io_type* forward_out, const io_type* grad_output,
    io_type* output_concat,
    int batch, int seq_len, int num_heads, int head_dim
) {
    int total_4d = batch * seq_len * num_heads * head_dim;
    io_type* grad_q     = output_concat;
    io_type* grad_k     = output_concat + total_4d;
    io_type* grad_v     = output_concat + 2 * total_4d;
    io_type* grad_beta  = output_concat + 3 * total_4d;
    io_type* grad_alpha = output_concat + 4 * total_4d;

    dim3 grid(batch, num_heads);
    dim3 block(head_dim);

    // 2*S[d][d] + k_shared[d] + q_shared[d] + temp_shared[d] + reduce[1]
    size_t smem_bytes = 2 * (size_t)head_dim * head_dim * sizeof(float)
                      + 3 * head_dim * sizeof(float)
                      + sizeof(float);

    fused_gated_delta_net_scan_backward_kernel<<<grid, block, smem_bytes, stream>>>(
        q, k, v, beta, alpha, forward_out, grad_output,
        grad_q, grad_k, grad_v, grad_beta, grad_alpha,
        seq_len, num_heads, head_dim
    );

    return (int)cudaGetLastError();
}

}  // extern "C"

#endif  // !EXLA_FFI

// ============================================================================
// XLA FFI integration
// ============================================================================

#ifdef EXLA_FFI

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {  // anonymous namespace — prevents symbol collision between f32/bf16

ffi::Error fused_gated_delta_net_scan_backward_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> q,
    ffi::Buffer<FFI_IO_TYPE> k,
    ffi::Buffer<FFI_IO_TYPE> v,
    ffi::Buffer<FFI_IO_TYPE> beta,
    ffi::Buffer<FFI_IO_TYPE> alpha,
    ffi::Buffer<FFI_IO_TYPE> forward_out,
    ffi::Buffer<FFI_IO_TYPE> grad_output,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_q,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_k,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_v,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_beta,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_alpha
) {
    auto dims = q.dimensions();
    int batch     = static_cast<int>(dims[0]);
    int seq_len   = static_cast<int>(dims[1]);
    int num_heads = static_cast<int>(dims[2]);
    int head_dim  = static_cast<int>(dims[3]);

    dim3 grid(batch, num_heads);
    dim3 block(head_dim);
    size_t smem_bytes = 2 * (size_t)head_dim * head_dim * sizeof(float)
                      + 3 * head_dim * sizeof(float)
                      + sizeof(float);

    fused_gated_delta_net_scan_backward_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const io_type*>(q.untyped_data()),
        reinterpret_cast<const io_type*>(k.untyped_data()),
        reinterpret_cast<const io_type*>(v.untyped_data()),
        reinterpret_cast<const io_type*>(beta.untyped_data()),
        reinterpret_cast<const io_type*>(alpha.untyped_data()),
        reinterpret_cast<const io_type*>(forward_out.untyped_data()),
        reinterpret_cast<const io_type*>(grad_output.untyped_data()),
        reinterpret_cast<io_type*>(grad_q->untyped_data()),
        reinterpret_cast<io_type*>(grad_k->untyped_data()),
        reinterpret_cast<io_type*>(grad_v->untyped_data()),
        reinterpret_cast<io_type*>(grad_beta->untyped_data()),
        reinterpret_cast<io_type*>(grad_alpha->untyped_data()),
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
    HANDLER_SYMBOL(fused_gated_delta_net_scan_backward), fused_gated_delta_net_scan_backward_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // q
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // k
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // v
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // beta
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // alpha
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // forward_out
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // grad_output
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_q
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_k
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_v
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_beta
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_alpha
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_gated_delta_net_scan_backward_" PRECISION_SUFFIX, "CUDA",
    HANDLER_SYMBOL(fused_gated_delta_net_scan_backward));

#endif  // EXLA_FFI
