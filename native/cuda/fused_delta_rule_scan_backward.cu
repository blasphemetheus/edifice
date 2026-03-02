// Fused DeltaNet Scan Backward Kernel
//
// Computes gradients for the delta rule matrix-state recurrence:
//   retrieval = S_{t-1} @ k_t
//   error = v_t - retrieval
//   S_t = S_{t-1} + beta_t * outer(error, k_t)
//   o_t = S_t @ q_t
//
// Two-pass approach:
//   Pass 1 (forward, t=0..T-1): Recompute S_t, store per-timestep S snapshots.
//   Pass 2 (reverse, t=T-1..0): Reverse accumulate dS and compute per-input grads.
//
// Reverse pass math (for vanilla DeltaNet, alpha=NULL):
//   do_t = grad_output[t]   (already in [B,T,H,d] format)
//
//   // dS from output: o = S @ q, so dS += outer(do, q)
//   dS[i][j] += do[i] * q_t[j]
//
//   // grad_q: dq[j] = S_t^T @ do = sum_i(S_t[i][j] * do[i])
//   // But thread i owns row i, so we need cross-thread reduction.
//   // Instead: dq[i] = sum_j(S_t[j][i] * do[j]) — same issue.
//   // Solution: thread i computes sum_j(S[i][j] * do[j]) ... wait, no.
//   // Actually dq = S^T @ do. Thread i can contribute: S[i][*] * do[i].
//   // dq[j] += S[i][j] * do[i]  — needs atomicAdd across threads for each j.
//
//   // dS from update: S_t = S_{t-1} + beta*outer(error, k)
//   // so dS_{t-1} = dS_t, d(beta*error*k) = dS_t
//   // d_beta_error[i] = sum_j(dS[i][j] * k[j])  — dot product of dS row with k
//   // d_k[j] += sum_i(dS[i][j] * beta[i]*error[i])  — cross-thread, use atomicAdd
//   // d_error[i] = d_beta_error[i] (when expanded: d_beta = ...)
//   // error = v - S_{t-1}@k, so dv[i] += d_error[i]*beta[i],
//   //   d(S_{t-1}@k) contributes back to dS_{t-1} and dk
//
// Thread layout: one block per (batch, head), head_dim threads per block.
// Each thread owns one row of dS[d][d].
//
// Inputs:
//   q:           [B, T, H, d] — query vectors (same as forward)
//   k:           [B, T, H, d] — key vectors (L2-normalized, same as forward)
//   v:           [B, T, H, d] — value vectors (same as forward)
//   beta:        [B, T, H, d] — update gate (post-sigmoid, same as forward)
//   forward_out: [B, T, H, d] — forward pass outputs
//   grad_output: [B, T, H, d] — upstream gradient dL/do
//
// Outputs:
//   grad_q:    [B, T, H, d]
//   grad_k:    [B, T, H, d]
//   grad_v:    [B, T, H, d]
//   grad_beta: [B, T, H, d]

#include <cuda_runtime.h>
#include "precision.cuh"

#define MAX_SEQ_LEN 1024

// ============================================================================
// Kernel
// ============================================================================

__global__ void fused_delta_rule_scan_backward_kernel(
    const io_type* __restrict__ q,            // [B, T, H, d]
    const io_type* __restrict__ k,            // [B, T, H, d]
    const io_type* __restrict__ v,            // [B, T, H, d]
    const io_type* __restrict__ beta,         // [B, T, H, d]
    const io_type* __restrict__ forward_out,  // [B, T, H, d]
    const io_type* __restrict__ grad_output,  // [B, T, H, d]
    io_type* __restrict__ grad_q,             // [B, T, H, d]
    io_type* __restrict__ grad_k,             // [B, T, H, d]
    io_type* __restrict__ grad_v,             // [B, T, H, d]
    io_type* __restrict__ grad_beta,          // [B, T, H, d]
    int seq_len,
    int num_heads,
    int head_dim
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = threadIdx.x;

    if (i >= head_dim) return;

    // Shared memory: S[d][d] + k_shared[d] + q_shared[d] + temp[d]
    extern __shared__ float smem[];
    float* S = smem;                                 // [d][d] — state matrix
    float* k_shared = smem + head_dim * head_dim;    // [d]
    float* q_shared = k_shared + head_dim;           // [d]
    float* temp_shared = q_shared + head_dim;        // [d] — for cross-thread reductions

    // Strides for [B, T, H, d]
    int THd = seq_len * num_heads * head_dim;
    int Hd = num_heads * head_dim;
    int d = head_dim;
    int base_bh = b * THd + h * d;

    // ========================================
    // Pass 1: Forward — recompute S at each timestep
    // Store per-timestep S snapshots in local memory (S before update at each t)
    // ========================================

    // Initialize S to zero
    for (int j = 0; j < head_dim; j++) {
        S[i * head_dim + j] = 0.0f;
    }
    __syncthreads();

    // Store S snapshots: S_prev[t] = S before timestep t's update
    // Each thread stores its row: local_S[t][j] for j in 0..d-1
    // This is d*T floats per thread — stored in local memory (GPU L1/L2)
    // We'll do a single reverse pass that recomputes S from forward outputs.

    // Actually, let's use a simpler approach: store per-timestep retrieval values
    // and reconstruct what we need. For the delta rule backward, we need S_{t} at each
    // timestep during the reverse pass to compute grad_q.
    //
    // Approach: Two passes. Forward builds S. Reverse uses the saved forward_out
    // and recomputes needed quantities.
    //
    // For grad_q: dq_t = S_t^T @ do_t. We need S_t during reverse.
    // Keeping S in shared memory: we can do a reverse scan of S by undoing updates.
    //
    // Reverse S reconstruction: S_{t-1} = S_t - beta_t * outer(error_t, k_t)
    // where error_t = v_t - S_{t-1}@k_t. But error_t depends on S_{t-1}!
    //
    // Alternative: Run forward pass fully to get S_T, then reverse:
    //   At each t (reverse), first compute grad_q using current S_t,
    //   then undo the update to get S_{t-1}.
    //
    // Undo update: S_{t-1} = S_t - beta_t * outer(v_t - S_{t-1}@k_t, k_t)
    // This is implicit because error depends on S_{t-1}. But:
    //   S_t = S_{t-1} + beta_t * outer(v_t - S_{t-1}@k_t, k_t)
    //       = S_{t-1} + beta_t * (v_t * k_t^T - (S_{t-1}@k_t) * k_t^T)
    //   S_t = S_{t-1} * (I - beta_t * k_t * k_t^T) + beta_t * v_t * k_t^T
    //
    // Inverting: S_{t-1} = (S_t - beta_t * v_t * k_t^T) * (I - beta_t * k_t * k_t^T)^{-1}
    //
    // For ||k||=1: (I - beta*k*k^T)^{-1} = I + (beta/(1-beta)) * k*k^T
    // when beta != 1. This is the Sherman-Morrison identity.
    //
    // SIMPLER APPROACH: Just run the forward pass to build final S_T, then during
    // the backward pass at each timestep t (from T-1 to 0), compute grads using
    // S_t, then reconstruct S_{t-1} using the Sherman-Morrison inverse.
    //
    // But Sherman-Morrison requires beta < 1 strictly, which may not hold.
    //
    // MOST PRACTICAL: Run forward pass fully. Then reverse pass with dS accumulator.
    // At each t (reverse), we need S_t for grad_q computation. We store retrieval
    // values from the forward pass to reconstruct error_t.

    // Forward pass to build final S_T
    // Store per-timestep error values for the backward pass
    float local_retrieval[MAX_SEQ_LEN];  // retrieval_i at each t (per-thread element)

    for (int t = 0; t < seq_len; t++) {
        int offset = base_bh + t * Hd;

        k_shared[i] = IO_LOAD(k, offset + i);
        __syncthreads();

        // retrieval[i] = S[i,:] @ k
        float retrieval = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            retrieval += S[i * head_dim + j] * k_shared[j];
        }

        local_retrieval[t] = retrieval;

        // Update S: S += beta * outer(v - retrieval, k)
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
    // Pass 2: Backward — reverse accumulate dS
    // ========================================
    // dS accumulator (thread i owns row i)
    float dS_row[128];  // head_dim <= 128
    for (int j = 0; j < head_dim; j++) {
        dS_row[j] = 0.0f;
    }

    for (int t = seq_len - 1; t >= 0; t--) {
        int offset = base_bh + t * Hd;

        // Load current timestep values
        float q_i = IO_LOAD(q, offset + i);
        float k_i = IO_LOAD(k, offset + i);
        float v_i = IO_LOAD(v, offset + i);
        float beta_i = IO_LOAD(beta, offset + i);
        float do_i = IO_LOAD(grad_output, offset + i);

        // Reconstruct error from forward pass
        float retrieval_i = local_retrieval[t];
        float error_i = v_i - retrieval_i;

        // ---- grad_q: dq = S_t^T @ do ----
        // S_t is current S in shared memory.
        // dq[i] = sum_j(S[j][i] * do[j]) = column i of S dotted with do.
        // Thread j has S[j][i], so thread j contributes S[j][i] * do[j].
        // We need to reduce across threads for each column i.
        //
        // Approach: thread i computes sum_j(S[i][j] * do[j]) — but that gives (S @ do)[i], not (S^T @ do)[i].
        // We need S^T @ do, which is: for each i, sum_j S[j][i] * do[j].
        //
        // Use shared memory: load do into shared, then thread i reads S[j][i] from other threads' rows.
        // But S is in shared memory as S[row][col], so S[j][i] = smem[j*d + i].
        q_shared[i] = do_i;
        __syncthreads();

        float dq_i = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            dq_i += S[j * head_dim + i] * q_shared[j];
        }

        IO_STORE(grad_q, offset + i, dq_i);

        // ---- dS from output: o = S @ q, so dS += outer(do, q) ----
        k_shared[i] = q_i;  // reuse k_shared to hold q values
        __syncthreads();

        for (int j = 0; j < head_dim; j++) {
            dS_row[j] += do_i * k_shared[j];
        }

        // ---- Gradients from the update S_t = S_{t-1} + beta*outer(error, k) ----
        // dS_{t-1} gets dS_t (passed through)
        // d(beta*error)[i] = sum_j(dS[i][j] * k[j])  — dot of dS row with k
        k_shared[i] = k_i;
        __syncthreads();

        float d_beta_error_i = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            d_beta_error_i += dS_row[j] * k_shared[j];
        }

        // d_k[j] = sum_i(dS[i][j] * beta[i]*error[i]) — cross-thread, use temp_shared
        // Each thread i contributes: dS[i][j] * beta_i * error_i to d_k[j]
        // Use atomicAdd approach
        if (i == 0) {
            for (int j = 0; j < head_dim; j++) {
                temp_shared[j] = 0.0f;
            }
        }
        __syncthreads();

        float scaled_err_i = beta_i * error_i;
        for (int j = 0; j < head_dim; j++) {
            atomicAdd(&temp_shared[j], dS_row[j] * scaled_err_i);
        }
        __syncthreads();

        // d_k also has contribution from retrieval gradient
        // error = v - S_{t-1}@k, and d_error[i] = d_beta_error_i / beta_i (if beta != 0)
        // Actually: d(beta*error)[i] = d_beta_error_i
        //   d_beta = sum_i(d_beta_error_i * error_i)  — scalar, but beta is per-element
        //   Actually beta is [B,T,H,d], so d_beta[i] = d_beta_error_i * error_i
        //   d_error[i] = d_beta_error_i * beta_i
        float d_error_i = d_beta_error_i * beta_i;

        // error = v - retrieval, so dv += d_error, d_retrieval -= d_error
        float dv_i = d_error_i;
        float d_retrieval_i = -d_error_i;

        // retrieval = S_{t-1} @ k, so:
        //   dS_{t-1}[i][j] += d_retrieval[i] * k[j]  (absorbed into dS)
        //   dk[j] += sum_i(S_{t-1}[i][j] * d_retrieval[i])  — cross-thread
        // The dS contribution is already handled by passing dS through
        // But we need to add the retrieval gradient to dS:
        for (int j = 0; j < head_dim; j++) {
            dS_row[j] += d_retrieval_i * k_shared[j];
        }

        // dk from retrieval: sum_i(S_{t-1}[i][j] * d_retrieval[i])
        // We need S_{t-1}, which is S_t minus the update.
        // Undo: S_{t-1} = S_t - beta*outer(error, k)
        // First undo the update to get S_{t-1}
        for (int j = 0; j < head_dim; j++) {
            S[i * head_dim + j] -= scaled_err_i * k_shared[j];
        }
        __syncthreads();

        // Now S holds S_{t-1}. Compute dk from retrieval.
        // dk_retrieval[j] = sum_i(S_{t-1}[i][j] * d_retrieval[i])
        // Thread i contributes S_{t-1}[i][j] * d_retrieval_i for each j
        if (i == 0) {
            for (int j = 0; j < head_dim; j++) {
                q_shared[j] = 0.0f;  // reuse as dk_retrieval accumulator
            }
        }
        __syncthreads();

        for (int j = 0; j < head_dim; j++) {
            atomicAdd(&q_shared[j], S[i * head_dim + j] * d_retrieval_i);
        }
        __syncthreads();

        // Total dk = dk from update (temp_shared) + dk from retrieval (q_shared)
        float dk_i = temp_shared[i] + q_shared[i];

        // grad_beta[i] = d_beta_error_i * error_i
        float d_beta_i = d_beta_error_i * error_i;

        IO_STORE(grad_k, offset + i, dk_i);
        IO_STORE(grad_v, offset + i, dv_i);
        IO_STORE(grad_beta, offset + i, d_beta_i);
        __syncthreads();
    }
}

// ============================================================================
// Standalone launch wrapper (C-linkage for NIF / dlopen)
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

// Output: concatenated [grad_q | grad_k | grad_v | grad_beta] each [B*T*H*d]
int fused_delta_rule_scan_backward_launch(
    cudaStream_t stream,
    const io_type* q, const io_type* k, const io_type* v, const io_type* beta,
    const io_type* forward_out, const io_type* grad_output,
    io_type* output_concat,
    int batch, int seq_len, int num_heads, int head_dim
) {
    int total = batch * seq_len * num_heads * head_dim;
    io_type* grad_q    = output_concat;
    io_type* grad_k    = output_concat + total;
    io_type* grad_v    = output_concat + 2 * total;
    io_type* grad_beta = output_concat + 3 * total;

    dim3 grid(batch, num_heads);
    dim3 block(head_dim);

    // S[d][d] + k_shared[d] + q_shared[d] + temp_shared[d]
    size_t smem_bytes = (size_t)head_dim * head_dim * sizeof(float)
                      + 3 * head_dim * sizeof(float);

    fused_delta_rule_scan_backward_kernel<<<grid, block, smem_bytes, stream>>>(
        q, k, v, beta, forward_out, grad_output,
        grad_q, grad_k, grad_v, grad_beta,
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

ffi::Error fused_delta_rule_scan_backward_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> q,
    ffi::Buffer<FFI_IO_TYPE> k,
    ffi::Buffer<FFI_IO_TYPE> v,
    ffi::Buffer<FFI_IO_TYPE> beta,
    ffi::Buffer<FFI_IO_TYPE> forward_out,
    ffi::Buffer<FFI_IO_TYPE> grad_output,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_q,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_k,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_v,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_beta
) {
    auto dims = q.dimensions();
    int batch     = static_cast<int>(dims[0]);
    int seq_len   = static_cast<int>(dims[1]);
    int num_heads = static_cast<int>(dims[2]);
    int head_dim  = static_cast<int>(dims[3]);

    dim3 grid(batch, num_heads);
    dim3 block(head_dim);
    size_t smem_bytes = (size_t)head_dim * head_dim * sizeof(float)
                      + 3 * head_dim * sizeof(float);

    fused_delta_rule_scan_backward_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const io_type*>(q.untyped_data()),
        reinterpret_cast<const io_type*>(k.untyped_data()),
        reinterpret_cast<const io_type*>(v.untyped_data()),
        reinterpret_cast<const io_type*>(beta.untyped_data()),
        reinterpret_cast<const io_type*>(forward_out.untyped_data()),
        reinterpret_cast<const io_type*>(grad_output.untyped_data()),
        reinterpret_cast<io_type*>(grad_q->untyped_data()),
        reinterpret_cast<io_type*>(grad_k->untyped_data()),
        reinterpret_cast<io_type*>(grad_v->untyped_data()),
        reinterpret_cast<io_type*>(grad_beta->untyped_data()),
        seq_len, num_heads, head_dim
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_delta_rule_scan_backward, fused_delta_rule_scan_backward_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // q
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // k
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // v
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // beta
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // forward_out
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // grad_output
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_q
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_k
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_v
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_beta
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_delta_rule_scan_backward_" PRECISION_SUFFIX, "CUDA", fused_delta_rule_scan_backward);

#endif  // EXLA_FFI
