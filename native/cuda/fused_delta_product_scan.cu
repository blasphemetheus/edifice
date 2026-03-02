// Fused DeltaProduct Scan Kernel
//
// Extends DeltaNet with multiple Householder transformation steps per token.
// At each timestep t, applies n_h sequential rank-1 updates to the state matrix:
//
//   For j in 0..n_h-1:
//     k_norm = k_{t,j} / ||k_{t,j}||_2        (L2 normalize key)
//     S = S - beta_{t,j} * (k_norm * k_norm^T @ S) + beta_{t,j} * (k_norm * v_{t,j}^T)
//
//   o_t = RMS_norm(S_t @ q_t)
//
// This is equivalent to: S = (I - beta * k*k^T) * S + beta * k*v^T
// which is a generalized Householder reflection + rank-1 update.
//
// Thread layout: one block per (batch, head), head_dim threads per block.
// Each thread owns one row of S[head_dim][head_dim] in shared memory.
//
// Inputs:
//   q:    [B, T, H, d]         — query vectors (shared across Householder steps)
//   k:    [B, T, n_h, H, d]    — key vectors per Householder step (pre-normalization)
//   v:    [B, T, n_h, H, d]    — value vectors per Householder step
//   beta: [B, T, n_h, H]       — scalar gate per head per step (post-sigmoid)
//
// Output:
//   out:  [B, T, H, d]         — RMS-normalized output
//
// Shared memory budget (head_dim=64):
//   S matrix:  64*64*4 = 16KB
//   k_shared:  64*4    = 256B
//   q_shared:  64*4    = 256B
//   rms_buf:   1*4     = 4B   (for warp reduction)
//   Total: ~17KB — well within 48KB limit

#include <cuda_runtime.h>
#include "precision.cuh"

constexpr float NORM_EPS = 1.0e-6f;

// ============================================================================
// Kernel
// ============================================================================

__global__ void fused_delta_product_scan_kernel(
    const io_type* __restrict__ q,       // [B, T, H, d]
    const io_type* __restrict__ k,       // [B, T, n_h, H, d]
    const io_type* __restrict__ v,       // [B, T, n_h, H, d]
    const io_type* __restrict__ beta,    // [B, T, n_h, H]
    io_type* __restrict__ output,        // [B, T, H, d]
    int seq_len,
    int num_householder,
    int num_heads,
    int head_dim
) {
    int b = blockIdx.x;   // batch index
    int h = blockIdx.y;   // head index
    int i = threadIdx.x;  // row index in S (0..head_dim-1)

    if (i >= head_dim) return;

    // Shared memory layout
    extern __shared__ float smem[];
    float* S = smem;                                    // [d][d]
    float* k_shared = smem + head_dim * head_dim;       // [d]
    float* q_shared = k_shared + head_dim;              // [d]
    float* rms_shared = q_shared + head_dim;            // [1] for RMS reduction

    // Initialize S to zero
    for (int j = 0; j < head_dim; j++) {
        S[i * head_dim + j] = 0.0f;
    }
    __syncthreads();

    // Strides for q: [B, T, H, d]
    int q_stride_B = seq_len * num_heads * head_dim;
    int q_stride_T = num_heads * head_dim;
    int q_stride_H = head_dim;

    // Strides for k/v: [B, T, n_h, H, d]
    int kv_stride_B = seq_len * num_householder * num_heads * head_dim;
    int kv_stride_T = num_householder * num_heads * head_dim;
    int kv_stride_J = num_heads * head_dim;
    int kv_stride_H = head_dim;

    // Strides for beta: [B, T, n_h, H]
    int beta_stride_B = seq_len * num_householder * num_heads;
    int beta_stride_T = num_householder * num_heads;
    int beta_stride_J = num_heads;

    int q_base = b * q_stride_B + h * q_stride_H;
    int kv_base = b * kv_stride_B + h * kv_stride_H;
    int beta_base = b * beta_stride_B + h;

    for (int t = 0; t < seq_len; t++) {
        // Apply n_h Householder updates
        for (int j = 0; j < num_householder; j++) {
            int kv_offset = kv_base + t * kv_stride_T + j * kv_stride_J;
            int beta_offset = beta_base + t * beta_stride_T + j * beta_stride_J;

            // Load k_{t,j}[i] into shared + register
            float k_i = IO_LOAD(k, kv_offset + i);
            k_shared[i] = k_i;
            __syncthreads();

            // L2 normalize k: compute ||k||^2 via shared memory reduction
            // Each thread computes k_i^2, then we sum across threads
            float k_sq = k_i * k_i;

            // Simple shared memory reduction for L2 norm
            // Reuse rms_shared buffer for the partial sum
            // Use atomicAdd since head_dim may not be power of 2
            if (i == 0) rms_shared[0] = 0.0f;
            __syncthreads();
            atomicAdd(rms_shared, k_sq);
            __syncthreads();

            float k_norm_inv = rsqrtf(rms_shared[0] + NORM_EPS);
            float k_normed_i = k_i * k_norm_inv;
            k_shared[i] = k_normed_i;  // Store normalized k
            __syncthreads();

            float beta_val = IO_LOAD(beta, beta_offset);

            // S = S - beta * (k*k^T @ S) + beta * (k*v^T)
            //
            // For row i of S:
            //   S[i][j] -= beta * k[i] * sum_l(k[l] * S[l][j])   (k*k^T @ S term)
            //   S[i][j] += beta * k[i] * v[j]                      (k*v^T term)
            //
            // But k*k^T @ S requires reading column j across all rows → needs sync.
            // Instead: retrieval[i] = sum_j(S[i][j] * k[j]), then:
            //   S[i][j] = S[i][j] - beta * k[i] * (S @ k)[i]'s contribution... no.
            //
            // Let's think about this differently:
            //   (k*k^T @ S)[i][j] = k[i] * sum_l(k[l] * S[l][j])
            //
            // We need sum_l(k[l] * S[l][j]) for each column j.
            // Each thread i has row S[i][*]. Thread i contributes k[i]*S[i][j] to column j's sum.
            // This requires a cross-thread reduction per column — expensive.
            //
            // Alternative formulation:
            //   S_new = S - beta*(k @ k^T @ S) + beta*(k @ v^T)
            //         = S + beta*k @ (v^T - k^T @ S)
            //         = S + beta*k @ (v - S^T @ k)^T
            //
            // Let error = v - S^T @ k (same as DeltaNet!)
            // S_new[i][j] = S[i][j] + beta * k[i] * error[j]
            //
            // But error[j] = v[j] - sum_l(S[l][j] * k[l]) = v[j] - (S^T @ k)[j]
            //
            // Since each thread owns row i, computing S^T @ k needs:
            //   (S^T @ k)[j] = sum_i(S[i][j] * k[i])
            // Each thread can compute its contribution: S[i][j] * k[i] for all j,
            // then reduce across threads.
            //
            // Simpler approach using the thread-per-row layout:
            // Thread i computes: retrieval_i = sum_j(S[i][j] * k[j])  (S @ k)
            // Then:
            //   S_new[i][j] = S[i][j] - beta * k_normed[i] * retrieval_i * k_normed[j]
            //                          + beta * k_normed[i] * v[j]
            //              = S[i][j] + beta * k_normed[i] * (v[j] - retrieval_i * k_normed[j])
            //
            // Wait — this isn't right either. Let me reconsider:
            // (I - beta*k*k^T) @ S means we apply (I - beta*k*k^T) to each COLUMN of S.
            // For column j: S_new[:,j] = S[:,j] - beta*(k*k^T)@S[:,j]
            //             = S[:,j] - beta*k*(k^T @ S[:,j])
            //             = S[:,j] - beta*k * sum_l(k[l]*S[l,j])
            //
            // So S_new[i][j] = S[i][j] - beta*k[i]*sum_l(k[l]*S[l,j]) + beta*k[i]*v[j]
            //
            // The sum_l(k[l]*S[l,j]) is a dot product of k with column j of S.
            // Thread i owns row i, so it knows S[i,j] for all j.
            // For column j, we need contributions from all threads (all rows).
            //
            // Efficient approach: each thread i computes partial = k[i] * S[i,j] for each j,
            // then we need sum across i. This is a parallel reduction.
            //
            // With head_dim threads, we can do this column-by-column but that's O(d^2) syncs.
            //
            // Better: Thread i computes retrieval_i = S[i,:] @ k = sum_j(S[i,j]*k[j]).
            // Then S^T @ k has element j = sum_i(S[i,j]*k[i]).
            // But S[i,:] @ k is NOT the same as (S^T @ k)[i].
            //
            // S @ k gives us: (S@k)[i] = sum_j(S[i,j]*k[j]) — thread i can compute this.
            // S^T @ k gives us: (S^T@k)[j] = sum_i(S[i,j]*k[i]) — needs cross-thread reduction.
            //
            // The Householder update needs (k*k^T) @ S, which applied to column j:
            //   ((k*k^T) @ S)[:,j] = k * (k^T @ S[:,j])
            //
            // k^T @ S[:,j] = sum_i(k[i]*S[i,j]) — this is (S^T @ k)[j], same reduction issue.
            //
            // SOLUTION: Use the reformulation from the DeltaNet kernel!
            //   S_new = S + beta * outer(k, v - S^T@k)
            //
            // But S^T@k requires cross-thread communication.
            //
            // ALTERNATIVE: Store S in COLUMN-major in shared memory.
            // Thread i then owns COLUMN i of S: S[:,i].
            // Then (S^T @ k)[i] = sum_j(S[j,i]*k[j]) = S[:,i] dot k = thread i can compute!
            //
            // Let me use column-major storage:
            // S stored as S_col[col][row], thread i owns column i.

            // Using row-major with a different formulation:
            // Actually let's just compute S@k (each thread does its own row dot product),
            // then share the result, and use it.
            //
            // For the Householder reflector (I - beta*k*k^T) applied to S:
            //   S_new = S - beta * k * k^T @ S + beta * k * v^T
            //
            // Row i of S_new:
            //   S_new[i,:] = S[i,:] - beta*k[i]*(k^T @ S)[1,:] ... no, this is wrong too.
            //
            //   (k*k^T @ S)[i,j] = k[i] * (k^T @ S[*,j])
            //                     = k[i] * sum_l(k[l] * S[l,j])
            //
            // Each thread i can compute its OWN contribution to (k^T @ S)[for each col j]:
            //   contrib_i_j = k[i] * S[i,j]
            // Then (k^T @ S)[*,j] = sum_i(contrib_i_j)
            //
            // This is equivalent to: for each j, reduce k[i]*S[i,j] across all threads i.
            // That's d reductions of d values each = O(d^2 log d) total. Too expensive.
            //
            // PRACTICAL APPROACH: Just compute S@k per thread (O(d) per thread),
            // this gives us (S@k)[i]. Then use:
            //
            // Actually, let me reconsider the math:
            //   S = (I - beta*k*k^T) @ S + beta*k*v^T
            //
            // Row i: S_new[i,:] = S[i,:] - beta*k[i]*(sum_l k[l]*S[l,:]) + beta*k[i]*v[:]
            //
            // The term sum_l(k[l]*S[l,:]) is a weighted sum of ALL rows of S.
            // This is fundamentally a cross-row operation.
            //
            // MOST PRACTICAL: Use atomicAdd to accumulate k[i]*S[i,j] into shared buffer.
            // We need a shared buffer of size head_dim for the result.

            // Phase 1: Compute S^T @ k via shared memory accumulation
            // (S^T @ k)[j] = sum_i(S[i,j] * k[i])
            // Reuse q_shared as temp buffer for this
            // First zero it out
            if (i == 0) {
                for (int jj = 0; jj < head_dim; jj++) {
                    q_shared[jj] = 0.0f;
                }
            }
            __syncthreads();

            // Each thread i atomically adds k_normed[i] * S[i,j] for all j
            for (int jj = 0; jj < head_dim; jj++) {
                atomicAdd(&q_shared[jj], k_normed_i * S[i * head_dim + jj]);
            }
            __syncthreads();

            // Now q_shared[j] = (S^T @ k)[j]
            // S_new[i][j] = S[i][j] - beta * k[i] * (S^T @ k)[j] + beta * k[i] * v[j]
            //             = S[i][j] + beta * k[i] * (v[j] - (S^T @ k)[j])
            // Load v into shared for the update (but we need v[j] not v[i])
            // Thread i can only write v[i] to shared, then read v[j] from shared
            // Reuse another shared buffer... but we're running low.
            // Alternative: each thread reads v[j] directly from global memory.
            // This is fine — v is only read once per Householder step.
            float beta_k_i = beta_val * k_normed_i;
            for (int jj = 0; jj < head_dim; jj++) {
                float v_j = IO_LOAD(v, kv_offset + jj);
                S[i * head_dim + jj] += beta_k_i * (v_j - q_shared[jj]);
            }
            __syncthreads();
        }

        // Output: o_t = S @ q_t with RMS normalization
        int q_offset = q_base + t * q_stride_T;

        // Load q_t into shared
        q_shared[i] = IO_LOAD(q, q_offset + i);
        __syncthreads();

        // Compute o_i = sum_j(S[i][j] * q[j])
        float o_i = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            o_i += S[i * head_dim + j] * q_shared[j];
        }

        // RMS normalization: rms = sqrt(mean(o^2) + eps)
        // Accumulate o_i^2 into shared
        if (i == 0) rms_shared[0] = 0.0f;
        __syncthreads();
        atomicAdd(rms_shared, o_i * o_i);
        __syncthreads();

        float rms_inv = rsqrtf(rms_shared[0] / (float)head_dim + NORM_EPS);
        float o_normed = o_i * rms_inv;

        // Write output
        IO_STORE(output, q_offset + i, o_normed);
        __syncthreads();
    }
}

// ============================================================================
// Standalone launch wrapper
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

int fused_delta_product_scan_launch(
    cudaStream_t stream,
    const io_type* q, const io_type* k, const io_type* v, const io_type* beta,
    io_type* output,
    int batch, int seq_len, int num_householder, int num_heads, int head_dim
) {
    dim3 grid(batch, num_heads);
    dim3 block(head_dim);

    // S[d][d] + k_shared[d] + q_shared[d] + rms_shared[1]
    size_t smem_bytes = (size_t)head_dim * head_dim * sizeof(float)
                      + 2 * head_dim * sizeof(float)
                      + sizeof(float);

    fused_delta_product_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        q, k, v, beta, output,
        seq_len, num_householder, num_heads, head_dim
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

ffi::Error fused_delta_product_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> q,       // [B, T, H, d]
    ffi::Buffer<FFI_IO_TYPE> k,       // [B, T, n_h, H, d]
    ffi::Buffer<FFI_IO_TYPE> v,       // [B, T, n_h, H, d]
    ffi::Buffer<FFI_IO_TYPE> beta,    // [B, T, n_h, H]
    ffi::ResultBuffer<FFI_IO_TYPE> output  // [B, T, H, d]
) {
    // Extract dims from q: [B, T, H, d]
    auto q_dims = q.dimensions();
    int batch     = static_cast<int>(q_dims[0]);
    int seq_len   = static_cast<int>(q_dims[1]);
    int num_heads = static_cast<int>(q_dims[2]);
    int head_dim  = static_cast<int>(q_dims[3]);

    // Extract n_h from k: [B, T, n_h, H, d]
    auto k_dims = k.dimensions();
    int num_householder = static_cast<int>(k_dims[2]);

    dim3 grid(batch, num_heads);
    dim3 block(head_dim);
    size_t smem_bytes = (size_t)head_dim * head_dim * sizeof(float)
                      + 2 * head_dim * sizeof(float)
                      + sizeof(float);

    fused_delta_product_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const io_type*>(q.untyped_data()),
        reinterpret_cast<const io_type*>(k.untyped_data()),
        reinterpret_cast<const io_type*>(v.untyped_data()),
        reinterpret_cast<const io_type*>(beta.untyped_data()),
        reinterpret_cast<io_type*>(output->untyped_data()),
        seq_len, num_householder, num_heads, head_dim
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_delta_product_scan, fused_delta_product_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // q
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // k
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // v
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // beta
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // output
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_delta_product_scan_" PRECISION_SUFFIX, "CUDA", fused_delta_product_scan);

#endif  // EXLA_FFI
