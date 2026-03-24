// Fused DeltaNet Scan Backward Kernel
//
// Computes gradients for the delta rule matrix-state recurrence:
//   retrieval = S_{t-1} @ k_t
//   error = v_t - retrieval
//   S_t = S_{t-1} + beta_t * outer(error, k_t)
//   o_t = S_t @ q_t
//
// Two-pass approach:
//   Pass 1 (forward, t=0..T-1): Recompute S_t, store retrieval values to workspace.
//   Pass 2 (reverse, t=T-1..0): Reverse accumulate dS and compute per-input grads.
//
// Optimizations over v1:
//   - TILE_K threads per row (4x occupancy: 256 vs 64 threads/block)
//   - Bank conflict elimination via S_STRIDE = head_dim + 1
//   - Global memory workspace for retrieval storage (replaces per-thread stack array)
//   - Warp shuffle reductions for dot products
//
// Thread layout: one block per (batch, head), head_dim * tile_k threads per block.
// Each group of tile_k threads cooperates on one row of S[d][d].
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

// ============================================================================
// Kernel
// ============================================================================

#ifdef EXLA_FFI
namespace {  // anonymous namespace — internal linkage per compilation unit
#endif

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
    float* __restrict__ retrieval_workspace,  // [B, H, d, T] global memory workspace
    int seq_len,
    int num_heads,
    int head_dim
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;

    // Tiling
    int tile_k = blockDim.x / head_dim;
    int row = tid / tile_k;
    int lane = tid % tile_k;

    if (row >= head_dim) return;

    int chunk = head_dim / tile_k;
    int j_start = lane * chunk;
    int j_end = j_start + chunk;

    // Shared memory with bank conflict elimination
    int s_stride = head_dim + 1;
    extern __shared__ float smem[];
    float* S = smem;                                           // [d][s_stride]
    float* k_shared = smem + head_dim * s_stride;              // [d]
    float* q_shared = k_shared + head_dim;                     // [d]
    float* temp_shared = q_shared + head_dim;                  // [d]

    // Strides for [B, T, H, d]
    int THd = seq_len * num_heads * head_dim;
    int Hd = num_heads * head_dim;
    int d = head_dim;
    int base_bh = b * THd + h * d;

    // Workspace strides: [B, H, d, T]
    int ws_base = (b * num_heads + h) * head_dim * seq_len + row * seq_len;

    // ========================================
    // Pass 1: Forward — recompute S, store retrieval values
    // ========================================
    for (int j = j_start; j < j_end; j++) {
        S[row * s_stride + j] = 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < seq_len; t++) {
        int offset = base_bh + t * Hd;

        if (tid < head_dim) {
            k_shared[tid] = IO_LOAD(k, offset + tid);
        }
        __syncthreads();

        // retrieval = S[row,:] @ k (tiled)
        float partial = 0.0f;
        for (int j = j_start; j < j_end; j++) {
            partial += S[row * s_stride + j] * k_shared[j];
        }
        for (int off = 1; off < tile_k; off *= 2) {
            partial += __shfl_xor_sync(0xffffffff, partial, off);
        }

        // Store retrieval to workspace
        if (lane == 0) {
            retrieval_workspace[ws_base + t] = partial;
        }

        // Update S
        float v_row = IO_LOAD(v, offset + row);
        float beta_row = IO_LOAD(beta, offset + row);
        float error_row = v_row - partial;
        float scaled_error_row = beta_row * error_row;

        for (int j = j_start; j < j_end; j++) {
            S[row * s_stride + j] += scaled_error_row * k_shared[j];
        }
        __syncthreads();
    }

    // ========================================
    // Pass 2: Backward
    // ========================================
    float dS_chunk[64];  // max head_dim/tile_k (head_dim up to 256)
    for (int j = 0; j < chunk; j++) {
        dS_chunk[j] = 0.0f;
    }

    for (int t = seq_len - 1; t >= 0; t--) {
        int offset = base_bh + t * Hd;

        float q_row = IO_LOAD(q, offset + row);
        float k_row = IO_LOAD(k, offset + row);
        float v_row = IO_LOAD(v, offset + row);
        float beta_row = IO_LOAD(beta, offset + row);
        float do_row = IO_LOAD(grad_output, offset + row);

        // Load retrieval from workspace
        float retrieval_row = retrieval_workspace[ws_base + t];
        retrieval_row = __shfl_sync(0xffffffff, retrieval_row, (row * tile_k) & 31);

        float error_row = v_row - retrieval_row;
        float scaled_err_row = beta_row * error_row;

        // ---- grad_q: dq = S_t^T @ do ----
        if (tid < head_dim) {
            q_shared[tid] = IO_LOAD(grad_output, offset + tid);
        }
        __syncthreads();

        // dq[row] = sum_j(S[j][row] * do[j]) — column access
        float dq_partial = 0.0f;
        for (int j = j_start; j < j_end; j++) {
            dq_partial += S[j * s_stride + row] * q_shared[j];
        }
        for (int off = 1; off < tile_k; off *= 2) {
            dq_partial += __shfl_xor_sync(0xffffffff, dq_partial, off);
        }
        if (lane == 0) {
            IO_STORE(grad_q, offset + row, dq_partial);
        }

        // ---- dS from output: dS += outer(do, q) ----
        if (tid < head_dim) {
            k_shared[tid] = IO_LOAD(q, offset + tid);
        }
        __syncthreads();

        for (int j = 0; j < chunk; j++) {
            dS_chunk[j] += do_row * k_shared[j_start + j];
        }

        // ---- Gradients from update ----
        if (tid < head_dim) {
            k_shared[tid] = IO_LOAD(k, offset + tid);
        }
        __syncthreads();

        // d_beta_error = dS[row,:] @ k[:]
        float d_beta_error_partial = 0.0f;
        for (int j = 0; j < chunk; j++) {
            d_beta_error_partial += dS_chunk[j] * k_shared[j_start + j];
        }
        for (int off = 1; off < tile_k; off *= 2) {
            d_beta_error_partial += __shfl_xor_sync(0xffffffff, d_beta_error_partial, off);
        }
        float d_beta_error_row = d_beta_error_partial;

        // dk from outer: dk[j] += sum_i(dS[i][j] * scaled_err[i])
        if (tid < head_dim) {
            temp_shared[tid] = 0.0f;
        }
        __syncthreads();

        for (int j = 0; j < chunk; j++) {
            atomicAdd(&temp_shared[j_start + j], dS_chunk[j] * scaled_err_row);
        }
        __syncthreads();

        float d_error_row = d_beta_error_row * beta_row;
        float dv_row = d_error_row;
        float d_retrieval_row = -d_error_row;

        // dS from retrieval gradient
        for (int j = 0; j < chunk; j++) {
            dS_chunk[j] += d_retrieval_row * k_shared[j_start + j];
        }

        // Undo update to get S_{t-1}
        for (int j = j_start; j < j_end; j++) {
            S[row * s_stride + j] -= scaled_err_row * k_shared[j];
        }
        __syncthreads();

        // dk from retrieval: dk[j] += sum_i(S_{t-1}[i][j] * d_retrieval[i])
        if (tid < head_dim) {
            q_shared[tid] = 0.0f;
        }
        __syncthreads();

        for (int j = j_start; j < j_end; j++) {
            atomicAdd(&q_shared[j], S[row * s_stride + j] * d_retrieval_row);
        }
        __syncthreads();

        if (lane == 0) {
            float dk_row = temp_shared[row] + q_shared[row];
            float d_beta_row_val = d_beta_error_row * error_row;

            IO_STORE(grad_k, offset + row, dk_row);
            IO_STORE(grad_v, offset + row, dv_row);
            IO_STORE(grad_beta, offset + row, d_beta_row_val);
        }
        __syncthreads();
    }
}

// ============================================================================
// Launch helpers
// ============================================================================

static inline int compute_tile_k(int head_dim) {
    if (head_dim >= 512) return 1;
    if (head_dim >= 256) return 2;
    return 4;
}

// ============================================================================
// Standalone launch wrapper (C-linkage for NIF / dlopen)
// ============================================================================

#ifdef EXLA_FFI
}  // anonymous namespace
#endif

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

    int tile_k = compute_tile_k(head_dim);
    dim3 grid(batch, num_heads);
    dim3 block(head_dim * tile_k);

    int s_stride = head_dim + 1;
    // S[d][s_stride] + k_shared[d] + q_shared[d] + temp_shared[d]
    size_t smem_bytes = (size_t)head_dim * s_stride * sizeof(float)
                      + 3 * head_dim * sizeof(float);

    // Allocate workspace: [batch, num_heads, head_dim, seq_len]
    size_t ws_size = (size_t)batch * num_heads * head_dim * seq_len * sizeof(float);
    float* retrieval_workspace = nullptr;
    cudaError_t alloc_err = cudaMallocAsync(&retrieval_workspace, ws_size, stream);
    if (alloc_err != cudaSuccess) {
        return (int)alloc_err;
    }

    fused_delta_rule_scan_backward_kernel<<<grid, block, smem_bytes, stream>>>(
        q, k, v, beta, forward_out, grad_output,
        grad_q, grad_k, grad_v, grad_beta,
        retrieval_workspace,
        seq_len, num_heads, head_dim
    );

    cudaError_t kernel_err = cudaGetLastError();
    cudaFreeAsync(retrieval_workspace, stream);

    return (int)kernel_err;
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

    int tile_k = compute_tile_k(head_dim);
    dim3 grid(batch, num_heads);
    dim3 block(head_dim * tile_k);

    int s_stride = head_dim + 1;
    size_t smem_bytes = (size_t)head_dim * s_stride * sizeof(float)
                      + 3 * head_dim * sizeof(float);

    // Allocate workspace
    size_t ws_size = (size_t)batch * num_heads * head_dim * seq_len * sizeof(float);
    float* retrieval_workspace = nullptr;
    cudaError_t alloc_err = cudaMallocAsync(&retrieval_workspace, ws_size, stream);
    if (alloc_err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, "Failed to allocate retrieval workspace");
    }

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
        retrieval_workspace,
        seq_len, num_heads, head_dim
    );

    cudaError_t kernel_err = cudaGetLastError();
    cudaFreeAsync(retrieval_workspace, stream);

    if (kernel_err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(kernel_err));
    }

    return ffi::Error::Success();
}

}  // anonymous namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    HANDLER_SYMBOL(fused_delta_rule_scan_backward), fused_delta_rule_scan_backward_ffi_impl,
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
    "exla_fused_delta_rule_scan_backward_" PRECISION_SUFFIX, "CUDA", HANDLER_SYMBOL(fused_delta_rule_scan_backward));

#endif  // EXLA_FFI
