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
// Same thread layout as forward: one block per (batch, head), TILE_K threads
// per row for higher occupancy. Bank conflict elimination via S_STRIDE = d+1.
//
// Retrieval values from the forward recompute pass are stored in a global
// memory workspace (allocated via cudaMallocAsync) instead of per-thread
// stack arrays, avoiding register spills that degraded performance.
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
    float* __restrict__ retrieval_workspace,  // [B, H, d, T] global memory workspace
    int seq_len,
    int num_heads,
    int head_dim
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;

    // Tiling: TILE_K threads per row
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
    float* S = smem;                                          // [d][s_stride]
    // Reserve second S block for alignment (backward needs extra scratch)
    float* k_shared = smem + 2 * head_dim * s_stride;         // [d]
    float* q_shared = k_shared + head_dim;                     // [d]
    float* temp_shared = q_shared + head_dim;                  // [d]
    float* reduce_shared = temp_shared + head_dim;             // [1]

    // Strides
    int THd = seq_len * num_heads * head_dim;
    int Hd = num_heads * head_dim;
    int d = head_dim;
    int base_bh = b * THd + h * d;

    // Alpha strides: [B, T, H]
    int alpha_TH = seq_len * num_heads;
    int alpha_base = b * alpha_TH + h;

    // Workspace strides: [B, H, d, T] — each (b,h,row) has seq_len entries
    int ws_base = (b * num_heads + h) * head_dim * seq_len + row * seq_len;

    // ========================================
    // Pass 1: Forward — recompute S states, store retrieval values to workspace
    // ========================================
    for (int j = j_start; j < j_end; j++) {
        S[row * s_stride + j] = 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < seq_len; t++) {
        int offset = base_bh + t * Hd;
        float alpha_val = IO_LOAD(alpha, alpha_base + t * num_heads);

        // Apply alpha decay
        for (int j = j_start; j < j_end; j++) {
            S[row * s_stride + j] *= alpha_val;
        }
        __syncthreads();

        if (tid < head_dim) {
            k_shared[tid] = IO_LOAD(k, offset + tid);
        }
        __syncthreads();

        // Retrieval = S[row,:] @ k (tiled)
        float partial = 0.0f;
        for (int j = j_start; j < j_end; j++) {
            partial += S[row * s_stride + j] * k_shared[j];
        }
        for (int offset_k = 1; offset_k < tile_k; offset_k *= 2) {
            partial += __shfl_xor_sync(0xffffffff, partial, offset_k);
        }

        // Store retrieval to global memory workspace (lane 0 writes)
        if (lane == 0) {
            retrieval_workspace[ws_base + t] = partial;
        }

        // Delta rule update
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
    // dS_row: per-lane partial gradient for this row's chunk
    // head_dim/tile_k elements per lane (e.g., 16 for tile_k=4, head_dim=64)
    float dS_chunk[64];  // max head_dim/tile_k (supports head_dim up to 256)
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
        float alpha_val = IO_LOAD(alpha, alpha_base + t * num_heads);

        // Load retrieval from workspace
        float retrieval_row = retrieval_workspace[ws_base + t];
        // Broadcast to all lanes
        retrieval_row = __shfl_sync(0xffffffff, retrieval_row, (row * tile_k) & 31);

        float error_row = v_row - retrieval_row;
        float scaled_err_row = beta_row * error_row;

        // ---- grad_q: dq = S_t^T @ do ----
        // Load do into shared for column access
        if (tid < head_dim) {
            q_shared[tid] = IO_LOAD(grad_output, offset + tid);
        }
        __syncthreads();

        // dq[row] = sum_j(S[j][row] * do[j]) — column access on S
        // Each lane handles its chunk of rows j
        float dq_partial = 0.0f;
        for (int j = j_start; j < j_end; j++) {
            dq_partial += S[j * s_stride + row] * q_shared[j];
        }
        for (int offset_k = 1; offset_k < tile_k; offset_k *= 2) {
            dq_partial += __shfl_xor_sync(0xffffffff, dq_partial, offset_k);
        }
        if (lane == 0) {
            IO_STORE(grad_q, offset + row, dq_partial);
        }

        // ---- dS from output: dS += outer(do, q) ----
        // Load q into shared
        if (tid < head_dim) {
            k_shared[tid] = IO_LOAD(q, offset + tid);  // reuse k_shared for q values
        }
        __syncthreads();

        for (int j = 0; j < chunk; j++) {
            dS_chunk[j] += do_row * k_shared[j_start + j];
        }

        // ---- Gradients from update S_t = S_gated + beta*outer(error, k) ----
        if (tid < head_dim) {
            k_shared[tid] = IO_LOAD(k, offset + tid);
        }
        __syncthreads();

        // d_beta_error = dS[row,:] @ k[:]
        float d_beta_error_partial = 0.0f;
        for (int j = 0; j < chunk; j++) {
            d_beta_error_partial += dS_chunk[j] * k_shared[j_start + j];
        }
        for (int offset_k = 1; offset_k < tile_k; offset_k *= 2) {
            d_beta_error_partial += __shfl_xor_sync(0xffffffff, d_beta_error_partial, offset_k);
        }
        float d_beta_error_row = d_beta_error_partial;

        // dk from outer product: dk[j] += sum_i(dS[i][j] * scaled_err[i])
        if (tid < head_dim) {
            temp_shared[tid] = 0.0f;
        }
        __syncthreads();

        // Each lane atomically adds its chunk contribution
        for (int j = 0; j < chunk; j++) {
            atomicAdd(&temp_shared[j_start + j], dS_chunk[j] * scaled_err_row);
        }
        __syncthreads();

        float d_error_row = d_beta_error_row * beta_row;
        float dv_row = d_error_row;
        float d_retrieval_row = -d_error_row;

        // dS += d_retrieval * outer(1, k) for this row
        for (int j = 0; j < chunk; j++) {
            dS_chunk[j] += d_retrieval_row * k_shared[j_start + j];
        }

        // Undo update to get S_gated
        for (int j = j_start; j < j_end; j++) {
            S[row * s_stride + j] -= scaled_err_row * k_shared[j];
        }
        __syncthreads();

        // dk from retrieval (S_gated @ k): dk[j] += sum_i(S[i][j] * d_retrieval[i])
        if (tid < head_dim) {
            q_shared[tid] = 0.0f;
        }
        __syncthreads();

        // Each lane adds its chunk's contribution to the column sum
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

        // ---- Gradient through alpha decay: S_gated = alpha * S_{t-1} ----
        // d_alpha = sum_{i,j}(dS_gated[i][j] * S_{t-1}[i][j])
        // Current S holds S_gated. S_{t-1} = S_gated / alpha.
        float d_alpha_partial = 0.0f;
        for (int j = 0; j < chunk; j++) {
            float s_gated_val = S[row * s_stride + (j_start + j)];
            float s_prev_val = (alpha_val != 0.0f) ? s_gated_val / alpha_val : 0.0f;
            d_alpha_partial += dS_chunk[j] * s_prev_val;
        }
        // Reduce across lanes within row
        for (int offset_k = 1; offset_k < tile_k; offset_k *= 2) {
            d_alpha_partial += __shfl_xor_sync(0xffffffff, d_alpha_partial, offset_k);
        }

        // Reduce across rows (lane 0 of each row contributes)
        if (lane == 0) {
            atomicAdd(reduce_shared, d_alpha_partial);
        }
        // Need barrier before thread 0 reads reduce_shared
        // But first, clear it for this timestep
        if (tid == 0) reduce_shared[0] = 0.0f;
        __syncthreads();
        if (lane == 0) {
            atomicAdd(reduce_shared, d_alpha_partial);
        }
        __syncthreads();

        if (tid == 0) {
            IO_STORE(grad_alpha, alpha_base + t * num_heads, reduce_shared[0]);
        }

        // dS propagated through alpha: scale by alpha for next iteration
        for (int j = 0; j < chunk; j++) {
            dS_chunk[j] *= alpha_val;
        }

        // Undo alpha decay to get S_{t-1}
        if (alpha_val != 0.0f) {
            float inv_alpha = 1.0f / alpha_val;
            for (int j = j_start; j < j_end; j++) {
                S[row * s_stride + j] *= inv_alpha;
            }
        }
        __syncthreads();
    }
}

// ============================================================================
// Launch helpers (shared with forward kernel)
// ============================================================================

static inline int compute_tile_k(int head_dim) {
    if (head_dim >= 512) return 1;
    if (head_dim >= 256) return 2;
    return 4;
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

    int tile_k = compute_tile_k(head_dim);
    dim3 grid(batch, num_heads);
    dim3 block(head_dim * tile_k);

    int s_stride = head_dim + 1;
    // 2*S[d][s_stride] + k_shared[d] + q_shared[d] + temp_shared[d] + reduce[1]
    size_t smem_bytes = 2 * (size_t)head_dim * s_stride * sizeof(float)
                      + 3 * head_dim * sizeof(float)
                      + sizeof(float);

    // Allocate global memory workspace for retrieval storage
    // Shape: [batch, num_heads, head_dim, seq_len]
    size_t ws_size = (size_t)batch * num_heads * head_dim * seq_len * sizeof(float);
    float* retrieval_workspace = nullptr;
    cudaError_t alloc_err = cudaMallocAsync(&retrieval_workspace, ws_size, stream);
    if (alloc_err != cudaSuccess) {
        return (int)alloc_err;
    }

    fused_gated_delta_net_scan_backward_kernel<<<grid, block, smem_bytes, stream>>>(
        q, k, v, beta, alpha, forward_out, grad_output,
        grad_q, grad_k, grad_v, grad_beta, grad_alpha,
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

    int tile_k = compute_tile_k(head_dim);
    dim3 grid(batch, num_heads);
    dim3 block(head_dim * tile_k);

    int s_stride = head_dim + 1;
    size_t smem_bytes = 2 * (size_t)head_dim * s_stride * sizeof(float)
                      + 3 * head_dim * sizeof(float)
                      + sizeof(float);

    // Allocate workspace
    size_t ws_size = (size_t)batch * num_heads * head_dim * seq_len * sizeof(float);
    float* retrieval_workspace = nullptr;
    cudaError_t alloc_err = cudaMallocAsync(&retrieval_workspace, ws_size, stream);
    if (alloc_err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, "Failed to allocate retrieval workspace");
    }

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
