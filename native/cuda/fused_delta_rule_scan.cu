// Fused Delta Rule Scan Kernel (DeltaNet + GatedDeltaNet)
//
// Matrix-state recurrence with cross-element communication.
// Unlike P0/P1 kernels (element-wise h = a*h + b), this kernel maintains
// a d x d state matrix S in shared memory and performs matrix-vector products
// (S @ k, S @ q) at each timestep.
//
// Thread layout: one thread BLOCK per (batch, head) pair.
// TILE_K threads per row, head_dim * TILE_K threads per block (e.g., 64*4 = 256).
// Each group of TILE_K threads cooperates on one row of S[d][d], parallelizing
// dot products across lanes with warp shuffle reduction.
//
// Bank conflict elimination: S matrix uses stride (head_dim + 1) instead of
// head_dim, offsetting consecutive rows by 1 bank. Without this, all rows
// map to the same bank at any given column j (since head_dim=64 is a multiple
// of 32 banks), causing 32-way serialization on every shared memory access.
//
// DeltaNet recurrence:
//   retrieval = S_{t-1} @ k_t
//   error = v_t - retrieval
//   S_t = S_{t-1} + beta_t * outer(error, k_t)
//   o_t = S_t @ q_t
//
// GatedDeltaNet adds scalar decay:
//   alpha_scalar = alpha_t[h]  (pre-computed mean per head on Elixir side)
//   S_gated = alpha_scalar * S_{t-1}
//   (then same delta rule on S_gated)
//
// When alpha pointer is NULL, the kernel behaves as vanilla DeltaNet.
//
// Inputs (all pre-computed on Elixir/XLA side):
//   q:     [B, T, H, d] — query vectors
//   k:     [B, T, H, d] — key vectors (L2-normalized on XLA side)
//   v:     [B, T, H, d] — value vectors
//   beta:  [B, T, H, d] — per-element update gate (post-sigmoid)
//   alpha: [B, T, H]    — per-head scalar forget gate (NULL for DeltaNet)
//
// Output:
//   output: [B, T, H, d] — retrieval outputs per head

#include <cuda_runtime.h>
#include "precision.cuh"

// ============================================================================
// Kernel
// ============================================================================

#ifdef EXLA_FFI
namespace {  // anonymous namespace — internal linkage per compilation unit
#endif

__global__ void fused_delta_rule_scan_kernel(
    const io_type* __restrict__ q,       // [B, T, H, d]
    const io_type* __restrict__ k,       // [B, T, H, d]
    const io_type* __restrict__ v,       // [B, T, H, d]
    const io_type* __restrict__ beta,    // [B, T, H, d]
    const io_type* __restrict__ alpha,   // [B, T, H] or NULL
    io_type* __restrict__ output,        // [B, T, H, d]
    int seq_len,
    int num_heads,
    int head_dim
) {
    // Thread block assignment: one block per (batch, head)
    int b = blockIdx.x;   // batch index
    int h = blockIdx.y;   // head index
    int tid = threadIdx.x;

    // Tiling: TILE_K threads per row for dot product parallelism
    int tile_k = blockDim.x / head_dim;
    int row = tid / tile_k;   // which row of S (0..head_dim-1)
    int lane = tid % tile_k;  // which lane within the row (0..tile_k-1)

    if (row >= head_dim) return;

    // Each lane handles a chunk of the inner dimension
    int chunk = head_dim / tile_k;
    int j_start = lane * chunk;
    int j_end = j_start + chunk;

    // Shared memory layout:
    //   S[head_dim][s_stride] — state matrix with padded stride for bank conflict elimination
    //   k_shared[head_dim]    — current timestep's k vector
    //   q_shared[head_dim]    — current timestep's q vector
    //
    // Bank conflict fix: stride = head_dim + 1 ensures consecutive rows
    // offset by 1 bank (32 banks × 4 bytes). Without padding, S[0][j] and
    // S[1][j] map to the same bank when head_dim % 32 == 0.
    int s_stride = head_dim + 1;
    extern __shared__ float smem[];
    float* S = smem;                                         // [head_dim][s_stride]
    float* k_shared = smem + head_dim * s_stride;            // [head_dim]
    float* q_shared = k_shared + head_dim;                   // [head_dim]

    // Initialize state matrix to zero (each lane handles its chunk of the row)
    for (int j = j_start; j < j_end; j++) {
        S[row * s_stride + j] = 0.0f;
    }
    __syncthreads();

    // Strides for indexing into [B, T, H, d] tensors
    int BHd = seq_len * num_heads * head_dim;   // stride for batch dim (T*H*d)
    int THd = num_heads * head_dim;             // stride for time dim (H*d)
    int Hd  = head_dim;                         // stride for head dim (d)

    // Stride for alpha: [B, T, H]
    int alpha_BH = seq_len * num_heads;  // stride for batch dim in alpha (T*H)
    int alpha_TH = num_heads;            // stride for time dim in alpha (H)

    int base_bh = b * BHd + h * Hd;     // base offset for this (batch, head) in q/k/v/beta
    int alpha_base_bh = b * alpha_BH;    // base offset for this batch in alpha

    for (int t = 0; t < seq_len; t++) {
        int offset = base_bh + t * THd;  // offset for this (b, t, h) in [B,T,H,d]

        // Step 1: Load k_t into shared memory (first head_dim threads load)
        if (tid < head_dim) {
            k_shared[tid] = IO_LOAD(k, offset + tid);
        }
        __syncthreads();

        // Step 2: Compute retrieval[row] = sum_j(S[row][j] * k_shared[j])
        // Each lane computes partial sum over its chunk, then reduce across lanes
        float partial = 0.0f;
        for (int j = j_start; j < j_end; j++) {
            partial += S[row * s_stride + j] * k_shared[j];
        }
        // Warp-level butterfly reduction across tile_k lanes
        // All tile_k lanes for a row are consecutive in the warp (row*tile_k + lane),
        // so __shfl_xor_sync correctly pairs them.
        for (int offset_k = 1; offset_k < tile_k; offset_k *= 2) {
            partial += __shfl_xor_sync(0xffffffff, partial, offset_k);
        }
        float retrieval = partial;  // all lanes now have the full dot product

        // Step 3: If alpha provided, apply scalar decay to state row
        if (alpha != NULL) {
            float alpha_val = IO_LOAD(alpha, alpha_base_bh + t * alpha_TH + h);
            for (int j = j_start; j < j_end; j++) {
                S[row * s_stride + j] *= alpha_val;
            }
            // Retrieval was computed on pre-decay S, so scale it too
            retrieval *= alpha_val;
        }

        // Step 4: Compute error and scaled error (all lanes compute the same values)
        float v_row = IO_LOAD(v, offset + row);
        float beta_row = IO_LOAD(beta, offset + row);
        float error_row = v_row - retrieval;
        float scaled_error_row = beta_row * error_row;

        // Step 5: Rank-1 update: S[row][j] += scaled_error[row] * k[j]
        // Each lane updates its chunk (no conflicts — different j ranges)
        for (int j = j_start; j < j_end; j++) {
            S[row * s_stride + j] += scaled_error_row * k_shared[j];
        }
        __syncthreads();

        // Step 6: Load q_t into shared memory
        if (tid < head_dim) {
            q_shared[tid] = IO_LOAD(q, offset + tid);
        }
        __syncthreads();

        // Step 7: Compute output[row] = sum_j(S[row][j] * q_shared[j])
        float out_partial = 0.0f;
        for (int j = j_start; j < j_end; j++) {
            out_partial += S[row * s_stride + j] * q_shared[j];
        }
        for (int offset_k = 1; offset_k < tile_k; offset_k *= 2) {
            out_partial += __shfl_xor_sync(0xffffffff, out_partial, offset_k);
        }

        // Step 8: Write output (only lane 0 per row writes)
        if (lane == 0) {
            IO_STORE(output, offset + row, out_partial);
        }
        __syncthreads();
    }
}

// ============================================================================
// Launch helpers
// ============================================================================

// Compute tile_k: how many threads per row of S
// Target: at least 128 threads per block for decent occupancy
// Constraint: head_dim * tile_k <= 1024 (max threads per block)
static inline int compute_tile_k(int head_dim) {
    if (head_dim >= 512) return 1;
    if (head_dim >= 256) return 2;
    // Default: 4 threads per row (256 threads for head_dim=64)
    return 4;
}

static inline size_t compute_smem_bytes(int head_dim) {
    int s_stride = head_dim + 1;
    return (size_t)head_dim * s_stride * sizeof(float)   // S[d][d+1]
         + 2 * head_dim * sizeof(float);                 // k_shared[d] + q_shared[d]
}

// ============================================================================
// Standalone launch wrappers (C-linkage for NIF / dlopen)
// ============================================================================

#ifdef EXLA_FFI
}  // anonymous namespace
#endif

#ifndef EXLA_FFI

extern "C" {

int fused_delta_rule_scan_launch(
    cudaStream_t stream,
    const io_type* q, const io_type* k, const io_type* v, const io_type* beta,
    const io_type* alpha,  // NULL for vanilla DeltaNet
    io_type* output,
    int batch, int seq_len, int num_heads, int head_dim
) {
    int tile_k = compute_tile_k(head_dim);
    dim3 grid(batch, num_heads);
    dim3 block(head_dim * tile_k);
    size_t smem_bytes = compute_smem_bytes(head_dim);

    fused_delta_rule_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        q, k, v, beta, alpha, output,
        seq_len, num_heads, head_dim
    );

    return (int)cudaGetLastError();
}

// Convenience wrapper for vanilla DeltaNet (alpha=NULL)
int fused_delta_net_scan_launch(
    cudaStream_t stream,
    const io_type* q, const io_type* k, const io_type* v, const io_type* beta,
    io_type* output,
    int batch, int seq_len, int num_heads, int head_dim
) {
    return fused_delta_rule_scan_launch(
        stream, q, k, v, beta, NULL, output,
        batch, seq_len, num_heads, head_dim
    );
}

// Convenience wrapper for GatedDeltaNet (alpha provided)
int fused_gated_delta_net_scan_launch(
    cudaStream_t stream,
    const io_type* q, const io_type* k, const io_type* v, const io_type* beta,
    const io_type* alpha,
    io_type* output,
    int batch, int seq_len, int num_heads, int head_dim
) {
    return fused_delta_rule_scan_launch(
        stream, q, k, v, beta, alpha, output,
        batch, seq_len, num_heads, head_dim
    );
}

}  // extern "C"

#endif  // !EXLA_FFI

// ============================================================================
// XLA FFI integration (for EXLA fork)
// ============================================================================

#ifdef EXLA_FFI

#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace ffi = xla::ffi;

namespace {  // anonymous namespace — prevents symbol collision between f32/bf16

// DeltaNet (no alpha gate)
ffi::Error fused_delta_net_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> q,
    ffi::Buffer<FFI_IO_TYPE> k,
    ffi::Buffer<FFI_IO_TYPE> v,
    ffi::Buffer<FFI_IO_TYPE> beta,
    ffi::ResultBuffer<FFI_IO_TYPE> output
) {
    auto dims = q.dimensions();
    int batch     = static_cast<int>(dims[0]);
    int seq_len   = static_cast<int>(dims[1]);
    int num_heads = static_cast<int>(dims[2]);
    int head_dim  = static_cast<int>(dims[3]);

    int tile_k = compute_tile_k(head_dim);
    dim3 grid(batch, num_heads);
    dim3 block(head_dim * tile_k);
    size_t smem_bytes = compute_smem_bytes(head_dim);

    fused_delta_rule_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const io_type*>(q.untyped_data()),
        reinterpret_cast<const io_type*>(k.untyped_data()),
        reinterpret_cast<const io_type*>(v.untyped_data()),
        reinterpret_cast<const io_type*>(beta.untyped_data()),
        nullptr,  // no alpha for DeltaNet
        reinterpret_cast<io_type*>(output->untyped_data()),
        seq_len, num_heads, head_dim
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

// GatedDeltaNet (with alpha gate)
ffi::Error fused_gated_delta_net_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> q,
    ffi::Buffer<FFI_IO_TYPE> k,
    ffi::Buffer<FFI_IO_TYPE> v,
    ffi::Buffer<FFI_IO_TYPE> beta,
    ffi::Buffer<FFI_IO_TYPE> alpha,
    ffi::ResultBuffer<FFI_IO_TYPE> output
) {
    auto dims = q.dimensions();
    int batch     = static_cast<int>(dims[0]);
    int seq_len   = static_cast<int>(dims[1]);
    int num_heads = static_cast<int>(dims[2]);
    int head_dim  = static_cast<int>(dims[3]);

    int tile_k = compute_tile_k(head_dim);
    dim3 grid(batch, num_heads);
    dim3 block(head_dim * tile_k);
    size_t smem_bytes = compute_smem_bytes(head_dim);

    fused_delta_rule_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const io_type*>(q.untyped_data()),
        reinterpret_cast<const io_type*>(k.untyped_data()),
        reinterpret_cast<const io_type*>(v.untyped_data()),
        reinterpret_cast<const io_type*>(beta.untyped_data()),
        reinterpret_cast<const io_type*>(alpha.untyped_data()),
        reinterpret_cast<io_type*>(output->untyped_data()),
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
    HANDLER_SYMBOL(fused_delta_net_scan), fused_delta_net_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // q
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // k
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // v
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // beta
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // output
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    HANDLER_SYMBOL(fused_gated_delta_net_scan), fused_gated_delta_net_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // q
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // k
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // v
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // beta
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // alpha
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // output
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_delta_net_scan_" PRECISION_SUFFIX, "CUDA",
    HANDLER_SYMBOL(fused_delta_net_scan));

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_gated_delta_net_scan_" PRECISION_SUFFIX, "CUDA",
    HANDLER_SYMBOL(fused_gated_delta_net_scan));

#endif  // EXLA_FFI
