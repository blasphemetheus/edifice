// Fused Delta Rule Scan Kernel (DeltaNet + GatedDeltaNet)
//
// Matrix-state recurrence with cross-element communication.
// Unlike P0/P1 kernels (element-wise h = a*h + b), this kernel maintains
// a d x d state matrix S in shared memory and performs matrix-vector products
// (S @ k, S @ q) at each timestep.
//
// Thread layout: one thread BLOCK per (batch, head) pair, d threads per block.
// Each thread owns one row of the state matrix S[d][d].
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
//
// Shared memory budget (head_dim=64):
//   S matrix: 64*64*4 = 16KB
//   k_shared: 64*4 = 256 bytes
//   q_shared: 64*4 = 256 bytes
//   Total: ~17KB — well within 48KB limit

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
    int i = threadIdx.x;  // row index in S matrix (0..head_dim-1)

    if (i >= head_dim) return;

    // Shared memory layout:
    //   S[head_dim][head_dim] — state matrix (each thread owns row i)
    //   k_shared[head_dim]    — current timestep's k vector
    //   q_shared[head_dim]    — current timestep's q vector
    extern __shared__ float smem[];
    float* S = smem;                                    // [head_dim][head_dim]
    float* k_shared = smem + head_dim * head_dim;       // [head_dim]
    float* q_shared = k_shared + head_dim;              // [head_dim]

    // Initialize state matrix to zero
    for (int j = 0; j < head_dim; j++) {
        S[i * head_dim + j] = 0.0f;
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

        // Step 1: Load k_t into shared memory (coalesced read, one element per thread)
        k_shared[i] = IO_LOAD(k, offset + i);
        __syncthreads();

        // Step 2: Compute retrieval[i] = sum_j(S[i][j] * k_shared[j])
        float retrieval = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            retrieval += S[i * head_dim + j] * k_shared[j];
        }

        // Step 3: If alpha provided, apply scalar decay to state row
        if (alpha != NULL) {
            float alpha_val = IO_LOAD(alpha, alpha_base_bh + t * alpha_TH + h);
            for (int j = 0; j < head_dim; j++) {
                S[i * head_dim + j] *= alpha_val;
            }
            // Recompute retrieval on decayed state
            retrieval *= alpha_val;
        }

        // Step 4: Compute error and scaled error
        float v_i = IO_LOAD(v, offset + i);
        float beta_i = IO_LOAD(beta, offset + i);
        float error_i = v_i - retrieval;
        float scaled_error_i = beta_i * error_i;

        // Step 5: Rank-1 update: S[i][j] += scaled_error[i] * k[j]
        for (int j = 0; j < head_dim; j++) {
            S[i * head_dim + j] += scaled_error_i * k_shared[j];
        }
        __syncthreads();

        // Step 6: Load q_t into shared memory
        q_shared[i] = IO_LOAD(q, offset + i);
        __syncthreads();

        // Step 7: Compute output[i] = sum_j(S[i][j] * q_shared[j])
        float out_i = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            out_i += S[i * head_dim + j] * q_shared[j];
        }

        // Step 8: Write output
        IO_STORE(output, offset + i, out_i);
        __syncthreads();
    }
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
    // One thread block per (batch, head), head_dim threads per block
    dim3 grid(batch, num_heads);
    dim3 block(head_dim);

    // Shared memory: S[d][d] + k_shared[d] + q_shared[d]
    size_t smem_bytes = (size_t)head_dim * head_dim * sizeof(float)
                      + 2 * head_dim * sizeof(float);

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

    dim3 grid(batch, num_heads);
    dim3 block(head_dim);
    size_t smem_bytes = (size_t)head_dim * head_dim * sizeof(float)
                      + 2 * head_dim * sizeof(float);

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

    dim3 grid(batch, num_heads);
    dim3 block(head_dim);
    size_t smem_bytes = (size_t)head_dim * head_dim * sizeof(float)
                      + 2 * head_dim * sizeof(float);

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
