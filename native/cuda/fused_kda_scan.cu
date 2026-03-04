// Fused KDA (Kimi Delta Attention) Scan Kernel
//
// Channel-wise decay delta rule with per-dimension alpha gating.
// Unlike GatedDeltaNet which uses scalar alpha per head, KDA uses
// vector alpha[d] — each row of the state matrix decays independently.
//
// KDA recurrence:
//   decay:      S[i][j] *= exp(alpha_t[i])   (channel-wise decay)
//   retrieval = S @ k
//   error    = v - retrieval
//   S_t      = S_decayed + beta_t * outer(error, k)
//   o_t      = S_t @ q
//
// Thread layout: one thread BLOCK per (batch, head) pair, head_dim threads.
// Each thread owns row i of S[d][d].
//
// Inputs (all pre-computed on Elixir/XLA side):
//   q:     [B, T, H, d] — query vectors (L2-normalized)
//   k:     [B, T, H, d] — key vectors (L2-normalized)
//   v:     [B, T, H, d] — value vectors
//   alpha: [B, T, H, d] — per-channel decay (log-space, apply exp)
//   beta:  [B, T, H]    — scalar update gate per head (post-sigmoid)
//
// Output:
//   output: [B, T, H, d] — retrieval outputs per head
//
// Shared memory budget (head_dim=64):
//   S matrix: 64*64*4 = 16KB
//   k_shared: 64*4 = 256 bytes
//   q_shared: 64*4 = 256 bytes
//   Total: ~17KB — within 48KB limit

#include <cuda_runtime.h>
#include "precision.cuh"

// ============================================================================
// Kernel
// ============================================================================

#ifdef EXLA_FFI
namespace {  // anonymous namespace — internal linkage per compilation unit
#endif

__global__ void fused_kda_scan_kernel(
    const io_type* __restrict__ q,       // [B, T, H, d]
    const io_type* __restrict__ k,       // [B, T, H, d]
    const io_type* __restrict__ v,       // [B, T, H, d]
    const io_type* __restrict__ alpha,   // [B, T, H, d]
    const io_type* __restrict__ beta,    // [B, T, H]
    io_type* __restrict__ output,        // [B, T, H, d]
    int seq_len,
    int num_heads,
    int head_dim
) {
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

    // Strides for [B, T, H, d] tensors
    int BHd = seq_len * num_heads * head_dim;   // stride for batch dim
    int THd = num_heads * head_dim;             // stride for time dim
    int Hd  = head_dim;                         // stride for head dim

    // Stride for beta: [B, T, H]
    int beta_BH = seq_len * num_heads;
    int beta_TH = num_heads;

    int base_bh = b * BHd + h * Hd;
    int beta_base_b = b * beta_BH;

    for (int t = 0; t < seq_len; t++) {
        int offset = base_bh + t * THd;

        // Step 1: Load k_t into shared memory
        k_shared[i] = IO_LOAD(k, offset + i);
        __syncthreads();

        // Step 2: Channel-wise decay — each thread decays its own row
        float alpha_i = expf(IO_LOAD(alpha, offset + i));
        for (int j = 0; j < head_dim; j++) {
            S[i * head_dim + j] *= alpha_i;
        }
        __syncthreads();

        // Step 3: Compute retrieval[i] = sum_j(S[i][j] * k_shared[j])
        float retrieval = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            retrieval += S[i * head_dim + j] * k_shared[j];
        }

        // Step 4: Delta rule update
        float v_i = IO_LOAD(v, offset + i);
        float beta_val = IO_LOAD(beta, beta_base_b + t * beta_TH + h);
        float error_i = v_i - retrieval;
        float scaled_error_i = beta_val * error_i;

        // S[i][j] += scaled_error[i] * k[j]
        for (int j = 0; j < head_dim; j++) {
            S[i * head_dim + j] += scaled_error_i * k_shared[j];
        }
        __syncthreads();

        // Step 5: Load q_t into shared memory
        q_shared[i] = IO_LOAD(q, offset + i);
        __syncthreads();

        // Step 6: Compute output[i] = sum_j(S[i][j] * q_shared[j])
        float out_i = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            out_i += S[i * head_dim + j] * q_shared[j];
        }

        // Step 7: Write output
        IO_STORE(output, offset + i, out_i);
        __syncthreads();
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

int fused_kda_scan_launch(
    cudaStream_t stream,
    const io_type* q, const io_type* k, const io_type* v,
    const io_type* alpha, const io_type* beta,
    io_type* output,
    int batch, int seq_len, int num_heads, int head_dim
) {
    dim3 grid(batch, num_heads);
    dim3 block(head_dim);

    // Shared memory: S[d][d] + k_shared[d] + q_shared[d]
    size_t smem_bytes = (size_t)head_dim * head_dim * sizeof(float)
                      + 2 * head_dim * sizeof(float);

    fused_kda_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        q, k, v, alpha, beta, output,
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

ffi::Error fused_kda_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> q,       // [B, T, H, d]
    ffi::Buffer<FFI_IO_TYPE> k,       // [B, T, H, d]
    ffi::Buffer<FFI_IO_TYPE> v,       // [B, T, H, d]
    ffi::Buffer<FFI_IO_TYPE> alpha,   // [B, T, H, d]
    ffi::Buffer<FFI_IO_TYPE> beta,    // [B, T, H]
    ffi::ResultBuffer<FFI_IO_TYPE> output  // [B, T, H, d]
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

    fused_kda_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const io_type*>(q.untyped_data()),
        reinterpret_cast<const io_type*>(k.untyped_data()),
        reinterpret_cast<const io_type*>(v.untyped_data()),
        reinterpret_cast<const io_type*>(alpha.untyped_data()),
        reinterpret_cast<const io_type*>(beta.untyped_data()),
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
    HANDLER_SYMBOL(fused_kda_scan), fused_kda_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // q
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // k
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // v
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // alpha
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // beta
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // output
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_kda_scan_" PRECISION_SUFFIX, "CUDA", HANDLER_SYMBOL(fused_kda_scan));

#endif  // EXLA_FFI
