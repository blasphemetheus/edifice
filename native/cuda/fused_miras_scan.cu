// Fused MIRAS (Memory as Iterative Reasoning over Associative Structures) Scan Kernel
//
// Implements the MIRAS Moneta variant recurrence with data-dependent gates:
//
//   pred_t     = M_{t-1} @ k_t                         — memory read
//   error_t    = pred_t - v_t                           — reconstruction error
//   grad_t     = 2 * error_t @ k_t^T                   — MSE gradient (Moneta, p=2)
//   mom_t      = momentum * mom_{t-1} + grad_t          — momentum update
//   alpha_t    = sigmoid(alpha_raw)                     — data-dependent forgetting
//   eta_t      = sigmoid(eta_raw)                       — data-dependent learning rate
//   M_t        = alpha_t * M_{t-1} - eta_t * mom_t     — gated memory update
//   M_t        = M_t / max(||M_t||_row, eps)           — L2 row normalization (Moneta)
//   o_t        = M_t @ q_t                             — output
//
// Thread layout: one thread per (batch, output_dim_i).
// Each thread holds row i of M and momentum matrices in registers.
//
// Inputs:
//   combined: [batch, seq_len, 5*memory_size] — concatenated Q, K, V, alpha, eta
//   momentum: scalar float                    — momentum coefficient
//
// Output:
//   out: [batch, seq_len, memory_size]        — output hidden states

#include <cuda_runtime.h>

#define MIRAS_MAX_MEM 128

__global__ void fused_miras_scan_kernel(
    const float* __restrict__ combined,  // [B, T, 5*M]
    float* __restrict__ output,          // [B, T, M]
    int batch, int seq_len, int mem_size,
    float momentum
) {
    int b = blockIdx.x;
    int i = threadIdx.x;  // output dimension index (row of M)

    if (b >= batch || i >= mem_size) return;

    // Shared memory: k[M] + v[M] + alpha[M] + eta[M] + reduce[1]
    extern __shared__ float shared_mem[];
    float* k_shared     = shared_mem;                       // [M]
    float* v_shared     = shared_mem + mem_size;            // [M]
    float* alpha_shared = shared_mem + 2 * mem_size;        // [M]
    float* eta_shared   = shared_mem + 3 * mem_size;        // [M]
    float* reduce_shared = shared_mem + 4 * mem_size;       // [1]

    int combined_stride = 5 * mem_size;

    // M[i][j] and momentum in registers
    float M_row[MIRAS_MAX_MEM];
    float mom_row[MIRAS_MAX_MEM];
    for (int j = 0; j < mem_size; j++) {
        M_row[j] = 0.0f;
        mom_row[j] = 0.0f;
    }

    for (int t = 0; t < seq_len; t++) {
        int base = b * seq_len * combined_stride + t * combined_stride;

        // Load k, v, alpha, eta into shared memory
        k_shared[i]     = combined[base + mem_size + i];        // K offset
        v_shared[i]     = combined[base + 2 * mem_size + i];    // V offset
        alpha_shared[i] = combined[base + 3 * mem_size + i];    // alpha offset
        eta_shared[i]   = combined[base + 4 * mem_size + i];    // eta offset
        __syncthreads();

        // Step 1: pred_i = M[i,:] @ k
        float pred_i = 0.0f;
        for (int j = 0; j < mem_size; j++) {
            pred_i += M_row[j] * k_shared[j];
        }

        // Step 2: error = pred - v, MSE gradient coeff = 2 * error
        float error_i = pred_i - v_shared[i];
        float grad_coeff = 2.0f * error_i;

        // Step 3: Data-dependent gates
        float alpha_val = 1.0f / (1.0f + expf(-alpha_shared[i]));  // sigmoid
        float eta_val   = 1.0f / (1.0f + expf(-eta_shared[i]));    // sigmoid

        // Step 4: Momentum update + gated memory update
        for (int j = 0; j < mem_size; j++) {
            float grad_ij = grad_coeff * k_shared[j];
            mom_row[j] = momentum * mom_row[j] + grad_ij;
            M_row[j] = alpha_val * M_row[j] - eta_val * mom_row[j];
        }

        // Step 5: L2 row normalization (Moneta variant)
        // norm_i = sqrt(sum_j(M[i][j]^2))
        float row_norm_sq = 0.0f;
        for (int j = 0; j < mem_size; j++) {
            row_norm_sq += M_row[j] * M_row[j];
        }
        float row_norm = sqrtf(row_norm_sq + 1.0e-12f);
        if (row_norm > 1.0e-6f) {
            float inv_norm = 1.0f / row_norm;
            for (int j = 0; j < mem_size; j++) {
                M_row[j] *= inv_norm;
            }
        }

        // Step 6: Output o_i = M[i,:] @ q
        // Load q into shared (reuse k_shared)
        __syncthreads();
        k_shared[i] = combined[base + i];  // Q at offset 0
        __syncthreads();

        float o_i = 0.0f;
        for (int j = 0; j < mem_size; j++) {
            o_i += M_row[j] * k_shared[j];
        }

        output[b * seq_len * mem_size + t * mem_size + i] = o_i;
        __syncthreads();
    }
}

// ============================================================================
// Standalone launch wrapper
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

int fused_miras_scan_launch(
    cudaStream_t stream,
    const float* combined, float* output,
    int batch, int seq_len, int mem_size,
    float momentum
) {
    int threads_per_block = mem_size;
    dim3 grid(batch);
    dim3 block(threads_per_block);
    // k + v + alpha + eta + reduce
    size_t smem_bytes = (4 * mem_size + 1) * sizeof(float);

    fused_miras_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        combined, output,
        batch, seq_len, mem_size, momentum
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

ffi::Error fused_miras_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> combined,     // [B, T, 5*M]
    ffi::Buffer<ffi::F32> momentum_t,   // scalar [1]
    ffi::ResultBuffer<ffi::F32> output  // [B, T, M]
) {
    auto dims = combined.dimensions();
    int batch    = static_cast<int>(dims[0]);
    int seq_len  = static_cast<int>(dims[1]);
    int mem_size = static_cast<int>(dims[2]) / 5;

    float momentum = reinterpret_cast<const float*>(momentum_t.untyped_data())[0];

    if (mem_size > MIRAS_MAX_MEM) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                         "MIRAS memory_size exceeds max supported (128)");
    }

    int threads_per_block = mem_size;
    dim3 grid(batch);
    dim3 block(threads_per_block);
    size_t smem_bytes = (4 * mem_size + 1) * sizeof(float);

    fused_miras_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const float*>(combined.untyped_data()),
        reinterpret_cast<float*>(output->untyped_data()),
        batch, seq_len, mem_size, momentum
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_miras_scan, fused_miras_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // combined
        .Arg<ffi::Buffer<ffi::F32>>()   // momentum
        .Ret<ffi::Buffer<ffi::F32>>()   // output
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_miras_scan_f32", "CUDA", fused_miras_scan);

#endif  // EXLA_FFI
