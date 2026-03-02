// Fused Titans (Surprise-Gated Memory) Scan Kernel
//
// Implements the Titans recurrence with surprise-gated momentum updates:
//
//   pred_t   = M_{t-1} @ k_t                         — memory read
//   error_t  = pred_t - v_t                           — reconstruction error
//   surprise = mean(error^2)                          — surprise signal
//   gate_t   = sigmoid(g_input + log(surprise + eps)) — surprise gate
//   grad_t   = error_t @ k_t^T                        — outer product gradient
//   mom_t    = momentum * mom_{t-1} + grad_t           — momentum update
//   M_t      = M_{t-1} - gate_t * mom_t               — gated memory update
//   o_t      = M_t @ q_t                              — output
//
// Thread layout: one thread per (batch, output_dim_i).
// Each thread holds row i of the M matrix in registers (memory_size floats).
// Shared memory used for k, v, gate vectors and reductions.
//
// Inputs:
//   combined: [batch, seq_len, 4*memory_size] — concatenated Q, K, V, gate_input
//   momentum: scalar float                    — momentum coefficient
//
// Output:
//   out: [batch, seq_len, memory_size]        — output hidden states

#include <cuda_runtime.h>

#define TITANS_MAX_MEM 128

__global__ void fused_titans_scan_kernel(
    const float* __restrict__ combined,  // [B, T, 4*M]
    float* __restrict__ output,          // [B, T, M]
    int batch, int seq_len, int mem_size,
    float momentum
) {
    int b = blockIdx.x;
    int i = threadIdx.x;  // output dimension index (row of M)

    if (b >= batch || i >= mem_size) return;

    // Shared memory layout:
    // k_shared[M] + v_shared[M] + gate_shared[M] + reduce_shared[1]
    extern __shared__ float shared_mem[];
    float* k_shared     = shared_mem;                    // [M]
    float* v_shared     = shared_mem + mem_size;         // [M]
    float* gate_shared  = shared_mem + 2 * mem_size;     // [M]
    float* reduce_shared = shared_mem + 3 * mem_size;    // [1]

    int combined_stride = 4 * mem_size;

    // M[i][j] in registers (row i of memory matrix)
    float M_row[TITANS_MAX_MEM];
    float mom_row[TITANS_MAX_MEM];
    for (int j = 0; j < mem_size; j++) {
        M_row[j] = 0.0f;
        mom_row[j] = 0.0f;
    }

    for (int t = 0; t < seq_len; t++) {
        int base = b * seq_len * combined_stride + t * combined_stride;

        // Load k, v, gate_input into shared memory
        k_shared[i]    = combined[base + mem_size + i];      // K offset
        v_shared[i]    = combined[base + 2 * mem_size + i];  // V offset
        gate_shared[i] = combined[base + 3 * mem_size + i];  // gate offset
        __syncthreads();

        // Step 1: pred_i = M[i,:] @ k (dot product row i with k)
        float pred_i = 0.0f;
        for (int j = 0; j < mem_size; j++) {
            pred_i += M_row[j] * k_shared[j];
        }

        // Step 2: error_i = pred_i - v_i
        float error_i = pred_i - v_shared[i];

        // Step 3: Compute surprise = mean(error^2) via shared reduction
        // Each thread contributes error_i^2 / mem_size
        float local_sq = error_i * error_i;

        // Parallel reduction for mean(error^2)
        // Use k_shared as temp (we're done reading k)
        k_shared[i] = local_sq;
        __syncthreads();

        if (i == 0) {
            float sum = 0.0f;
            for (int j = 0; j < mem_size; j++) {
                sum += k_shared[j];
            }
            reduce_shared[0] = sum / mem_size;
        }
        __syncthreads();

        float surprise = reduce_shared[0];

        // Step 4: Surprise gate: sigmoid(gate_input + log(surprise + eps))
        float surprise_log = logf(surprise + 1.0e-6f);
        float gate_val = 1.0f / (1.0f + expf(-(gate_shared[i] + surprise_log)));

        // Step 5: Gradient = error_i * k^T (rank-1 update to row i)
        // Reload k since we used k_shared as temp
        __syncthreads();
        k_shared[i] = combined[base + mem_size + i];
        __syncthreads();

        // Step 6: Momentum update and gated memory update
        for (int j = 0; j < mem_size; j++) {
            float grad_ij = error_i * k_shared[j];
            mom_row[j] = momentum * mom_row[j] + grad_ij;
            M_row[j] -= gate_val * mom_row[j];
        }

        // Step 7: Output o_i = M_updated[i,:] @ q
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

int fused_titans_scan_launch(
    cudaStream_t stream,
    const float* combined, float* output,
    int batch, int seq_len, int mem_size,
    float momentum
) {
    int threads_per_block = mem_size;  // one thread per output dim
    dim3 grid(batch);
    dim3 block(threads_per_block);
    // k + v + gate + reduce
    size_t smem_bytes = (3 * mem_size + 1) * sizeof(float);

    fused_titans_scan_kernel<<<grid, block, smem_bytes, stream>>>(
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

ffi::Error fused_titans_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> combined,     // [B, T, 4*M]
    ffi::Buffer<ffi::F32> momentum_t,   // scalar [1]
    ffi::ResultBuffer<ffi::F32> output  // [B, T, M]
) {
    auto dims = combined.dimensions();
    int batch    = static_cast<int>(dims[0]);
    int seq_len  = static_cast<int>(dims[1]);
    int mem_size = static_cast<int>(dims[2]) / 4;

    float momentum = reinterpret_cast<const float*>(momentum_t.untyped_data())[0];

    if (mem_size > TITANS_MAX_MEM) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                         "Titans memory_size exceeds max supported (128)");
    }

    int threads_per_block = mem_size;
    dim3 grid(batch);
    dim3 block(threads_per_block);
    size_t smem_bytes = (3 * mem_size + 1) * sizeof(float);

    fused_titans_scan_kernel<<<grid, block, smem_bytes, stream>>>(
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
    fused_titans_scan, fused_titans_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // combined
        .Arg<ffi::Buffer<ffi::F32>>()   // momentum
        .Ret<ffi::Buffer<ffi::F32>>()   // output
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_titans_scan_f32", "CUDA", fused_titans_scan);

#endif  // EXLA_FFI
