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
//   combined: [batch, seq_len, combined_stride] — concatenated Q, K, V, gate_input
//     NIF path:  combined_stride = 4*M (momentum passed separately)
//     FFI path:  combined_stride = 4*M+1 (momentum packed as extra column)
//   momentum: scalar float                    — momentum coefficient
//
// Output:
//   out: [batch, seq_len, memory_size]        — output hidden states

#include <cuda_runtime.h>
#include "precision.cuh"

#define TITANS_MAX_MEM 128

#ifdef EXLA_FFI
namespace {  // anonymous namespace — internal linkage per compilation unit
#endif

__global__ void fused_titans_scan_kernel(
    const io_type* __restrict__ combined,  // [B, T, combined_stride]
    io_type* __restrict__ output,          // [B, T, M]
    int batch, int seq_len, int mem_size,
    float momentum, int combined_stride
) {
    int b = blockIdx.x;
    int i = threadIdx.x;  // output dimension index (row of M)

    if (b >= batch || i >= mem_size) return;

    // Shared memory layout:
    extern __shared__ float shared_mem[];
    float* k_shared     = shared_mem;                    // [M]
    float* v_shared     = shared_mem + mem_size;         // [M]
    float* gate_shared  = shared_mem + 2 * mem_size;     // [M]
    float* reduce_shared = shared_mem + 3 * mem_size;    // [1]

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
        k_shared[i]    = IO_LOAD(combined, base + mem_size + i);      // K offset
        v_shared[i]    = IO_LOAD(combined, base + 2 * mem_size + i);  // V offset
        gate_shared[i] = IO_LOAD(combined, base + 3 * mem_size + i);  // gate offset
        __syncthreads();

        // Step 1: pred_i = M[i,:] @ k
        float pred_i = 0.0f;
        for (int j = 0; j < mem_size; j++) {
            pred_i += M_row[j] * k_shared[j];
        }

        // Step 2: error_i = pred_i - v_i
        float error_i = pred_i - v_shared[i];

        // Step 3: Compute surprise = mean(error^2)
        float local_sq = error_i * error_i;

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

        // Step 4: Surprise gate
        float surprise_log = logf(surprise + 1.0e-6f);
        float gate_val = 1.0f / (1.0f + expf(-(gate_shared[i] + surprise_log)));

        // Step 5: Reload k since we used k_shared as temp
        __syncthreads();
        k_shared[i] = IO_LOAD(combined, base + mem_size + i);
        __syncthreads();

        // Step 6: Momentum update and gated memory update
        for (int j = 0; j < mem_size; j++) {
            float grad_ij = error_i * k_shared[j];
            mom_row[j] = momentum * mom_row[j] + grad_ij;
            M_row[j] -= gate_val * mom_row[j];
        }

        // Step 7: Output o_i = M_updated[i,:] @ q
        __syncthreads();
        k_shared[i] = IO_LOAD(combined, base + i);  // Q at offset 0
        __syncthreads();

        float o_i = 0.0f;
        for (int j = 0; j < mem_size; j++) {
            o_i += M_row[j] * k_shared[j];
        }

        IO_STORE(output, b * seq_len * mem_size + t * mem_size + i, o_i);
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

int fused_titans_scan_launch(
    cudaStream_t stream,
    const io_type* combined, io_type* output,
    int batch, int seq_len, int mem_size,
    float momentum
) {
    int threads_per_block = mem_size;
    int combined_stride = 4 * mem_size;
    dim3 grid(batch);
    dim3 block(threads_per_block);
    size_t smem_bytes = (3 * mem_size + 1) * sizeof(float);

    fused_titans_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        combined, output,
        batch, seq_len, mem_size, momentum, combined_stride
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

// 1 operand: packed tensor [B, T, 4*M+1] with momentum in the extra column.
// This avoids the scalar buffer operand segfault in XLA while still
// passing the actual momentum value configured by the user.
ffi::Error fused_titans_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> combined,     // [B, T, 4*M+1]
    ffi::ResultBuffer<FFI_IO_TYPE> output  // [B, T, M]
) {
    auto dims = combined.dimensions();
    int batch    = static_cast<int>(dims[0]);
    int seq_len  = static_cast<int>(dims[1]);
    int combined_stride = static_cast<int>(dims[2]);  // 4*M+1
    int mem_size = (combined_stride - 1) / 4;

    // Read momentum from packed tensor (last column of first row)
    const io_type* combined_ptr = reinterpret_cast<const io_type*>(combined.untyped_data());
    io_type momentum_raw;
    cudaMemcpyAsync(&momentum_raw, combined_ptr + 4 * mem_size,
                    sizeof(io_type), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
#ifdef USE_BF16
    float momentum = __bfloat162float(momentum_raw);
#else
    float momentum = momentum_raw;
#endif

    if (mem_size > TITANS_MAX_MEM) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                         "Titans memory_size exceeds max supported (128)");
    }

    int threads_per_block = mem_size;
    dim3 grid(batch);
    dim3 block(threads_per_block);
    size_t smem_bytes = (3 * mem_size + 1) * sizeof(float);

    fused_titans_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        combined_ptr,
        reinterpret_cast<io_type*>(output->untyped_data()),
        batch, seq_len, mem_size, momentum, combined_stride
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

}  // anonymous namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    HANDLER_SYMBOL(fused_titans_scan), fused_titans_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // combined
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // output
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_titans_scan_" PRECISION_SUFFIX, "CUDA", HANDLER_SYMBOL(fused_titans_scan));

#endif  // EXLA_FFI
