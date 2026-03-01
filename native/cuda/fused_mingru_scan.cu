// Fused MinGRU Parallel Scan Kernel
//
// Eliminates per-timestep kernel launch overhead by scanning the entire
// sequence in a single kernel. Each thread handles one (batch, hidden)
// element and walks through all timesteps sequentially, keeping state
// in registers.
//
// For seq_len <= 64, this sequential-per-thread approach is faster than
// a parallel prefix scan because it avoids the 2x memory traffic of the
// up-sweep/down-sweep and has zero synchronization overhead.
//
// Two compilation modes:
//   1. Standalone (-DSTANDALONE): Compiles kernel + C-linkage launch wrapper.
//      Use for testing and NIF-based integration.
//   2. EXLA FFI (-DEXLA_FFI): Compiles kernel + XLA FFI handler + registration.
//      Use when building inside deps/exla/c_src/exla/custom_calls/.
//
// Inputs:
//   gates:      [batch, seq_len, hidden] — pre-computed sigmoid(W_z @ x)
//   candidates: [batch, seq_len, hidden] — pre-computed W_h @ x
//   h0:         [batch, hidden]          — initial hidden state
//
// Output:
//   output:     [batch, seq_len, hidden] — all hidden states

#include <cuda_runtime.h>

// ============================================================================
// Kernel (always compiled)
// ============================================================================

__global__ void fused_mingru_scan_kernel(
    const float* __restrict__ gates,       // [B, T, H] sigmoid values
    const float* __restrict__ candidates,  // [B, T, H] candidate values
    const float* __restrict__ h0,          // [B, H] initial state
    float* __restrict__ output,            // [B, T, H] all hidden states
    int batch, int seq_len, int hidden
) {
    // Each thread processes one (batch, hidden) pair across all timesteps
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || h >= hidden) return;

    // Load initial state into register — stays on-chip for entire scan
    float h_state = h0[b * hidden + h];

    // Sequential scan through timesteps (all state stays in registers)
    for (int t = 0; t < seq_len; t++) {
        int idx = b * seq_len * hidden + t * hidden + h;
        float z = gates[idx];
        float h_tilde = candidates[idx];

        // MinGRU update: h = (1-z)*h + z*h_tilde
        h_state = (1.0f - z) * h_state + z * h_tilde;

        output[idx] = h_state;
    }
}

// ============================================================================
// Standalone launch wrapper (C-linkage for NIF / dlopen)
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

// Launch the fused MinGRU scan kernel.
// All pointers must be device pointers. stream may be 0 for default stream.
// Returns cudaSuccess (0) on success.
int fused_mingru_scan_launch(
    cudaStream_t stream,
    const float* gates,
    const float* candidates,
    const float* h0,
    float* output,
    int batch, int seq_len, int hidden
) {
    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_mingru_scan_kernel<<<grid, block, 0, stream>>>(
        gates, candidates, h0, output,
        batch, seq_len, hidden
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
#include "xla/ffi/ffi_api.h"

namespace ffi = xla::ffi;

ffi::Error fused_mingru_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> gates,
    ffi::Buffer<ffi::F32> candidates,
    ffi::Buffer<ffi::F32> h0,
    ffi::ResultBuffer<ffi::F32> output
) {
    auto dims = gates.dimensions();
    int batch   = static_cast<int>(dims[0]);
    int seq_len = static_cast<int>(dims[1]);
    int hidden  = static_cast<int>(dims[2]);

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_mingru_scan_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(gates.untyped_data()),
        reinterpret_cast<const float*>(candidates.untyped_data()),
        reinterpret_cast<const float*>(h0.untyped_data()),
        reinterpret_cast<float*>(output->untyped_data()),
        batch, seq_len, hidden
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_mingru_scan, fused_mingru_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // gates
        .Arg<ffi::Buffer<ffi::F32>>()   // candidates
        .Arg<ffi::Buffer<ffi::F32>>()   // h0
        .Ret<ffi::Buffer<ffi::F32>>()   // output
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_mingru_scan_f32", "CUDA", fused_mingru_scan);

#endif  // EXLA_FFI
