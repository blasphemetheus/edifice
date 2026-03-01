// Fused MinLSTM Parallel Scan Kernel
//
// Same strategy as MinGRU: one thread per (batch, hidden) element,
// sequential scan through timesteps with state in registers.
//
// MinLSTM differs from MinGRU by having separate forget and input gates
// that are normalized to sum to 1: f' = f/(f+i), i' = i/(f+i).
// The normalization happens inside the kernel to avoid extra kernel
// launches for the divide operations.
//
// Two compilation modes:
//   1. Standalone (-DSTANDALONE): Compiles kernel + C-linkage launch wrapper.
//   2. EXLA FFI (-DEXLA_FFI): Compiles kernel + XLA FFI handler + registration.
//
// Inputs:
//   forget_gates: [batch, seq_len, hidden] — pre-computed sigmoid(W_f @ x)
//   input_gates:  [batch, seq_len, hidden] — pre-computed sigmoid(W_i @ x)
//   candidates:   [batch, seq_len, hidden] — pre-computed W_h @ x
//   h0:           [batch, hidden]          — initial cell/hidden state
//
// Output:
//   output:       [batch, seq_len, hidden] — all hidden states

#include <cuda_runtime.h>

constexpr float NORM_EPS = 1.0e-6f;

// ============================================================================
// Kernel (always compiled)
// ============================================================================

__global__ void fused_minlstm_scan_kernel(
    const float* __restrict__ forget_gates,  // [B, T, H] sigmoid values
    const float* __restrict__ input_gates,   // [B, T, H] sigmoid values
    const float* __restrict__ candidates,    // [B, T, H] candidate values
    const float* __restrict__ h0,            // [B, H] initial state
    float* __restrict__ output,              // [B, T, H] all hidden states
    int batch, int seq_len, int hidden
) {
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || h >= hidden) return;

    // Load initial state into register
    float c_state = h0[b * hidden + h];

    for (int t = 0; t < seq_len; t++) {
        int idx = b * seq_len * hidden + t * hidden + h;
        float f = forget_gates[idx];
        float i = input_gates[idx];
        float cand = candidates[idx];

        // Normalize gates: f' = f/(f+i+eps), i' = i/(f+i+eps)
        float gate_sum = f + i + NORM_EPS;
        float f_norm = f / gate_sum;
        float i_norm = i / gate_sum;

        // MinLSTM update: c = f'*c + i'*candidate
        c_state = f_norm * c_state + i_norm * cand;

        output[idx] = c_state;
    }
}

// ============================================================================
// Standalone launch wrapper (C-linkage for NIF / dlopen)
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

int fused_minlstm_scan_launch(
    cudaStream_t stream,
    const float* forget_gates,
    const float* input_gates,
    const float* candidates,
    const float* h0,
    float* output,
    int batch, int seq_len, int hidden
) {
    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_minlstm_scan_kernel<<<grid, block, 0, stream>>>(
        forget_gates, input_gates, candidates, h0, output,
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

namespace ffi = xla::ffi;

ffi::Error fused_minlstm_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> forget_gates,
    ffi::Buffer<ffi::F32> input_gates,
    ffi::Buffer<ffi::F32> candidates,
    ffi::Buffer<ffi::F32> h0,
    ffi::ResultBuffer<ffi::F32> output
) {
    auto dims = forget_gates.dimensions();
    int batch   = static_cast<int>(dims[0]);
    int seq_len = static_cast<int>(dims[1]);
    int hidden  = static_cast<int>(dims[2]);

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_minlstm_scan_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(forget_gates.untyped_data()),
        reinterpret_cast<const float*>(input_gates.untyped_data()),
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
    fused_minlstm_scan, fused_minlstm_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // forget_gates
        .Arg<ffi::Buffer<ffi::F32>>()   // input_gates
        .Arg<ffi::Buffer<ffi::F32>>()   // candidates
        .Arg<ffi::Buffer<ffi::F32>>()   // h0
        .Ret<ffi::Buffer<ffi::F32>>()   // output
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_minlstm_scan_f32", "CUDA", fused_minlstm_scan);

#endif  // EXLA_FFI
