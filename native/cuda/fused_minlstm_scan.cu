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
#include "precision.cuh"

constexpr float NORM_EPS = 1.0e-6f;

// ============================================================================
// Kernel (always compiled)
// ============================================================================

__global__ void fused_minlstm_scan_kernel(
    const io_type* __restrict__ forget_gates,  // [B, T, H] sigmoid values
    const io_type* __restrict__ input_gates,   // [B, T, H] sigmoid values
    const io_type* __restrict__ candidates,    // [B, T, H] candidate values
    const io_type* __restrict__ h0,            // [B, H] initial state
    io_type* __restrict__ output,              // [B, T, H] all hidden states
    int batch, int seq_len, int hidden
) {
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || h >= hidden) return;

    // Load initial state into register
    float c_state = IO_LOAD(h0, b * hidden + h);

    for (int t = 0; t < seq_len; t++) {
        int idx = b * seq_len * hidden + t * hidden + h;
        float f = IO_LOAD(forget_gates, idx);
        float i = IO_LOAD(input_gates, idx);
        float cand = IO_LOAD(candidates, idx);

        // Normalize gates: f' = f/(f+i+eps), i' = i/(f+i+eps)
        float gate_sum = f + i + NORM_EPS;
        float f_norm = f / gate_sum;
        float i_norm = i / gate_sum;

        // MinLSTM update: c = f'*c + i'*candidate
        c_state = f_norm * c_state + i_norm * cand;

        IO_STORE(output, idx, c_state);
    }
}

// ============================================================================
// Standalone launch wrapper (C-linkage for NIF / dlopen)
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

int fused_minlstm_scan_launch(
    cudaStream_t stream,
    const io_type* forget_gates,
    const io_type* input_gates,
    const io_type* candidates,
    const io_type* h0,
    io_type* output,
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
    ffi::Buffer<FFI_IO_TYPE> forget_gates,
    ffi::Buffer<FFI_IO_TYPE> input_gates,
    ffi::Buffer<FFI_IO_TYPE> candidates,
    ffi::Buffer<FFI_IO_TYPE> h0,
    ffi::ResultBuffer<FFI_IO_TYPE> output
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
        reinterpret_cast<const io_type*>(forget_gates.untyped_data()),
        reinterpret_cast<const io_type*>(input_gates.untyped_data()),
        reinterpret_cast<const io_type*>(candidates.untyped_data()),
        reinterpret_cast<const io_type*>(h0.untyped_data()),
        reinterpret_cast<io_type*>(output->untyped_data()),
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
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // forget_gates
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // input_gates
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // candidates
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // h0
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // output
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_minlstm_scan_" PRECISION_SUFFIX, "CUDA", fused_minlstm_scan);

#endif  // EXLA_FFI
