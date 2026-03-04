// Fused Linear Recurrence Scan Kernel
//
// Implements the generic linear recurrence: h = a * h + b
// where a and b are pre-computed tensors. No nonlinearities are applied
// inside the kernel — all activations (sigmoid, softplus, etc.) are
// computed on the Elixir/XLA side before calling this kernel.
//
// This single kernel covers multiple architecture patterns:
//   - Griffin RG-LRU:        h = decay * h + x_scaled
//   - MEGA EMA:              h = alpha * h + (1-alpha)*x  (alpha pre-expanded)
//   - SSTransformer EMA:     h = a_t * h + (1-a_t)*x_t
//   - HybridBuilder EMA:     h = a_t * h + (1-a_t)*x_t
//   - GSS SSM:               h = A_bar * h + B*x  (reshaped to 2D)
//   - MambaVision SSM:       h = A_bar * h + B*x  (reshaped to 2D)
//
// Same thread layout as MinGRU: one thread per (batch, hidden) element,
// sequential scan through timesteps with state in registers.
//
// Two compilation modes:
//   1. Standalone (-DSTANDALONE): kernel + C-linkage launch wrapper.
//   2. EXLA FFI (-DEXLA_FFI): kernel + XLA FFI handler + registration.
//
// Inputs:
//   a_vals: [batch, seq_len, hidden] — multiplicative decay coefficients
//   b_vals: [batch, seq_len, hidden] — additive input terms
//   h0:     [batch, hidden]          — initial hidden state
//
// Output:
//   output: [batch, seq_len, hidden] — all hidden states

#include <cuda_runtime.h>
#include "precision.cuh"

// ============================================================================
// Kernel
// ============================================================================

#ifdef EXLA_FFI
namespace {  // anonymous namespace — internal linkage per compilation unit
#endif

__global__ void fused_linear_scan_kernel(
    const io_type* __restrict__ a_vals,  // [B, T, H] multiplicative coefficients
    const io_type* __restrict__ b_vals,  // [B, T, H] additive terms
    const io_type* __restrict__ h0,      // [B, H] initial state
    io_type* __restrict__ output,        // [B, T, H] all hidden states
    int batch, int seq_len, int hidden
) {
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || h >= hidden) return;

    float h_state = IO_LOAD(h0, b * hidden + h);

    for (int t = 0; t < seq_len; t++) {
        int idx = b * seq_len * hidden + t * hidden + h;
        float a = IO_LOAD(a_vals, idx);
        float bv = IO_LOAD(b_vals, idx);

        h_state = a * h_state + bv;
        IO_STORE(output, idx, h_state);
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

int fused_linear_scan_launch(
    cudaStream_t stream,
    const io_type* a_vals, const io_type* b_vals, const io_type* h0,
    io_type* output, int batch, int seq_len, int hidden
) {
    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_linear_scan_kernel<<<grid, block, 0, stream>>>(
        a_vals, b_vals, h0, output,
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

namespace {  // anonymous namespace — prevents symbol collision between f32/bf16

ffi::Error fused_linear_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> a_vals,
    ffi::Buffer<FFI_IO_TYPE> b_vals,
    ffi::Buffer<FFI_IO_TYPE> h0,
    ffi::ResultBuffer<FFI_IO_TYPE> output
) {
    auto dims = a_vals.dimensions();
    int batch   = static_cast<int>(dims[0]);
    int seq_len = static_cast<int>(dims[1]);
    int hidden  = static_cast<int>(dims[2]);

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_linear_scan_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const io_type*>(a_vals.untyped_data()),
        reinterpret_cast<const io_type*>(b_vals.untyped_data()),
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

}  // anonymous namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    HANDLER_SYMBOL(fused_linear_scan), fused_linear_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // a_vals
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // b_vals
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // h0
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // output
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_linear_scan_" PRECISION_SUFFIX, "CUDA", HANDLER_SYMBOL(fused_linear_scan));

#endif  // EXLA_FFI
