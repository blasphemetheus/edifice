// Fused Liquid (LTC) Exact Solver Scan Kernel
//
// Implements the analytical solution to the LTC ODE:
//   dx/dt = (activation - x) / tau
//   x(t+1) = activation + (x(t) - activation) * exp(-1.0/tau)
//
// The dt=1.0 is baked in (one frame per timestep). The Elixir side
// computes tau = softplus(tau_proj) + 0.1 and activation = f_proj
// before passing to the kernel.
//
// Same thread layout as MinGRU: one thread per (batch, hidden) element,
// sequential scan through timesteps with state in registers.
//
// Two compilation modes:
//   1. Standalone (-DSTANDALONE): kernel + C-linkage launch wrapper.
//   2. EXLA FFI (-DEXLA_FFI): kernel + XLA FFI handler + registration.
//
// Inputs:
//   tau:        [batch, seq_len, hidden] — post-softplus time constants
//   activation: [batch, seq_len, hidden] — f_proj network output
//   h0:         [batch, hidden]          — initial hidden state
//
// Output:
//   output:     [batch, seq_len, hidden] — all hidden states

#include <cuda_runtime.h>
#include "precision.cuh"

// ============================================================================
// Kernel
// ============================================================================

__global__ void fused_liquid_scan_kernel(
    const io_type* __restrict__ tau,         // [B, T, H] post-softplus time constants
    const io_type* __restrict__ activation,  // [B, T, H] activation network output
    const io_type* __restrict__ h0,          // [B, H] initial state
    io_type* __restrict__ output,            // [B, T, H] all hidden states
    int batch, int seq_len, int hidden
) {
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || h >= hidden) return;

    float h_state = IO_LOAD(h0, b * hidden + h);

    for (int t = 0; t < seq_len; t++) {
        int idx = b * seq_len * hidden + t * hidden + h;
        float tau_t = IO_LOAD(tau, idx);
        float act_t = IO_LOAD(activation, idx);

        // Exact solution: h = activation + (h - activation) * exp(-dt/tau)
        // with dt = 1.0
        float decay = expf(-1.0f / tau_t);
        h_state = act_t + (h_state - act_t) * decay;

        IO_STORE(output, idx, h_state);
    }
}

// ============================================================================
// Standalone launch wrapper (C-linkage for NIF / dlopen)
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

int fused_liquid_scan_launch(
    cudaStream_t stream,
    const io_type* tau, const io_type* activation, const io_type* h0,
    io_type* output, int batch, int seq_len, int hidden
) {
    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_liquid_scan_kernel<<<grid, block, 0, stream>>>(
        tau, activation, h0, output,
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

ffi::Error fused_liquid_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> tau,
    ffi::Buffer<FFI_IO_TYPE> activation,
    ffi::Buffer<FFI_IO_TYPE> h0,
    ffi::ResultBuffer<FFI_IO_TYPE> output
) {
    auto dims = tau.dimensions();
    int batch   = static_cast<int>(dims[0]);
    int seq_len = static_cast<int>(dims[1]);
    int hidden  = static_cast<int>(dims[2]);

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_liquid_scan_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const io_type*>(tau.untyped_data()),
        reinterpret_cast<const io_type*>(activation.untyped_data()),
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
    fused_liquid_scan, fused_liquid_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // tau
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // activation
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // h0
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // output
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_liquid_scan_" PRECISION_SUFFIX, "CUDA", fused_liquid_scan);

#endif  // EXLA_FFI
