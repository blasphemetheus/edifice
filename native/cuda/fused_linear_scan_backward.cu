// Fused Linear Scan Backward Kernel
//
// Computes gradients for the linear recurrence: h_t = a_t * h_{t-1} + b_t
//
// Reverse pass (T-1 -> 0):
//   dh = grad_output[t] + dh_acc
//   da[t] = dh * h_{t-1}
//   db[t] = dh
//   dh_acc = dh * a_t
//   grad_h0 = dh_acc (after loop)
//
// Same thread layout as the forward kernel: one thread per (batch, hidden)
// element, reverse-iterates over T timesteps. State (dh_acc) stays in
// registers — no shared memory needed.
//
// Two compilation modes:
//   1. Standalone (default): kernel + C-linkage launch wrapper for NIF.
//   2. EXLA FFI (-DEXLA_FFI): kernel + XLA FFI handler + registration.
//
// Inputs:
//   a_vals:       [B, T, H] — multiplicative coefficients (from forward)
//   h0:           [B, H]    — initial hidden state
//   forward_out:  [B, T, H] — forward pass hidden states (h_1..h_T)
//   grad_output:  [B, T, H] — upstream gradient dL/dh
//
// Outputs (concatenated for NIF, separate buffers for FFI):
//   grad_a:  [B, T, H] — gradient w.r.t. a_vals
//   grad_b:  [B, T, H] — gradient w.r.t. b_vals
//   grad_h0: [B, H]    — gradient w.r.t. initial state

#include <cuda_runtime.h>
#include "precision.cuh"

// ============================================================================
// Kernel
// ============================================================================

#ifdef EXLA_FFI
namespace {  // anonymous namespace — internal linkage per compilation unit
#endif

__global__ void fused_linear_scan_backward_kernel(
    const io_type* __restrict__ a_vals,       // [B, T, H]
    const io_type* __restrict__ h0,           // [B, H]
    const io_type* __restrict__ forward_out,  // [B, T, H]
    const io_type* __restrict__ grad_output,  // [B, T, H]
    io_type* __restrict__ grad_a,             // [B, T, H]
    io_type* __restrict__ grad_b,             // [B, T, H]
    io_type* __restrict__ grad_h0,            // [B, H]
    int batch, int seq_len, int hidden
) {
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || h >= hidden) return;

    float dh_acc = 0.0f;

    for (int t = seq_len - 1; t >= 0; t--) {
        int idx = b * seq_len * hidden + t * hidden + h;

        float dh = IO_LOAD(grad_output, idx) + dh_acc;

        // h_{t-1}: use h0 for t=0, otherwise forward_out[t-1]
        float h_prev;
        if (t == 0) {
            h_prev = IO_LOAD(h0, b * hidden + h);
        } else {
            h_prev = IO_LOAD(forward_out, idx - hidden);  // forward_out[b,t-1,h]
        }

        float a = IO_LOAD(a_vals, idx);

        IO_STORE(grad_a, idx, dh * h_prev);
        IO_STORE(grad_b, idx, dh);
        dh_acc = dh * a;
    }

    IO_STORE(grad_h0, b * hidden + h, dh_acc);
}

// ============================================================================
// Standalone launch wrapper (C-linkage for NIF / dlopen)
// ============================================================================

#ifdef EXLA_FFI
}  // anonymous namespace
#endif

#ifndef EXLA_FFI

extern "C" {

// Outputs are written to a single concatenated buffer:
//   [grad_a (B*T*H) | grad_b (B*T*H) | grad_h0 (B*H)] floats
int fused_linear_scan_backward_launch(
    cudaStream_t stream,
    const io_type* a_vals, const io_type* h0,
    const io_type* forward_out, const io_type* grad_output,
    io_type* output_concat,  // concatenated: grad_a + grad_b + grad_h0
    int batch, int seq_len, int hidden
) {
    int bth = batch * seq_len * hidden;
    io_type* grad_a  = output_concat;
    io_type* grad_b  = output_concat + bth;
    io_type* grad_h0 = output_concat + 2 * bth;

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_linear_scan_backward_kernel<<<grid, block, 0, stream>>>(
        a_vals, h0, forward_out, grad_output,
        grad_a, grad_b, grad_h0,
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

namespace {  // anonymous namespace — prevents symbol collision between f32/bf16

ffi::Error fused_linear_scan_backward_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> a_vals,
    ffi::Buffer<FFI_IO_TYPE> h0,
    ffi::Buffer<FFI_IO_TYPE> forward_out,
    ffi::Buffer<FFI_IO_TYPE> grad_output,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_a,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_b,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_h0
) {
    auto dims = a_vals.dimensions();
    int batch   = static_cast<int>(dims[0]);
    int seq_len = static_cast<int>(dims[1]);
    int hidden  = static_cast<int>(dims[2]);

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_linear_scan_backward_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const io_type*>(a_vals.untyped_data()),
        reinterpret_cast<const io_type*>(h0.untyped_data()),
        reinterpret_cast<const io_type*>(forward_out.untyped_data()),
        reinterpret_cast<const io_type*>(grad_output.untyped_data()),
        reinterpret_cast<io_type*>(grad_a->untyped_data()),
        reinterpret_cast<io_type*>(grad_b->untyped_data()),
        reinterpret_cast<io_type*>(grad_h0->untyped_data()),
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
    HANDLER_SYMBOL(fused_linear_scan_backward), fused_linear_scan_backward_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // a_vals
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // h0
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // forward_out
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // grad_output
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_a
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_b
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_h0
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_linear_scan_backward_" PRECISION_SUFFIX, "CUDA", HANDLER_SYMBOL(fused_linear_scan_backward));

#endif  // EXLA_FFI
