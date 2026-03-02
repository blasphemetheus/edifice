// Fused DiagLinear Scan Backward Kernel
//
// Computes gradients for: h_t = a_t * h_{t-1} + b_t
// where a_t = sigmoid(raw_a_t) (applied inside the forward kernel).
//
// This backward kernel works with post-sigmoid a values.
// The sigmoid chain rule is applied in Elixir.
//
// Reverse pass (T-1 -> 0):
//   dh = grad_output[t] + dh_acc
//   da[t] = dh * h_{t-1}
//   db[t] = dh
//   dh_acc = dh * a_t
//   grad_h0 = dh_acc (after loop)
//
// Note: Elixir applies: grad_raw_a = grad_a * a * (1 - a) (sigmoid derivative)
//
// Inputs:
//   a:            [B, T, H] — post-sigmoid decay values
//   h0:           [B, H]    — initial hidden state
//   forward_out:  [B, T, H] — forward pass hidden states
//   grad_output:  [B, T, H] — upstream gradient dL/dh
//
// Outputs:
//   grad_a:    [B, T, H] — gradient w.r.t. a (post-sigmoid)
//   grad_b:    [B, T, H] — gradient w.r.t. b
//   grad_h0:   [B, H]    — gradient w.r.t. initial state

#include <cuda_runtime.h>
#include "precision.cuh"

// ============================================================================
// Kernel
// ============================================================================

__global__ void fused_diag_linear_scan_backward_kernel(
    const io_type* __restrict__ a,            // [B, T, H] post-sigmoid decay
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

        float a_t = IO_LOAD(a, idx);

        // h_{t-1}
        float h_prev;
        if (t == 0) {
            h_prev = IO_LOAD(h0, b * hidden + h);
        } else {
            h_prev = IO_LOAD(forward_out, idx - hidden);
        }

        IO_STORE(grad_a, idx, dh * h_prev);
        IO_STORE(grad_b, idx, dh);
        dh_acc = dh * a_t;
    }

    IO_STORE(grad_h0, b * hidden + h, dh_acc);
}

// ============================================================================
// Standalone launch wrapper (C-linkage for NIF / dlopen)
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

// Output: concatenated [grad_a (B*T*H) | grad_b (B*T*H) | grad_h0 (B*H)]
int fused_diag_linear_scan_backward_launch(
    cudaStream_t stream,
    const io_type* a,
    const io_type* h0, const io_type* forward_out,
    const io_type* grad_output,
    io_type* output_concat,
    int batch, int seq_len, int hidden
) {
    int bth = batch * seq_len * hidden;
    io_type* grad_a    = output_concat;
    io_type* grad_b    = output_concat + bth;
    io_type* grad_h0   = output_concat + 2 * bth;

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_diag_linear_scan_backward_kernel<<<grid, block, 0, stream>>>(
        a, h0, forward_out, grad_output,
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

ffi::Error fused_diag_linear_scan_backward_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> a,
    ffi::Buffer<FFI_IO_TYPE> h0,
    ffi::Buffer<FFI_IO_TYPE> forward_out,
    ffi::Buffer<FFI_IO_TYPE> grad_output,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_a,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_b,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_h0
) {
    auto dims = a.dimensions();
    int batch   = static_cast<int>(dims[0]);
    int seq_len = static_cast<int>(dims[1]);
    int hidden  = static_cast<int>(dims[2]);

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_diag_linear_scan_backward_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const io_type*>(a.untyped_data()),
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_diag_linear_scan_backward, fused_diag_linear_scan_backward_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // a
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // h0
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // forward_out
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // grad_output
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_a
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_b
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_h0
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_diag_linear_scan_backward_" PRECISION_SUFFIX, "CUDA", fused_diag_linear_scan_backward);

#endif  // EXLA_FFI
