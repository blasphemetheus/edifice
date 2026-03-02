// Fused MinLSTM Scan Backward Kernel
//
// Computes gradients for: c_t = f'_t * c_{t-1} + i'_t * cand_t
// where f' = f/(f+i+eps), i' = i/(f+i+eps), f = sigmoid(raw_f), i = sigmoid(raw_i)
//
// The kernel works with post-sigmoid f and i values.
// Normalization gradients are computed inside the kernel.
// Chain rule for sigmoid is applied in Elixir.
//
// Reverse pass (T-1 -> 0):
//   dc = grad_output[t] + dc_acc
//   S = f + i + eps
//   f' = f / S, i' = i / S
//   df' = dc * c_{t-1}
//   di' = dc * cand_t
//   dcand[t] = dc * i'
//   dc_acc = dc * f'
//
//   Through normalization (quotient rule):
//     df = (df' * (i + eps) - di' * i) / S^2
//     di = (-df' * f + di' * (f + eps)) / S^2
//
// Inputs:
//   f:            [B, T, H] — post-sigmoid forget gate
//   i:            [B, T, H] — post-sigmoid input gate
//   candidates:   [B, T, H] — candidate values
//   h0:           [B, H]    — initial cell state
//   forward_out:  [B, T, H] — forward pass cell states
//   grad_output:  [B, T, H] — upstream gradient dL/dc
//
// Outputs:
//   grad_f:    [B, T, H] — gradient w.r.t. f (post-sigmoid)
//   grad_i:    [B, T, H] — gradient w.r.t. i (post-sigmoid)
//   grad_cand: [B, T, H] — gradient w.r.t. candidates
//   grad_h0:   [B, H]    — gradient w.r.t. initial cell state

#include <cuda_runtime.h>
#include "precision.cuh"

constexpr float NORM_EPS = 1.0e-6f;

// ============================================================================
// Kernel
// ============================================================================

__global__ void fused_minlstm_scan_backward_kernel(
    const io_type* __restrict__ f,            // [B, T, H] post-sigmoid forget
    const io_type* __restrict__ i_gate,       // [B, T, H] post-sigmoid input
    const io_type* __restrict__ candidates,   // [B, T, H]
    const io_type* __restrict__ h0,           // [B, H]
    const io_type* __restrict__ forward_out,  // [B, T, H]
    const io_type* __restrict__ grad_output,  // [B, T, H]
    io_type* __restrict__ grad_f,             // [B, T, H]
    io_type* __restrict__ grad_i,             // [B, T, H]
    io_type* __restrict__ grad_cand,          // [B, T, H]
    io_type* __restrict__ grad_h0,            // [B, H]
    int batch, int seq_len, int hidden
) {
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || h >= hidden) return;

    float dc_acc = 0.0f;

    for (int t = seq_len - 1; t >= 0; t--) {
        int idx = b * seq_len * hidden + t * hidden + h;

        float dc = IO_LOAD(grad_output, idx) + dc_acc;

        float f_t = IO_LOAD(f, idx);
        float i_t = IO_LOAD(i_gate, idx);
        float cand_t = IO_LOAD(candidates, idx);

        // Normalization
        float S = f_t + i_t + NORM_EPS;
        float f_norm = f_t / S;
        float i_norm = i_t / S;

        // c_{t-1}
        float c_prev;
        if (t == 0) {
            c_prev = IO_LOAD(h0, b * hidden + h);
        } else {
            c_prev = IO_LOAD(forward_out, idx - hidden);
        }

        // Gradients w.r.t. normalized gates
        float df_norm = dc * c_prev;
        float di_norm = dc * cand_t;

        // Gradient w.r.t. candidates
        IO_STORE(grad_cand, idx, dc * i_norm);

        // Accumulate gradient flowing back through c_{t-1}
        dc_acc = dc * f_norm;

        // Through normalization: quotient rule
        // f' = f/S => df = (df' * S - f * (df'+di')) / S^2
        //           = (df' * (i+eps) - di' * f) / S^2  ... wait, let's be precise
        //
        // f' = f/(f+i+eps), i' = i/(f+i+eps)
        // d(f')/df = (S - f)/S^2 = (i+eps)/S^2
        // d(f')/di = -f/S^2
        // d(i')/df = -i/S^2
        // d(i')/di = (S - i)/S^2 = (f+eps)/S^2
        //
        // df_total = df' * d(f')/df + di' * d(i')/df
        //          = df' * (i+eps)/S^2 + di' * (-i)/S^2
        //          = (df' * (i+eps) - di' * i) / S^2
        //
        // di_total = df' * d(f')/di + di' * d(i')/di
        //          = df' * (-f)/S^2 + di' * (f+eps)/S^2
        //          = (-df' * f + di' * (f+eps)) / S^2

        float S2 = S * S;
        IO_STORE(grad_f, idx, (df_norm * (i_t + NORM_EPS) - di_norm * i_t) / S2);
        IO_STORE(grad_i, idx, (-df_norm * f_t + di_norm * (f_t + NORM_EPS)) / S2);
    }

    IO_STORE(grad_h0, b * hidden + h, dc_acc);
}

// ============================================================================
// Standalone launch wrapper (C-linkage for NIF / dlopen)
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

// Output: concatenated [grad_f (B*T*H) | grad_i (B*T*H) | grad_cand (B*T*H) | grad_h0 (B*H)]
int fused_minlstm_scan_backward_launch(
    cudaStream_t stream,
    const io_type* f, const io_type* i_gate,
    const io_type* candidates,
    const io_type* h0, const io_type* forward_out,
    const io_type* grad_output,
    io_type* output_concat,
    int batch, int seq_len, int hidden
) {
    int bth = batch * seq_len * hidden;
    io_type* grad_f    = output_concat;
    io_type* grad_i    = output_concat + bth;
    io_type* grad_cand = output_concat + 2 * bth;
    io_type* grad_h0   = output_concat + 3 * bth;

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_minlstm_scan_backward_kernel<<<grid, block, 0, stream>>>(
        f, i_gate, candidates, h0, forward_out, grad_output,
        grad_f, grad_i, grad_cand, grad_h0,
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

ffi::Error fused_minlstm_scan_backward_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> f,
    ffi::Buffer<FFI_IO_TYPE> i_gate,
    ffi::Buffer<FFI_IO_TYPE> candidates,
    ffi::Buffer<FFI_IO_TYPE> h0,
    ffi::Buffer<FFI_IO_TYPE> forward_out,
    ffi::Buffer<FFI_IO_TYPE> grad_output,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_f,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_i,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_cand,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_h0
) {
    auto dims = f.dimensions();
    int batch   = static_cast<int>(dims[0]);
    int seq_len = static_cast<int>(dims[1]);
    int hidden  = static_cast<int>(dims[2]);

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_minlstm_scan_backward_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const io_type*>(f.untyped_data()),
        reinterpret_cast<const io_type*>(i_gate.untyped_data()),
        reinterpret_cast<const io_type*>(candidates.untyped_data()),
        reinterpret_cast<const io_type*>(h0.untyped_data()),
        reinterpret_cast<const io_type*>(forward_out.untyped_data()),
        reinterpret_cast<const io_type*>(grad_output.untyped_data()),
        reinterpret_cast<io_type*>(grad_f->untyped_data()),
        reinterpret_cast<io_type*>(grad_i->untyped_data()),
        reinterpret_cast<io_type*>(grad_cand->untyped_data()),
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
    fused_minlstm_scan_backward, fused_minlstm_scan_backward_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // f
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // i_gate
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // candidates
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // h0
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // forward_out
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // grad_output
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_f
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_i
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_cand
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_h0
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_minlstm_scan_backward_" PRECISION_SUFFIX, "CUDA", fused_minlstm_scan_backward);

#endif  // EXLA_FFI
