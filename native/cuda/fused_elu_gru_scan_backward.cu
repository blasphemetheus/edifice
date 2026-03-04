// Fused ELU-GRU Scan Backward Kernel
//
// Computes gradients for: h_t = (1 - z_t) * h_{t-1} + z_t * c_t
// where z_t = sigmoid(gates_t), c_t = 1 + elu(candidates_t)
// (both applied inside the forward kernel from raw pre-activation values)
//
// This backward kernel works with post-activation z and c values.
// Chain rules for sigmoid and ELU are applied in Elixir.
//
// Reverse pass (T-1 -> 0):
//   dh = grad_output[t] + dh_acc
//   dz[t] = dh * (c_t - h_{t-1})
//   dc[t] = dh * z_t
//   dh_acc = dh * (1 - z_t)
//   grad_h0 = dh_acc (after loop)
//
// Note: The Elixir side applies:
//   grad_gates = grad_z * z * (1 - z)          (sigmoid derivative)
//   grad_cand_raw = grad_c * elu'(raw_cand)     (elu derivative)
//   where elu'(x) = 1 if x >= 0, exp(x) if x < 0
//
// Inputs:
//   z:            [B, T, H] — post-sigmoid gate values
//   c:            [B, T, H] — post-ELU+1 candidate values (1 + elu(raw))
//   h0:           [B, H]    — initial hidden state
//   forward_out:  [B, T, H] — forward pass hidden states
//   grad_output:  [B, T, H] — upstream gradient dL/dh
//
// Outputs:
//   grad_z:    [B, T, H] — gradient w.r.t. z (post-sigmoid)
//   grad_c:    [B, T, H] — gradient w.r.t. c (post-elu+1)
//   grad_h0:   [B, H]    — gradient w.r.t. initial state

#include <cuda_runtime.h>
#include "precision.cuh"

// ============================================================================
// Kernel
// ============================================================================

#ifdef EXLA_FFI
namespace {  // anonymous namespace — internal linkage per compilation unit
#endif

__global__ void fused_elu_gru_scan_backward_kernel(
    const io_type* __restrict__ z,            // [B, T, H] post-sigmoid gates
    const io_type* __restrict__ c,            // [B, T, H] post-elu+1 candidates
    const io_type* __restrict__ h0,           // [B, H]
    const io_type* __restrict__ forward_out,  // [B, T, H]
    const io_type* __restrict__ grad_output,  // [B, T, H]
    io_type* __restrict__ grad_z,             // [B, T, H]
    io_type* __restrict__ grad_c,             // [B, T, H]
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

        float z_t = IO_LOAD(z, idx);
        float c_t = IO_LOAD(c, idx);

        // h_{t-1}
        float h_prev;
        if (t == 0) {
            h_prev = IO_LOAD(h0, b * hidden + h);
        } else {
            h_prev = IO_LOAD(forward_out, idx - hidden);
        }

        IO_STORE(grad_z, idx, dh * (c_t - h_prev));
        IO_STORE(grad_c, idx, dh * z_t);
        dh_acc = dh * (1.0f - z_t);
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

// Output: concatenated [grad_z (B*T*H) | grad_c (B*T*H) | grad_h0 (B*H)]
int fused_elu_gru_scan_backward_launch(
    cudaStream_t stream,
    const io_type* z, const io_type* c,
    const io_type* h0, const io_type* forward_out,
    const io_type* grad_output,
    io_type* output_concat,
    int batch, int seq_len, int hidden
) {
    int bth = batch * seq_len * hidden;
    io_type* grad_z    = output_concat;
    io_type* grad_c    = output_concat + bth;
    io_type* grad_h0   = output_concat + 2 * bth;

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_elu_gru_scan_backward_kernel<<<grid, block, 0, stream>>>(
        z, c, h0, forward_out, grad_output,
        grad_z, grad_c, grad_h0,
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

ffi::Error fused_elu_gru_scan_backward_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> z,
    ffi::Buffer<FFI_IO_TYPE> c,
    ffi::Buffer<FFI_IO_TYPE> h0,
    ffi::Buffer<FFI_IO_TYPE> forward_out,
    ffi::Buffer<FFI_IO_TYPE> grad_output,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_z,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_c,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_h0
) {
    auto dims = z.dimensions();
    int batch   = static_cast<int>(dims[0]);
    int seq_len = static_cast<int>(dims[1]);
    int hidden  = static_cast<int>(dims[2]);

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_elu_gru_scan_backward_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const io_type*>(z.untyped_data()),
        reinterpret_cast<const io_type*>(c.untyped_data()),
        reinterpret_cast<const io_type*>(h0.untyped_data()),
        reinterpret_cast<const io_type*>(forward_out.untyped_data()),
        reinterpret_cast<const io_type*>(grad_output.untyped_data()),
        reinterpret_cast<io_type*>(grad_z->untyped_data()),
        reinterpret_cast<io_type*>(grad_c->untyped_data()),
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
    HANDLER_SYMBOL(fused_elu_gru_scan_backward), fused_elu_gru_scan_backward_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // z
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // c
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // h0
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // forward_out
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // grad_output
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_z
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_c
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_h0
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_elu_gru_scan_backward_" PRECISION_SUFFIX, "CUDA", HANDLER_SYMBOL(fused_elu_gru_scan_backward));

#endif  // EXLA_FFI
