// Fused Liquid (LTC) Scan Backward Kernel
//
// Computes gradients for the LTC exact solver recurrence:
//   decay = exp(-1/tau)
//   h_t = act_t + (h_{t-1} - act_t) * decay
//       = act_t * (1 - decay) + h_{t-1} * decay
//
// Reverse pass (T-1 -> 0):
//   dh = grad_output[t] + dh_acc
//   d_decay = dh * (h_{t-1} - act_t)
//   d_act   = dh * (1 - decay)
//   dh_acc  = dh * decay
//   d_tau   = d_decay * exp(-1/tau) * (1/tau^2)   [chain rule: d/dtau exp(-1/tau)]
//   grad_h0 = dh_acc (after loop)
//
// Same thread layout as the forward kernel: one thread per (batch, hidden)
// element, reverse-iterates over T timesteps.
//
// Inputs:
//   tau:         [B, T, H] — post-softplus time constants (same as forward)
//   activation:  [B, T, H] — activation values (same as forward)
//   h0:          [B, H]    — initial hidden state
//   forward_out: [B, T, H] — forward pass hidden states (h_1..h_T)
//   grad_output: [B, T, H] — upstream gradient dL/dh
//
// Outputs:
//   grad_tau:  [B, T, H] — gradient w.r.t. tau
//   grad_act:  [B, T, H] — gradient w.r.t. activation
//   grad_h0:   [B, H]    — gradient w.r.t. initial state

#include <cuda_runtime.h>
#include "precision.cuh"

// ============================================================================
// Kernel
// ============================================================================

#ifdef EXLA_FFI
namespace {  // anonymous namespace — internal linkage per compilation unit
#endif

__global__ void fused_liquid_scan_backward_kernel(
    const io_type* __restrict__ tau,          // [B, T, H]
    const io_type* __restrict__ activation,   // [B, T, H]
    const io_type* __restrict__ h0,           // [B, H]
    const io_type* __restrict__ forward_out,  // [B, T, H]
    const io_type* __restrict__ grad_output,  // [B, T, H]
    io_type* __restrict__ grad_tau,           // [B, T, H]
    io_type* __restrict__ grad_act,           // [B, T, H]
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

        float tau_t = IO_LOAD(tau, idx);
        float act_t = IO_LOAD(activation, idx);

        // h_{t-1}: use h0 for t=0, otherwise forward_out[t-1]
        float h_prev;
        if (t == 0) {
            h_prev = IO_LOAD(h0, b * hidden + h);
        } else {
            h_prev = IO_LOAD(forward_out, idx - hidden);
        }

        // Recompute decay from forward pass
        float decay = expf(-1.0f / tau_t);

        // Gradient w.r.t. activation: dh * (1 - decay)
        float d_act = dh * (1.0f - decay);

        // Gradient w.r.t. decay: dh * (h_prev - act)
        float d_decay = dh * (h_prev - act_t);

        // Chain rule: d/dtau exp(-1/tau) = exp(-1/tau) * (1/tau^2)
        float d_tau = d_decay * decay / (tau_t * tau_t);

        IO_STORE(grad_act, idx, d_act);
        IO_STORE(grad_tau, idx, d_tau);

        // Accumulate gradient through h_{t-1}
        dh_acc = dh * decay;
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

// Output: concatenated [grad_tau (B*T*H) | grad_act (B*T*H) | grad_h0 (B*H)]
int fused_liquid_scan_backward_launch(
    cudaStream_t stream,
    const io_type* tau, const io_type* activation,
    const io_type* h0, const io_type* forward_out,
    const io_type* grad_output,
    io_type* output_concat,
    int batch, int seq_len, int hidden
) {
    int bth = batch * seq_len * hidden;
    io_type* grad_tau = output_concat;
    io_type* grad_act = output_concat + bth;
    io_type* grad_h0  = output_concat + 2 * bth;

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_liquid_scan_backward_kernel<<<grid, block, 0, stream>>>(
        tau, activation, h0, forward_out, grad_output,
        grad_tau, grad_act, grad_h0,
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

ffi::Error fused_liquid_scan_backward_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> tau,
    ffi::Buffer<FFI_IO_TYPE> activation,
    ffi::Buffer<FFI_IO_TYPE> h0,
    ffi::Buffer<FFI_IO_TYPE> forward_out,
    ffi::Buffer<FFI_IO_TYPE> grad_output,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_tau,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_act,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_h0
) {
    auto dims = tau.dimensions();
    int batch   = static_cast<int>(dims[0]);
    int seq_len = static_cast<int>(dims[1]);
    int hidden  = static_cast<int>(dims[2]);

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_liquid_scan_backward_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const io_type*>(tau.untyped_data()),
        reinterpret_cast<const io_type*>(activation.untyped_data()),
        reinterpret_cast<const io_type*>(h0.untyped_data()),
        reinterpret_cast<const io_type*>(forward_out.untyped_data()),
        reinterpret_cast<const io_type*>(grad_output.untyped_data()),
        reinterpret_cast<io_type*>(grad_tau->untyped_data()),
        reinterpret_cast<io_type*>(grad_act->untyped_data()),
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
    HANDLER_SYMBOL(fused_liquid_scan_backward), fused_liquid_scan_backward_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // tau
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // activation
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // h0
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // forward_out
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // grad_output
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_tau
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_act
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_h0
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_liquid_scan_backward_" PRECISION_SUFFIX, "CUDA", HANDLER_SYMBOL(fused_liquid_scan_backward));

#endif  // EXLA_FFI
