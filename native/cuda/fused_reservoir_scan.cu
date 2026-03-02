// Fused Reservoir (Echo State Network) Scan Kernel
//
// Implements the reservoir recurrence:
//   h_t = (1-leak)*h_{t-1} + leak*tanh(wx_t + W_res @ h_{t-1})
//
// When leak_rate = 1.0 (default), simplifies to:
//   h_t = tanh(wx_t + W_res @ h_{t-1})
//
// The input projection W_in @ x is pre-computed on the Axon/Nx side.
// The reservoir weight matmul W_res @ h is done in-kernel via shared memory.
// W_res is a FIXED (non-trainable) sparse matrix with controlled spectral radius.
//
// Thread layout: one thread per (batch, hidden_dim) element.
// Each thread handles one hidden dimension across all timesteps.
//
// Inputs:
//   wx:    [batch, seq_len, reservoir_size]  -- pre-computed W_in @ x
//   w_res: [reservoir_size, reservoir_size]  -- fixed reservoir weights
//   h0:    [batch, reservoir_size]           -- initial hidden state
//   leak:  scalar float                      -- leak rate (1.0 = no leaking)
//
// Output:
//   out:   [batch, reservoir_size]           -- final hidden state only

#include <cuda_runtime.h>
#include "precision.cuh"

__global__ void fused_reservoir_scan_kernel(
    const io_type* __restrict__ wx,       // [B, T, H]
    const io_type* __restrict__ w_res,    // [H, H]
    const io_type* __restrict__ h0,       // [B, H]
    io_type* __restrict__ output,         // [B, H]
    int batch, int seq_len, int hidden,
    float leak_rate
) {
    int b = blockIdx.x;
    int i = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || i >= hidden) return;

    // Shared memory for h_prev (needed for W_res @ h matmul)
    extern __shared__ float h_shared[];  // [hidden]

    // Load initial state
    float h_val = IO_LOAD(h0, b * hidden + i);

    for (int t = 0; t < seq_len; t++) {
        // Write current h to shared memory for matmul
        h_shared[i] = h_val;
        __syncthreads();

        // Compute W_res @ h for dimension i
        // rh_i = sum_j(h_shared[j] * w_res[j * H + i])
        float rh_i = 0.0f;
        for (int j = 0; j < hidden; j++) {
            rh_i += h_shared[j] * IO_LOAD(w_res, j * hidden + i);
        }

        // Load pre-computed wx
        int wx_idx = b * seq_len * hidden + t * hidden + i;
        float pre_act = IO_LOAD(wx, wx_idx) + rh_i;

        // h_new = tanh(wx + W_res @ h)
        float h_new = tanhf(pre_act);

        // Leaky integration: h = (1-leak)*h_prev + leak*h_new
        if (leak_rate < 1.0f) {
            h_val = (1.0f - leak_rate) * h_val + leak_rate * h_new;
        } else {
            h_val = h_new;
        }

        __syncthreads();
    }

    // Write final hidden state only (ESN only needs last state for readout)
    IO_STORE(output, b * hidden + i, h_val);
}

// ============================================================================
// Standalone launch wrapper
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

int fused_reservoir_scan_launch(
    cudaStream_t stream,
    const io_type* wx, const io_type* w_res,
    const io_type* h0, io_type* output,
    int batch, int seq_len, int hidden,
    float leak_rate
) {
    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    size_t smem_bytes = hidden * sizeof(float);

    fused_reservoir_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        wx, w_res, h0, output,
        batch, seq_len, hidden, leak_rate
    );

    return (int)cudaGetLastError();
}

}  // extern "C"

#endif  // !EXLA_FFI

// ============================================================================
// XLA FFI integration
// ============================================================================

#ifdef EXLA_FFI

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

ffi::Error fused_reservoir_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> wx,       // [B, T, H]
    ffi::Buffer<FFI_IO_TYPE> w_res,    // [H, H]
    ffi::Buffer<FFI_IO_TYPE> h0,       // [B, H]
    ffi::Buffer<ffi::F32> leak_t,   // scalar [1]
    ffi::ResultBuffer<FFI_IO_TYPE> output  // [B, H]
) {
    auto wx_dims = wx.dimensions();
    int batch   = static_cast<int>(wx_dims[0]);
    int seq_len = static_cast<int>(wx_dims[1]);
    int hidden  = static_cast<int>(wx_dims[2]);

    float leak_rate = reinterpret_cast<const float*>(leak_t.untyped_data())[0];

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    size_t smem_bytes = hidden * sizeof(float);

    fused_reservoir_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const io_type*>(wx.untyped_data()),
        reinterpret_cast<const io_type*>(w_res.untyped_data()),
        reinterpret_cast<const io_type*>(h0.untyped_data()),
        reinterpret_cast<io_type*>(output->untyped_data()),
        batch, seq_len, hidden, leak_rate
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_reservoir_scan, fused_reservoir_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // wx
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // w_res
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // h0
        .Arg<ffi::Buffer<ffi::F32>>()   // leak_rate
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // output
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_reservoir_scan_" PRECISION_SUFFIX, "CUDA", fused_reservoir_scan);

#endif  // EXLA_FFI
