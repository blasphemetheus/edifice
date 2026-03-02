// Fused Standard GRU Scan Kernel
//
// Implements the classic GRU with in-kernel recurrent matmul:
//   r = σ(wx[0:H]   + R@h_prev[0:H])          reset gate
//   z = σ(wx[H:2H]  + R@h_prev[H:2H])         update gate
//   n = tanh(wx[2H:3H] + r * R@h_prev[2H:3H]) candidate (reset applied to recurrent part only)
//   h = (1-z)*n + z*h_prev                      hidden update
//
// Note: The reset gate r multiplies only the recurrent contribution
// R_n@h (the third column block of R@h), not the input part wx_n.
// This matches the "fully gated" GRU formulation used by PyTorch and Axon.
//
// The input projection W@x + bias is pre-computed on the Axon/Nx side.
// The hidden-to-hidden matmul R@h is done inside the kernel using
// shared memory for the current hidden state vector.
//
// Thread layout: one thread per (batch, hidden) element.
// Each thread handles one hidden dimension across all timesteps.
//
// Inputs:
//   wx:  [batch, seq_len, 3*hidden] — pre-computed W@x + bias (r, z, n gates)
//   R:   [hidden, 3*hidden]         — recurrent weight matrix (constant)
//   h0:  [batch, hidden]            — initial hidden state
//
// Output:
//   out: [batch, seq_len, hidden]   — hidden states for all timesteps

#include <cuda_runtime.h>
#include "precision.cuh"

// ============================================================================
// Kernel
// ============================================================================

__global__ void fused_gru_scan_kernel(
    const io_type* __restrict__ wx,     // [B, T, 3*H]
    const io_type* __restrict__ R,      // [H, 3*H]
    const io_type* __restrict__ h0,     // [B, H]
    io_type* __restrict__ output,       // [B, T, H]
    int batch, int seq_len, int hidden
) {
    int b = blockIdx.x;
    int i = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || i >= hidden) return;

    // Shared memory for h_prev (needed for R@h matmul)
    extern __shared__ float h_shared[];  // [hidden]

    // Load initial state
    float h_val = IO_LOAD(h0, b * hidden + i);

    int hidden3 = 3 * hidden;

    for (int t = 0; t < seq_len; t++) {
        // Write current h to shared memory for matmul
        h_shared[i] = h_val;
        __syncthreads();

        // Compute R@h for all 3 gates at position i
        // rh[g] = sum_j(h_shared[j] * R[j * 3*H + g*H + i])
        float rh_r = 0.0f;   // reset gate
        float rh_z = 0.0f;   // update gate
        float rh_n = 0.0f;   // candidate (will be multiplied by reset gate)

        for (int j = 0; j < hidden; j++) {
            float h_j = h_shared[j];
            int r_base = j * hidden3;
            rh_r += h_j * IO_LOAD(R, r_base + i);
            rh_z += h_j * IO_LOAD(R, r_base + hidden + i);
            rh_n += h_j * IO_LOAD(R, r_base + 2 * hidden + i);
        }

        // Load pre-computed W@x + bias gates
        int wx_idx = b * seq_len * hidden3 + t * hidden3;
        float r_t = 1.0f / (1.0f + expf(-(IO_LOAD(wx, wx_idx + i) + rh_r)));
        float z_t = 1.0f / (1.0f + expf(-(IO_LOAD(wx, wx_idx + hidden + i) + rh_z)));

        // Candidate: reset gate applied only to recurrent contribution
        float n_t = tanhf(IO_LOAD(wx, wx_idx + 2 * hidden + i) + r_t * rh_n);

        // Hidden update: blend between candidate and previous hidden
        h_val = (1.0f - z_t) * n_t + z_t * h_val;

        // Write output
        IO_STORE(output, b * seq_len * hidden + t * hidden + i, h_val);
        __syncthreads();
    }
}

// ============================================================================
// Standalone launch wrapper
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

int fused_gru_scan_launch(
    cudaStream_t stream,
    const io_type* wx, const io_type* R,
    const io_type* h0,
    io_type* output,
    int batch, int seq_len, int hidden
) {
    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    size_t smem_bytes = hidden * sizeof(float);

    fused_gru_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        wx, R, h0, output,
        batch, seq_len, hidden
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

ffi::Error fused_gru_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> wx,      // [B, T, 3*H]
    ffi::Buffer<FFI_IO_TYPE> R,       // [H, 3*H]
    ffi::Buffer<FFI_IO_TYPE> h0,      // [B, H]
    ffi::ResultBuffer<FFI_IO_TYPE> output  // [B, T, H]
) {
    auto wx_dims = wx.dimensions();
    int batch   = static_cast<int>(wx_dims[0]);
    int seq_len = static_cast<int>(wx_dims[1]);
    int hidden  = static_cast<int>(wx_dims[2]) / 3;

    auto h0_dims = h0.dimensions();
    int hidden_h0 = static_cast<int>(h0_dims[1]);
    if (hidden != hidden_h0) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                         "wx last dim must be 3 * h0 last dim");
    }

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    size_t smem_bytes = hidden * sizeof(float);

    fused_gru_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const io_type*>(wx.untyped_data()),
        reinterpret_cast<const io_type*>(R.untyped_data()),
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
    fused_gru_scan, fused_gru_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // wx
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // R
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // h0
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // output
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_gru_scan_" PRECISION_SUFFIX, "CUDA", fused_gru_scan);

#endif  // EXLA_FFI
