// Fused Standard LSTM Scan Kernel
//
// Implements the classic LSTM with in-kernel recurrent matmul:
//   i = σ(wx[0:H]   + R@h_prev[0:H])      input gate
//   f = σ(wx[H:2H]  + R@h_prev[H:2H])     forget gate
//   g = tanh(wx[2H:3H] + R@h_prev[2H:3H]) cell candidate
//   o = σ(wx[3H:4H] + R@h_prev[3H:4H])    output gate
//   c = f*c_prev + i*g                      cell update
//   h = o * tanh(c)                          hidden update
//
// The input projection W@x + bias is pre-computed on the Axon/Nx side.
// The hidden-to-hidden matmul R@h is done inside the kernel using
// shared memory for the current hidden state vector.
//
// Thread layout: one thread per (batch, hidden) element.
// Each thread handles one hidden dimension across all timesteps.
//
// Inputs:
//   wx:  [batch, seq_len, 4*hidden] — pre-computed W@x + bias (i, f, g, o gates)
//   R:   [hidden, 4*hidden]         — recurrent weight matrix (constant)
//   h0:  [batch, hidden]            — initial hidden state
//   c0:  [batch, hidden]            — initial cell state
//
// Output:
//   out: [batch, seq_len, hidden]   — hidden states for all timesteps

#include <cuda_runtime.h>
#include "precision.cuh"

// ============================================================================
// Kernel
// ============================================================================

// Note: This kernel requires hidden_size <= 256 for the shared memory
// reduction approach. For the recurrent matmul h@R, we use a two-phase
// approach: each thread computes its contribution and we accumulate via
// shared memory.
//
// Phase 1: Each thread writes h[i] to shared memory
// Phase 2: Each thread computes sum_j(h[j] * R[j, gate_offset + i])
//          by reading all h[j] from shared memory

__global__ void fused_lstm_scan_kernel(
    const io_type* __restrict__ wx,     // [B, T, 4*H]
    const io_type* __restrict__ R,      // [H, 4*H]
    const io_type* __restrict__ h0,     // [B, H]
    const io_type* __restrict__ c0,     // [B, H]
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
    float c_val = IO_LOAD(c0, b * hidden + i);

    int hidden4 = 4 * hidden;

    for (int t = 0; t < seq_len; t++) {
        // Write current h to shared memory for matmul
        h_shared[i] = h_val;
        __syncthreads();

        // Compute R@h for all 4 gates at position i
        // rh[g] = sum_j(h_shared[j] * R[j * 4*H + g*H + i])
        float rh_i = 0.0f;   // input gate
        float rh_f = 0.0f;   // forget gate
        float rh_g = 0.0f;   // cell candidate
        float rh_o = 0.0f;   // output gate

        for (int j = 0; j < hidden; j++) {
            float h_j = h_shared[j];
            int r_base = j * hidden4;
            rh_i += h_j * IO_LOAD(R, r_base + i);
            rh_f += h_j * IO_LOAD(R, r_base + hidden + i);
            rh_g += h_j * IO_LOAD(R, r_base + 2 * hidden + i);
            rh_o += h_j * IO_LOAD(R, r_base + 3 * hidden + i);
        }

        // Load pre-computed W@x + bias gates
        int wx_idx = b * seq_len * hidden4 + t * hidden4;
        float i_t = 1.0f / (1.0f + expf(-(IO_LOAD(wx, wx_idx + i) + rh_i)));
        float f_t = 1.0f / (1.0f + expf(-(IO_LOAD(wx, wx_idx + hidden + i) + rh_f)));
        float g_t = tanhf(IO_LOAD(wx, wx_idx + 2 * hidden + i) + rh_g);
        float o_t = 1.0f / (1.0f + expf(-(IO_LOAD(wx, wx_idx + 3 * hidden + i) + rh_o)));

        // Cell update: c = f*c_prev + i*g
        c_val = f_t * c_val + i_t * g_t;

        // Hidden update: h = o * tanh(c)
        h_val = o_t * tanhf(c_val);

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

int fused_lstm_scan_launch(
    cudaStream_t stream,
    const io_type* wx, const io_type* R,
    const io_type* h0, const io_type* c0,
    io_type* output,
    int batch, int seq_len, int hidden
) {
    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    size_t smem_bytes = hidden * sizeof(float);

    fused_lstm_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        wx, R, h0, c0, output,
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

ffi::Error fused_lstm_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> wx,      // [B, T, 4*H]
    ffi::Buffer<FFI_IO_TYPE> R,       // [H, 4*H]
    ffi::Buffer<FFI_IO_TYPE> h0,      // [B, H]
    ffi::Buffer<FFI_IO_TYPE> c0,      // [B, H]
    ffi::ResultBuffer<FFI_IO_TYPE> output  // [B, T, H]
) {
    auto wx_dims = wx.dimensions();
    int batch   = static_cast<int>(wx_dims[0]);
    int seq_len = static_cast<int>(wx_dims[1]);
    int hidden  = static_cast<int>(wx_dims[2]) / 4;

    auto h0_dims = h0.dimensions();
    int hidden_h0 = static_cast<int>(h0_dims[1]);
    if (hidden != hidden_h0) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                         "wx last dim must be 4 * h0 last dim");
    }

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    size_t smem_bytes = hidden * sizeof(float);

    fused_lstm_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const io_type*>(wx.untyped_data()),
        reinterpret_cast<const io_type*>(R.untyped_data()),
        reinterpret_cast<const io_type*>(h0.untyped_data()),
        reinterpret_cast<const io_type*>(c0.untyped_data()),
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
    fused_lstm_scan, fused_lstm_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // wx
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // R
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // h0
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // c0
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // output
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_lstm_scan_" PRECISION_SUFFIX, "CUDA", fused_lstm_scan);

#endif  // EXLA_FFI
