// Fused Standard LSTM Scan Backward Kernel
//
// Computes gradients for the standard LSTM:
//   i = σ(wx[0:H]   + R@h)      input gate
//   f = σ(wx[H:2H]  + R@h)      forget gate
//   g = tanh(wx[2H:3H] + R@h)   cell candidate
//   o = σ(wx[3H:4H] + R@h)      output gate
//   c = f*c_prev + i*g           cell update
//   h = o * tanh(c)              hidden update
//
// Two-pass approach:
//   Pass 1 (forward, t=0..T-1): Recompute gate activations and cell states,
//     storing per-timestep values needed for the reverse pass.
//   Pass 2 (reverse, t=T-1..0): Accumulate gradients.
//
// The forward pass uses shared memory for R@h (same as forward kernel).
// The reverse pass uses shared memory for dgate @ R^T.
//
// Reverse pass math:
//   dh = grad_output[t] + dh_acc
//   do_pre = dh * tanh(c_t) * o_t * (1 - o_t)
//   dc = dh * o_t * (1 - tanh²(c_t)) + dc_acc
//   di_pre = dc * g_t * i_t * (1 - i_t)
//   df_pre = dc * c_{t-1} * f_t * (1 - f_t)
//   dg_pre = dc * i_t * (1 - g_t²)
//   dc_acc = dc * f_t
//   dh_acc = [di_pre, df_pre, dg_pre, do_pre] @ R^T
//   grad_wx[t] = [di_pre, df_pre, dg_pre, do_pre]
//
// Inputs:
//   wx:          [B, T, 4*H] — pre-computed W@x + bias
//   R:           [H, 4*H]   — recurrent weight matrix
//   h0:          [B, H]     — initial hidden state
//   c0:          [B, H]     — initial cell state
//   forward_out: [B, T, H]  — forward pass hidden states (for h_prev)
//   grad_output: [B, T, H]  — upstream gradient dL/dh
//
// Outputs:
//   grad_wx:  [B, T, 4*H] — gradient w.r.t. pre-activation gate values
//   grad_h0:  [B, H]      — gradient w.r.t. initial hidden state
//   grad_c0:  [B, H]      — gradient w.r.t. initial cell state
//
// Note: grad_R is computed in Elixir via matmul:
//   grad_R = sum_b,t( h_prev^T @ grad_wx_t )

#include <cuda_runtime.h>
#include "precision.cuh"

// Maximum sequence length for thread-local storage.
// Longer sequences fall back to Elixir.
#define MAX_SEQ_LEN 1024

// ============================================================================
// Kernel
// ============================================================================

#ifdef EXLA_FFI
namespace {  // anonymous namespace — internal linkage per compilation unit
#endif

__global__ void fused_lstm_scan_backward_kernel(
    const io_type* __restrict__ wx,           // [B, T, 4*H]
    const io_type* __restrict__ R,            // [H, 4*H]
    const io_type* __restrict__ h0,           // [B, H]
    const io_type* __restrict__ c0,           // [B, H]
    const io_type* __restrict__ forward_out,  // [B, T, H]
    const io_type* __restrict__ grad_output,  // [B, T, H]
    io_type* __restrict__ grad_wx,            // [B, T, 4*H]
    io_type* __restrict__ grad_h0,            // [B, H]
    io_type* __restrict__ grad_c0,            // [B, H]
    int batch, int seq_len, int hidden
) {
    int b = blockIdx.x;
    int i = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || i >= hidden) return;

    extern __shared__ float h_shared[];  // [hidden] — stays float for R@h matmul

    int hidden4 = 4 * hidden;

    // Thread-local arrays for per-timestep gate activations and cell states
    // These live in GPU local memory (spills to L1/L2 cache)
    float local_i[MAX_SEQ_LEN];
    float local_f[MAX_SEQ_LEN];
    float local_g[MAX_SEQ_LEN];
    float local_o[MAX_SEQ_LEN];
    float local_c[MAX_SEQ_LEN];

    // ========================================
    // Pass 1: Forward — recompute gate activations and cell states
    // ========================================
    float h_val = IO_LOAD(h0, b * hidden + i);
    float c_val = IO_LOAD(c0, b * hidden + i);

    for (int t = 0; t < seq_len; t++) {
        // Write current h to shared memory for R@h matmul
        h_shared[i] = h_val;
        __syncthreads();

        // Compute R@h for all 4 gates
        float rh_i = 0.0f, rh_f = 0.0f, rh_g = 0.0f, rh_o = 0.0f;
        for (int j = 0; j < hidden; j++) {
            float h_j = h_shared[j];
            int r_base = j * hidden4;
            rh_i += h_j * IO_LOAD(R, r_base + i);
            rh_f += h_j * IO_LOAD(R, r_base + hidden + i);
            rh_g += h_j * IO_LOAD(R, r_base + 2 * hidden + i);
            rh_o += h_j * IO_LOAD(R, r_base + 3 * hidden + i);
        }
        __syncthreads();

        // Compute gate activations
        int wx_idx = b * seq_len * hidden4 + t * hidden4;
        float i_t = 1.0f / (1.0f + expf(-(IO_LOAD(wx, wx_idx + i) + rh_i)));
        float f_t = 1.0f / (1.0f + expf(-(IO_LOAD(wx, wx_idx + hidden + i) + rh_f)));
        float g_t = tanhf(IO_LOAD(wx, wx_idx + 2 * hidden + i) + rh_g);
        float o_t = 1.0f / (1.0f + expf(-(IO_LOAD(wx, wx_idx + 3 * hidden + i) + rh_o)));

        // Cell and hidden update
        c_val = f_t * c_val + i_t * g_t;
        h_val = o_t * tanhf(c_val);

        // Store for backward pass
        local_i[t] = i_t;
        local_f[t] = f_t;
        local_g[t] = g_t;
        local_o[t] = o_t;
        local_c[t] = c_val;
    }

    // ========================================
    // Pass 2: Backward — reverse accumulate gradients
    // ========================================
    float dh_acc = 0.0f;
    float dc_acc = 0.0f;

    // Shared memory layout for dgate values (reuse h_shared region)
    // We need 4*hidden floats, allocated as smem_bytes = 4*hidden*sizeof(float)
    // but h_shared was declared with extern, so it's the same block
    float* dg_sh_i = h_shared;
    float* dg_sh_f = h_shared + hidden;
    float* dg_sh_g = h_shared + 2 * hidden;
    float* dg_sh_o = h_shared + 3 * hidden;

    for (int t = seq_len - 1; t >= 0; t--) {
        int h_idx = b * seq_len * hidden + t * hidden + i;

        float dh = IO_LOAD(grad_output, h_idx) + dh_acc;

        float i_t = local_i[t];
        float f_t = local_f[t];
        float g_t = local_g[t];
        float o_t = local_o[t];
        float c_t = local_c[t];
        float tanh_c = tanhf(c_t);

        // c_{t-1}
        float c_prev = (t == 0) ? IO_LOAD(c0, b * hidden + i) : local_c[t - 1];

        // Output gate gradient (pre-activation)
        float do_pre = dh * tanh_c * o_t * (1.0f - o_t);

        // Cell gradient
        float dc = dh * o_t * (1.0f - tanh_c * tanh_c) + dc_acc;

        // Input gate gradient (pre-activation)
        float di_pre = dc * g_t * i_t * (1.0f - i_t);

        // Forget gate gradient (pre-activation)
        float df_pre = dc * c_prev * f_t * (1.0f - f_t);

        // Cell candidate gradient (pre-activation)
        float dg_pre = dc * i_t * (1.0f - g_t * g_t);

        // Cell accumulator
        dc_acc = dc * f_t;

        // Write grad_wx
        int gwx_idx = b * seq_len * hidden4 + t * hidden4;
        IO_STORE(grad_wx, gwx_idx + i, di_pre);
        IO_STORE(grad_wx, gwx_idx + hidden + i, df_pre);
        IO_STORE(grad_wx, gwx_idx + 2 * hidden + i, dg_pre);
        IO_STORE(grad_wx, gwx_idx + 3 * hidden + i, do_pre);

        // --- Compute dh_acc = dgate @ R^T ---
        dg_sh_i[i] = di_pre;
        dg_sh_f[i] = df_pre;
        dg_sh_g[i] = dg_pre;
        dg_sh_o[i] = do_pre;
        __syncthreads();

        // dh_acc[i] = sum_j( di[j]*R[i,j] + df[j]*R[i,H+j] + dg[j]*R[i,2H+j] + do[j]*R[i,3H+j] )
        float dh_from_r = 0.0f;
        int r_row = i * hidden4;
        for (int j = 0; j < hidden; j++) {
            dh_from_r += dg_sh_i[j] * IO_LOAD(R, r_row + j);
            dh_from_r += dg_sh_f[j] * IO_LOAD(R, r_row + hidden + j);
            dh_from_r += dg_sh_g[j] * IO_LOAD(R, r_row + 2 * hidden + j);
            dh_from_r += dg_sh_o[j] * IO_LOAD(R, r_row + 3 * hidden + j);
        }
        dh_acc = dh_from_r;
        __syncthreads();
    }

    IO_STORE(grad_h0, b * hidden + i, dh_acc);
    IO_STORE(grad_c0, b * hidden + i, dc_acc);
}

// ============================================================================
// Standalone launch wrapper (C-linkage for NIF / dlopen)
// ============================================================================

#ifdef EXLA_FFI
}  // anonymous namespace
#endif

#ifndef EXLA_FFI

extern "C" {

// Output: concatenated [grad_wx (B*T*4H) | grad_h0 (B*H) | grad_c0 (B*H)]
int fused_lstm_scan_backward_launch(
    cudaStream_t stream,
    const io_type* wx, const io_type* R,
    const io_type* h0, const io_type* c0,
    const io_type* forward_out,
    const io_type* grad_output,
    io_type* output_concat,
    int batch, int seq_len, int hidden
) {
    int bt4h = batch * seq_len * 4 * hidden;
    int bh = batch * hidden;
    io_type* grad_wx = output_concat;
    io_type* grad_h0 = output_concat + bt4h;
    io_type* grad_c0 = output_concat + bt4h + bh;

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    // Shared memory: 4*hidden floats (for dgate values in backward, h_prev in forward)
    size_t smem_bytes = 4 * hidden * sizeof(float);

    fused_lstm_scan_backward_kernel<<<grid, block, smem_bytes, stream>>>(
        wx, R, h0, c0, forward_out, grad_output,
        grad_wx, grad_h0, grad_c0,
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

ffi::Error fused_lstm_scan_backward_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> wx,
    ffi::Buffer<FFI_IO_TYPE> R,
    ffi::Buffer<FFI_IO_TYPE> h0,
    ffi::Buffer<FFI_IO_TYPE> c0,
    ffi::Buffer<FFI_IO_TYPE> forward_out,
    ffi::Buffer<FFI_IO_TYPE> grad_output,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_wx,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_h0,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_c0
) {
    auto wx_dims = wx.dimensions();
    int batch   = static_cast<int>(wx_dims[0]);
    int seq_len = static_cast<int>(wx_dims[1]);
    int hidden  = static_cast<int>(wx_dims[2]) / 4;

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    size_t smem_bytes = 4 * hidden * sizeof(float);

    fused_lstm_scan_backward_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const io_type*>(wx.untyped_data()),
        reinterpret_cast<const io_type*>(R.untyped_data()),
        reinterpret_cast<const io_type*>(h0.untyped_data()),
        reinterpret_cast<const io_type*>(c0.untyped_data()),
        reinterpret_cast<const io_type*>(forward_out.untyped_data()),
        reinterpret_cast<const io_type*>(grad_output.untyped_data()),
        reinterpret_cast<io_type*>(grad_wx->untyped_data()),
        reinterpret_cast<io_type*>(grad_h0->untyped_data()),
        reinterpret_cast<io_type*>(grad_c0->untyped_data()),
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
    HANDLER_SYMBOL(fused_lstm_scan_backward), fused_lstm_scan_backward_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // wx
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // R
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // h0
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // c0
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // forward_out
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // grad_output
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_wx
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_h0
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_c0
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_lstm_scan_backward_" PRECISION_SUFFIX, "CUDA", HANDLER_SYMBOL(fused_lstm_scan_backward));

#endif  // EXLA_FFI
