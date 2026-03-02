// Fused sLSTM Scan Backward Kernel
//
// Computes gradients for the sLSTM (scalar LSTM with exponential gating):
//   m_t = max(log_f + m_{t-1}, log_i)
//   i_t = exp(log_i - m_t)
//   f_t = exp(log_f + m_{t-1} - m_t)
//   c_t = f_t * c_{t-1} + i_t * z_t
//   n_t = f_t * n_{t-1} + i_t
//   h_t = o_t * c_t / max(|n_t|, 1)
//
// Two-pass approach:
//   Pass 1 (forward): Recompute gate activations, cell/normalizer states.
//   Pass 2 (reverse): Accumulate gradients.
//
// Reverse pass math:
//   dh = grad_output[t] + dh_acc
//   safe_denom = max(|n_t|, 1)
//   do_pre = dh * (c_t/safe_denom) * o_t * (1-o_t)
//   d(c_t/safe_denom) = dh * o_t
//   dc = d(c_t/safe_denom) / safe_denom  (when |n_t| < 1, dc = d(c_t/safe_denom))
//   dn = -d(c_t/safe_denom) * c_t / (safe_denom^2) * sign(n_t)  (when |n_t| >= 1)
//   di_pre = dc * z_t + dn    (pre-activation input gate gradient)
//   df_pre = dc * c_{t-1} + dn * n_{t-1}  (pre-activation forget gate gradient)
//   dz = dc * i_t * (1 - z_t^2)
//   dc_acc = dc * f_t
//   dn_acc = dn * f_t
//   d_log_i = di_pre * i_t    (chain rule through exp)
//   d_log_f = df_pre * f_t    (chain rule through exp)
//   d_m adjustment: d_log_i and d_log_f need stabilizer correction
//   dh_acc = dgate @ R^T (via shared memory)
//   grad_wx[t] = [d_log_i, d_log_f, dz, do_pre]
//
// Inputs:
//   wx:          [B, T, 4*H] — pre-computed W@x + bias
//   R:           [H, 4*H]   — recurrent weight matrix
//   h0:          [B, H]     — initial hidden state
//   c0:          [B, H]     — initial cell state
//   forward_out: [B, T, H]  — forward pass hidden states
//   grad_output: [B, T, H]  — upstream gradient
//
// Outputs:
//   grad_wx: [B, T, 4*H] — gradient w.r.t. pre-activation gate values
//   grad_h0: [B, H]      — gradient w.r.t. initial hidden state
//   grad_c0: [B, H]      — gradient w.r.t. initial cell state

#include <cuda_runtime.h>
#include "precision.cuh"
#include <cfloat>

#define MAX_SEQ_LEN 1024

// ============================================================================
// Kernel
// ============================================================================

__global__ void fused_slstm_scan_backward_kernel(
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

    extern __shared__ float h_shared[];  // [4*hidden] for dgate sharing

    int hidden4 = 4 * hidden;

    // Thread-local arrays for per-timestep values
    float local_i[MAX_SEQ_LEN];   // stabilized input gate
    float local_f[MAX_SEQ_LEN];   // stabilized forget gate
    float local_z[MAX_SEQ_LEN];   // tanh cell candidate
    float local_o[MAX_SEQ_LEN];   // sigmoid output gate
    float local_c[MAX_SEQ_LEN];   // cell state
    float local_n[MAX_SEQ_LEN];   // normalizer

    // ========================================
    // Pass 1: Forward — recompute states
    // ========================================
    float h_val = IO_LOAD(h0, b * hidden + i);
    float c_val = IO_LOAD(c0, b * hidden + i);
    float n_val = 1.0f;
    float m_val = 0.0f;

    for (int t = 0; t < seq_len; t++) {
        h_shared[i] = h_val;
        __syncthreads();

        // R@h for all 4 gates
        float rh_i = 0.0f, rh_f = 0.0f, rh_z = 0.0f, rh_o = 0.0f;
        for (int j = 0; j < hidden; j++) {
            float h_j = h_shared[j];
            int r_base = j * hidden4;
            rh_i += h_j * IO_LOAD(R, r_base + i);
            rh_f += h_j * IO_LOAD(R, r_base + hidden + i);
            rh_z += h_j * IO_LOAD(R, r_base + 2 * hidden + i);
            rh_o += h_j * IO_LOAD(R, r_base + 3 * hidden + i);
        }
        __syncthreads();

        int wx_idx = b * seq_len * hidden4 + t * hidden4;
        float log_i_raw = IO_LOAD(wx, wx_idx + i) + rh_i;
        float log_f_raw = IO_LOAD(wx, wx_idx + hidden + i) + rh_f;
        float z_t = tanhf(IO_LOAD(wx, wx_idx + 2 * hidden + i) + rh_z);
        float o_t = 1.0f / (1.0f + expf(-(IO_LOAD(wx, wx_idx + 3 * hidden + i) + rh_o)));

        float log_f_plus_m = log_f_raw + m_val;
        float m_new = fmaxf(log_f_plus_m, log_i_raw);

        float i_t = expf(log_i_raw - m_new);
        float f_t = expf(log_f_plus_m - m_new);

        c_val = f_t * c_val + i_t * z_t;
        n_val = f_t * n_val + i_t;
        float safe_denom = fmaxf(fabsf(n_val), 1.0f);
        h_val = o_t * (c_val / safe_denom);
        m_val = m_new;

        local_i[t] = i_t;
        local_f[t] = f_t;
        local_z[t] = z_t;
        local_o[t] = o_t;
        local_c[t] = c_val;
        local_n[t] = n_val;
    }

    // ========================================
    // Pass 2: Backward
    // ========================================
    float dh_acc = 0.0f;
    float dc_acc = 0.0f;
    float dn_acc = 0.0f;

    float* dg_sh_i = h_shared;
    float* dg_sh_f = h_shared + hidden;
    float* dg_sh_z = h_shared + 2 * hidden;
    float* dg_sh_o = h_shared + 3 * hidden;

    for (int t = seq_len - 1; t >= 0; t--) {
        int h_idx = b * seq_len * hidden + t * hidden + i;

        float dh = IO_LOAD(grad_output, h_idx) + dh_acc;

        float i_t = local_i[t];
        float f_t = local_f[t];
        float z_t = local_z[t];
        float o_t = local_o[t];
        float c_t = local_c[t];
        float n_t = local_n[t];

        float c_prev = (t == 0) ? IO_LOAD(c0, b * hidden + i) : local_c[t - 1];
        float n_prev = (t == 0) ? 1.0f : local_n[t - 1];

        float safe_denom = fmaxf(fabsf(n_t), 1.0f);
        float ratio = c_t / safe_denom;

        // Output gate gradient (pre-sigmoid)
        float do_pre = dh * ratio * o_t * (1.0f - o_t);

        // d(o * c/safe_denom) w.r.t. c and n
        float d_ratio = dh * o_t;

        // dc from ratio = c/safe_denom
        float dc = d_ratio / safe_denom + dc_acc;

        // dn from ratio when |n| >= 1
        float dn = dn_acc;
        if (fabsf(n_t) >= 1.0f) {
            float sign_n = (n_t >= 0.0f) ? 1.0f : -1.0f;
            dn += -d_ratio * c_t / (safe_denom * safe_denom) * sign_n;
        }

        // Gate gradients (pre-exp, through stabilized gates)
        // c_t = f_t * c_prev + i_t * z_t
        // n_t = f_t * n_prev + i_t
        float di = dc * z_t + dn;           // d(i_t * z_t)/di + d(i_t)/di for normalizer
        float df = dc * c_prev + dn * n_prev;

        // tanh candidate gradient
        float dz = dc * i_t * (1.0f - z_t * z_t);

        // Through exponential gating: i_t = exp(log_i - m), f_t = exp(log_f + m_prev - m)
        // d_log_i = di * i_t, d_log_f = df * f_t
        float d_log_i = di * i_t;
        float d_log_f = df * f_t;

        // Cell and normalizer accumulators
        dc_acc = dc * f_t;
        dn_acc = dn * f_t;

        // Write grad_wx (pre-activation gradients)
        int gwx_idx = b * seq_len * hidden4 + t * hidden4;
        IO_STORE(grad_wx, gwx_idx + i, d_log_i);
        IO_STORE(grad_wx, gwx_idx + hidden + i, d_log_f);
        IO_STORE(grad_wx, gwx_idx + 2 * hidden + i, dz);
        IO_STORE(grad_wx, gwx_idx + 3 * hidden + i, do_pre);

        // --- dh_acc = dgate @ R^T ---
        dg_sh_i[i] = d_log_i;
        dg_sh_f[i] = d_log_f;
        dg_sh_z[i] = dz;
        dg_sh_o[i] = do_pre;
        __syncthreads();

        float dh_from_r = 0.0f;
        int r_row = i * hidden4;
        for (int j = 0; j < hidden; j++) {
            dh_from_r += dg_sh_i[j] * IO_LOAD(R, r_row + j);
            dh_from_r += dg_sh_f[j] * IO_LOAD(R, r_row + hidden + j);
            dh_from_r += dg_sh_z[j] * IO_LOAD(R, r_row + 2 * hidden + j);
            dh_from_r += dg_sh_o[j] * IO_LOAD(R, r_row + 3 * hidden + j);
        }
        dh_acc = dh_from_r;
        __syncthreads();
    }

    IO_STORE(grad_h0, b * hidden + i, dh_acc);
    IO_STORE(grad_c0, b * hidden + i, dc_acc);
}

// ============================================================================
// Standalone launch wrapper
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

// Output: concatenated [grad_wx (B*T*4H) | grad_h0 (B*H) | grad_c0 (B*H)]
int fused_slstm_scan_backward_launch(
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

    size_t smem_bytes = 4 * hidden * sizeof(float);

    fused_slstm_scan_backward_kernel<<<grid, block, smem_bytes, stream>>>(
        wx, R, h0, c0, forward_out, grad_output,
        grad_wx, grad_h0, grad_c0,
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

ffi::Error fused_slstm_scan_backward_ffi_impl(
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

    fused_slstm_scan_backward_kernel<<<grid, block, smem_bytes, stream>>>(
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_slstm_scan_backward, fused_slstm_scan_backward_ffi_impl,
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
    "exla_fused_slstm_scan_backward_" PRECISION_SUFFIX, "CUDA", fused_slstm_scan_backward);

#endif  // EXLA_FFI
