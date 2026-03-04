// Fused Standard GRU Scan Backward Kernel
//
// Computes gradients for the standard GRU:
//   r = σ(wx[0:H]   + R@h[0:H])          reset gate
//   z = σ(wx[H:2H]  + R@h[H:2H])         update gate
//   n = tanh(wx[2H:3H] + r * R@h[2H:3H]) candidate (reset applied to recurrent only)
//   h = (1-z)*n + z*h_prev                hidden update
//
// Two-pass approach:
//   Pass 1 (forward, t=0..T-1): Recompute gate activations via R@h in shared memory.
//   Pass 2 (reverse, t=T-1..0): Accumulate gradients with dgate @ R^T via shared memory.
//
// Reverse pass math:
//   dh = grad_output[t] + dh_acc
//   dn = dh * (1 - z_t)                                    candidate gradient
//   dh_prev_from_z = dh * z_t                                through update gate
//   dz_pre = dh * (h_prev - n_t) * z_t * (1 - z_t)         update gate (pre-sigmoid)
//   dn_pre = dn * (1 - n_t²)                                candidate (pre-tanh)
//   dr_pre = dn_pre * rh_n * r_t * (1 - r_t)               reset gate (pre-sigmoid)
//   d_rh_n = dn_pre * r_t                                   gradient through r*R@h[n]
//
//   dh_acc = dh_prev_from_z + [di_pre,dz_pre,d_rh_n_pre] @ R^T
//   grad_wx[t] = [dr_pre, dz_pre, dn_pre]
//
// Note: The reset gate multiplies only the recurrent contribution to candidate.
//   r affects R@h[2H:3H] but not wx[2H:3H]. So:
//   - grad through r goes to R@h[n] column only
//   - The R^T matmul for dh_acc uses d_rh_n (not dn_pre) for the n-gate column
//
// Inputs:
//   wx:          [B, T, 3*H] — pre-computed W@x + bias
//   R:           [H, 3*H]   — recurrent weight matrix
//   h0:          [B, H]     — initial hidden state
//   forward_out: [B, T, H]  — forward pass hidden states
//   grad_output: [B, T, H]  — upstream gradient dL/dh
//
// Outputs:
//   grad_wx:  [B, T, 3*H] — gradient w.r.t. pre-activation gate values
//   grad_h0:  [B, H]      — gradient w.r.t. initial hidden state
//
// Note: grad_R computed in Elixir via matmul:
//   For r,z columns: grad_R[:,0:2H] = sum_b,t( h_prev^T @ grad_wx[:,0:2H] )
//   For n column: grad_R[:,2H:3H] = sum_b,t( (r*h_prev)^T @ grad_wx_n )
//   This is handled in the Elixir backward dispatch function.

#include <cuda_runtime.h>
#include "precision.cuh"

#define MAX_SEQ_LEN 1024

// ============================================================================
// Kernel
// ============================================================================

#ifdef EXLA_FFI
namespace {  // anonymous namespace — internal linkage per compilation unit
#endif

__global__ void fused_gru_scan_backward_kernel(
    const io_type* __restrict__ wx,           // [B, T, 3*H]
    const io_type* __restrict__ R,            // [H, 3*H]
    const io_type* __restrict__ h0,           // [B, H]
    const io_type* __restrict__ forward_out,  // [B, T, H]
    const io_type* __restrict__ grad_output,  // [B, T, H]
    io_type* __restrict__ grad_wx,            // [B, T, 3*H] — gradient w.r.t. wx pre-activations
    io_type* __restrict__ grad_rh,            // [B, T, 3*H] — gradient w.r.t. R@h (for computing grad_R)
    io_type* __restrict__ grad_h0,            // [B, H]
    int batch, int seq_len, int hidden
) {
    int b = blockIdx.x;
    int i = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || i >= hidden) return;

    extern __shared__ float h_shared[];  // Size: 3*hidden floats — stays float for R@h matmul

    int hidden3 = 3 * hidden;

    // Thread-local arrays for per-timestep values
    float local_r[MAX_SEQ_LEN];   // reset gate
    float local_z[MAX_SEQ_LEN];   // update gate
    float local_n[MAX_SEQ_LEN];   // candidate
    float local_rh_n[MAX_SEQ_LEN]; // R@h for candidate column (before reset multiply)

    // ========================================
    // Pass 1: Forward — recompute gate activations
    // ========================================
    float h_val = IO_LOAD(h0, b * hidden + i);

    for (int t = 0; t < seq_len; t++) {
        h_shared[i] = h_val;
        __syncthreads();

        // Compute R@h for all 3 gates
        float rh_r = 0.0f, rh_z = 0.0f, rh_n = 0.0f;
        for (int j = 0; j < hidden; j++) {
            float h_j = h_shared[j];
            int r_base = j * hidden3;
            rh_r += h_j * IO_LOAD(R, r_base + i);
            rh_z += h_j * IO_LOAD(R, r_base + hidden + i);
            rh_n += h_j * IO_LOAD(R, r_base + 2 * hidden + i);
        }
        __syncthreads();

        int wx_idx = b * seq_len * hidden3 + t * hidden3;
        float r_t = 1.0f / (1.0f + expf(-(IO_LOAD(wx, wx_idx + i) + rh_r)));
        float z_t = 1.0f / (1.0f + expf(-(IO_LOAD(wx, wx_idx + hidden + i) + rh_z)));
        float n_t = tanhf(IO_LOAD(wx, wx_idx + 2 * hidden + i) + r_t * rh_n);

        h_val = (1.0f - z_t) * n_t + z_t * h_val;

        local_r[t] = r_t;
        local_z[t] = z_t;
        local_n[t] = n_t;
        local_rh_n[t] = rh_n;
    }

    // ========================================
    // Pass 2: Backward — reverse accumulate gradients
    // ========================================
    float dh_acc = 0.0f;

    float* dg_sh_r = h_shared;
    float* dg_sh_z = h_shared + hidden;
    float* dg_sh_n = h_shared + 2 * hidden;

    for (int t = seq_len - 1; t >= 0; t--) {
        int h_idx = b * seq_len * hidden + t * hidden + i;

        float h_prev = (t == 0) ? IO_LOAD(h0, b * hidden + i) : IO_LOAD(forward_out, h_idx - hidden);

        float r_t = local_r[t];
        float z_t = local_z[t];
        float n_t = local_n[t];
        float rh_n = local_rh_n[t];

        float dh = IO_LOAD(grad_output, h_idx) + dh_acc;

        // h = (1-z)*n + z*h_prev
        // dn = dh * (1 - z)
        float dn = dh * (1.0f - z_t);

        // dz (pre-activation): dh * (h_prev - n) * z*(1-z)
        float dz_pre = dh * (h_prev - n_t) * z_t * (1.0f - z_t);

        // dn (pre-activation): dn * (1 - n²)
        float dn_pre = dn * (1.0f - n_t * n_t);

        // dr (pre-activation): dn_pre affects wx_n + r*rh_n
        //   d/dr = dn_pre * rh_n, then through sigmoid: * r*(1-r)
        float dr_pre = dn_pre * rh_n * r_t * (1.0f - r_t);

        // Gradient flowing through r*rh_n to R@h[n] column
        // d/d(rh_n) = dn_pre * r
        float d_rh_n = dn_pre * r_t;

        // Write grad_wx (pre-activation gradients for wx)
        int gwx_idx = b * seq_len * hidden3 + t * hidden3;
        IO_STORE(grad_wx, gwx_idx + i, dr_pre);
        IO_STORE(grad_wx, gwx_idx + hidden + i, dz_pre);
        IO_STORE(grad_wx, gwx_idx + 2 * hidden + i, dn_pre);

        // Write grad_rh (gradients w.r.t. R@h — differs for n column)
        IO_STORE(grad_rh, gwx_idx + i, dr_pre);
        IO_STORE(grad_rh, gwx_idx + hidden + i, dz_pre);
        IO_STORE(grad_rh, gwx_idx + 2 * hidden + i, d_rh_n);

        // --- Compute dh_acc = dgate @ R^T ---
        // For r,z gates: use dr_pre, dz_pre
        // For n gate: use d_rh_n (gradient w.r.t. R@h[n], not dn_pre)
        dg_sh_r[i] = dr_pre;
        dg_sh_z[i] = dz_pre;
        dg_sh_n[i] = d_rh_n;
        __syncthreads();

        // dh_acc[i] = sum_j(dr[j]*R[i,j] + dz[j]*R[i,H+j] + d_rh_n[j]*R[i,2H+j])
        //           + dh * z_t  (direct gradient from h = (1-z)*n + z*h_prev)
        float dh_from_r = 0.0f;
        int r_row = i * hidden3;
        for (int j = 0; j < hidden; j++) {
            dh_from_r += dg_sh_r[j] * IO_LOAD(R, r_row + j);
            dh_from_r += dg_sh_z[j] * IO_LOAD(R, r_row + hidden + j);
            dh_from_r += dg_sh_n[j] * IO_LOAD(R, r_row + 2 * hidden + j);
        }
        dh_acc = dh * z_t + dh_from_r;
        __syncthreads();
    }

    IO_STORE(grad_h0, b * hidden + i, dh_acc);
}

// ============================================================================
// Standalone launch wrapper (C-linkage for NIF / dlopen)
// ============================================================================

#ifdef EXLA_FFI
}  // anonymous namespace
#endif

#ifndef EXLA_FFI

extern "C" {

// Output: concatenated [grad_wx (B*T*3H) | grad_rh (B*T*3H) | grad_h0 (B*H)]
int fused_gru_scan_backward_launch(
    cudaStream_t stream,
    const io_type* wx, const io_type* R,
    const io_type* h0,
    const io_type* forward_out,
    const io_type* grad_output,
    io_type* output_concat,
    int batch, int seq_len, int hidden
) {
    int bt3h = batch * seq_len * 3 * hidden;
    io_type* grad_wx = output_concat;
    io_type* grad_rh = output_concat + bt3h;
    io_type* grad_h0 = output_concat + 2 * bt3h;

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    // Shared memory: 3*hidden floats
    size_t smem_bytes = 3 * hidden * sizeof(float);

    fused_gru_scan_backward_kernel<<<grid, block, smem_bytes, stream>>>(
        wx, R, h0, forward_out, grad_output,
        grad_wx, grad_rh, grad_h0,
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

ffi::Error fused_gru_scan_backward_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> wx,
    ffi::Buffer<FFI_IO_TYPE> R,
    ffi::Buffer<FFI_IO_TYPE> h0,
    ffi::Buffer<FFI_IO_TYPE> forward_out,
    ffi::Buffer<FFI_IO_TYPE> grad_output,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_wx,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_rh,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_h0
) {
    auto wx_dims = wx.dimensions();
    int batch   = static_cast<int>(wx_dims[0]);
    int seq_len = static_cast<int>(wx_dims[1]);
    int hidden  = static_cast<int>(wx_dims[2]) / 3;

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    size_t smem_bytes = 3 * hidden * sizeof(float);

    fused_gru_scan_backward_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const io_type*>(wx.untyped_data()),
        reinterpret_cast<const io_type*>(R.untyped_data()),
        reinterpret_cast<const io_type*>(h0.untyped_data()),
        reinterpret_cast<const io_type*>(forward_out.untyped_data()),
        reinterpret_cast<const io_type*>(grad_output.untyped_data()),
        reinterpret_cast<io_type*>(grad_wx->untyped_data()),
        reinterpret_cast<io_type*>(grad_rh->untyped_data()),
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
    HANDLER_SYMBOL(fused_gru_scan_backward), fused_gru_scan_backward_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // wx
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // R
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // h0
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // forward_out
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // grad_output
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_wx
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_rh
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_h0
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_gru_scan_backward_" PRECISION_SUFFIX, "CUDA", HANDLER_SYMBOL(fused_gru_scan_backward));

#endif  // EXLA_FFI
