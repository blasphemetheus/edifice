// Fused TTT-Linear (Test-Time Training) Scan Backward Kernel
//
// Computes gradients for the TTT-Linear forward scan:
//   pred_t = W_{t-1} @ k_t                         -- inner model forward
//   pred_normed = LayerNorm(pred_t; gamma, beta)    -- stabilize predictions
//   error_t = pred_normed - v_t                     -- reconstruction error
//   scaled_error_t = eta_t * error_t                -- learning-rate-gated gradient
//   W_t = W_{t-1} - outer(scaled_error_t, k_t)     -- weight update
//   o_t = W_t @ q_t                                -- output from updated model
//
// Two-pass approach:
//   Pass 1 (forward, t=0..T-1):
//     Re-run the forward recurrence storing lightweight intermediates:
//       pred[t][i]  -- per-thread prediction (thread i owns row i of W)
//       mean[t], var[t]  -- LayerNorm statistics per timestep (shared memory)
//     At the end of pass 1, W holds W_T (the final weight matrix).
//
//   Pass 2 (backward, t=T-1..0):
//     Start with dW accumulator = 0.  Iterate in reverse:
//     (a) Reconstruct W_prev from W_new by undoing the weight update:
//           scaled_error_i = eta_i * (gamma_i*(pred_i - mean)*inv_std + beta_i - v_i)
//           W_prev[i][j] = W_new[i][j] + scaled_error_i * k_j
//     (b) Backpropagate through steps 6-1 at timestep t.
//     (c) dW accumulator carries to next (earlier) timestep.
//     After all timesteps, dW = grad_w0.
//
// Backward math per timestep (given dW accumulator and do_t from upstream):
//
//   Step 6 backward: o = W_new @ q
//     dW_new[i][j] += do[i] * q[j]   (accumulated into dW)
//     dq[j] = sum_i(W_new[i][j] * do[i])  -- cross-thread reduction
//
//   Step 5 backward: W_new = W_prev - outer(scaled_error, k)
//     dW_prev[i][j] = dW_new[i][j]   (gradient passes through)
//     d_scaled_error[i] = -sum_j(dW_new[i][j] * k[j])  -- per-thread dot product
//     dk_update[j] = -sum_i(scaled_error[i] * dW_new[i][j])  -- cross-thread reduction
//
//   Step 4 backward: scaled_error = eta * error
//     d_eta[i] = d_scaled_error[i] * error[i]
//     d_error[i] = d_scaled_error[i] * eta[i]
//
//   Step 3 backward: error = pred_normed - v
//     d_pred_normed[i] = d_error[i]
//     dv[i] = -d_error[i]
//
//   Step 2 backward: pred_normed = gamma * (pred - mean) * inv_std + beta
//     Standard LayerNorm backward (using stored pred[t], mean[t], var[t]):
//       x_hat[i] = (pred[i] - mean) * inv_std
//       d_pred[i] = (gamma[i] * inv_std / D) * (D * d_pred_normed[i]
//                   - sum(d_pred_normed) - x_hat[i] * sum(d_pred_normed * x_hat))
//     Also accumulates:
//       d_gamma[i] += x_hat[i] * d_pred_normed[i]    (across all t and b)
//       d_beta[i]  += d_pred_normed[i]                (across all t and b)
//
//   Step 1 backward: pred = W_prev @ k
//     dk_pred[j] = sum_i(W_prev[i][j] * d_pred[i])  -- cross-thread reduction
//     dW[i][j] += d_pred[i] * k[j]                   -- added to dW accumulator
//
//   Total dk = dk_update + dk_pred + dq (if q shares memory; here they are separate)
//   dq comes only from step 6.
//
// Thread layout: one block per batch element, inner_size threads.
//   Thread i owns row i of W (and dW).
//
// Inputs:
//   q:           [B, T, D]
//   k:           [B, T, D]
//   v:           [B, T, D]
//   eta:         [B, T, D]
//   w0:          [B, D, D]
//   ln_g:        [D]          -- LayerNorm gamma
//   ln_b:        [D]          -- LayerNorm beta
//   grad_output: [B, T, D]
//
// Outputs:
//   grad_q:    [B, T, D]
//   grad_k:    [B, T, D]
//   grad_v:    [B, T, D]
//   grad_eta:  [B, T, D]
//   grad_w0:   [B, D, D]
//   grad_lng:  [D]           -- accumulated across all B and T (atomicAdd)
//   grad_lnb:  [D]           -- accumulated across all B and T (atomicAdd)

#include <cuda_runtime.h>
#include "precision.cuh"

constexpr float TTT_LN_EPS = 1.0e-6f;
#define MAX_SEQ_LEN 1024
#define TTT_MAX_INNER 128

// ============================================================================
// Kernel
// ============================================================================

__global__ void fused_ttt_scan_backward_kernel(
    const io_type* __restrict__ q,            // [B, T, D]
    const io_type* __restrict__ k,            // [B, T, D]
    const io_type* __restrict__ v,            // [B, T, D]
    const io_type* __restrict__ eta,          // [B, T, D]
    const io_type* __restrict__ w0,           // [B, D, D]
    const io_type* __restrict__ ln_g,         // [D]
    const io_type* __restrict__ ln_b,         // [D]
    const io_type* __restrict__ grad_output,  // [B, T, D]
    io_type* __restrict__ grad_q,             // [B, T, D]
    io_type* __restrict__ grad_k,             // [B, T, D]
    io_type* __restrict__ grad_v,             // [B, T, D]
    io_type* __restrict__ grad_eta,           // [B, T, D]
    io_type* __restrict__ grad_w0,            // [B, D, D]
    float* __restrict__ grad_lng,             // [D] — atomicAdd target
    float* __restrict__ grad_lnb,             // [D] — atomicAdd target
    int batch, int seq_len, int inner_size
) {
    int b = blockIdx.x;
    int i = threadIdx.x;  // output dimension index (row of W)

    if (b >= batch || i >= inner_size) return;

    // Shared memory layout:
    //   k_shared[D]        -- loaded key (or reused for q, reductions)
    //   reduce_buf[D]      -- cross-thread reduction workspace
    //   ln_stats[2]        -- mean, var for current timestep LN
    extern __shared__ float shared_mem[];
    float* k_shared   = shared_mem;                            // [D]
    float* reduce_buf = shared_mem + inner_size;               // [D]
    float* ln_stats   = shared_mem + 2 * inner_size;           // [2]

    int D = inner_size;
    int TD = seq_len * D;
    int base_b = b * TD;                     // base offset for [B,T,D] tensors
    int w0_base = b * D * D + i * D;         // row i of w0 for this batch

    // Load gamma and beta for this thread's dimension
    float gamma_i = IO_LOAD(ln_g, i);
    float beta_i  = IO_LOAD(ln_b, i);

    // ========================================================================
    // Pass 1: Forward — recompute W, store pred[t] per thread + mean/var
    // ========================================================================
    // W_row[j]: row i of the weight matrix, held in registers
    float W_row[TTT_MAX_INNER];
    for (int j = 0; j < D; j++) {
        W_row[j] = IO_LOAD(w0, w0_base + j);
    }

    // Per-thread local storage for intermediates needed in backward
    float pred_store[MAX_SEQ_LEN];
    float mean_store[MAX_SEQ_LEN];
    float var_store[MAX_SEQ_LEN];

    for (int t = 0; t < seq_len; t++) {
        int t_off = base_b + t * D;

        // Load k into shared memory
        k_shared[i] = IO_LOAD(k, t_off + i);
        __syncthreads();

        // pred_i = W[i,:] @ k
        float pred_i = 0.0f;
        for (int j = 0; j < D; j++) {
            pred_i += W_row[j] * k_shared[j];
        }
        pred_store[t] = pred_i;

        // LayerNorm statistics: need all threads' pred values
        reduce_buf[i] = pred_i;
        __syncthreads();

        // Thread 0 computes mean and variance
        if (i == 0) {
            float sum = 0.0f;
            for (int j = 0; j < D; j++) {
                sum += reduce_buf[j];
            }
            float mean = sum / D;
            ln_stats[0] = mean;

            float var_sum = 0.0f;
            for (int j = 0; j < D; j++) {
                float diff = reduce_buf[j] - mean;
                var_sum += diff * diff;
            }
            ln_stats[1] = var_sum / D;
        }
        __syncthreads();

        float mean = ln_stats[0];
        float var  = ln_stats[1];
        mean_store[t] = mean;
        var_store[t]  = var;

        float inv_std = rsqrtf(var + TTT_LN_EPS);
        float pred_normed = gamma_i * (pred_i - mean) * inv_std + beta_i;

        // error and weight update
        float v_i = IO_LOAD(v, t_off + i);
        float error_i = pred_normed - v_i;
        float eta_i = IO_LOAD(eta, t_off + i);
        float scaled_error_i = eta_i * error_i;

        for (int j = 0; j < D; j++) {
            W_row[j] -= scaled_error_i * k_shared[j];
        }
        __syncthreads();
    }

    // After pass 1, W_row holds W_T (final weight state after all timesteps).

    // ========================================================================
    // Pass 2: Backward — reverse iterate, accumulating gradients
    // ========================================================================

    // dW accumulator: gradient flowing backward through the W recurrence
    float dW_row[TTT_MAX_INNER];
    for (int j = 0; j < D; j++) {
        dW_row[j] = 0.0f;
    }

    for (int t = seq_len - 1; t >= 0; t--) {
        int t_off = base_b + t * D;

        // Load inputs for this timestep
        float do_i  = IO_LOAD(grad_output, t_off + i);
        float q_i   = IO_LOAD(q, t_off + i);
        float k_i   = IO_LOAD(k, t_off + i);
        float v_i   = IO_LOAD(v, t_off + i);
        float eta_i = IO_LOAD(eta, t_off + i);

        // Recover stored intermediates
        float pred_i = pred_store[t];
        float mean   = mean_store[t];
        float var    = var_store[t];
        float inv_std = rsqrtf(var + TTT_LN_EPS);
        float x_hat_i = (pred_i - mean) * inv_std;

        // Reconstruct error and scaled_error for undo
        float pred_normed_i = gamma_i * x_hat_i + beta_i;
        float error_i = pred_normed_i - v_i;
        float scaled_error_i = eta_i * error_i;

        // ---- Undo weight update to get W_prev ----
        // W_prev[i][j] = W_new[i][j] + scaled_error_i * k[j]
        // (W_row currently holds W_new = W at end of timestep t)
        k_shared[i] = k_i;
        __syncthreads();

        for (int j = 0; j < D; j++) {
            W_row[j] += scaled_error_i * k_shared[j];
        }
        // Now W_row holds W_prev (W at start of timestep t)

        // ---- Step 6 backward: o = W_updated @ q ----
        // W_updated = W_prev - outer(scaled_error, k)
        // We need W_updated for dq, but we just undid the update.
        // Recompute W_updated[i][j] = W_row[j] - scaled_error_i * k_shared[j]
        // (don't modify W_row, compute on the fly)

        // dq[j] = sum_i(W_updated[i][j] * do[i]) -- cross-thread reduction
        // First, put q into shared for the dW update
        reduce_buf[i] = q_i;
        __syncthreads();

        // dW[i][j] += do_i * q[j]
        for (int j = 0; j < D; j++) {
            dW_row[j] += do_i * reduce_buf[j];
        }

        // dq: need cross-thread reduction
        // dq[j] = sum_i(W_updated[i][j] * do[i])
        // Thread i contributes: do_i * W_updated[i][j] for each j
        // But we need sum over i, and each thread is a different i.
        // We iterate over j, using shared mem reduction.
        // For each j: accumulate do_i * W_updated_ij from thread i into reduce_buf[j]
        // With D threads and D values, each thread writes its own contribution.
        // We use atomicAdd to reduce_buf.
        if (i == 0) {
            for (int j = 0; j < D; j++) {
                reduce_buf[j] = 0.0f;
            }
        }
        __syncthreads();

        // W_updated[i][j] = W_row[j] - scaled_error_i * k_shared[j]
        for (int j = 0; j < D; j++) {
            float w_updated_ij = W_row[j] - scaled_error_i * k_shared[j];
            atomicAdd(&reduce_buf[j], w_updated_ij * do_i);
        }
        __syncthreads();

        float dq_i = reduce_buf[i];
        IO_STORE(grad_q, t_off + i, dq_i);

        // ---- Step 5 backward: W_new = W_prev - outer(scaled_error, k) ----
        // dW_prev = dW_new (already in dW_row from above)
        // d_scaled_error[i] = -sum_j(dW[i][j] * k[j])
        float d_scaled_error_i = 0.0f;
        for (int j = 0; j < D; j++) {
            d_scaled_error_i -= dW_row[j] * k_shared[j];
        }

        // dk_update[j] = -sum_i(scaled_error[i] * dW[i][j]) -- cross-thread
        if (i == 0) {
            for (int j = 0; j < D; j++) {
                reduce_buf[j] = 0.0f;
            }
        }
        __syncthreads();

        for (int j = 0; j < D; j++) {
            atomicAdd(&reduce_buf[j], -scaled_error_i * dW_row[j]);
        }
        __syncthreads();

        // Save dk_update for later combination
        float dk_update_i = reduce_buf[i];

        // ---- Step 4 backward: scaled_error = eta * error ----
        float d_eta_i = d_scaled_error_i * error_i;
        float d_error_i = d_scaled_error_i * eta_i;

        // ---- Step 3 backward: error = pred_normed - v ----
        float d_pred_normed_i = d_error_i;
        float dv_i = -d_error_i;

        // ---- Step 2 backward: LayerNorm ----
        // pred_normed = gamma * (pred - mean) * inv_std + beta
        // x_hat = (pred - mean) * inv_std
        //
        // d_gamma[i] += x_hat_i * d_pred_normed_i  (accumulated globally)
        // d_beta[i]  += d_pred_normed_i             (accumulated globally)
        atomicAdd(&grad_lng[i], x_hat_i * d_pred_normed_i);
        atomicAdd(&grad_lnb[i], d_pred_normed_i);

        // d_pred (standard LayerNorm backward):
        //   d_xhat[i] = gamma[i] * d_pred_normed[i]
        //   d_pred[i] = inv_std * (d_xhat[i] - mean(d_xhat) - x_hat[i] * mean(d_xhat * x_hat))
        float d_xhat_i = gamma_i * d_pred_normed_i;

        // We need cross-thread sums: sum(d_xhat) and sum(d_xhat * x_hat)
        // Store d_xhat in reduce_buf for thread 0 reduction
        reduce_buf[i] = d_xhat_i;
        __syncthreads();

        float sum_dxhat = 0.0f;
        if (i == 0) {
            for (int j = 0; j < D; j++) {
                sum_dxhat += reduce_buf[j];
            }
            ln_stats[0] = sum_dxhat / D;
        }
        __syncthreads();

        // Now store d_xhat * x_hat for the second sum
        // But we need x_hat for all threads — each thread knows its own x_hat_i
        // Store in reduce_buf
        reduce_buf[i] = d_xhat_i * x_hat_i;
        __syncthreads();

        if (i == 0) {
            float s = 0.0f;
            for (int j = 0; j < D; j++) {
                s += reduce_buf[j];
            }
            ln_stats[1] = s / D;
        }
        __syncthreads();

        float mean_dxhat      = ln_stats[0];
        float mean_dxhat_xhat = ln_stats[1];

        float d_pred_i = inv_std * (d_xhat_i - mean_dxhat - x_hat_i * mean_dxhat_xhat);

        // ---- Step 1 backward: pred = W_prev @ k ----
        // dk_pred[j] = sum_i(W_prev[i][j] * d_pred[i]) -- cross-thread
        // dW[i][j] += d_pred[i] * k[j]

        // First: dk_pred via cross-thread reduction
        if (i == 0) {
            for (int j = 0; j < D; j++) {
                reduce_buf[j] = 0.0f;
            }
        }
        __syncthreads();

        for (int j = 0; j < D; j++) {
            atomicAdd(&reduce_buf[j], W_row[j] * d_pred_i);
        }
        __syncthreads();

        float dk_pred_i = reduce_buf[i];

        // dW[i][j] += d_pred[i] * k[j]
        for (int j = 0; j < D; j++) {
            dW_row[j] += d_pred_i * k_shared[j];
        }

        // ---- Combine dk contributions ----
        float dk_i = dk_update_i + dk_pred_i;
        IO_STORE(grad_k, t_off + i, dk_i);
        IO_STORE(grad_v, t_off + i, dv_i);
        IO_STORE(grad_eta, t_off + i, d_eta_i);

        __syncthreads();
    }

    // After all timesteps, dW_row = grad_w0[b][i][:]
    for (int j = 0; j < D; j++) {
        IO_STORE(grad_w0, w0_base + j, dW_row[j]);
    }
}

// ============================================================================
// Standalone launch wrapper (C-linkage for NIF / dlopen)
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

// Output: concat [grad_q(B*T*D) | grad_k(B*T*D) | grad_v(B*T*D) |
//                 grad_eta(B*T*D) | grad_w0(B*D*D) | grad_lng(D) | grad_lnb(D)]
// Note: grad_lng and grad_lnb use atomicAdd so stay as float.
int fused_ttt_scan_backward_launch(
    cudaStream_t stream,
    const io_type* q, const io_type* k, const io_type* v,
    const io_type* eta, const io_type* w0,
    const io_type* ln_g, const io_type* ln_b,
    const io_type* grad_output,
    io_type* output_concat,
    int batch, int seq_len, int inner_size
) {
    int btd = batch * seq_len * inner_size;
    int bdd = batch * inner_size * inner_size;
    int D   = inner_size;

    io_type* grad_q   = output_concat;
    io_type* grad_k   = output_concat + btd;
    io_type* grad_v   = output_concat + 2 * btd;
    io_type* grad_eta = output_concat + 3 * btd;
    io_type* grad_w0  = output_concat + 4 * btd;
    // grad_lng and grad_lnb are float (atomicAdd targets) — placed after io_type region
    float* grad_lng = reinterpret_cast<float*>(output_concat + 4 * btd + bdd);
    float* grad_lnb = grad_lng + D;

    // Zero out atomicAdd targets before kernel launch
    cudaMemsetAsync(grad_lng, 0, D * sizeof(float), stream);
    cudaMemsetAsync(grad_lnb, 0, D * sizeof(float), stream);

    dim3 grid(batch);
    dim3 block(inner_size);
    // Shared memory: k_shared[D] + reduce_buf[D] + ln_stats[2]
    size_t smem_bytes = (2 * inner_size + 2) * sizeof(float);

    fused_ttt_scan_backward_kernel<<<grid, block, smem_bytes, stream>>>(
        q, k, v, eta, w0, ln_g, ln_b, grad_output,
        grad_q, grad_k, grad_v, grad_eta, grad_w0,
        grad_lng, grad_lnb,
        batch, seq_len, inner_size
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

ffi::Error fused_ttt_scan_backward_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> q,            // [B, T, D]
    ffi::Buffer<FFI_IO_TYPE> k,            // [B, T, D]
    ffi::Buffer<FFI_IO_TYPE> v,            // [B, T, D]
    ffi::Buffer<FFI_IO_TYPE> eta,          // [B, T, D]
    ffi::Buffer<FFI_IO_TYPE> w0,           // [B, D, D]
    ffi::Buffer<FFI_IO_TYPE> ln_g,         // [D]
    ffi::Buffer<FFI_IO_TYPE> ln_b,         // [D]
    ffi::Buffer<FFI_IO_TYPE> grad_output,  // [B, T, D]
    ffi::ResultBuffer<FFI_IO_TYPE> grad_q,     // [B, T, D]
    ffi::ResultBuffer<FFI_IO_TYPE> grad_k,     // [B, T, D]
    ffi::ResultBuffer<FFI_IO_TYPE> grad_v,     // [B, T, D]
    ffi::ResultBuffer<FFI_IO_TYPE> grad_eta,   // [B, T, D]
    ffi::ResultBuffer<FFI_IO_TYPE> grad_w0,    // [B, D, D]
    ffi::ResultBuffer<ffi::F32> grad_lng,      // [D] — atomicAdd target
    ffi::ResultBuffer<ffi::F32> grad_lnb       // [D] — atomicAdd target
) {
    auto q_dims = q.dimensions();
    int batch      = static_cast<int>(q_dims[0]);
    int seq_len    = static_cast<int>(q_dims[1]);
    int inner_size = static_cast<int>(q_dims[2]);

    if (inner_size > TTT_MAX_INNER) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                         "TTT inner_size exceeds max supported (128)");
    }

    int D = inner_size;

    // Zero out atomicAdd targets
    cudaMemsetAsync(reinterpret_cast<float*>(grad_lng->untyped_data()), 0,
                    D * sizeof(float), stream);
    cudaMemsetAsync(reinterpret_cast<float*>(grad_lnb->untyped_data()), 0,
                    D * sizeof(float), stream);

    dim3 grid(batch);
    dim3 block(inner_size);
    // Shared memory: k_shared[D] + reduce_buf[D] + ln_stats[2]
    size_t smem_bytes = (2 * inner_size + 2) * sizeof(float);

    fused_ttt_scan_backward_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const io_type*>(q.untyped_data()),
        reinterpret_cast<const io_type*>(k.untyped_data()),
        reinterpret_cast<const io_type*>(v.untyped_data()),
        reinterpret_cast<const io_type*>(eta.untyped_data()),
        reinterpret_cast<const io_type*>(w0.untyped_data()),
        reinterpret_cast<const io_type*>(ln_g.untyped_data()),
        reinterpret_cast<const io_type*>(ln_b.untyped_data()),
        reinterpret_cast<const io_type*>(grad_output.untyped_data()),
        reinterpret_cast<io_type*>(grad_q->untyped_data()),
        reinterpret_cast<io_type*>(grad_k->untyped_data()),
        reinterpret_cast<io_type*>(grad_v->untyped_data()),
        reinterpret_cast<io_type*>(grad_eta->untyped_data()),
        reinterpret_cast<io_type*>(grad_w0->untyped_data()),
        reinterpret_cast<float*>(grad_lng->untyped_data()),
        reinterpret_cast<float*>(grad_lnb->untyped_data()),
        batch, seq_len, inner_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_ttt_scan_backward, fused_ttt_scan_backward_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // q
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // k
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // v
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // eta
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // w0
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // ln_g
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // ln_b
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // grad_output
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_q
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_k
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_v
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_eta
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_w0
        .Ret<ffi::Buffer<ffi::F32>>()      // grad_lng (atomicAdd target)
        .Ret<ffi::Buffer<ffi::F32>>()      // grad_lnb (atomicAdd target)
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_ttt_scan_backward_" PRECISION_SUFFIX, "CUDA",
    fused_ttt_scan_backward);

#endif  // EXLA_FFI
