// Fused FoX (Forgetting Transformer) Attention Backward Kernel
//
// FoX backward: O = softmax(QK^T/sqrt(d) + cs[i] - cs[j]) @ V
//
// Key differences from standard flash attention backward:
//   - Score includes forget bias: s += cs[i] - cs[j] (diagonal zeroed)
//   - Need CS tile in shared memory alongside K/V
//   - Extra output: grad_cs[B,H,T] — gradient w.r.t. cumulative log-forget
//     grad_cs[t] = (row_sum of dS at row t) - (col_sum of dS at col t)
//   - Always causal (no causal flag needed)
//
// Thread layout: one thread block per (batch, head) pair.
// Each block has TILE_SIZE threads.
//
// Two compilation modes:
//   1. Standalone (default): kernel + C-linkage launch wrapper for NIF.
//   2. EXLA FFI (-DEXLA_FFI): kernel + XLA FFI handler + registration.
//
// Inputs:
//   Q:      [B, H, T, d] — query vectors
//   K:      [B, H, T, d] — key vectors
//   V:      [B, H, T, d] — value vectors
//   CS:     [B, H, T]    — cumulative log-forget
//   O:      [B, H, T, d] — forward output
//   grad_O: [B, H, T, d] — upstream gradient
//
// Outputs:
//   dQ:      [B, H, T, d] — gradient w.r.t. Q
//   dK:      [B, H, T, d] — gradient w.r.t. K
//   dV:      [B, H, T, d] — gradient w.r.t. V
//   grad_cs: [B, H, T]    — gradient w.r.t. CS

#include <cuda_runtime.h>
#include "precision.cuh"
#include <float.h>
#include <math.h>

#define TILE_SIZE 32

// ============================================================================
// Kernel
// ============================================================================

__global__ void fused_fox_attention_backward_kernel(
    const io_type* __restrict__ Q,       // [B, H, T, d]
    const io_type* __restrict__ K,       // [B, H, T, d]
    const io_type* __restrict__ V,       // [B, H, T, d]
    const io_type* __restrict__ CS,      // [B, H, T]
    const io_type* __restrict__ O,       // [B, H, T, d]
    const io_type* __restrict__ grad_O,  // [B, H, T, d]
    io_type* __restrict__ dQ,            // [B, H, T, d]
    io_type* __restrict__ dK,            // [B, H, T, d]
    io_type* __restrict__ dV,            // [B, H, T, d]
    io_type* __restrict__ grad_cs,       // [B, H, T]
    const float* __restrict__ D_buf,     // [B*H*T]
    const float* __restrict__ lse_buf,   // [B*H*T]
    int seq_len,
    int head_dim
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int ti = threadIdx.x;

    extern __shared__ float smem[];
    // Layout: smem_a[TILE_SIZE][head_dim] + smem_b[TILE_SIZE][head_dim] + smem_cs[TILE_SIZE]
    float* smem_a = smem;
    float* smem_b = smem + TILE_SIZE * head_dim;
    float* smem_cs = smem + 2 * TILE_SIZE * head_dim;

    float scale = rsqrtf((float)head_dim);

    int BH_stride = seq_len * head_dim;
    int base = (b * gridDim.y + h) * BH_stride;
    int cs_base = (b * gridDim.y + h) * seq_len;
    int aux_base = cs_base;  // D_buf, lse_buf use same stride

    int num_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;

    // ==================================================================
    // Phase 2: Compute dK, dV, and column sums for grad_cs
    // Also accumulate col_sum of dS for grad_cs
    // ==================================================================
    for (int kv_tile = 0; kv_tile < num_tiles; kv_tile++) {
        int kv_idx = kv_tile * TILE_SIZE + ti;
        bool kv_valid = kv_idx < seq_len;

        float k_row[128], v_row[128];
        float dk_acc[128], dv_acc[128];
        float cs_kv = kv_valid ? IO_LOAD(CS, cs_base + kv_idx) : 0.0f;
        float col_sum_ds = 0.0f;  // col sum of dS for grad_cs

        for (int dd = 0; dd < head_dim; dd++) {
            k_row[dd] = kv_valid ? IO_LOAD(K, base + kv_idx * head_dim + dd) : 0.0f;
            v_row[dd] = kv_valid ? IO_LOAD(V, base + kv_idx * head_dim + dd) : 0.0f;
            dk_acc[dd] = 0.0f;
            dv_acc[dd] = 0.0f;
        }

        for (int q_tile = 0; q_tile < num_tiles; q_tile++) {
            // Causal: skip Q tiles entirely before this KV row
            int q_end = (q_tile + 1) * TILE_SIZE - 1;
            if (q_end < kv_tile * TILE_SIZE) continue;

            // Load Q tile, dO tile, and CS tile
            int q_row = q_tile * TILE_SIZE + ti;
            for (int dd = 0; dd < head_dim; dd++) {
                if (q_row < seq_len) {
                    smem_a[ti * head_dim + dd] = IO_LOAD(Q, base + q_row * head_dim + dd);
                    smem_b[ti * head_dim + dd] = IO_LOAD(grad_O, base + q_row * head_dim + dd);
                } else {
                    smem_a[ti * head_dim + dd] = 0.0f;
                    smem_b[ti * head_dim + dd] = 0.0f;
                }
            }
            smem_cs[ti] = (q_row < seq_len) ? IO_LOAD(CS, cs_base + q_row) : 0.0f;
            __syncthreads();

            if (kv_valid) {
                int tile_len = min(TILE_SIZE, seq_len - q_tile * TILE_SIZE);
                for (int qi = 0; qi < tile_len; qi++) {
                    int q_pos = q_tile * TILE_SIZE + qi;
                    if (q_pos < kv_idx) continue;  // causal

                    float s = 0.0f;
                    for (int dd = 0; dd < head_dim; dd++) {
                        s += smem_a[qi * head_dim + dd] * k_row[dd];
                    }
                    s *= scale;

                    // Add forget bias
                    if (q_pos != kv_idx) {
                        s += (smem_cs[qi] - cs_kv);
                    }

                    float lse_q = lse_buf[aux_base + q_pos];
                    float p = expf(s - lse_q);

                    float dp = 0.0f;
                    for (int dd = 0; dd < head_dim; dd++) {
                        dv_acc[dd] += p * smem_b[qi * head_dim + dd];
                        dp += smem_b[qi * head_dim + dd] * v_row[dd];
                    }

                    float D_q = D_buf[aux_base + q_pos];
                    float ds = p * (dp - D_q);

                    for (int dd = 0; dd < head_dim; dd++) {
                        dk_acc[dd] += ds * smem_a[qi * head_dim + dd] * scale;
                    }

                    // Accumulate col sum of ds for grad_cs (minus direction for col j)
                    if (q_pos != kv_idx) {
                        col_sum_ds += ds;
                    }
                }
            }
            __syncthreads();
        }

        if (kv_valid) {
            for (int dd = 0; dd < head_dim; dd++) {
                IO_STORE(dK, base + kv_idx * head_dim + dd, dk_acc[dd]);
                IO_STORE(dV, base + kv_idx * head_dim + dd, dv_acc[dd]);
            }
            // Store negative col sum (will be subtracted from row sum later)
            // grad_cs[j] -= col_sum_ds[j]
            // Initialize grad_cs with -col_sum
            IO_STORE(grad_cs, cs_base + kv_idx, -col_sum_ds);
        }
    }

    // ==================================================================
    // Phase 3: Compute dQ and row sums for grad_cs
    // ==================================================================
    for (int q_tile = 0; q_tile < num_tiles; q_tile++) {
        int q_idx = q_tile * TILE_SIZE + ti;
        bool q_valid = q_idx < seq_len;

        float dq_acc[128];
        float q_row[128];
        float cs_qi = q_valid ? IO_LOAD(CS, cs_base + q_idx) : 0.0f;
        float row_sum_ds = 0.0f;

        for (int dd = 0; dd < head_dim; dd++) {
            dq_acc[dd] = 0.0f;
            q_row[dd] = q_valid ? IO_LOAD(Q, base + q_idx * head_dim + dd) : 0.0f;
        }

        float lse_i = q_valid ? lse_buf[aux_base + q_idx] : 0.0f;
        float D_i = q_valid ? D_buf[aux_base + q_idx] : 0.0f;

        for (int kv_tile = 0; kv_tile < num_tiles; kv_tile++) {
            // Causal: skip KV tiles entirely in the future
            int kv_start = kv_tile * TILE_SIZE;
            if (q_valid && kv_start > q_idx) break;

            // Load K, V, and CS tiles
            int kv_row = kv_tile * TILE_SIZE + ti;
            for (int dd = 0; dd < head_dim; dd++) {
                if (kv_row < seq_len) {
                    smem_a[ti * head_dim + dd] = IO_LOAD(K, base + kv_row * head_dim + dd);
                    smem_b[ti * head_dim + dd] = IO_LOAD(V, base + kv_row * head_dim + dd);
                } else {
                    smem_a[ti * head_dim + dd] = 0.0f;
                    smem_b[ti * head_dim + dd] = 0.0f;
                }
            }
            smem_cs[ti] = (kv_row < seq_len) ? IO_LOAD(CS, cs_base + kv_row) : 0.0f;
            __syncthreads();

            if (q_valid) {
                int tile_len = min(TILE_SIZE, seq_len - kv_tile * TILE_SIZE);
                for (int j = 0; j < tile_len; j++) {
                    int kv_pos = kv_tile * TILE_SIZE + j;
                    if (kv_pos > q_idx) break;  // causal

                    float s = 0.0f;
                    for (int dd = 0; dd < head_dim; dd++) {
                        s += q_row[dd] * smem_a[j * head_dim + dd];
                    }
                    s *= scale;

                    if (kv_pos != q_idx) {
                        s += (cs_qi - smem_cs[j]);
                    }

                    float p = expf(s - lse_i);

                    float dp = 0.0f;
                    for (int dd = 0; dd < head_dim; dd++) {
                        dp += IO_LOAD(grad_O, base + q_idx * head_dim + dd) * smem_b[j * head_dim + dd];
                    }

                    float ds = p * (dp - D_i);

                    for (int dd = 0; dd < head_dim; dd++) {
                        dq_acc[dd] += ds * smem_a[j * head_dim + dd] * scale;
                    }

                    // Row sum of ds for grad_cs
                    if (kv_pos != q_idx) {
                        row_sum_ds += ds;
                    }
                }
            }
            __syncthreads();
        }

        if (q_valid) {
            for (int dd = 0; dd < head_dim; dd++) {
                IO_STORE(dQ, base + q_idx * head_dim + dd, dq_acc[dd]);
            }
            // Add row sum to grad_cs (already has -col_sum from Phase 2)
            float existing = IO_LOAD(grad_cs, cs_base + q_idx);
            IO_STORE(grad_cs, cs_base + q_idx, existing + row_sum_ds);
        }
    }
}

// ============================================================================
// Helper kernel: Precompute D[i] and lse[i] for FoX backward
// ============================================================================

__global__ void fox_attention_backward_precompute_kernel(
    const io_type* __restrict__ Q,       // [B, H, T, d]
    const io_type* __restrict__ K,       // [B, H, T, d]
    const io_type* __restrict__ CS,      // [B, H, T]
    const io_type* __restrict__ O,       // [B, H, T, d]
    const io_type* __restrict__ grad_O,  // [B, H, T, d]
    float* __restrict__ D_buf,           // [B*H*T]
    float* __restrict__ lse_buf,         // [B*H*T]
    int seq_len,
    int head_dim
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int qi = threadIdx.x;

    extern __shared__ float smem[];
    float* Kj = smem;              // [TILE_SIZE][head_dim]
    float* CSj = Kj + TILE_SIZE * head_dim;  // [TILE_SIZE]

    float scale = rsqrtf((float)head_dim);

    int BH_stride = seq_len * head_dim;
    int base = (b * gridDim.y + h) * BH_stride;
    int cs_base = (b * gridDim.y + h) * seq_len;
    int aux_base = cs_base;

    int num_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;

    for (int q_tile = 0; q_tile < num_tiles; q_tile++) {
        int i = q_tile * TILE_SIZE + qi;
        if (i >= seq_len) continue;

        // D[i] = sum(dO[i] * O[i])
        float d_val = 0.0f;
        for (int dd = 0; dd < head_dim; dd++) {
            d_val += IO_LOAD(grad_O, base + i * head_dim + dd) *
                     IO_LOAD(O, base + i * head_dim + dd);
        }
        D_buf[aux_base + i] = d_val;

        // Recompute lse[i] with forget bias
        float cs_i = IO_LOAD(CS, cs_base + i);
        float m_i = -FLT_MAX;
        float l_i = 0.0f;

        for (int kv_tile = 0; kv_tile < num_tiles; kv_tile++) {
            int kv_start = kv_tile * TILE_SIZE;
            if (kv_start > i) break;

            int kv_row = kv_tile * TILE_SIZE + qi;
            if (kv_row < seq_len) {
                for (int dd = 0; dd < head_dim; dd++) {
                    Kj[qi * head_dim + dd] = IO_LOAD(K, base + kv_row * head_dim + dd);
                }
                CSj[qi] = IO_LOAD(CS, cs_base + kv_row);
            } else {
                for (int dd = 0; dd < head_dim; dd++) {
                    Kj[qi * head_dim + dd] = 0.0f;
                }
                CSj[qi] = 0.0f;
            }
            __syncthreads();

            int tile_len = min(TILE_SIZE, seq_len - kv_tile * TILE_SIZE);
            for (int j = 0; j < tile_len; j++) {
                int kv_pos = kv_tile * TILE_SIZE + j;
                if (kv_pos > i) break;

                float s = 0.0f;
                for (int dd = 0; dd < head_dim; dd++) {
                    s += IO_LOAD(Q, base + i * head_dim + dd) * Kj[j * head_dim + dd];
                }
                s *= scale;

                // Add forget bias (zeroed on diagonal)
                if (kv_pos != i) {
                    s += (cs_i - CSj[j]);
                }

                float m_new = fmaxf(m_i, s);
                float exp_diff = expf(m_i - m_new);
                float exp_s = expf(s - m_new);
                l_i = l_i * exp_diff + exp_s;
                m_i = m_new;
            }
            __syncthreads();
        }

        lse_buf[aux_base + i] = m_i + logf(l_i);
    }
}

// ============================================================================
// Standalone launch wrapper (C-linkage for NIF / dlopen)
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

// Outputs: [dQ (B*H*T*d) | dK (B*H*T*d) | dV (B*H*T*d) | grad_cs (B*H*T)]
int fused_fox_attention_backward_launch(
    cudaStream_t stream,
    const io_type* q, const io_type* k, const io_type* v,
    const io_type* cs, const io_type* o, const io_type* grad_o,
    io_type* output_concat,
    int batch, int num_heads, int seq_len, int head_dim
) {
    size_t bhtd = (size_t)batch * num_heads * seq_len * head_dim;
    size_t bht = (size_t)batch * num_heads * seq_len;
    io_type* dq = output_concat;
    io_type* dk = output_concat + bhtd;
    io_type* dv = output_concat + 2 * bhtd;
    io_type* gcs = output_concat + 3 * bhtd;

    float* d_buf = NULL;
    float* lse_buf = NULL;
    cudaError_t err;

    err = cudaMalloc(&d_buf, bht * sizeof(float));
    if (err != cudaSuccess) return (int)err;
    err = cudaMalloc(&lse_buf, bht * sizeof(float));
    if (err != cudaSuccess) { cudaFree(d_buf); return (int)err; }

    dim3 grid(batch, num_heads);
    dim3 block(TILE_SIZE);

    // Precompute D and lse (shared memory: Kj + CSj)
    size_t smem_precompute = (TILE_SIZE * head_dim + TILE_SIZE) * sizeof(float);
    fox_attention_backward_precompute_kernel<<<grid, block, smem_precompute, stream>>>(
        q, k, cs, o, grad_o, d_buf, lse_buf,
        seq_len, head_dim
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_buf); cudaFree(lse_buf);
        return (int)err;
    }

    // Main backward kernel (shared memory: smem_a + smem_b + smem_cs)
    size_t smem_main = (2 * TILE_SIZE * head_dim + TILE_SIZE) * sizeof(float);
    fused_fox_attention_backward_kernel<<<grid, block, smem_main, stream>>>(
        q, k, v, cs, o, grad_o,
        dq, dk, dv, gcs,
        d_buf, lse_buf,
        seq_len, head_dim
    );

    err = cudaGetLastError();
    cudaFree(d_buf);
    cudaFree(lse_buf);
    return (int)err;
}

}  // extern "C"

#endif  // !EXLA_FFI

// ============================================================================
// XLA FFI integration (for EXLA fork)
// ============================================================================

#ifdef EXLA_FFI

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

ffi::Error fused_fox_attention_backward_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> q,
    ffi::Buffer<FFI_IO_TYPE> k,
    ffi::Buffer<FFI_IO_TYPE> v,
    ffi::Buffer<FFI_IO_TYPE> cs,
    ffi::Buffer<FFI_IO_TYPE> o,
    ffi::Buffer<FFI_IO_TYPE> grad_o,
    ffi::ResultBuffer<FFI_IO_TYPE> dq,
    ffi::ResultBuffer<FFI_IO_TYPE> dk,
    ffi::ResultBuffer<FFI_IO_TYPE> dv,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_cs
) {
    auto dims = q.dimensions();
    int batch    = static_cast<int>(dims[0]);
    int num_heads = static_cast<int>(dims[1]);
    int seq_len  = static_cast<int>(dims[2]);
    int head_dim = static_cast<int>(dims[3]);

    size_t bht = (size_t)batch * num_heads * seq_len;
    float* d_buf = NULL;
    float* lse_buf = NULL;
    cudaError_t err;

    err = cudaMalloc(&d_buf, bht * sizeof(float));
    if (err != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, "cudaMalloc failed for d_buf");
    err = cudaMalloc(&lse_buf, bht * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_buf);
        return ffi::Error(ffi::ErrorCode::kInternal, "cudaMalloc failed for lse_buf");
    }

    dim3 grid(batch, num_heads);
    dim3 block(TILE_SIZE);

    size_t smem_precompute = (TILE_SIZE * head_dim + TILE_SIZE) * sizeof(float);
    fox_attention_backward_precompute_kernel<<<grid, block, smem_precompute, stream>>>(
        reinterpret_cast<const io_type*>(q.untyped_data()),
        reinterpret_cast<const io_type*>(k.untyped_data()),
        reinterpret_cast<const io_type*>(cs.untyped_data()),
        reinterpret_cast<const io_type*>(o.untyped_data()),
        reinterpret_cast<const io_type*>(grad_o.untyped_data()),
        d_buf, lse_buf,
        seq_len, head_dim
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_buf); cudaFree(lse_buf);
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    size_t smem_main = (2 * TILE_SIZE * head_dim + TILE_SIZE) * sizeof(float);
    fused_fox_attention_backward_kernel<<<grid, block, smem_main, stream>>>(
        reinterpret_cast<const io_type*>(q.untyped_data()),
        reinterpret_cast<const io_type*>(k.untyped_data()),
        reinterpret_cast<const io_type*>(v.untyped_data()),
        reinterpret_cast<const io_type*>(cs.untyped_data()),
        reinterpret_cast<const io_type*>(o.untyped_data()),
        reinterpret_cast<const io_type*>(grad_o.untyped_data()),
        reinterpret_cast<io_type*>(dq->untyped_data()),
        reinterpret_cast<io_type*>(dk->untyped_data()),
        reinterpret_cast<io_type*>(dv->untyped_data()),
        reinterpret_cast<io_type*>(grad_cs->untyped_data()),
        d_buf, lse_buf,
        seq_len, head_dim
    );

    err = cudaGetLastError();
    cudaFree(d_buf);
    cudaFree(lse_buf);

    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_fox_attention_backward, fused_fox_attention_backward_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // q       [B, H, T, d]
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // k       [B, H, T, d]
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // v       [B, H, T, d]
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // cs      [B, H, T]
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // o       [B, H, T, d]
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // grad_o  [B, H, T, d]
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // dQ      [B, H, T, d]
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // dK      [B, H, T, d]
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // dV      [B, H, T, d]
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_cs [B, H, T]
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_fox_attention_backward_" PRECISION_SUFFIX, "CUDA", fused_fox_attention_backward);

#endif  // EXLA_FFI
