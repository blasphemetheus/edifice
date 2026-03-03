// Fused Flash Attention V2 Backward Kernel
//
// FlashAttention-2 backward pass with recomputed logsumexp (no forward
// modification needed). Computes dQ, dK, dV given Q, K, V, O, and dO.
//
// Algorithm (per batch, head):
//   Phase 0: D[i] = rowsum(dO[i] * O[i]) for all i
//   Phase 1: Recompute lse[i] = m_i + log(l_i) via online softmax over KV tiles
//   Phase 2 (dK, dV): Outer loop over KV tiles, inner loop over Q tiles.
//     Each thread owns one KV row, accumulates dK/dV in registers.
//   Phase 3 (dQ): Outer loop over Q tiles, inner loop over KV tiles.
//     Each thread owns one Q row, accumulates dQ in registers.
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
//   O:      [B, H, T, d] — forward output
//   grad_O: [B, H, T, d] — upstream gradient
//   causal: scalar int — 0 = full, 1 = causal mask
//
// Outputs:
//   dQ: [B, H, T, d] — gradient w.r.t. Q
//   dK: [B, H, T, d] — gradient w.r.t. K
//   dV: [B, H, T, d] — gradient w.r.t. V

#include <cuda_runtime.h>
#include "precision.cuh"
#include <float.h>
#include <math.h>

#define TILE_SIZE 32

// ============================================================================
// Kernel
// ============================================================================

__global__ void fused_flash_attention_backward_kernel(
    const io_type* __restrict__ Q,       // [B, H, T, d]
    const io_type* __restrict__ K,       // [B, H, T, d]
    const io_type* __restrict__ V,       // [B, H, T, d]
    const io_type* __restrict__ O,       // [B, H, T, d]
    const io_type* __restrict__ grad_O,  // [B, H, T, d]
    io_type* __restrict__ dQ,            // [B, H, T, d]
    io_type* __restrict__ dK,            // [B, H, T, d]
    io_type* __restrict__ dV,            // [B, H, T, d]
    const float* __restrict__ D_buf,     // [B*H*T] precomputed
    const float* __restrict__ lse_buf,   // [B*H*T] precomputed
    int seq_len,
    int head_dim,
    int causal
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int ti = threadIdx.x;  // thread index within tile (0..TILE_SIZE-1)

    extern __shared__ float smem[];
    // Phase 2: Qi[TILE_SIZE][head_dim] + dOi[TILE_SIZE][head_dim]
    // Phase 3: Kj[TILE_SIZE][head_dim] + Vj[TILE_SIZE][head_dim]
    float* smem_a = smem;                              // [TILE_SIZE][head_dim]
    float* smem_b = smem + TILE_SIZE * head_dim;       // [TILE_SIZE][head_dim]

    float scale = rsqrtf((float)head_dim);

    int BH_stride = seq_len * head_dim;
    int base = (b * gridDim.y + h) * BH_stride;
    int aux_base = (b * gridDim.y + h) * seq_len;  // for D_buf, lse_buf

    int num_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;

    // ==================================================================
    // Phase 2: Compute dK, dV
    // Outer loop over KV tiles. Each thread handles one KV row.
    // ==================================================================
    for (int kv_tile = 0; kv_tile < num_tiles; kv_tile++) {
        int kv_idx = kv_tile * TILE_SIZE + ti;
        bool kv_valid = kv_idx < seq_len;

        // Load own K and V row into registers
        float k_row[128], v_row[128];
        float dk_acc[128], dv_acc[128];
        for (int dd = 0; dd < head_dim; dd++) {
            k_row[dd] = kv_valid ? IO_LOAD(K, base + kv_idx * head_dim + dd) : 0.0f;
            v_row[dd] = kv_valid ? IO_LOAD(V, base + kv_idx * head_dim + dd) : 0.0f;
            dk_acc[dd] = 0.0f;
            dv_acc[dd] = 0.0f;
        }

        // Inner loop over Q tiles
        for (int q_tile = 0; q_tile < num_tiles; q_tile++) {
            // Causal: skip Q tiles that are entirely before this KV position
            if (causal) {
                int q_end = (q_tile + 1) * TILE_SIZE - 1;
                if (q_end < kv_tile * TILE_SIZE) continue;
            }

            // Cooperatively load Q tile and dO tile into shared memory
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
            __syncthreads();

            if (kv_valid) {
                int tile_len = min(TILE_SIZE, seq_len - q_tile * TILE_SIZE);
                for (int qi = 0; qi < tile_len; qi++) {
                    int q_pos = q_tile * TILE_SIZE + qi;

                    // Causal: skip if q_pos < kv_idx (q must attend to kv)
                    if (causal && q_pos < kv_idx) continue;

                    // s = Q[q] . K[kv] * scale
                    float s = 0.0f;
                    for (int dd = 0; dd < head_dim; dd++) {
                        s += smem_a[qi * head_dim + dd] * k_row[dd];
                    }
                    s *= scale;

                    // p = exp(s - lse[q])
                    float lse_q = lse_buf[aux_base + q_pos];
                    float p = expf(s - lse_q);

                    // dV += p * dO[q]
                    // dp = dot(dO[q], V[kv])
                    float dp = 0.0f;
                    for (int dd = 0; dd < head_dim; dd++) {
                        dv_acc[dd] += p * smem_b[qi * head_dim + dd];
                        dp += smem_b[qi * head_dim + dd] * v_row[dd];
                    }

                    // ds = p * (dp - D[q])
                    float D_q = D_buf[aux_base + q_pos];
                    float ds = p * (dp - D_q);

                    // dK += ds * Q[q] * scale
                    for (int dd = 0; dd < head_dim; dd++) {
                        dk_acc[dd] += ds * smem_a[qi * head_dim + dd] * scale;
                    }
                }
            }
            __syncthreads();
        }

        // Write dK, dV
        if (kv_valid) {
            for (int dd = 0; dd < head_dim; dd++) {
                IO_STORE(dK, base + kv_idx * head_dim + dd, dk_acc[dd]);
                IO_STORE(dV, base + kv_idx * head_dim + dd, dv_acc[dd]);
            }
        }
    }

    // ==================================================================
    // Phase 3: Compute dQ
    // Outer loop over Q tiles. Each thread handles one Q row.
    // ==================================================================
    for (int q_tile = 0; q_tile < num_tiles; q_tile++) {
        int q_idx = q_tile * TILE_SIZE + ti;
        bool q_valid = q_idx < seq_len;

        float dq_acc[128];
        float q_row[128];
        for (int dd = 0; dd < head_dim; dd++) {
            dq_acc[dd] = 0.0f;
            q_row[dd] = q_valid ? IO_LOAD(Q, base + q_idx * head_dim + dd) : 0.0f;
        }

        float lse_i = q_valid ? lse_buf[aux_base + q_idx] : 0.0f;
        float D_i = q_valid ? D_buf[aux_base + q_idx] : 0.0f;

        // Inner loop over KV tiles
        for (int kv_tile = 0; kv_tile < num_tiles; kv_tile++) {
            // Causal: skip KV tiles entirely in the future
            if (causal) {
                int kv_start = kv_tile * TILE_SIZE;
                if (q_valid && kv_start > q_idx) break;
            }

            // Cooperatively load K and V tiles into shared memory
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
            __syncthreads();

            if (q_valid) {
                int tile_len = min(TILE_SIZE, seq_len - kv_tile * TILE_SIZE);
                for (int j = 0; j < tile_len; j++) {
                    int kv_pos = kv_tile * TILE_SIZE + j;
                    if (causal && kv_pos > q_idx) break;

                    // s = Q[i] . K[j] * scale
                    float s = 0.0f;
                    for (int dd = 0; dd < head_dim; dd++) {
                        s += q_row[dd] * smem_a[j * head_dim + dd];
                    }
                    s *= scale;

                    // p = exp(s - lse[i])
                    float p = expf(s - lse_i);

                    // dp = dot(dO[i], V[j])
                    float dp = 0.0f;
                    for (int dd = 0; dd < head_dim; dd++) {
                        dp += IO_LOAD(grad_O, base + q_idx * head_dim + dd) * smem_b[j * head_dim + dd];
                    }

                    // ds = p * (dp - D[i])
                    float ds = p * (dp - D_i);

                    // dQ[i] += ds * K[j] * scale
                    for (int dd = 0; dd < head_dim; dd++) {
                        dq_acc[dd] += ds * smem_a[j * head_dim + dd] * scale;
                    }
                }
            }
            __syncthreads();
        }

        // Write dQ
        if (q_valid) {
            for (int dd = 0; dd < head_dim; dd++) {
                IO_STORE(dQ, base + q_idx * head_dim + dd, dq_acc[dd]);
            }
        }
    }
}

// ============================================================================
// Helper kernel: Compute D[i] = sum(dO[i] * O[i]) and lse[i] via online softmax
// ============================================================================

__global__ void flash_attention_backward_precompute_kernel(
    const io_type* __restrict__ Q,       // [B, H, T, d]
    const io_type* __restrict__ K,       // [B, H, T, d]
    const io_type* __restrict__ O,       // [B, H, T, d]
    const io_type* __restrict__ grad_O,  // [B, H, T, d]
    float* __restrict__ D_buf,           // [B*H*T]
    float* __restrict__ lse_buf,         // [B*H*T]
    int seq_len,
    int head_dim,
    int causal
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int qi = threadIdx.x;

    extern __shared__ float smem[];
    float* Kj = smem;  // [TILE_SIZE][head_dim]

    float scale = rsqrtf((float)head_dim);

    int BH_stride = seq_len * head_dim;
    int base = (b * gridDim.y + h) * BH_stride;
    int aux_base = (b * gridDim.y + h) * seq_len;

    int num_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;

    // Process each Q row assigned to this thread
    for (int q_tile = 0; q_tile < num_tiles; q_tile++) {
        int i = q_tile * TILE_SIZE + qi;
        if (i >= seq_len) continue;

        // Phase 0: D[i] = sum(dO[i] * O[i])
        float d_val = 0.0f;
        for (int dd = 0; dd < head_dim; dd++) {
            d_val += IO_LOAD(grad_O, base + i * head_dim + dd) *
                     IO_LOAD(O, base + i * head_dim + dd);
        }
        D_buf[aux_base + i] = d_val;

        // Phase 1: Recompute lse[i] via online softmax
        float m_i = -FLT_MAX;
        float l_i = 0.0f;

        for (int kv_tile = 0; kv_tile < num_tiles; kv_tile++) {
            if (causal) {
                int kv_start = kv_tile * TILE_SIZE;
                if (kv_start > i) break;
            }

            // Cooperatively load K tile
            int kv_row = kv_tile * TILE_SIZE + qi;
            if (kv_row < seq_len) {
                for (int dd = 0; dd < head_dim; dd++) {
                    Kj[qi * head_dim + dd] = IO_LOAD(K, base + kv_row * head_dim + dd);
                }
            } else {
                for (int dd = 0; dd < head_dim; dd++) {
                    Kj[qi * head_dim + dd] = 0.0f;
                }
            }
            __syncthreads();

            int tile_len = min(TILE_SIZE, seq_len - kv_tile * TILE_SIZE);
            for (int j = 0; j < tile_len; j++) {
                int kv_pos = kv_tile * TILE_SIZE + j;
                if (causal && kv_pos > i) break;

                float s = 0.0f;
                for (int dd = 0; dd < head_dim; dd++) {
                    s += IO_LOAD(Q, base + i * head_dim + dd) * Kj[j * head_dim + dd];
                }
                s *= scale;

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

// Outputs written to concatenated buffer: [dQ | dK | dV] each B*H*T*d
int fused_flash_attention_backward_launch(
    cudaStream_t stream,
    const io_type* q, const io_type* k, const io_type* v,
    const io_type* o, const io_type* grad_o,
    io_type* output_concat,
    int batch, int num_heads, int seq_len, int head_dim,
    int causal
) {
    size_t bhtd = (size_t)batch * num_heads * seq_len * head_dim;
    io_type* dq = output_concat;
    io_type* dk = output_concat + bhtd;
    io_type* dv = output_concat + 2 * bhtd;

    // Allocate temp buffers for D and lse
    size_t bht = (size_t)batch * num_heads * seq_len;
    float* d_buf = NULL;
    float* lse_buf = NULL;
    cudaError_t err;

    err = cudaMalloc(&d_buf, bht * sizeof(float));
    if (err != cudaSuccess) return (int)err;
    err = cudaMalloc(&lse_buf, bht * sizeof(float));
    if (err != cudaSuccess) { cudaFree(d_buf); return (int)err; }

    dim3 grid(batch, num_heads);
    dim3 block(TILE_SIZE);

    // Precompute D and lse
    size_t smem_precompute = TILE_SIZE * head_dim * sizeof(float);
    flash_attention_backward_precompute_kernel<<<grid, block, smem_precompute, stream>>>(
        q, k, o, grad_o, d_buf, lse_buf,
        seq_len, head_dim, causal
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_buf); cudaFree(lse_buf);
        return (int)err;
    }

    // Main backward kernel
    size_t smem_main = 2 * TILE_SIZE * head_dim * sizeof(float);
    fused_flash_attention_backward_kernel<<<grid, block, smem_main, stream>>>(
        q, k, v, o, grad_o,
        dq, dk, dv,
        d_buf, lse_buf,
        seq_len, head_dim, causal
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

// Causal hardcoded to 1 to avoid scalar buffer operand segfault in XLA.
ffi::Error fused_flash_attention_backward_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> q,
    ffi::Buffer<FFI_IO_TYPE> k,
    ffi::Buffer<FFI_IO_TYPE> v,
    ffi::Buffer<FFI_IO_TYPE> o,
    ffi::Buffer<FFI_IO_TYPE> grad_o,
    ffi::ResultBuffer<FFI_IO_TYPE> dq,
    ffi::ResultBuffer<FFI_IO_TYPE> dk,
    ffi::ResultBuffer<FFI_IO_TYPE> dv
) {
    auto dims = q.dimensions();
    int batch    = static_cast<int>(dims[0]);
    int num_heads = static_cast<int>(dims[1]);
    int seq_len  = static_cast<int>(dims[2]);
    int head_dim = static_cast<int>(dims[3]);

    int causal = 1;  // hardcoded causal

    // Allocate temp buffers for D and lse
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

    // Precompute D and lse
    size_t smem_precompute = TILE_SIZE * head_dim * sizeof(float);
    flash_attention_backward_precompute_kernel<<<grid, block, smem_precompute, stream>>>(
        reinterpret_cast<const io_type*>(q.untyped_data()),
        reinterpret_cast<const io_type*>(k.untyped_data()),
        reinterpret_cast<const io_type*>(o.untyped_data()),
        reinterpret_cast<const io_type*>(grad_o.untyped_data()),
        d_buf, lse_buf,
        seq_len, head_dim, causal
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_buf); cudaFree(lse_buf);
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    // Main backward kernel
    size_t smem_main = 2 * TILE_SIZE * head_dim * sizeof(float);
    fused_flash_attention_backward_kernel<<<grid, block, smem_main, stream>>>(
        reinterpret_cast<const io_type*>(q.untyped_data()),
        reinterpret_cast<const io_type*>(k.untyped_data()),
        reinterpret_cast<const io_type*>(v.untyped_data()),
        reinterpret_cast<const io_type*>(o.untyped_data()),
        reinterpret_cast<const io_type*>(grad_o.untyped_data()),
        reinterpret_cast<io_type*>(dq->untyped_data()),
        reinterpret_cast<io_type*>(dk->untyped_data()),
        reinterpret_cast<io_type*>(dv->untyped_data()),
        d_buf, lse_buf,
        seq_len, head_dim, causal
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
    fused_flash_attention_backward, fused_flash_attention_backward_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // q       [B, H, T, d]
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // k       [B, H, T, d]
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // v       [B, H, T, d]
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // o       [B, H, T, d]
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // grad_o  [B, H, T, d]
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // dQ      [B, H, T, d]
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // dK      [B, H, T, d]
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // dV      [B, H, T, d]
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_flash_attention_backward_" PRECISION_SUFFIX, "CUDA", fused_flash_attention_backward);

#endif  // EXLA_FFI
