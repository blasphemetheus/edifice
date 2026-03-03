// Fused LASER Attention Kernel
//
// LASER: Attention with Exponential Transformation.
// Computes: O = log(softmax(QK^T / sqrt(d)) @ exp(V)) using LWSE trick.
//
// Instead of accumulating V directly (as in standard flash attention),
// we accumulate exp(V - v_max) and apply log(result) + v_max at output.
// This decouples gradient flow from attention weight magnitude.
//
// The online softmax (Flash Attention V2) algorithm is preserved exactly —
// only the V accumulation and output writeback steps differ.
//
// Thread layout: one thread block per (batch, head) pair.
// Each block has TILE_SIZE threads (one per Q row in the current tile).
//
// Inputs:
//   q: [B, H, T, d]  — query vectors
//   k: [B, H, T, d]  — key vectors
//   v: [B, H, T, d]  — value vectors (raw, not exponentiated)
//   v_max: [B, H, 1, d] — per-column max of V (precomputed on host)
//
// Output:
//   output: [B, H, T, d]  — LASER attention output (in log space)
//
// Causal: when causal=1, positions j > i are masked.

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include "precision.cuh"

#define TILE_SIZE 32

__global__ void fused_laser_attention_kernel(
    const io_type* __restrict__ Q,       // [B, H, T, d]
    const io_type* __restrict__ K,       // [B, H, T, d]
    const io_type* __restrict__ V,       // [B, H, T, d]
    const io_type* __restrict__ V_max,   // [B, H, 1, d]
    io_type* __restrict__ O,             // [B, H, T, d]
    int seq_len,
    int head_dim,
    int causal
) {
    int b = blockIdx.x;   // batch index
    int h = blockIdx.y;   // head index
    int qi = threadIdx.x; // which Q row within this tile

    extern __shared__ float smem[];
    float* Kj = smem;                         // [TILE_SIZE][head_dim]
    float* Vj = Kj + TILE_SIZE * head_dim;    // [TILE_SIZE][head_dim]

    float scale = rsqrtf((float)head_dim);

    int BH_stride = seq_len * head_dim;
    int base = (b * gridDim.y + h) * BH_stride;

    // Base offset for V_max: [B, H, 1, d]
    int vmax_base = (b * gridDim.y + h) * head_dim;

    int num_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;

    for (int qi_tile = 0; qi_tile < num_tiles; qi_tile++) {
        int i = qi_tile * TILE_SIZE + qi;
        if (i >= seq_len) continue;

        float m_i = -FLT_MAX;
        float l_i = 0.0f;

        float o_acc[128];
        for (int dd = 0; dd < head_dim; dd++) {
            o_acc[dd] = 0.0f;
        }

        for (int kv_tile = 0; kv_tile < num_tiles; kv_tile++) {
            if (causal) {
                int kv_start = kv_tile * TILE_SIZE;
                if (kv_start > i) break;
            }

            // Cooperatively load K and V tiles
            int kv_row = kv_tile * TILE_SIZE + qi;
            if (kv_row < seq_len) {
                for (int dd = 0; dd < head_dim; dd++) {
                    Kj[qi * head_dim + dd] = IO_LOAD(K, base + kv_row * head_dim + dd);
                    // Load exp(V - v_max) into shared memory
                    float v_val = IO_LOAD(V, base + kv_row * head_dim + dd);
                    float vm = IO_LOAD(V_max, vmax_base + dd);
                    Vj[qi * head_dim + dd] = expf(v_val - vm);
                }
            } else {
                for (int dd = 0; dd < head_dim; dd++) {
                    Kj[qi * head_dim + dd] = 0.0f;
                    Vj[qi * head_dim + dd] = 0.0f;
                }
            }
            __syncthreads();

            int tile_len = min(TILE_SIZE, seq_len - kv_tile * TILE_SIZE);

            for (int j = 0; j < tile_len; j++) {
                int kv_pos = kv_tile * TILE_SIZE + j;
                if (causal && kv_pos > i) break;

                // Dot product: s = Q[i] . K[j] * scale
                float s = 0.0f;
                for (int dd = 0; dd < head_dim; dd++) {
                    s += IO_LOAD(Q, base + i * head_dim + dd) * Kj[j * head_dim + dd];
                }
                s *= scale;

                // Online softmax update
                float m_new = fmaxf(m_i, s);
                float exp_diff = expf(m_i - m_new);
                float exp_s = expf(s - m_new);

                l_i = l_i * exp_diff + exp_s;

                for (int dd = 0; dd < head_dim; dd++) {
                    o_acc[dd] = o_acc[dd] * exp_diff + exp_s * Vj[j * head_dim + dd];
                }

                m_i = m_new;
            }

            __syncthreads();
        }

        // Final: apply normalization, then log + v_max
        if (i < seq_len && l_i > 0.0f) {
            float inv_l = 1.0f / l_i;
            for (int dd = 0; dd < head_dim; dd++) {
                float normalized = o_acc[dd] * inv_l;
                float vm = IO_LOAD(V_max, vmax_base + dd);
                IO_STORE(O, base + i * head_dim + dd, logf(fmaxf(normalized, 1.0e-7f)) + vm);
            }
        }
    }
}

// ============================================================================
// Standalone launch wrapper (C-linkage for NIF / dlopen)
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

int fused_laser_attention_launch(
    cudaStream_t stream,
    const io_type* q,
    const io_type* k,
    const io_type* v,
    const io_type* v_max,
    io_type* output,
    int batch,
    int num_heads,
    int seq_len,
    int head_dim,
    int causal
) {
    dim3 grid(batch, num_heads);
    dim3 block(TILE_SIZE);
    size_t smem_bytes = 2 * TILE_SIZE * head_dim * sizeof(float);

    fused_laser_attention_kernel<<<grid, block, smem_bytes, stream>>>(
        q, k, v, v_max, output,
        seq_len, head_dim, causal
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

// Causal hardcoded to 1 to avoid scalar buffer operand segfault in XLA.
ffi::Error fused_laser_attention_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> q,
    ffi::Buffer<FFI_IO_TYPE> k,
    ffi::Buffer<FFI_IO_TYPE> v,
    ffi::Buffer<FFI_IO_TYPE> v_max,
    ffi::ResultBuffer<FFI_IO_TYPE> output
) {
    auto dims = q.dimensions();
    int batch     = static_cast<int>(dims[0]);
    int num_heads = static_cast<int>(dims[1]);
    int seq_len   = static_cast<int>(dims[2]);
    int head_dim  = static_cast<int>(dims[3]);

    int causal = 1;  // hardcoded causal

    dim3 grid(batch, num_heads);
    dim3 block(TILE_SIZE);
    size_t smem_bytes = 2 * TILE_SIZE * head_dim * sizeof(float);

    fused_laser_attention_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const io_type*>(q.untyped_data()),
        reinterpret_cast<const io_type*>(k.untyped_data()),
        reinterpret_cast<const io_type*>(v.untyped_data()),
        reinterpret_cast<const io_type*>(v_max.untyped_data()),
        reinterpret_cast<io_type*>(output->untyped_data()),
        seq_len, head_dim, causal
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_laser_attention, fused_laser_attention_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // q      [B, H, T, d]
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // k      [B, H, T, d]
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // v      [B, H, T, d]
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // v_max  [B, H, 1, d]
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // output [B, H, T, d]
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_laser_attention_" PRECISION_SUFFIX, "CUDA", fused_laser_attention);

#endif  // EXLA_FFI
