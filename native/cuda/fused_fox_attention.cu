// Fused FoX (Forgetting Transformer) Attention Kernel
//
// Augments standard flash attention with an additive forget bias on logits:
//   scores'[i,j] = QK^T[i,j] / sqrt(d) + (cs[i] - cs[j])
// where cs = cumsum(log(sigmoid(f))) is the cumulative log-forget.
//
// The forget bias is precomputed on the host side:
//   cs: [B, H, T] — cumulative sum of log(sigmoid(forget_logits))
//
// At each (i, j) position, the bias is cs[i] - cs[j], which creates an
// exponentially decaying window: recent tokens get full weight, distant
// tokens get exponentially discounted.
//
// FoX is always causal (the forget gate is inherently directional).
//
// Thread layout: one thread block per (batch, head) pair.
// Each block has TILE_SIZE threads (one per Q row in the current tile).
//
// Inputs:
//   q:  [B, H, T, d]  — query vectors
//   k:  [B, H, T, d]  — key vectors
//   v:  [B, H, T, d]  — value vectors
//   cs: [B, H, T]     — cumulative log-forget (precomputed)
//
// Output:
//   output: [B, H, T, d]  — FoX attention output

#include <cuda_runtime.h>
#include "precision.cuh"
#include <float.h>
#include <math.h>

#define TILE_SIZE 32

__global__ void fused_fox_attention_kernel(
    const io_type* __restrict__ Q,    // [B, H, T, d]
    const io_type* __restrict__ K,    // [B, H, T, d]
    const io_type* __restrict__ V,    // [B, H, T, d]
    const io_type* __restrict__ CS,   // [B, H, T]
    io_type* __restrict__ O,          // [B, H, T, d]
    int seq_len,
    int head_dim
) {
    int b = blockIdx.x;   // batch index
    int h = blockIdx.y;   // head index
    int qi = threadIdx.x; // which Q row within this tile

    extern __shared__ float smem[];
    float* Kj = smem;                         // [TILE_SIZE][head_dim]
    float* Vj = Kj + TILE_SIZE * head_dim;    // [TILE_SIZE][head_dim]
    float* CSj = Vj + TILE_SIZE * head_dim;   // [TILE_SIZE]

    float scale = rsqrtf((float)head_dim);

    int BH_stride = seq_len * head_dim;
    int base = (b * gridDim.y + h) * BH_stride;

    // CS base: [B, H, T] → stride = seq_len per (b,h)
    int cs_base = (b * gridDim.y + h) * seq_len;

    int num_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;

    for (int qi_tile = 0; qi_tile < num_tiles; qi_tile++) {
        int i = qi_tile * TILE_SIZE + qi;
        if (i >= seq_len) continue;

        float m_i = -FLT_MAX;
        float l_i = 0.0f;

        // Load cs[i] for this thread's Q position
        float cs_i = IO_LOAD(CS, cs_base + i);

        float o_acc[128];
        for (int dd = 0; dd < head_dim; dd++) {
            o_acc[dd] = 0.0f;
        }

        for (int kv_tile = 0; kv_tile < num_tiles; kv_tile++) {
            // Always causal: skip future tiles
            int kv_start = kv_tile * TILE_SIZE;
            if (kv_start > i) break;

            // Cooperatively load K, V, and CS tiles
            int kv_row = kv_tile * TILE_SIZE + qi;
            if (kv_row < seq_len) {
                for (int dd = 0; dd < head_dim; dd++) {
                    Kj[qi * head_dim + dd] = IO_LOAD(K, base + kv_row * head_dim + dd);
                    Vj[qi * head_dim + dd] = IO_LOAD(V, base + kv_row * head_dim + dd);
                }
                CSj[qi] = IO_LOAD(CS, cs_base + kv_row);
            } else {
                for (int dd = 0; dd < head_dim; dd++) {
                    Kj[qi * head_dim + dd] = 0.0f;
                    Vj[qi * head_dim + dd] = 0.0f;
                }
                CSj[qi] = 0.0f;
            }
            __syncthreads();

            int tile_len = min(TILE_SIZE, seq_len - kv_tile * TILE_SIZE);

            for (int j = 0; j < tile_len; j++) {
                int kv_pos = kv_tile * TILE_SIZE + j;
                if (kv_pos > i) break;  // causal

                // Dot product: s = Q[i] . K[j] * scale
                float s = 0.0f;
                for (int dd = 0; dd < head_dim; dd++) {
                    s += IO_LOAD(Q, base + i * head_dim + dd) * Kj[j * head_dim + dd];
                }
                s *= scale;

                // Add forget bias: cs[i] - cs[j]
                // For diagonal (i == j), bias is 0 (no forgetting for self-attention)
                if (kv_pos != i) {
                    s += (cs_i - CSj[j]);
                }

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

        // Final normalization
        if (i < seq_len && l_i > 0.0f) {
            float inv_l = 1.0f / l_i;
            for (int dd = 0; dd < head_dim; dd++) {
                IO_STORE(O, base + i * head_dim + dd, o_acc[dd] * inv_l);
            }
        }
    }
}

// ============================================================================
// Standalone launch wrapper (C-linkage for NIF / dlopen)
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

int fused_fox_attention_launch(
    cudaStream_t stream,
    const io_type* q,
    const io_type* k,
    const io_type* v,
    const io_type* cs,
    io_type* output,
    int batch,
    int num_heads,
    int seq_len,
    int head_dim
) {
    dim3 grid(batch, num_heads);
    dim3 block(TILE_SIZE);
    // Shared: Kj[T][d] + Vj[T][d] + CSj[T]
    size_t smem_bytes = (2 * TILE_SIZE * head_dim + TILE_SIZE) * sizeof(float);

    fused_fox_attention_kernel<<<grid, block, smem_bytes, stream>>>(
        q, k, v, cs, output,
        seq_len, head_dim
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

ffi::Error fused_fox_attention_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> q,
    ffi::Buffer<FFI_IO_TYPE> k,
    ffi::Buffer<FFI_IO_TYPE> v,
    ffi::Buffer<FFI_IO_TYPE> cs,
    ffi::ResultBuffer<FFI_IO_TYPE> output
) {
    auto dims = q.dimensions();
    int batch     = static_cast<int>(dims[0]);
    int num_heads = static_cast<int>(dims[1]);
    int seq_len   = static_cast<int>(dims[2]);
    int head_dim  = static_cast<int>(dims[3]);

    dim3 grid(batch, num_heads);
    dim3 block(TILE_SIZE);
    size_t smem_bytes = (2 * TILE_SIZE * head_dim + TILE_SIZE) * sizeof(float);

    fused_fox_attention_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const io_type*>(q.untyped_data()),
        reinterpret_cast<const io_type*>(k.untyped_data()),
        reinterpret_cast<const io_type*>(v.untyped_data()),
        reinterpret_cast<const io_type*>(cs.untyped_data()),
        reinterpret_cast<io_type*>(output->untyped_data()),
        seq_len, head_dim
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_fox_attention, fused_fox_attention_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // q   [B, H, T, d]
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // k   [B, H, T, d]
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // v   [B, H, T, d]
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // cs  [B, H, T]
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // output [B, H, T, d]
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_fox_attention_" PRECISION_SUFFIX, "CUDA", fused_fox_attention);

#endif  // EXLA_FFI
