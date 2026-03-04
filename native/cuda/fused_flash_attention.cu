// Fused Flash Attention V2 Kernel
//
// IO-aware exact attention that avoids materializing the full [seq, seq]
// attention matrix. Uses online softmax with tiled loading of K/V blocks,
// keeping running max and sum statistics to normalize at the end.
//
// This is a GPU performance optimization — the Elixir fallback
// (Primitives.multi_head_sdpa) produces identical results.
//
// Thread layout: one thread block per (batch, head) pair.
// Each block has Br threads (one per Q row in the current tile).
// Outer loop: iterate over K/V blocks of size Bc.
// Inner: each thread handles one Q row, accumulating output and softmax stats.
//
// Two compilation modes:
//   1. Standalone (default): Compiles kernel + C-linkage launch wrapper.
//      Use for testing and NIF-based integration.
//   2. EXLA FFI (-DEXLA_FFI): Compiles kernel + XLA FFI handler + registration.
//      Use when building inside deps/exla/c_src/exla/custom_calls/.
//
// Inputs:
//   q: [B, H, T, d]  — query vectors
//   k: [B, H, T, d]  — key vectors
//   v: [B, H, T, d]  — value vectors
//
// Output:
//   output: [B, H, T, d]  — attention output
//
// Causal: when causal=1, positions j > i are masked (only attend to past).
//
// Supported head dims: 32, 64, 128 (covers GPT, LLaMA, Whisper, ViT).
// Tile sizes: Br = Bc = 32 (fits comfortably in shared memory for all head dims).

#include <cuda_runtime.h>
#include <float.h>
#include "precision.cuh"

// ============================================================================
// Configuration
// ============================================================================

#define TILE_SIZE 32

// ============================================================================
// Kernel
// ============================================================================

#ifdef EXLA_FFI
namespace {  // anonymous namespace — internal linkage per compilation unit
#endif

__global__ void fused_flash_attention_kernel(
    const io_type* __restrict__ Q,       // [B, H, T, d]
    const io_type* __restrict__ K,       // [B, H, T, d]
    const io_type* __restrict__ V,       // [B, H, T, d]
    io_type* __restrict__ O,             // [B, H, T, d]
    int seq_len,
    int head_dim,
    int causal
) {
    int b = blockIdx.x;   // batch index
    int h = blockIdx.y;   // head index
    int qi = threadIdx.x; // which Q row within this tile (0..TILE_SIZE-1)

    // Shared memory layout:
    extern __shared__ float smem[];
    float* Kj = smem;                                    // [TILE_SIZE][head_dim]
    float* Vj = Kj + TILE_SIZE * head_dim;               // [TILE_SIZE][head_dim]

    // Scale factor: 1 / sqrt(d)
    float scale = rsqrtf((float)head_dim);

    // Base offset for this (batch, head) in [B, H, T, d]
    int BH_stride = seq_len * head_dim;
    int base = (b * gridDim.y + h) * BH_stride;

    // Number of Q/KV tiles
    int num_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;

    // Outer loop over Q tiles
    for (int qi_tile = 0; qi_tile < num_tiles; qi_tile++) {
        int i = qi_tile * TILE_SIZE + qi;
        if (i >= seq_len) continue;

        float m_i = -FLT_MAX;
        float l_i = 0.0f;

        float o_acc[128];
        for (int dd = 0; dd < head_dim; dd++) {
            o_acc[dd] = 0.0f;
        }

        // Inner loop over K/V tiles
        for (int kv_tile = 0; kv_tile < num_tiles; kv_tile++) {
            if (causal) {
                int kv_start = kv_tile * TILE_SIZE;
                if (kv_start > i) break;
            }

            // Cooperatively load K tile into shared memory
            int kv_row = kv_tile * TILE_SIZE + qi;
            if (kv_row < seq_len) {
                for (int dd = 0; dd < head_dim; dd++) {
                    Kj[qi * head_dim + dd] = IO_LOAD(K, base + kv_row * head_dim + dd);
                    Vj[qi * head_dim + dd] = IO_LOAD(V, base + kv_row * head_dim + dd);
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

#ifdef EXLA_FFI
}  // anonymous namespace
#endif

#ifndef EXLA_FFI

extern "C" {

int fused_flash_attention_launch(
    cudaStream_t stream,
    const io_type* q,
    const io_type* k,
    const io_type* v,
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

    fused_flash_attention_kernel<<<grid, block, smem_bytes, stream>>>(
        q, k, v, output,
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

namespace {  // anonymous namespace — prevents symbol collision between f32/bf16

// Causal hardcoded to 1 to avoid scalar buffer operand segfault in XLA.
ffi::Error fused_flash_attention_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> q,
    ffi::Buffer<FFI_IO_TYPE> k,
    ffi::Buffer<FFI_IO_TYPE> v,
    ffi::ResultBuffer<FFI_IO_TYPE> output
) {
    auto dims = q.dimensions();
    int batch    = static_cast<int>(dims[0]);
    int num_heads = static_cast<int>(dims[1]);
    int seq_len  = static_cast<int>(dims[2]);
    int head_dim = static_cast<int>(dims[3]);

    int causal = 1;  // hardcoded causal

    dim3 grid(batch, num_heads);
    dim3 block(TILE_SIZE);
    size_t smem_bytes = 2 * TILE_SIZE * head_dim * sizeof(float);

    fused_flash_attention_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const io_type*>(q.untyped_data()),
        reinterpret_cast<const io_type*>(k.untyped_data()),
        reinterpret_cast<const io_type*>(v.untyped_data()),
        reinterpret_cast<io_type*>(output->untyped_data()),
        seq_len, head_dim, causal
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

}  // anonymous namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    HANDLER_SYMBOL(fused_flash_attention), fused_flash_attention_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // q  [B, H, T, d]
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // k  [B, H, T, d]
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // v  [B, H, T, d]
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // output [B, H, T, d]
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_flash_attention_" PRECISION_SUFFIX, "CUDA", HANDLER_SYMBOL(fused_flash_attention));

#endif  // EXLA_FFI
