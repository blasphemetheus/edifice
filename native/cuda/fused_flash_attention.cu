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
#include "precision.cuh"
#include <float.h>

// ============================================================================
// Configuration
// ============================================================================

// Tile sizes — kept small enough for shared memory at d=128:
//   Kj tile:  32 * 128 * 4 = 16KB
//   Vj tile:  32 * 128 * 4 = 16KB
//   S_ij:     32 * 32 * 4 =  4KB
//   Total: ~36KB (within 48KB limit)
#define TILE_SIZE 32

// ============================================================================
// Kernel
// ============================================================================

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
    //   Kj[TILE_SIZE][head_dim]  — current K block
    //   Vj[TILE_SIZE][head_dim]  — current V block
    extern __shared__ float smem[];
    float* Kj = smem;                                    // [TILE_SIZE][head_dim]
    float* Vj = Kj + TILE_SIZE * head_dim;               // [TILE_SIZE][head_dim]

    // Scale factor: 1 / sqrt(d)
    float scale = rsqrtf((float)head_dim);

    // Base offset for this (batch, head) in [B, H, T, d]
    int BH_stride = seq_len * head_dim;   // stride for one (B,H) block
    int base = (b * gridDim.y + h) * BH_stride;

    // Number of Q/KV tiles
    int num_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;

    // Outer loop over Q tiles
    for (int qi_tile = 0; qi_tile < num_tiles; qi_tile++) {
        int i = qi_tile * TILE_SIZE + qi;  // global Q row index
        if (i >= seq_len) continue;

        // Load this thread's Q row into registers
        // q_i[d] — one row of Q for position i
        float m_i = -FLT_MAX;   // running max for online softmax
        float l_i = 0.0f;       // running sum of exp(s - m)

        // Accumulator for output: O_i[d]
        // We'll store in registers for small d, or loop for larger d
        // For simplicity, use a fixed-size register array
        // Max supported head_dim is 128
        float o_acc[128];
        for (int dd = 0; dd < head_dim; dd++) {
            o_acc[dd] = 0.0f;
        }

        // Inner loop over K/V tiles
        for (int kv_tile = 0; kv_tile < num_tiles; kv_tile++) {
            // Causal: skip K/V blocks that are entirely in the future
            if (causal) {
                int kv_start = kv_tile * TILE_SIZE;
                if (kv_start > i) break;
            }

            // Cooperatively load K tile into shared memory
            // Each thread loads one row of Kj and Vj
            int kv_row = kv_tile * TILE_SIZE + qi;  // reuse qi as loading index
            if (kv_row < seq_len) {
                for (int dd = 0; dd < head_dim; dd++) {
                    Kj[qi * head_dim + dd] = IO_LOAD(K, base + kv_row * head_dim + dd);
                    Vj[qi * head_dim + dd] = IO_LOAD(V, base + kv_row * head_dim + dd);
                }
            } else {
                // Pad with zeros for out-of-bounds
                for (int dd = 0; dd < head_dim; dd++) {
                    Kj[qi * head_dim + dd] = 0.0f;
                    Vj[qi * head_dim + dd] = 0.0f;
                }
            }
            __syncthreads();

            // Compute attention scores for this Q row against all K rows in tile
            int tile_len = min(TILE_SIZE, seq_len - kv_tile * TILE_SIZE);

            for (int j = 0; j < tile_len; j++) {
                int kv_pos = kv_tile * TILE_SIZE + j;

                // Causal mask: skip future positions
                if (causal && kv_pos > i) break;

                // Dot product: s = Q[i] . K[j] * scale
                float s = 0.0f;
                for (int dd = 0; dd < head_dim; dd++) {
                    s += IO_LOAD(Q, base + i * head_dim + dd) * Kj[j * head_dim + dd];
                }
                s *= scale;

                // Online softmax update (Flash Attention V2 algorithm)
                float m_new = fmaxf(m_i, s);
                float exp_diff = expf(m_i - m_new);
                float exp_s = expf(s - m_new);

                // Rescale existing accumulator
                l_i = l_i * exp_diff + exp_s;

                // Rescale output accumulator and add new contribution
                for (int dd = 0; dd < head_dim; dd++) {
                    o_acc[dd] = o_acc[dd] * exp_diff + exp_s * Vj[j * head_dim + dd];
                }

                m_i = m_new;
            }

            __syncthreads();
        }

        // Final normalization: O_i = O_i / l_i
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
    // One block per (batch, head), TILE_SIZE threads per block
    dim3 grid(batch, num_heads);
    dim3 block(TILE_SIZE);

    // Shared memory: Kj[TILE_SIZE][head_dim] + Vj[TILE_SIZE][head_dim]
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

ffi::Error fused_flash_attention_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> q,
    ffi::Buffer<FFI_IO_TYPE> k,
    ffi::Buffer<FFI_IO_TYPE> v,
    ffi::AnyBuffer causal_flag,
    ffi::ResultBuffer<FFI_IO_TYPE> output
) {
    auto dims = q.dimensions();
    int batch    = static_cast<int>(dims[0]);
    int num_heads = static_cast<int>(dims[1]);
    int seq_len  = static_cast<int>(dims[2]);
    int head_dim = static_cast<int>(dims[3]);

    // Extract causal flag from scalar i32 buffer
    int causal = static_cast<int>(
        reinterpret_cast<const int32_t*>(causal_flag.untyped_data())[0]
    );

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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_flash_attention, fused_flash_attention_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // q  [B, H, T, d]
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // k  [B, H, T, d]
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // v  [B, H, T, d]
        .Arg<ffi::AnyBuffer>()             // causal (scalar i32)
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // output [B, H, T, d]
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_flash_attention_" PRECISION_SUFFIX, "CUDA", fused_flash_attention);

#endif  // EXLA_FFI
