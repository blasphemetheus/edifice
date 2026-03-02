// Fused MinLSTM Multi-Layer Block Scan Kernel
//
// Processes ALL layers of a MinLSTM stack in a single kernel launch, keeping
// intermediate activations in shared memory and registers instead of writing
// them to global memory between LayerNorm, Dense, Scan, and Residual ops.
//
// For each timestep, each thread handles one (batch, hidden) element and
// processes all layers sequentially: LayerNorm → 3 GEMVs → Gate Normalize →
// Scan update → Residual. The hidden state per layer is kept in registers.
//
// Constraint: hidden_size ≤ 256 (single CUDA block handles all hidden dims).
//
// Weight packing per layer (contiguous in global memory):
//   W_f[H,H]  b_f[H]  W_i[H,H]  b_i[H]  W_h[H,H]  b_h[H]  gamma[H]  beta[H]
//   Total per layer: 3*H*H + 5*H floats
//
// Two compilation modes:
//   1. Standalone (default): kernel + C-linkage launch wrapper for NIF/dlopen.
//   2. EXLA FFI (-DEXLA_FFI): kernel + XLA FFI handler + registration.
//
// Inputs:
//   input:   [B, T, H]          — input sequence
//   weights: [num_layers * (3*H*H + 5*H)]  — packed per-layer weights
//   h0:      [B, num_layers, H] — per-layer initial hidden states
//
// Output:
//   output:  [B, T, H]          — final layer output for all timesteps

#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// Kernel
// ============================================================================

__global__ void fused_minlstm_block_scan_kernel(
    const float* __restrict__ input,    // [B, T, H]
    const float* __restrict__ weights,  // [num_layers * (3*H*H + 5*H)]
    const float* __restrict__ h0,       // [B, num_layers, H]
    float* __restrict__ output,         // [B, T, H]
    int B, int T, int H, int num_layers
) {
    int b = blockIdx.x;
    int i = threadIdx.x;

    if (b >= B || i >= H) return;

    extern __shared__ float shared[];
    float* input_shared  = shared;          // [H]
    float* normed_shared = shared + H;      // [H]
    float* reduce_buf    = shared + 2 * H;  // [H]

    // Per-layer weight stride
    int layer_stride = 3 * H * H + 5 * H;

    // Load per-layer initial hidden states into registers
    float h_state[16];
    for (int l = 0; l < num_layers && l < 16; l++) {
        h_state[l] = h0[b * num_layers * H + l * H + i];
    }

    // Process all timesteps
    for (int t = 0; t < T; t++) {
        float x_val = input[b * T * H + t * H + i];

        for (int layer = 0; layer < num_layers; layer++) {
            const float* lw = weights + layer * layer_stride;
            const float* W_f   = lw;                            // [H, H]
            const float* b_f   = lw + H * H;                    // [H]
            const float* W_i   = lw + H * H + H;                // [H, H]
            const float* b_i   = lw + 2 * H * H + H;            // [H]
            const float* W_h   = lw + 2 * H * H + 2 * H;        // [H, H]
            const float* b_h   = lw + 3 * H * H + 2 * H;        // [H]
            const float* gamma = lw + 3 * H * H + 3 * H;        // [H]
            const float* beta  = lw + 3 * H * H + 4 * H;        // [H]

            // ---- LayerNorm ----
            input_shared[i] = x_val;
            __syncthreads();

            // Mean reduction
            reduce_buf[i] = input_shared[i];
            __syncthreads();
            for (int stride = H / 2; stride > 0; stride >>= 1) {
                if (i < stride) {
                    reduce_buf[i] += reduce_buf[i + stride];
                }
                __syncthreads();
            }
            float mean = reduce_buf[0] / (float)H;

            // Variance reduction
            float diff = input_shared[i] - mean;
            reduce_buf[i] = diff * diff;
            __syncthreads();
            for (int stride = H / 2; stride > 0; stride >>= 1) {
                if (i < stride) {
                    reduce_buf[i] += reduce_buf[i + stride];
                }
                __syncthreads();
            }
            float var = reduce_buf[0] / (float)H;
            float inv_std = rsqrtf(var + 1e-5f);

            // Normalize + affine
            float normed = (input_shared[i] - mean) * inv_std;
            normed = normed * gamma[i] + beta[i];
            normed_shared[i] = normed;
            __syncthreads();

            // ---- GEMV: forget, input, and candidate projections ----
            float f_val = b_f[i];
            float i_val = b_i[i];
            float c_val = b_h[i];
            for (int j = 0; j < H; j++) {
                float n_j = normed_shared[j];
                f_val += n_j * W_f[j * H + i];
                i_val += n_j * W_i[j * H + i];
                c_val += n_j * W_h[j * H + i];
            }

            // ---- Gate normalization ----
            // f' = sigmoid(f) / (sigmoid(f) + sigmoid(i) + eps)
            // i' = sigmoid(i) / (sigmoid(f) + sigmoid(i) + eps)
            float sig_f = 1.0f / (1.0f + expf(-f_val));
            float sig_i = 1.0f / (1.0f + expf(-i_val));
            float gate_sum = sig_f + sig_i + 1e-6f;
            float f_norm = sig_f / gate_sum;
            float i_norm = sig_i / gate_sum;

            // ---- Scan update ----
            h_state[layer] = f_norm * h_state[layer] + i_norm * c_val;

            // ---- Residual ----
            x_val = x_val + h_state[layer];
            __syncthreads();
        }

        output[b * T * H + t * H + i] = x_val;
    }
}

// ============================================================================
// Standalone launch wrapper (C-linkage for NIF / dlopen)
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

int fused_minlstm_block_scan_launch(
    cudaStream_t stream,
    const float* input,
    const float* weights,
    const float* h0,
    float* output,
    int B, int T, int H, int num_layers
) {
    dim3 grid(B);
    dim3 block(H);
    size_t shared_bytes = 3 * H * sizeof(float);

    fused_minlstm_block_scan_kernel<<<grid, block, shared_bytes, stream>>>(
        input, weights, h0, output,
        B, T, H, num_layers
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

ffi::Error fused_minlstm_block_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> input,
    ffi::Buffer<ffi::F32> weights,
    ffi::Buffer<ffi::F32> h0,
    ffi::ResultBuffer<ffi::F32> output,
    int32_t num_layers
) {
    auto dims = input.dimensions();
    int B = static_cast<int>(dims[0]);
    int T = static_cast<int>(dims[1]);
    int H = static_cast<int>(dims[2]);

    dim3 grid(B);
    dim3 block(H);
    size_t shared_bytes = 3 * H * sizeof(float);

    fused_minlstm_block_scan_kernel<<<grid, block, shared_bytes, stream>>>(
        reinterpret_cast<const float*>(input.untyped_data()),
        reinterpret_cast<const float*>(weights.untyped_data()),
        reinterpret_cast<const float*>(h0.untyped_data()),
        reinterpret_cast<float*>(output->untyped_data()),
        B, T, H, num_layers
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_minlstm_block_scan, fused_minlstm_block_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // input
        .Arg<ffi::Buffer<ffi::F32>>()   // weights
        .Arg<ffi::Buffer<ffi::F32>>()   // h0
        .Ret<ffi::Buffer<ffi::F32>>()   // output
        .Attr<int32_t>("num_layers")
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_minlstm_block_scan_f32", "CUDA", fused_minlstm_block_scan);

#endif  // EXLA_FFI
