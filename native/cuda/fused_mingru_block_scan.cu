// Fused MinGRU Multi-Layer Block Scan Kernel
//
// Processes ALL layers of a MinGRU stack in a single kernel launch, keeping
// intermediate activations in shared memory and registers instead of writing
// them to global memory between LayerNorm, Dense, Scan, and Residual ops.
//
// For each timestep, each thread handles one (batch, hidden) element and
// processes all layers sequentially: LayerNorm → 2 GEMVs → Sigmoid →
// Scan update → Residual. The hidden state per layer is kept in registers.
//
// Constraint: hidden_size ≤ 256 (single CUDA block handles all hidden dims).
//
// Weight packing per layer (contiguous in global memory):
//   W_z[H,H]  b_z[H]  W_h[H,H]  b_h[H]  gamma[H]  beta[H]
//   Total per layer: 2*H*H + 4*H floats
//
// Two compilation modes:
//   1. Standalone (default): kernel + C-linkage launch wrapper for NIF/dlopen.
//   2. EXLA FFI (-DEXLA_FFI): kernel + XLA FFI handler + registration.
//
// Inputs:
//   input:   [B, T, H]          — input sequence
//   weights: [num_layers * (2*H*H + 4*H)]  — packed per-layer weights
//   h0:      [B, num_layers, H] — per-layer initial hidden states
//
// Output:
//   output:  [B, T, H]          — final layer output for all timesteps

#include <cuda_runtime.h>
#include "precision.cuh"
#include <math.h>

// ============================================================================
// Kernel
// ============================================================================

// Shared memory layout: [2 * H] floats
//   [0..H-1]   = input_shared  (current input vector / reduction workspace)
//   [H..2H-1]  = normed_shared (normalized input for GEMV reads)

__global__ void fused_mingru_block_scan_kernel(
    const io_type* __restrict__ input,    // [B, T, H]
    const io_type* __restrict__ weights,  // [num_layers * (2*H*H + 4*H)]
    const io_type* __restrict__ h0,       // [B, num_layers, H]
    io_type* __restrict__ output,         // [B, T, H]
    int B, int T, int H, int num_layers
) {
    // Thread layout: blockIdx.x = batch, threadIdx.x = hidden dim
    int b = blockIdx.x;
    int i = threadIdx.x;

    if (b >= B || i >= H) return;

    extern __shared__ float shared[];
    float* input_shared  = shared;          // [H]
    float* normed_shared = shared + H;      // [H]
    float* reduce_buf    = shared + 2 * H;  // [H] for reduction workspace

    // Per-layer weight stride
    int layer_stride = 2 * H * H + 4 * H;

    // Load per-layer initial hidden states into registers
    // Max 16 layers supported (arbitrary but generous for MinGRU stacks)
    float h_state[16];
    for (int l = 0; l < num_layers && l < 16; l++) {
        h_state[l] = IO_LOAD(h0, b * num_layers * H + l * H + i);
    }

    // Process all timesteps
    for (int t = 0; t < T; t++) {
        // Load input for this timestep into register
        float x_val = IO_LOAD(input, b * T * H + t * H + i);

        // Process all layers for this timestep
        for (int layer = 0; layer < num_layers; layer++) {
            // Pointer to this layer's packed weights
            int lw_base = layer * layer_stride;

            // ---- LayerNorm ----
            // Step 1: Load x_val into shared memory for reduction
            input_shared[i] = x_val;
            __syncthreads();

            // Step 2: Compute mean via parallel reduction
            reduce_buf[i] = input_shared[i];
            __syncthreads();
            for (int stride = H / 2; stride > 0; stride >>= 1) {
                if (i < stride) {
                    reduce_buf[i] += reduce_buf[i + stride];
                }
                __syncthreads();
            }
            float mean = reduce_buf[0] / (float)H;

            // Step 3: Compute variance via parallel reduction
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

            // Step 4: Normalize and apply gamma/beta
            float normed = (input_shared[i] - mean) * inv_std;
            float gamma_val = IO_LOAD(weights, lw_base + 2 * H * H + 2 * H + i);
            float beta_val  = IO_LOAD(weights, lw_base + 2 * H * H + 3 * H + i);
            normed = normed * gamma_val + beta_val;
            normed_shared[i] = normed;
            __syncthreads();

            // ---- GEMV: gate and candidate projections ----
            // Each thread computes one output element: out[i] = sum_j(normed_shared[j] * W[j*H + i]) + bias
            float z_val = IO_LOAD(weights, lw_base + H * H + i);           // b_z[i]
            float c_val = IO_LOAD(weights, lw_base + 2 * H * H + H + i);   // b_h[i]
            for (int j = 0; j < H; j++) {
                float n_j = normed_shared[j];
                z_val += n_j * IO_LOAD(weights, lw_base + j * H + i);              // W_z[j*H + i]
                c_val += n_j * IO_LOAD(weights, lw_base + H * H + H + j * H + i);  // W_h[j*H + i]
            }

            // ---- Sigmoid + Scan update ----
            float z = 1.0f / (1.0f + expf(-z_val));
            h_state[layer] = (1.0f - z) * h_state[layer] + z * c_val;

            // ---- Residual ----
            x_val = x_val + h_state[layer];
            __syncthreads();
        }

        // Write final layer output for this timestep
        IO_STORE(output, b * T * H + t * H + i, x_val);
    }
}

// ============================================================================
// Standalone launch wrapper (C-linkage for NIF / dlopen)
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

int fused_mingru_block_scan_launch(
    cudaStream_t stream,
    const io_type* input,
    const io_type* weights,
    const io_type* h0,
    io_type* output,
    int B, int T, int H, int num_layers
) {
    // One block per batch element, H threads per block
    dim3 grid(B);
    dim3 block(H);
    // Shared memory: input_shared[H] + normed_shared[H] + reduce_buf[H]
    size_t shared_bytes = 3 * H * sizeof(float);

    fused_mingru_block_scan_kernel<<<grid, block, shared_bytes, stream>>>(
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

ffi::Error fused_mingru_block_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> input,
    ffi::Buffer<FFI_IO_TYPE> weights,
    ffi::Buffer<FFI_IO_TYPE> h0,
    ffi::ResultBuffer<FFI_IO_TYPE> output,
    int32_t num_layers
) {
    auto dims = input.dimensions();
    int B = static_cast<int>(dims[0]);
    int T = static_cast<int>(dims[1]);
    int H = static_cast<int>(dims[2]);

    dim3 grid(B);
    dim3 block(H);
    size_t shared_bytes = 3 * H * sizeof(float);

    fused_mingru_block_scan_kernel<<<grid, block, shared_bytes, stream>>>(
        reinterpret_cast<const io_type*>(input.untyped_data()),
        reinterpret_cast<const io_type*>(weights.untyped_data()),
        reinterpret_cast<const io_type*>(h0.untyped_data()),
        reinterpret_cast<io_type*>(output->untyped_data()),
        B, T, H, num_layers
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_mingru_block_scan, fused_mingru_block_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // input
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // weights
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // h0
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // output
        .Attr<int32_t>("num_layers")
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_mingru_block_scan_" PRECISION_SUFFIX, "CUDA", fused_mingru_block_scan);

#endif  // EXLA_FFI
