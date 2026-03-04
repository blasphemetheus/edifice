// Fused GRU Multi-Layer Block Scan Kernel
//
// Processes ALL layers of a GRU stack in a single kernel launch, keeping
// intermediate activations in shared memory and registers instead of writing
// them to global memory between layers.
//
// For each timestep, each thread handles one (batch, hidden) element and
// processes all layers sequentially: LayerNorm → W_x@normed (input proj) →
// R@h (recurrent proj via shared mem) → 3-gate GRU update → Residual.
// The hidden state per layer is kept in registers.
//
// Constraint: hidden_size ≤ 256 (single CUDA block handles all hidden dims).
//
// Weight packing per layer (contiguous in global memory):
//   W_x[H, 3*H]  b_x[3*H]  R[H, 3*H]  gamma[H]  beta[H]
//   Total per layer: 6*H*H + 5*H floats
//
// Two compilation modes:
//   1. Standalone (default): kernel + C-linkage launch wrapper for NIF/dlopen.
//   2. EXLA FFI (-DEXLA_FFI): kernel + XLA FFI handler + registration.
//
// Inputs:
//   input:   [B, T, H]          — input sequence
//   weights: [num_layers * (6*H*H + 5*H)] — packed per-layer weights
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

// Shared memory layout: [4 * H] floats
//   [0..H-1]     = input_shared  (current input vector / reduction workspace)
//   [H..2H-1]    = normed_shared (normalized input for W_x GEMV)
//   [2H..3H-1]   = h_shared      (hidden state for R GEMV)
//   [3H..4H-1]   = reduce_buf    (reduction workspace)

#ifdef EXLA_FFI
namespace {  // anonymous namespace — internal linkage per compilation unit
#endif

__global__ void fused_gru_block_scan_kernel(
    const io_type* __restrict__ input,    // [B, T, H]
    const io_type* __restrict__ weights,  // [num_layers * (6*H*H + 5*H)]
    const io_type* __restrict__ h0,       // [B, num_layers, H]
    io_type* __restrict__ output,         // [B, T, H]
    int B, int T, int H, int num_layers
) {
    int b = blockIdx.x;
    int i = threadIdx.x;

    if (b >= B || i >= H) return;

    extern __shared__ float shared[];
    float* input_shared  = shared;              // [H]
    float* normed_shared = shared + H;          // [H]
    float* h_shared      = shared + 2 * H;      // [H]
    float* reduce_buf    = shared + 3 * H;      // [H]

    // Per-layer weight stride: W_x[H,3H] + b_x[3H] + R[H,3H] + gamma[H] + beta[H]
    int layer_stride = 6 * H * H + 5 * H;

    // Load per-layer initial hidden states into registers
    float h_state[16];
    for (int l = 0; l < num_layers && l < 16; l++) {
        h_state[l] = IO_LOAD(h0, b * num_layers * H + l * H + i);
    }

    // Process all timesteps
    for (int t = 0; t < T; t++) {
        float x_val = IO_LOAD(input, b * T * H + t * H + i);

        // Process all layers for this timestep
        for (int layer = 0; layer < num_layers; layer++) {
            int lw_base = layer * layer_stride;

            // ---- LayerNorm ----
            input_shared[i] = x_val;
            __syncthreads();

            // Compute mean
            reduce_buf[i] = input_shared[i];
            __syncthreads();
            for (int stride = H / 2; stride > 0; stride >>= 1) {
                if (i < stride) {
                    reduce_buf[i] += reduce_buf[i + stride];
                }
                __syncthreads();
            }
            float mean = reduce_buf[0] / (float)H;

            // Compute variance
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

            // Normalize and apply gamma/beta
            // gamma at offset: 6*H*H + 3*H
            // beta  at offset: 6*H*H + 4*H
            float normed = (input_shared[i] - mean) * inv_std;
            float gamma_val = IO_LOAD(weights, lw_base + 6 * H * H + 3 * H + i);
            float beta_val  = IO_LOAD(weights, lw_base + 6 * H * H + 4 * H + i);
            normed = normed * gamma_val + beta_val;
            normed_shared[i] = normed;
            __syncthreads();

            // ---- GEMV: W_x @ normed + b_x → 3 gate values ----
            // W_x is [H, 3*H], stored row-major: W_x[j][g*H+i]
            // b_x[3*H] at offset 3*H*H
            float wx_r = IO_LOAD(weights, lw_base + 3 * H * H + 0 * H + i);  // b_x[r_gate, i]
            float wx_z = IO_LOAD(weights, lw_base + 3 * H * H + 1 * H + i);  // b_x[z_gate, i]
            float wx_n = IO_LOAD(weights, lw_base + 3 * H * H + 2 * H + i);  // b_x[n_gate, i]
            for (int j = 0; j < H; j++) {
                float n_j = normed_shared[j];
                wx_r += n_j * IO_LOAD(weights, lw_base + j * 3 * H + 0 * H + i);
                wx_z += n_j * IO_LOAD(weights, lw_base + j * 3 * H + 1 * H + i);
                wx_n += n_j * IO_LOAD(weights, lw_base + j * 3 * H + 2 * H + i);
            }

            // ---- Stage h for cooperative R@h GEMV ----
            h_shared[i] = h_state[layer];
            __syncthreads();

            // ---- GEMV: R @ h → 3 gate recurrent contributions ----
            // R is [H, 3*H] at offset 3*H*H + 3*H
            int r_base = lw_base + 3 * H * H + 3 * H;
            float rh_r = 0.0f, rh_z = 0.0f, rh_n = 0.0f;
            for (int j = 0; j < H; j++) {
                float h_j = h_shared[j];
                rh_r += h_j * IO_LOAD(weights, r_base + j * 3 * H + 0 * H + i);
                rh_z += h_j * IO_LOAD(weights, r_base + j * 3 * H + 1 * H + i);
                rh_n += h_j * IO_LOAD(weights, r_base + j * 3 * H + 2 * H + i);
            }

            // ---- GRU gate computation ----
            float r_gate = 1.0f / (1.0f + expf(-(wx_r + rh_r)));   // reset gate
            float z_gate = 1.0f / (1.0f + expf(-(wx_z + rh_z)));   // update gate

            // n_gate: reset applied to recurrent contribution only
            float n_gate = tanhf(wx_n + r_gate * rh_n);              // new gate

            // ---- Hidden state update ----
            h_state[layer] = (1.0f - z_gate) * h_state[layer] + z_gate * n_gate;

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

#ifdef EXLA_FFI
}  // anonymous namespace
#endif

#ifndef EXLA_FFI

extern "C" {

int fused_gru_block_scan_launch(
    cudaStream_t stream,
    const io_type* input,
    const io_type* weights,
    const io_type* h0,
    io_type* output,
    int B, int T, int H, int num_layers
) {
    dim3 grid(B);
    dim3 block(H);
    // Shared memory: input_shared[H] + normed_shared[H] + h_shared[H] + reduce_buf[H]
    size_t shared_bytes = 4 * H * sizeof(float);

    fused_gru_block_scan_kernel<<<grid, block, shared_bytes, stream>>>(
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

namespace {  // anonymous namespace — prevents symbol collision between f32/bf16

ffi::Error fused_gru_block_scan_ffi_impl(
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
    size_t shared_bytes = 4 * H * sizeof(float);

    fused_gru_block_scan_kernel<<<grid, block, shared_bytes, stream>>>(
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

}  // anonymous namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    HANDLER_SYMBOL(fused_gru_block_scan), fused_gru_block_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // input
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // weights
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // h0
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // output
        .Attr<int32_t>("num_layers")
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_gru_block_scan_" PRECISION_SUFFIX, "CUDA", HANDLER_SYMBOL(fused_gru_block_scan));

#endif  // EXLA_FFI
