// Fused NativeRecurrence Parallel Scan Kernels
//
// Three kernel variants for the NativeRecurrence module:
//   1. ELU-GRU:       z=sigmoid(gate), c=1+elu(cand), h=(1-z)*h + z*c
//   2. Real-GRU:      z=sigmoid(gate), h=(1-z)*h + z*cand
//   3. Diag-Linear:   h=sigmoid(a)*h + b
//
// All three apply their nonlinearities (sigmoid, elu) inside the kernel
// so the Elixir side passes raw pre-activation projections.
//
// Same thread layout as MinGRU: one thread per (batch, hidden) element,
// sequential scan through timesteps with state in registers.
//
// Two compilation modes:
//   1. Standalone (-DSTANDALONE): kernel + C-linkage launch wrapper.
//   2. EXLA FFI (-DEXLA_FFI): kernel + XLA FFI handler + registration.
//
// Inputs (all variants):
//   arg1:   [batch, seq_len, hidden] — first projection (pre-activation)
//   arg2:   [batch, seq_len, hidden] — second projection
//   h0:     [batch, hidden]          — initial hidden state
//
// Output:
//   output: [batch, seq_len, hidden] — all hidden states

#include <cuda_runtime.h>

// ============================================================================
// Device helpers
// ============================================================================

__device__ __forceinline__ float d_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float d_elu(float x) {
    return (x >= 0.0f) ? x : (expf(x) - 1.0f);
}

// ============================================================================
// Kernel 1: ELU-GRU scan
// z = sigmoid(gate), c = 1 + elu(cand), h = (1-z)*h + z*c
// ============================================================================

__global__ void fused_elu_gru_scan_kernel(
    const float* __restrict__ gates,       // [B, T, H] raw pre-sigmoid
    const float* __restrict__ candidates,  // [B, T, H] raw pre-elu
    const float* __restrict__ h0,          // [B, H] initial state
    float* __restrict__ output,            // [B, T, H] all hidden states
    int batch, int seq_len, int hidden
) {
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || h >= hidden) return;

    float h_state = h0[b * hidden + h];

    for (int t = 0; t < seq_len; t++) {
        int idx = b * seq_len * hidden + t * hidden + h;
        float z = d_sigmoid(gates[idx]);
        float c = 1.0f + d_elu(candidates[idx]);

        h_state = (1.0f - z) * h_state + z * c;
        output[idx] = h_state;
    }
}

// ============================================================================
// Kernel 2: Real-GRU scan (MinGRU with in-kernel sigmoid)
// z = sigmoid(gate), h = (1-z)*h + z*cand
// ============================================================================

__global__ void fused_real_gru_scan_kernel(
    const float* __restrict__ gates,       // [B, T, H] raw pre-sigmoid
    const float* __restrict__ candidates,  // [B, T, H] candidate values
    const float* __restrict__ h0,          // [B, H] initial state
    float* __restrict__ output,            // [B, T, H] all hidden states
    int batch, int seq_len, int hidden
) {
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || h >= hidden) return;

    float h_state = h0[b * hidden + h];

    for (int t = 0; t < seq_len; t++) {
        int idx = b * seq_len * hidden + t * hidden + h;
        float z = d_sigmoid(gates[idx]);
        float c = candidates[idx];

        h_state = (1.0f - z) * h_state + z * c;
        output[idx] = h_state;
    }
}

// ============================================================================
// Kernel 3: Diagonal linear recurrence scan
// h = sigmoid(a)*h + b
// ============================================================================

__global__ void fused_diag_linear_scan_kernel(
    const float* __restrict__ a_vals,  // [B, T, H] raw pre-sigmoid
    const float* __restrict__ b_vals,  // [B, T, H] additive term
    const float* __restrict__ h0,      // [B, H] initial state
    float* __restrict__ output,        // [B, T, H] all hidden states
    int batch, int seq_len, int hidden
) {
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || h >= hidden) return;

    float h_state = h0[b * hidden + h];

    for (int t = 0; t < seq_len; t++) {
        int idx = b * seq_len * hidden + t * hidden + h;
        float a = d_sigmoid(a_vals[idx]);
        float bv = b_vals[idx];

        h_state = a * h_state + bv;
        output[idx] = h_state;
    }
}

// ============================================================================
// Standalone launch wrappers (C-linkage for NIF / dlopen)
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

int fused_elu_gru_scan_launch(
    cudaStream_t stream,
    const float* gates, const float* candidates, const float* h0,
    float* output, int batch, int seq_len, int hidden
) {
    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_elu_gru_scan_kernel<<<grid, block, 0, stream>>>(
        gates, candidates, h0, output,
        batch, seq_len, hidden
    );

    return (int)cudaGetLastError();
}

int fused_real_gru_scan_launch(
    cudaStream_t stream,
    const float* gates, const float* candidates, const float* h0,
    float* output, int batch, int seq_len, int hidden
) {
    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_real_gru_scan_kernel<<<grid, block, 0, stream>>>(
        gates, candidates, h0, output,
        batch, seq_len, hidden
    );

    return (int)cudaGetLastError();
}

int fused_diag_linear_scan_launch(
    cudaStream_t stream,
    const float* a_vals, const float* b_vals, const float* h0,
    float* output, int batch, int seq_len, int hidden
) {
    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_diag_linear_scan_kernel<<<grid, block, 0, stream>>>(
        a_vals, b_vals, h0, output,
        batch, seq_len, hidden
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
#include "xla/ffi/ffi_api.h"

namespace ffi = xla::ffi;

// --- ELU-GRU FFI ---

ffi::Error fused_elu_gru_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> gates,
    ffi::Buffer<ffi::F32> candidates,
    ffi::Buffer<ffi::F32> h0,
    ffi::ResultBuffer<ffi::F32> output
) {
    auto dims = gates.dimensions();
    int batch   = static_cast<int>(dims[0]);
    int seq_len = static_cast<int>(dims[1]);
    int hidden  = static_cast<int>(dims[2]);

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_elu_gru_scan_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(gates.untyped_data()),
        reinterpret_cast<const float*>(candidates.untyped_data()),
        reinterpret_cast<const float*>(h0.untyped_data()),
        reinterpret_cast<float*>(output->untyped_data()),
        batch, seq_len, hidden
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_elu_gru_scan, fused_elu_gru_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // gates
        .Arg<ffi::Buffer<ffi::F32>>()   // candidates
        .Arg<ffi::Buffer<ffi::F32>>()   // h0
        .Ret<ffi::Buffer<ffi::F32>>()   // output
);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
    "exla_fused_elu_gru_scan_f32", "CUDA", fused_elu_gru_scan);

// --- Real-GRU FFI ---

ffi::Error fused_real_gru_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> gates,
    ffi::Buffer<ffi::F32> candidates,
    ffi::Buffer<ffi::F32> h0,
    ffi::ResultBuffer<ffi::F32> output
) {
    auto dims = gates.dimensions();
    int batch   = static_cast<int>(dims[0]);
    int seq_len = static_cast<int>(dims[1]);
    int hidden  = static_cast<int>(dims[2]);

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_real_gru_scan_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(gates.untyped_data()),
        reinterpret_cast<const float*>(candidates.untyped_data()),
        reinterpret_cast<const float*>(h0.untyped_data()),
        reinterpret_cast<float*>(output->untyped_data()),
        batch, seq_len, hidden
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_real_gru_scan, fused_real_gru_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // gates
        .Arg<ffi::Buffer<ffi::F32>>()   // candidates
        .Arg<ffi::Buffer<ffi::F32>>()   // h0
        .Ret<ffi::Buffer<ffi::F32>>()   // output
);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
    "exla_fused_real_gru_scan_f32", "CUDA", fused_real_gru_scan);

// --- Diag-Linear FFI ---

ffi::Error fused_diag_linear_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> a_vals,
    ffi::Buffer<ffi::F32> b_vals,
    ffi::Buffer<ffi::F32> h0,
    ffi::ResultBuffer<ffi::F32> output
) {
    auto dims = a_vals.dimensions();
    int batch   = static_cast<int>(dims[0]);
    int seq_len = static_cast<int>(dims[1]);
    int hidden  = static_cast<int>(dims[2]);

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_diag_linear_scan_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(a_vals.untyped_data()),
        reinterpret_cast<const float*>(b_vals.untyped_data()),
        reinterpret_cast<const float*>(h0.untyped_data()),
        reinterpret_cast<float*>(output->untyped_data()),
        batch, seq_len, hidden
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_diag_linear_scan, fused_diag_linear_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // a_vals
        .Arg<ffi::Buffer<ffi::F32>>()   // b_vals
        .Arg<ffi::Buffer<ffi::F32>>()   // h0
        .Ret<ffi::Buffer<ffi::F32>>()   // output
);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
    "exla_fused_diag_linear_scan_f32", "CUDA", fused_diag_linear_scan);

#endif  // EXLA_FFI
