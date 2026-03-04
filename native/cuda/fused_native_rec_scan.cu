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
#include "precision.cuh"

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

#ifdef EXLA_FFI
namespace {  // anonymous namespace — internal linkage per compilation unit
#endif

__global__ void fused_elu_gru_scan_kernel(
    const io_type* __restrict__ gates,       // [B, T, H] raw pre-sigmoid
    const io_type* __restrict__ candidates,  // [B, T, H] raw pre-elu
    const io_type* __restrict__ h0,          // [B, H] initial state
    io_type* __restrict__ output,            // [B, T, H] all hidden states
    int batch, int seq_len, int hidden
) {
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || h >= hidden) return;

    float h_state = IO_LOAD(h0, b * hidden + h);

    for (int t = 0; t < seq_len; t++) {
        int idx = b * seq_len * hidden + t * hidden + h;
        float z = d_sigmoid(IO_LOAD(gates, idx));
        float c = 1.0f + d_elu(IO_LOAD(candidates, idx));

        h_state = (1.0f - z) * h_state + z * c;
        IO_STORE(output, idx, h_state);
    }
}

// ============================================================================
// Kernel 2: Real-GRU scan (MinGRU with in-kernel sigmoid)
// z = sigmoid(gate), h = (1-z)*h + z*cand
// ============================================================================

__global__ void fused_real_gru_scan_kernel(
    const io_type* __restrict__ gates,       // [B, T, H] raw pre-sigmoid
    const io_type* __restrict__ candidates,  // [B, T, H] candidate values
    const io_type* __restrict__ h0,          // [B, H] initial state
    io_type* __restrict__ output,            // [B, T, H] all hidden states
    int batch, int seq_len, int hidden
) {
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || h >= hidden) return;

    float h_state = IO_LOAD(h0, b * hidden + h);

    for (int t = 0; t < seq_len; t++) {
        int idx = b * seq_len * hidden + t * hidden + h;
        float z = d_sigmoid(IO_LOAD(gates, idx));
        float c = IO_LOAD(candidates, idx);

        h_state = (1.0f - z) * h_state + z * c;
        IO_STORE(output, idx, h_state);
    }
}

// ============================================================================
// Kernel 3: Diagonal linear recurrence scan
// h = sigmoid(a)*h + b
// ============================================================================

__global__ void fused_diag_linear_scan_kernel(
    const io_type* __restrict__ a_vals,  // [B, T, H] raw pre-sigmoid
    const io_type* __restrict__ b_vals,  // [B, T, H] additive term
    const io_type* __restrict__ h0,      // [B, H] initial state
    io_type* __restrict__ output,        // [B, T, H] all hidden states
    int batch, int seq_len, int hidden
) {
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || h >= hidden) return;

    float h_state = IO_LOAD(h0, b * hidden + h);

    for (int t = 0; t < seq_len; t++) {
        int idx = b * seq_len * hidden + t * hidden + h;
        float a = d_sigmoid(IO_LOAD(a_vals, idx));
        float bv = IO_LOAD(b_vals, idx);

        h_state = a * h_state + bv;
        IO_STORE(output, idx, h_state);
    }
}

// ============================================================================
// Standalone launch wrappers (C-linkage for NIF / dlopen)
// ============================================================================

#ifdef EXLA_FFI
}  // anonymous namespace
#endif

#ifndef EXLA_FFI

extern "C" {

int fused_elu_gru_scan_launch(
    cudaStream_t stream,
    const io_type* gates, const io_type* candidates, const io_type* h0,
    io_type* output, int batch, int seq_len, int hidden
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
    const io_type* gates, const io_type* candidates, const io_type* h0,
    io_type* output, int batch, int seq_len, int hidden
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
    const io_type* a_vals, const io_type* b_vals, const io_type* h0,
    io_type* output, int batch, int seq_len, int hidden
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

namespace {  // anonymous namespace — prevents symbol collision between f32/bf16

// --- ELU-GRU FFI ---

ffi::Error fused_elu_gru_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> gates,
    ffi::Buffer<FFI_IO_TYPE> candidates,
    ffi::Buffer<FFI_IO_TYPE> h0,
    ffi::ResultBuffer<FFI_IO_TYPE> output
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
        reinterpret_cast<const io_type*>(gates.untyped_data()),
        reinterpret_cast<const io_type*>(candidates.untyped_data()),
        reinterpret_cast<const io_type*>(h0.untyped_data()),
        reinterpret_cast<io_type*>(output->untyped_data()),
        batch, seq_len, hidden
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

}  // anonymous namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    HANDLER_SYMBOL(fused_elu_gru_scan), fused_elu_gru_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // gates
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // candidates
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // h0
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // output
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_elu_gru_scan_" PRECISION_SUFFIX, "CUDA", HANDLER_SYMBOL(fused_elu_gru_scan));

// --- Real-GRU FFI ---

namespace {

ffi::Error fused_real_gru_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> gates,
    ffi::Buffer<FFI_IO_TYPE> candidates,
    ffi::Buffer<FFI_IO_TYPE> h0,
    ffi::ResultBuffer<FFI_IO_TYPE> output
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
        reinterpret_cast<const io_type*>(gates.untyped_data()),
        reinterpret_cast<const io_type*>(candidates.untyped_data()),
        reinterpret_cast<const io_type*>(h0.untyped_data()),
        reinterpret_cast<io_type*>(output->untyped_data()),
        batch, seq_len, hidden
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

}  // anonymous namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    HANDLER_SYMBOL(fused_real_gru_scan), fused_real_gru_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // gates
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // candidates
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // h0
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // output
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_real_gru_scan_" PRECISION_SUFFIX, "CUDA", HANDLER_SYMBOL(fused_real_gru_scan));

// --- Diag-Linear FFI ---

namespace {

ffi::Error fused_diag_linear_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> a_vals,
    ffi::Buffer<FFI_IO_TYPE> b_vals,
    ffi::Buffer<FFI_IO_TYPE> h0,
    ffi::ResultBuffer<FFI_IO_TYPE> output
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
        reinterpret_cast<const io_type*>(a_vals.untyped_data()),
        reinterpret_cast<const io_type*>(b_vals.untyped_data()),
        reinterpret_cast<const io_type*>(h0.untyped_data()),
        reinterpret_cast<io_type*>(output->untyped_data()),
        batch, seq_len, hidden
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

}  // anonymous namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    HANDLER_SYMBOL(fused_diag_linear_scan), fused_diag_linear_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // a_vals
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // b_vals
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // h0
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // output
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_diag_linear_scan_" PRECISION_SUFFIX, "CUDA", HANDLER_SYMBOL(fused_diag_linear_scan));

#endif  // EXLA_FFI
