// Fused Mamba Selective Scan Kernel
//
// Implements the Mamba SSM recurrence with input-dependent discretization:
//   A_bar = exp(dt * A)
//   B_bar = dt * B
//   h_t = A_bar * h_{t-1} + B_bar * x_t    (per state dimension)
//   y_t = sum(C_t * h_t, axis=state)
//
// Thread layout: one thread per (batch, hidden) element.
// Each thread maintains state[32] in registers and scans sequentially
// through timesteps.
//
// Two compilation modes:
//   1. Standalone (-DSTANDALONE): kernel + C-linkage launch wrapper.
//   2. EXLA FFI (-DEXLA_FFI): kernel + XLA FFI handler + registration.
//
// Inputs:
//   x:  [batch, seq_len, hidden]  — input activations
//   dt: [batch, seq_len, hidden]  — discretization timesteps (clamped)
//   A:  [hidden, state]           — state transition diagonal (negative)
//   B:  [batch, seq_len, state]   — input-to-state projection
//   C:  [batch, seq_len, state]   — state-to-output projection
//
// Output:
//   out: [batch, seq_len, hidden] — scan output

#include <cuda_runtime.h>
#include "precision.cuh"

constexpr float DT_MIN = 0.001f;
constexpr float DT_MAX = 0.1f;

// ============================================================================
// Kernel
// ============================================================================

#ifdef EXLA_FFI
namespace {  // anonymous namespace — internal linkage per compilation unit
#endif

__global__ void fused_selective_scan_kernel(
    const io_type* __restrict__ x,      // [B, T, H]
    const io_type* __restrict__ dt,     // [B, T, H]
    const io_type* __restrict__ A,      // [H, S]
    const io_type* __restrict__ B,      // [B, T, S]
    const io_type* __restrict__ C,      // [B, T, S]
    io_type* __restrict__ out,          // [B, T, H]
    int batch, int seq_len, int hidden, int state
) {
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || h >= hidden) return;

    // Initialize hidden state in registers (max state_size = 32)
    float h_state[32];
    for (int s = 0; s < state && s < 32; s++) {
        h_state[s] = 0.0f;
    }

    // Load A diagonal for this hidden dim (constant across timesteps)
    float A_diag[32];
    for (int s = 0; s < state && s < 32; s++) {
        A_diag[s] = IO_LOAD(A, h * state + s);
    }

    // Sequential scan through timesteps
    for (int t = 0; t < seq_len; t++) {
        int x_idx = b * seq_len * hidden + t * hidden + h;
        float x_t = IO_LOAD(x, x_idx);
        float dt_t = fminf(fmaxf(IO_LOAD(dt, x_idx), DT_MIN), DT_MAX);

        int bc_idx = b * seq_len * state + t * state;
        float y_t = 0.0f;

        for (int s = 0; s < state && s < 32; s++) {
            // Discretize: A_bar = exp(dt * A), B_bar = dt * B
            float A_bar = expf(dt_t * A_diag[s]);
            float B_bar = dt_t * IO_LOAD(B, bc_idx + s);
            float C_s = IO_LOAD(C, bc_idx + s);

            // Recurrence: h = A_bar * h + B_bar * x
            h_state[s] = A_bar * h_state[s] + B_bar * x_t;

            // Output: y = sum(C * h)
            y_t += C_s * h_state[s];
        }

        IO_STORE(out, x_idx, y_t);
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

int fused_selective_scan_launch(
    cudaStream_t stream,
    const io_type* x, const io_type* dt, const io_type* A,
    const io_type* B, const io_type* C,
    io_type* out,
    int batch, int seq_len, int hidden, int state
) {
    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_selective_scan_kernel<<<grid, block, 0, stream>>>(
        x, dt, A, B, C, out,
        batch, seq_len, hidden, state
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

ffi::Error fused_selective_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> x,       // [B, T, H]
    ffi::Buffer<FFI_IO_TYPE> dt,      // [B, T, H]
    ffi::Buffer<FFI_IO_TYPE> A,       // [H, S]
    ffi::Buffer<FFI_IO_TYPE> B,       // [B, T, S]
    ffi::Buffer<FFI_IO_TYPE> C,       // [B, T, S]
    ffi::ResultBuffer<FFI_IO_TYPE> out // [B, T, H]
) {
    // Extract dimensions from x: [batch, seq_len, hidden]
    auto x_dims = x.dimensions();
    int batch   = static_cast<int>(x_dims[0]);
    int seq_len = static_cast<int>(x_dims[1]);
    int hidden  = static_cast<int>(x_dims[2]);

    // Extract state_size from A: [hidden, state]
    auto a_dims = A.dimensions();
    int state   = static_cast<int>(a_dims[1]);

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_selective_scan_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const io_type*>(x.untyped_data()),
        reinterpret_cast<const io_type*>(dt.untyped_data()),
        reinterpret_cast<const io_type*>(A.untyped_data()),
        reinterpret_cast<const io_type*>(B.untyped_data()),
        reinterpret_cast<const io_type*>(C.untyped_data()),
        reinterpret_cast<io_type*>(out->untyped_data()),
        batch, seq_len, hidden, state
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

}  // anonymous namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    HANDLER_SYMBOL(fused_selective_scan), fused_selective_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // x
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // dt
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // A
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // B
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // C
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // out
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_selective_scan_" PRECISION_SUFFIX, "CUDA", HANDLER_SYMBOL(fused_selective_scan));

#endif  // EXLA_FFI
