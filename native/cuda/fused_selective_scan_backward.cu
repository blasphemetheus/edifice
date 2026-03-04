// Fused Mamba Selective Scan Backward Kernel
//
// Computes gradients for the Mamba SSM selective scan:
//   dt_t = clamp(dt[b,t,h], DT_MIN, DT_MAX)
//   A_bar[s] = exp(dt_t * A[h,s])
//   B_bar[s] = dt_t * B[b,t,s]
//   h[s] = A_bar[s] * h_prev[s] + B_bar[s] * x[b,t,h]
//   y[b,t,h] = sum_s(C[b,t,s] * h[s])
//
// Two-pass approach:
//   Pass 1 (forward, t=0..T-1): Recompute h_state[s] per timestep, store in local arrays.
//   Pass 2 (reverse, t=T-1..0): Accumulate gradients.
//
// Backward math (per thread h, reverse over t):
//   dy = grad_output[b,t,h]
//   For each state s:
//     dh_state[s] += dy * C[b,t,s]          // from output y = sum(C*h)
//     dC_contrib[s] = dy * h_state[b,t,s]   // gradient to C (needs cross-hidden reduction)
//     dh_prev[s] = dh_state[s] * A_bar[s]
//     dx += dh_state[s] * dt_t * B[b,t,s]   // sum over s
//     ddt += dh_state[s] * h_prev[s] * A_bar[s] * A[h,s]  // through exp(dt*A)
//     ddt += dh_state[s] * x * B[b,t,s]                     // through dt*B*x
//     dB_contrib[s] = dh_state[s] * x * dt_t               // needs cross-hidden reduction
//     dA_contrib[s] = dh_state[s] * h_prev[s] * A_bar[s] * dt_t  // needs cross-batch/time reduction
//     dh_state[s] = dh_prev[s]  // carry backward
//
// Thread layout: one thread per (batch, hidden) element — same as forward.
// Each thread maintains dh_state[32] accumulators in registers.
//
// Cross-hidden reductions for dB and dC:
//   dB[b,t,s] = sum_h(dh_state_h[s] * x[b,t,h] * dt_t_h)
//   dC[b,t,s] = sum_h(dy_h * h_state_h[t,s])
//   These are accumulated via atomicAdd to global memory.
//
// Cross-batch/time reduction for dA:
//   dA[h,s] = sum_{b,t}(dh_state[s] * h_prev[s] * A_bar[s] * dt_t)
//   Each thread accumulates locally across timesteps, then atomicAdds to global.
//
// Inputs:
//   x:           [B, T, H]  — input activations
//   dt:          [B, T, H]  — discretization timesteps
//   A:           [H, S]     — state transition diagonal
//   B:           [B, T, S]  — input-to-state projection
//   C:           [B, T, S]  — state-to-output projection
//   forward_out: [B, T, H]  — forward pass output (y values)
//   grad_output: [B, T, H]  — upstream gradient dL/dy
//
// Outputs:
//   grad_x:  [B, T, H]  — gradient w.r.t. x
//   grad_dt: [B, T, H]  — gradient w.r.t. dt (pre-clamp)
//   grad_B:  [B, T, S]  — gradient w.r.t. B (reduced across hidden)
//   grad_C:  [B, T, S]  — gradient w.r.t. C (reduced across hidden)

#include <cuda_runtime.h>
#include "precision.cuh"

constexpr float DT_MIN = 0.001f;
constexpr float DT_MAX = 0.1f;
#define MAX_SEQ_LEN 1024
#define MAX_STATE 32

// ============================================================================
// Kernel
// ============================================================================

#ifdef EXLA_FFI
namespace {  // anonymous namespace — internal linkage per compilation unit
#endif

__global__ void fused_selective_scan_backward_kernel(
    const io_type* __restrict__ x,           // [B, T, H]
    const io_type* __restrict__ dt,          // [B, T, H]
    const io_type* __restrict__ A,           // [H, S]
    const io_type* __restrict__ B,           // [B, T, S]
    const io_type* __restrict__ C,           // [B, T, S]
    const io_type* __restrict__ grad_output, // [B, T, H]
    io_type* __restrict__ grad_x,            // [B, T, H]
    io_type* __restrict__ grad_dt,           // [B, T, H]
    float* __restrict__ grad_B,            // [B, T, S] — atomicAdd target (stays float for atomicAdd)
    float* __restrict__ grad_C,            // [B, T, S] — atomicAdd target (stays float for atomicAdd)
    int batch, int seq_len, int hidden, int state_size
) {
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || h >= hidden) return;

    // Load A diagonal for this hidden dim
    float A_diag[MAX_STATE];
    for (int s = 0; s < state_size && s < MAX_STATE; s++) {
        A_diag[s] = IO_LOAD(A, h * state_size + s);
    }

    // ========================================
    // Pass 1: Forward — recompute and store h_state per timestep
    // ========================================
    // We store h_state[t][s] in local arrays for the backward pass
    // h_prev_store[t][s] = h_state before update at timestep t
    float h_state[MAX_STATE];
    float h_prev_store[MAX_SEQ_LEN][MAX_STATE];

    for (int s = 0; s < state_size && s < MAX_STATE; s++) {
        h_state[s] = 0.0f;
    }

    for (int t = 0; t < seq_len; t++) {
        int x_idx = b * seq_len * hidden + t * hidden + h;
        float x_t = IO_LOAD(x, x_idx);
        float dt_raw = IO_LOAD(dt, x_idx);
        float dt_t = fminf(fmaxf(dt_raw, DT_MIN), DT_MAX);

        int bc_idx = b * seq_len * state_size + t * state_size;

        for (int s = 0; s < state_size && s < MAX_STATE; s++) {
            h_prev_store[t][s] = h_state[s];

            float A_bar = expf(dt_t * A_diag[s]);
            float B_bar = dt_t * IO_LOAD(B, bc_idx + s);
            h_state[s] = A_bar * h_state[s] + B_bar * x_t;
        }
    }

    // ========================================
    // Pass 2: Backward — reverse accumulate gradients
    // ========================================
    float dh_state[MAX_STATE];
    for (int s = 0; s < state_size && s < MAX_STATE; s++) {
        dh_state[s] = 0.0f;
    }

    for (int t = seq_len - 1; t >= 0; t--) {
        int x_idx = b * seq_len * hidden + t * hidden + h;
        int bc_idx = b * seq_len * state_size + t * state_size;

        float dy = IO_LOAD(grad_output, x_idx);
        float x_t = IO_LOAD(x, x_idx);
        float dt_raw = IO_LOAD(dt, x_idx);
        float dt_t = fminf(fmaxf(dt_raw, DT_MIN), DT_MAX);

        float dx_t = 0.0f;
        float ddt_t = 0.0f;

        for (int s = 0; s < state_size && s < MAX_STATE; s++) {
            float C_s = IO_LOAD(C, bc_idx + s);
            float B_s = IO_LOAD(B, bc_idx + s);
            float h_prev_s = h_prev_store[t][s];
            float A_bar = expf(dt_t * A_diag[s]);

            // h_after_update[s] = A_bar * h_prev_s + dt_t * B_s * x_t
            float h_cur_s = A_bar * h_prev_s + dt_t * B_s * x_t;

            // Gradient from output: y = sum_s(C_s * h_cur_s)
            dh_state[s] += dy * C_s;

            // Gradient to C: dC[b,t,s] += dy * h_cur_s (summed across hidden)
            atomicAdd(&grad_C[bc_idx + s], dy * h_cur_s);

            // Through recurrence: h = A_bar * h_prev + B_bar * x
            // dx += dh * B_bar = dh * dt_t * B_s
            dx_t += dh_state[s] * dt_t * B_s;

            // ddt through A_bar = exp(dt*A): d/d(dt) = A_bar * A[h,s]
            ddt_t += dh_state[s] * h_prev_s * A_bar * A_diag[s];

            // ddt through B_bar = dt * B: d/d(dt) = B_s * x
            ddt_t += dh_state[s] * B_s * x_t;

            // dB[b,t,s] += dh * x * dt (summed across hidden)
            atomicAdd(&grad_B[bc_idx + s], dh_state[s] * x_t * dt_t);

            // Carry dh backward: dh_prev = dh * A_bar
            dh_state[s] = dh_state[s] * A_bar;
        }

        IO_STORE(grad_x, x_idx, dx_t);

        // dt gradient only flows if dt was not clamped
        float dt_in_range = (dt_raw >= DT_MIN && dt_raw <= DT_MAX) ? 1.0f : 0.0f;
        IO_STORE(grad_dt, x_idx, ddt_t * dt_in_range);
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

// Output: concatenated [grad_x (B*T*H) | grad_dt (B*T*H) | grad_B (B*T*S) | grad_C (B*T*S)]
// Note: grad_B and grad_C use atomicAdd so stay as float even in bf16 mode.
// The output buffer layout: io_type for grad_x/grad_dt, float for grad_B/grad_C.
int fused_selective_scan_backward_launch(
    cudaStream_t stream,
    const io_type* x, const io_type* dt, const io_type* A,
    const io_type* B, const io_type* C,
    const io_type* grad_output,
    io_type* output_concat,
    int batch, int seq_len, int hidden, int state_size
) {
    int bth = batch * seq_len * hidden;
    int bts = batch * seq_len * state_size;
    io_type* grad_x  = output_concat;
    io_type* grad_dt = output_concat + bth;
    // grad_B and grad_C are float (atomicAdd targets) — placed after io_type region
    float* grad_B  = reinterpret_cast<float*>(output_concat + 2 * bth);
    float* grad_C  = reinterpret_cast<float*>(output_concat + 2 * bth) + bts;

    // Zero out grad_B and grad_C since they use atomicAdd
    cudaMemsetAsync(grad_B, 0, bts * sizeof(float), stream);
    cudaMemsetAsync(grad_C, 0, bts * sizeof(float), stream);

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_selective_scan_backward_kernel<<<grid, block, 0, stream>>>(
        x, dt, A, B, C, grad_output,
        grad_x, grad_dt, grad_B, grad_C,
        batch, seq_len, hidden, state_size
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

ffi::Error fused_selective_scan_backward_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> x,
    ffi::Buffer<FFI_IO_TYPE> dt,
    ffi::Buffer<FFI_IO_TYPE> A,
    ffi::Buffer<FFI_IO_TYPE> B,
    ffi::Buffer<FFI_IO_TYPE> C,
    ffi::Buffer<FFI_IO_TYPE> grad_output,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_x,
    ffi::ResultBuffer<FFI_IO_TYPE> grad_dt,
    ffi::ResultBuffer<ffi::F32> grad_B,
    ffi::ResultBuffer<ffi::F32> grad_C
) {
    auto x_dims = x.dimensions();
    int batch      = static_cast<int>(x_dims[0]);
    int seq_len    = static_cast<int>(x_dims[1]);
    int hidden     = static_cast<int>(x_dims[2]);

    auto a_dims = A.dimensions();
    int state_size = static_cast<int>(a_dims[1]);

    int bts = batch * seq_len * state_size;

    // Zero out atomicAdd targets
    cudaMemsetAsync(reinterpret_cast<float*>(grad_B->untyped_data()), 0,
                    bts * sizeof(float), stream);
    cudaMemsetAsync(reinterpret_cast<float*>(grad_C->untyped_data()), 0,
                    bts * sizeof(float), stream);

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_selective_scan_backward_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const io_type*>(x.untyped_data()),
        reinterpret_cast<const io_type*>(dt.untyped_data()),
        reinterpret_cast<const io_type*>(A.untyped_data()),
        reinterpret_cast<const io_type*>(B.untyped_data()),
        reinterpret_cast<const io_type*>(C.untyped_data()),
        reinterpret_cast<const io_type*>(grad_output.untyped_data()),
        reinterpret_cast<io_type*>(grad_x->untyped_data()),
        reinterpret_cast<io_type*>(grad_dt->untyped_data()),
        reinterpret_cast<float*>(grad_B->untyped_data()),
        reinterpret_cast<float*>(grad_C->untyped_data()),
        batch, seq_len, hidden, state_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

}  // anonymous namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    HANDLER_SYMBOL(fused_selective_scan_backward), fused_selective_scan_backward_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // x
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // dt
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // A
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // B
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // C
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // grad_output
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_x
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // grad_dt
        .Ret<ffi::Buffer<ffi::F32>>()   // grad_B (atomicAdd target)
        .Ret<ffi::Buffer<ffi::F32>>()   // grad_C (atomicAdd target)
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_selective_scan_backward_" PRECISION_SUFFIX, "CUDA", HANDLER_SYMBOL(fused_selective_scan_backward));

#endif  // EXLA_FFI
