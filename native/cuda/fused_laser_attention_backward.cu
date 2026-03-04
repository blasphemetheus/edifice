// Fused LASER Attention Backward Kernel
//
// LASER backward: O = log(softmax(S) @ exp(V - v_max)) + v_max
//
// Key differences from standard flash attention backward:
//   - D[i] = sum(dO[i]) (not sum(dO*O)) — from log derivative
//   - Effective gradient: dR[i,dd] = dO[i,dd] * exp(v_max[dd] - O[i,dd])
//   - V tiles load exp(V - v_max) into shared memory (same as forward)
//   - dV gets extra chain rule: dV[j,dd] = dV_exp[j,dd] * exp(V[j,dd] - v_max[dd])
//   - v_max gradient treated as constant (numerical stability trick)
//
// Thread layout: one thread block per (batch, head) pair.
// Each block has TILE_SIZE threads.
//
// Two compilation modes:
//   1. Standalone (default): kernel + C-linkage launch wrapper for NIF.
//   2. EXLA FFI (-DEXLA_FFI): kernel + XLA FFI handler + registration.
//
// Inputs:
//   Q:      [B, H, T, d] — query vectors
//   K:      [B, H, T, d] — key vectors
//   V:      [B, H, T, d] — value vectors (raw, not exponentiated)
//   V_max:  [B, H, 1, d] — per-column max of V
//   O:      [B, H, T, d] — forward output (in log space)
//   grad_O: [B, H, T, d] — upstream gradient
//   causal: scalar int
//
// Outputs:
//   dQ: [B, H, T, d] — gradient w.r.t. Q
//   dK: [B, H, T, d] — gradient w.r.t. K
//   dV: [B, H, T, d] — gradient w.r.t. V

#include <cuda_runtime.h>
#include "precision.cuh"
#include <float.h>
#include <math.h>

#define TILE_SIZE 32

// ============================================================================
// Kernel
// ============================================================================

#ifdef EXLA_FFI
namespace {  // anonymous namespace — internal linkage per compilation unit
#endif

__global__ void fused_laser_attention_backward_kernel(
    const io_type* __restrict__ Q,       // [B, H, T, d]
    const io_type* __restrict__ K,       // [B, H, T, d]
    const io_type* __restrict__ V,       // [B, H, T, d]
    const io_type* __restrict__ V_max,   // [B, H, 1, d]
    const io_type* __restrict__ O,       // [B, H, T, d]
    const io_type* __restrict__ grad_O,  // [B, H, T, d]
    io_type* __restrict__ dQ,            // [B, H, T, d]
    io_type* __restrict__ dK,            // [B, H, T, d]
    io_type* __restrict__ dV,            // [B, H, T, d]
    const float* __restrict__ D_buf,     // [B*H*T] precomputed
    const float* __restrict__ lse_buf,   // [B*H*T] precomputed
    int seq_len,
    int head_dim,
    int causal
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int ti = threadIdx.x;

    extern __shared__ float smem[];
    float* smem_a = smem;                              // [TILE_SIZE][head_dim]
    float* smem_b = smem + TILE_SIZE * head_dim;       // [TILE_SIZE][head_dim]

    float scale = rsqrtf((float)head_dim);

    int BH_stride = seq_len * head_dim;
    int base = (b * gridDim.y + h) * BH_stride;
    int vmax_base = (b * gridDim.y + h) * head_dim;
    int aux_base = (b * gridDim.y + h) * seq_len;

    int num_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;

    // ==================================================================
    // Phase 2: Compute dK, dV
    // ==================================================================
    for (int kv_tile = 0; kv_tile < num_tiles; kv_tile++) {
        int kv_idx = kv_tile * TILE_SIZE + ti;
        bool kv_valid = kv_idx < seq_len;

        // Load own K row and exp(V - v_max) into registers
        float k_row[128], expv_row[128], v_raw[128];
        float dk_acc[128], dv_exp_acc[128];
        for (int dd = 0; dd < head_dim; dd++) {
            if (kv_valid) {
                k_row[dd] = IO_LOAD(K, base + kv_idx * head_dim + dd);
                v_raw[dd] = IO_LOAD(V, base + kv_idx * head_dim + dd);
                float vm = IO_LOAD(V_max, vmax_base + dd);
                expv_row[dd] = expf(v_raw[dd] - vm);
            } else {
                k_row[dd] = 0.0f;
                expv_row[dd] = 0.0f;
                v_raw[dd] = 0.0f;
            }
            dk_acc[dd] = 0.0f;
            dv_exp_acc[dd] = 0.0f;
        }

        // Inner loop over Q tiles
        for (int q_tile = 0; q_tile < num_tiles; q_tile++) {
            if (causal) {
                int q_end = (q_tile + 1) * TILE_SIZE - 1;
                if (q_end < kv_tile * TILE_SIZE) continue;
            }

            // Load Q tile and effective gradient dR into shared memory
            // dR[i,dd] = dO[i,dd] * exp(v_max[dd] - O[i,dd])
            int q_row = q_tile * TILE_SIZE + ti;
            for (int dd = 0; dd < head_dim; dd++) {
                if (q_row < seq_len) {
                    smem_a[ti * head_dim + dd] = IO_LOAD(Q, base + q_row * head_dim + dd);
                    float do_val = IO_LOAD(grad_O, base + q_row * head_dim + dd);
                    float o_val = IO_LOAD(O, base + q_row * head_dim + dd);
                    float vm = IO_LOAD(V_max, vmax_base + dd);
                    smem_b[ti * head_dim + dd] = do_val * expf(vm - o_val);
                } else {
                    smem_a[ti * head_dim + dd] = 0.0f;
                    smem_b[ti * head_dim + dd] = 0.0f;
                }
            }
            __syncthreads();

            if (kv_valid) {
                int tile_len = min(TILE_SIZE, seq_len - q_tile * TILE_SIZE);
                for (int qi = 0; qi < tile_len; qi++) {
                    int q_pos = q_tile * TILE_SIZE + qi;
                    if (causal && q_pos < kv_idx) continue;

                    float s = 0.0f;
                    for (int dd = 0; dd < head_dim; dd++) {
                        s += smem_a[qi * head_dim + dd] * k_row[dd];
                    }
                    s *= scale;

                    float lse_q = lse_buf[aux_base + q_pos];
                    float p = expf(s - lse_q);

                    // dV_exp += p * dR[q]
                    float dp = 0.0f;
                    for (int dd = 0; dd < head_dim; dd++) {
                        dv_exp_acc[dd] += p * smem_b[qi * head_dim + dd];
                        dp += smem_b[qi * head_dim + dd] * expv_row[dd];
                    }

                    float D_q = D_buf[aux_base + q_pos];
                    float ds = p * (dp - D_q);

                    for (int dd = 0; dd < head_dim; dd++) {
                        dk_acc[dd] += ds * smem_a[qi * head_dim + dd] * scale;
                    }
                }
            }
            __syncthreads();
        }

        // Write dK and dV (chain rule: dV = dV_exp * exp(V - v_max))
        if (kv_valid) {
            for (int dd = 0; dd < head_dim; dd++) {
                IO_STORE(dK, base + kv_idx * head_dim + dd, dk_acc[dd]);
                IO_STORE(dV, base + kv_idx * head_dim + dd, dv_exp_acc[dd] * expv_row[dd]);
            }
        }
    }

    // ==================================================================
    // Phase 3: Compute dQ
    // ==================================================================
    for (int q_tile = 0; q_tile < num_tiles; q_tile++) {
        int q_idx = q_tile * TILE_SIZE + ti;
        bool q_valid = q_idx < seq_len;

        float dq_acc[128];
        float q_row[128];
        for (int dd = 0; dd < head_dim; dd++) {
            dq_acc[dd] = 0.0f;
            q_row[dd] = q_valid ? IO_LOAD(Q, base + q_idx * head_dim + dd) : 0.0f;
        }

        float lse_i = q_valid ? lse_buf[aux_base + q_idx] : 0.0f;
        float D_i = q_valid ? D_buf[aux_base + q_idx] : 0.0f;

        // Load effective gradient for this Q row
        float dr_row[128];
        for (int dd = 0; dd < head_dim; dd++) {
            if (q_valid) {
                float do_val = IO_LOAD(grad_O, base + q_idx * head_dim + dd);
                float o_val = IO_LOAD(O, base + q_idx * head_dim + dd);
                float vm = IO_LOAD(V_max, vmax_base + dd);
                dr_row[dd] = do_val * expf(vm - o_val);
            } else {
                dr_row[dd] = 0.0f;
            }
        }

        for (int kv_tile = 0; kv_tile < num_tiles; kv_tile++) {
            if (causal) {
                int kv_start = kv_tile * TILE_SIZE;
                if (q_valid && kv_start > q_idx) break;
            }

            // Load K and exp(V - v_max) tiles into shared memory
            int kv_row = kv_tile * TILE_SIZE + ti;
            for (int dd = 0; dd < head_dim; dd++) {
                if (kv_row < seq_len) {
                    smem_a[ti * head_dim + dd] = IO_LOAD(K, base + kv_row * head_dim + dd);
                    float v_val = IO_LOAD(V, base + kv_row * head_dim + dd);
                    float vm = IO_LOAD(V_max, vmax_base + dd);
                    smem_b[ti * head_dim + dd] = expf(v_val - vm);
                } else {
                    smem_a[ti * head_dim + dd] = 0.0f;
                    smem_b[ti * head_dim + dd] = 0.0f;
                }
            }
            __syncthreads();

            if (q_valid) {
                int tile_len = min(TILE_SIZE, seq_len - kv_tile * TILE_SIZE);
                for (int j = 0; j < tile_len; j++) {
                    int kv_pos = kv_tile * TILE_SIZE + j;
                    if (causal && kv_pos > q_idx) break;

                    float s = 0.0f;
                    for (int dd = 0; dd < head_dim; dd++) {
                        s += q_row[dd] * smem_a[j * head_dim + dd];
                    }
                    s *= scale;

                    float p = expf(s - lse_i);

                    float dp = 0.0f;
                    for (int dd = 0; dd < head_dim; dd++) {
                        dp += dr_row[dd] * smem_b[j * head_dim + dd];
                    }

                    float ds = p * (dp - D_i);

                    for (int dd = 0; dd < head_dim; dd++) {
                        dq_acc[dd] += ds * smem_a[j * head_dim + dd] * scale;
                    }
                }
            }
            __syncthreads();
        }

        if (q_valid) {
            for (int dd = 0; dd < head_dim; dd++) {
                IO_STORE(dQ, base + q_idx * head_dim + dd, dq_acc[dd]);
            }
        }
    }
}

// ============================================================================
// Helper kernel: Precompute D[i] and lse[i] for LASER backward
// ============================================================================

__global__ void laser_attention_backward_precompute_kernel(
    const io_type* __restrict__ Q,       // [B, H, T, d]
    const io_type* __restrict__ K,       // [B, H, T, d]
    const io_type* __restrict__ V,       // [B, H, T, d]
    const io_type* __restrict__ V_max,   // [B, H, 1, d]
    const io_type* __restrict__ O,       // [B, H, T, d]
    const io_type* __restrict__ grad_O,  // [B, H, T, d]
    float* __restrict__ D_buf,           // [B*H*T]
    float* __restrict__ lse_buf,         // [B*H*T]
    int seq_len,
    int head_dim,
    int causal
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int qi = threadIdx.x;

    extern __shared__ float smem[];
    float* Kj = smem;

    float scale = rsqrtf((float)head_dim);

    int BH_stride = seq_len * head_dim;
    int base = (b * gridDim.y + h) * BH_stride;
    int vmax_base = (b * gridDim.y + h) * head_dim;
    int aux_base = (b * gridDim.y + h) * seq_len;

    int num_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;

    for (int q_tile = 0; q_tile < num_tiles; q_tile++) {
        int i = q_tile * TILE_SIZE + qi;
        if (i >= seq_len) continue;

        // D[i] = sum(dR[i]) where dR = dO * exp(v_max - O)
        // Simplification: D[i] = sum_dd(dO[i,dd] * exp(v_max[dd] - O[i,dd]))
        float d_val = 0.0f;
        for (int dd = 0; dd < head_dim; dd++) {
            float do_val = IO_LOAD(grad_O, base + i * head_dim + dd);
            float o_val = IO_LOAD(O, base + i * head_dim + dd);
            float vm = IO_LOAD(V_max, vmax_base + dd);
            float dr = do_val * expf(vm - o_val);
            // D[i] for LASER: sum(dR[i] * expV[j]) summed over the softmax result
            // Actually D[i] = sum_dd(dR[i,dd] * (softmax@expV)[i,dd])
            // = sum_dd(dR[i,dd] * exp(O[i,dd] - v_max[dd]))
            // = sum_dd(dO[i,dd])
            d_val += do_val;
        }
        D_buf[aux_base + i] = d_val;

        // Recompute lse[i] via online softmax (same as standard)
        float m_i = -FLT_MAX;
        float l_i = 0.0f;

        for (int kv_tile = 0; kv_tile < num_tiles; kv_tile++) {
            if (causal) {
                int kv_start = kv_tile * TILE_SIZE;
                if (kv_start > i) break;
            }

            int kv_row = kv_tile * TILE_SIZE + qi;
            if (kv_row < seq_len) {
                for (int dd = 0; dd < head_dim; dd++) {
                    Kj[qi * head_dim + dd] = IO_LOAD(K, base + kv_row * head_dim + dd);
                }
            } else {
                for (int dd = 0; dd < head_dim; dd++) {
                    Kj[qi * head_dim + dd] = 0.0f;
                }
            }
            __syncthreads();

            int tile_len = min(TILE_SIZE, seq_len - kv_tile * TILE_SIZE);
            for (int j = 0; j < tile_len; j++) {
                int kv_pos = kv_tile * TILE_SIZE + j;
                if (causal && kv_pos > i) break;

                float s = 0.0f;
                for (int dd = 0; dd < head_dim; dd++) {
                    s += IO_LOAD(Q, base + i * head_dim + dd) * Kj[j * head_dim + dd];
                }
                s *= scale;

                float m_new = fmaxf(m_i, s);
                float exp_diff = expf(m_i - m_new);
                float exp_s = expf(s - m_new);
                l_i = l_i * exp_diff + exp_s;
                m_i = m_new;
            }
            __syncthreads();
        }

        lse_buf[aux_base + i] = m_i + logf(l_i);
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

// Outputs: [dQ | dK | dV] each B*H*T*d
int fused_laser_attention_backward_launch(
    cudaStream_t stream,
    const io_type* q, const io_type* k, const io_type* v,
    const io_type* v_max, const io_type* o, const io_type* grad_o,
    io_type* output_concat,
    int batch, int num_heads, int seq_len, int head_dim,
    int causal
) {
    size_t bhtd = (size_t)batch * num_heads * seq_len * head_dim;
    io_type* dq = output_concat;
    io_type* dk = output_concat + bhtd;
    io_type* dv = output_concat + 2 * bhtd;

    size_t bht = (size_t)batch * num_heads * seq_len;
    float* d_buf = NULL;
    float* lse_buf = NULL;
    cudaError_t err;

    err = cudaMalloc(&d_buf, bht * sizeof(float));
    if (err != cudaSuccess) return (int)err;
    err = cudaMalloc(&lse_buf, bht * sizeof(float));
    if (err != cudaSuccess) { cudaFree(d_buf); return (int)err; }

    dim3 grid(batch, num_heads);
    dim3 block(TILE_SIZE);

    size_t smem_precompute = TILE_SIZE * head_dim * sizeof(float);
    laser_attention_backward_precompute_kernel<<<grid, block, smem_precompute, stream>>>(
        q, k, v, v_max, o, grad_o, d_buf, lse_buf,
        seq_len, head_dim, causal
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_buf); cudaFree(lse_buf);
        return (int)err;
    }

    size_t smem_main = 2 * TILE_SIZE * head_dim * sizeof(float);
    fused_laser_attention_backward_kernel<<<grid, block, smem_main, stream>>>(
        q, k, v, v_max, o, grad_o,
        dq, dk, dv,
        d_buf, lse_buf,
        seq_len, head_dim, causal
    );

    err = cudaGetLastError();
    cudaFree(d_buf);
    cudaFree(lse_buf);
    return (int)err;
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
ffi::Error fused_laser_attention_backward_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> q,
    ffi::Buffer<FFI_IO_TYPE> k,
    ffi::Buffer<FFI_IO_TYPE> v,
    ffi::Buffer<FFI_IO_TYPE> v_max,
    ffi::Buffer<FFI_IO_TYPE> o,
    ffi::Buffer<FFI_IO_TYPE> grad_o,
    ffi::ResultBuffer<FFI_IO_TYPE> dq,
    ffi::ResultBuffer<FFI_IO_TYPE> dk,
    ffi::ResultBuffer<FFI_IO_TYPE> dv
) {
    auto dims = q.dimensions();
    int batch    = static_cast<int>(dims[0]);
    int num_heads = static_cast<int>(dims[1]);
    int seq_len  = static_cast<int>(dims[2]);
    int head_dim = static_cast<int>(dims[3]);

    int causal = 1;  // hardcoded causal

    size_t bht = (size_t)batch * num_heads * seq_len;
    float* d_buf = NULL;
    float* lse_buf = NULL;
    cudaError_t err;

    err = cudaMalloc(&d_buf, bht * sizeof(float));
    if (err != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, "cudaMalloc failed for d_buf");
    err = cudaMalloc(&lse_buf, bht * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_buf);
        return ffi::Error(ffi::ErrorCode::kInternal, "cudaMalloc failed for lse_buf");
    }

    dim3 grid(batch, num_heads);
    dim3 block(TILE_SIZE);

    size_t smem_precompute = TILE_SIZE * head_dim * sizeof(float);
    laser_attention_backward_precompute_kernel<<<grid, block, smem_precompute, stream>>>(
        reinterpret_cast<const io_type*>(q.untyped_data()),
        reinterpret_cast<const io_type*>(k.untyped_data()),
        reinterpret_cast<const io_type*>(v.untyped_data()),
        reinterpret_cast<const io_type*>(v_max.untyped_data()),
        reinterpret_cast<const io_type*>(o.untyped_data()),
        reinterpret_cast<const io_type*>(grad_o.untyped_data()),
        d_buf, lse_buf,
        seq_len, head_dim, causal
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_buf); cudaFree(lse_buf);
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    size_t smem_main = 2 * TILE_SIZE * head_dim * sizeof(float);
    fused_laser_attention_backward_kernel<<<grid, block, smem_main, stream>>>(
        reinterpret_cast<const io_type*>(q.untyped_data()),
        reinterpret_cast<const io_type*>(k.untyped_data()),
        reinterpret_cast<const io_type*>(v.untyped_data()),
        reinterpret_cast<const io_type*>(v_max.untyped_data()),
        reinterpret_cast<const io_type*>(o.untyped_data()),
        reinterpret_cast<const io_type*>(grad_o.untyped_data()),
        reinterpret_cast<io_type*>(dq->untyped_data()),
        reinterpret_cast<io_type*>(dk->untyped_data()),
        reinterpret_cast<io_type*>(dv->untyped_data()),
        d_buf, lse_buf,
        seq_len, head_dim, causal
    );

    err = cudaGetLastError();
    cudaFree(d_buf);
    cudaFree(lse_buf);

    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

}  // anonymous namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    HANDLER_SYMBOL(fused_laser_attention_backward), fused_laser_attention_backward_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // q       [B, H, T, d]
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // k       [B, H, T, d]
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // v       [B, H, T, d]
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // v_max   [B, H, 1, d]
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // o       [B, H, T, d]
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // grad_o  [B, H, T, d]
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // dQ      [B, H, T, d]
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // dK      [B, H, T, d]
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // dV      [B, H, T, d]
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_laser_attention_backward_" PRECISION_SUFFIX, "CUDA", HANDLER_SYMBOL(fused_laser_attention_backward));

#endif  // EXLA_FFI
