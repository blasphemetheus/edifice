// Fused GSA (Gated Slot Attention) Scan Kernel
//
// Implements the per-timestep slot memory update + read:
//
// Write pass:
//   kv_outer = k_slot[h,s] * v[h,d]      — outer product [H, m, d]
//   mem[h] = alpha * mem[h] + (1-alpha) * kv_outer
//
// Read pass:
//   scores = mem[h] @ q[h]               — [H, m]
//   p = softmax(scores, dim=slots)       — [H, m]
//   output[h] = sum_s(p[s] * mem[h,s,:]) — [H, d]
//
// Thread layout: one thread per (batch, head, slot_idx).
// Each thread handles one slot of one head across all timesteps.
// The slot_dim (d) values are in registers.
//
// Inputs:
//   q:       [batch, seq_len, num_heads, head_dim]  — query (post ELU+1)
//   k_slot:  [batch, seq_len, num_heads, num_slots] — slot keys (post softmax)
//   v:       [batch, seq_len, num_heads, head_dim]  — values
//   alpha:   [batch, seq_len, num_heads]             — damped sigmoid gate
//
// Output:
//   out: [batch, seq_len, num_heads * head_dim]      — concatenated head outputs

#include <cuda_runtime.h>
#include "precision.cuh"

#define GSA_MAX_HEAD_DIM 128
#define GSA_MAX_SLOTS 64

// Each block handles one (batch, head) pair.
// Thread s handles slot s. Each thread stores mem[s, 0..d-1] in registers.
// Reduction for read pass uses shared memory.

#ifdef EXLA_FFI
namespace {  // anonymous namespace — internal linkage per compilation unit
#endif

__global__ void fused_gsa_scan_kernel(
    const io_type* __restrict__ q,         // [B, T, H, d]
    const io_type* __restrict__ k_slot,    // [B, T, H, m]
    const io_type* __restrict__ v,         // [B, T, H, d]
    const io_type* __restrict__ alpha,     // [B, T, H]
    io_type* __restrict__ output,          // [B, T, H*d]
    int batch, int seq_len,
    int num_heads, int num_slots, int head_dim
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int s = threadIdx.x;  // slot index

    if (b >= batch || h >= num_heads || s >= num_slots) return;

    // Shared memory for:
    // - scores[num_slots]: softmax scores for read pass
    // - output_accum[head_dim]: accumulated output per head
    // - reduce_buf[1]: max for softmax
    extern __shared__ float shared_mem[];
    float* scores_shared = shared_mem;                          // [num_slots]
    float* output_accum  = shared_mem + num_slots;              // [head_dim]
    float* reduce_buf    = shared_mem + num_slots + head_dim;   // [2] (max, sum)

    // Slot memory in registers: mem[d] for this slot
    float mem_slot[GSA_MAX_HEAD_DIM];
    for (int d = 0; d < head_dim; d++) {
        mem_slot[d] = 0.0f;
    }

    int q_stride = num_heads * head_dim;
    int ks_stride = num_heads * num_slots;
    int v_stride = num_heads * head_dim;
    int alpha_stride = num_heads;

    for (int t = 0; t < seq_len; t++) {
        // Load alpha for this (batch, timestep, head)
        float alpha_val = IO_LOAD(alpha, b * seq_len * alpha_stride + t * alpha_stride + h);

        // Load k_slot[b, t, h, s] for this slot
        float ks_val = IO_LOAD(k_slot, b * seq_len * ks_stride + t * ks_stride + h * num_slots + s);

        // Write pass: update slot memory
        // mem[s][d] = alpha * mem[s][d] + (1-alpha) * k_slot[s] * v[d]
        int v_base = b * seq_len * v_stride + t * v_stride + h * head_dim;
        for (int d = 0; d < head_dim; d++) {
            float v_val = IO_LOAD(v, v_base + d);
            mem_slot[d] = alpha_val * mem_slot[d] + (1.0f - alpha_val) * ks_val * v_val;
        }

        // Read pass: compute score for this slot
        // score[s] = mem[s,:] @ q[h,:]
        int q_base = b * seq_len * q_stride + t * q_stride + h * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += mem_slot[d] * IO_LOAD(q, q_base + d);
        }
        scores_shared[s] = score;
        __syncthreads();

        // Softmax over slots (thread 0 computes max and sum)
        if (s == 0) {
            float max_val = scores_shared[0];
            for (int j = 1; j < num_slots; j++) {
                if (scores_shared[j] > max_val) max_val = scores_shared[j];
            }
            reduce_buf[0] = max_val;
        }
        __syncthreads();

        float max_val = reduce_buf[0];
        float exp_score = expf(scores_shared[s] - max_val);
        scores_shared[s] = exp_score;
        __syncthreads();

        if (s == 0) {
            float sum = 0.0f;
            for (int j = 0; j < num_slots; j++) {
                sum += scores_shared[j];
            }
            reduce_buf[1] = sum + 1.0e-8f;
        }
        __syncthreads();

        float p_val = exp_score / reduce_buf[1];  // softmax probability

        // Weighted read: output[h,d] = sum_s(p[s] * mem[s,d])
        // Each slot thread contributes p[s] * mem[s, d] to the output
        // Use atomic add to output_accum
        if (s == 0) {
            for (int d = 0; d < head_dim; d++) {
                output_accum[d] = 0.0f;
            }
        }
        __syncthreads();

        for (int d = 0; d < head_dim; d++) {
            atomicAdd(&output_accum[d], p_val * mem_slot[d]);
        }
        __syncthreads();

        // Write output (thread 0 stores)
        if (s == 0) {
            int out_base = b * seq_len * (num_heads * head_dim) + t * (num_heads * head_dim) + h * head_dim;
            for (int d = 0; d < head_dim; d++) {
                IO_STORE(output, out_base + d, output_accum[d]);
            }
        }
        __syncthreads();
    }
}

// ============================================================================
// Standalone launch wrapper
// ============================================================================

#ifdef EXLA_FFI
}  // anonymous namespace
#endif

#ifndef EXLA_FFI

extern "C" {

int fused_gsa_scan_launch(
    cudaStream_t stream,
    const io_type* q, const io_type* k_slot,
    const io_type* v, const io_type* alpha,
    io_type* output,
    int batch, int seq_len,
    int num_heads, int num_slots, int head_dim
) {
    dim3 grid(batch, num_heads);
    dim3 block(num_slots);
    // scores[m] + output_accum[d] + reduce_buf[2]
    size_t smem_bytes = (num_slots + head_dim + 2) * sizeof(float);

    fused_gsa_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        q, k_slot, v, alpha, output,
        batch, seq_len, num_heads, num_slots, head_dim
    );

    return (int)cudaGetLastError();
}

}  // extern "C"

#endif  // !EXLA_FFI

// ============================================================================
// XLA FFI integration
// ============================================================================

#ifdef EXLA_FFI

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {  // anonymous namespace — prevents symbol collision between f32/bf16

ffi::Error fused_gsa_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> q,        // [B, T, H, d]
    ffi::Buffer<FFI_IO_TYPE> k_slot,   // [B, T, H, m]
    ffi::Buffer<FFI_IO_TYPE> v,        // [B, T, H, d]
    ffi::Buffer<FFI_IO_TYPE> alpha,    // [B, T, H]
    ffi::ResultBuffer<FFI_IO_TYPE> output  // [B, T, H*d]
) {
    auto q_dims = q.dimensions();
    int batch    = static_cast<int>(q_dims[0]);
    int seq_len  = static_cast<int>(q_dims[1]);
    int num_heads = static_cast<int>(q_dims[2]);
    int head_dim = static_cast<int>(q_dims[3]);

    auto ks_dims = k_slot.dimensions();
    int num_slots = static_cast<int>(ks_dims[3]);

    dim3 grid(batch, num_heads);
    dim3 block(num_slots);
    size_t smem_bytes = (num_slots + head_dim + 2) * sizeof(float);

    fused_gsa_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const io_type*>(q.untyped_data()),
        reinterpret_cast<const io_type*>(k_slot.untyped_data()),
        reinterpret_cast<const io_type*>(v.untyped_data()),
        reinterpret_cast<const io_type*>(alpha.untyped_data()),
        reinterpret_cast<io_type*>(output->untyped_data()),
        batch, seq_len, num_heads, num_slots, head_dim
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

}  // anonymous namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    HANDLER_SYMBOL(fused_gsa_scan), fused_gsa_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // q
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // k_slot
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // v
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // alpha
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // output
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_gsa_scan_" PRECISION_SUFFIX, "CUDA", HANDLER_SYMBOL(fused_gsa_scan));

#endif  // EXLA_FFI
