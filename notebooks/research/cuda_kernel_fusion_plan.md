# CUDA Kernel Fusion — Planning & Implementation Guide

## Purpose

This document is a self-contained reference for implementing fused CUDA kernels
for ExPhil's 60 FPS inference target. Written for use on the CUDA development
machine. Covers: bottleneck analysis, existing infrastructure, implementation
paths, and step-by-step instructions.

---

## 1. The Bottleneck: Why We Need Custom Kernels

### The 60 FPS Constraint

Melee runs at 60 FPS. Inference must complete in <16ms per frame. With Slippi
online's 18+ frame delay, we have slightly more budget, but 16ms is the hard
target for local play.

### Current Benchmark Results (NVIDIA GPU, EXLA/CUDA, embed=256, batch=1, layers=2)

**seq_len=32 (32-frame context window):**

| Rank | Architecture | Category | Avg (ms) | FPS | Status |
|------|-------------|----------|----------|-----|--------|
| 1 | gated_ssm | ssm | 12.96 | 77.2 | VIABLE |
| 2 | fnet | attention | 14.56 | 68.7 | VIABLE |
| 3 | jamba | ssm | 18.77 | 53.3 | Close |
| 4 | mamba_hillis_steele | ssm | 20.25 | 49.4 | Close |
| ... | ... | ... | ... | ... | |
| 22 | min_lstm | recurrent | 74.79 | 13.4 | 4.7x too slow |
| 24 | min_gru | recurrent | 84.75 | 11.8 | 5.3x too slow |
| 30 | ttt | recurrent | 160.0 | 6.2 | 10x too slow |
| 34 | gru | recurrent | 381.0 | 2.6 | 24x too slow |
| 35 | lstm | recurrent | 502.0 | 2.0 | 31x too slow |

**Only 2 of 35 architectures hit 60 FPS at seq_len=32.**

**seq_len=1 (single-step inference):**

| Rank | Architecture | Category | Avg (ms) | FPS | Status |
|------|-------------|----------|----------|-----|--------|
| 1 | reservoir | other | 1.19 | 839.2 | VIABLE |
| 2 | min_gru | recurrent | 10.13 | 98.7 | VIABLE |
| 3 | min_lstm | recurrent | 12.26 | 81.5 | VIABLE |
| 4 | liquid | recurrent | 13.15 | 76.1 | VIABLE |
| 5 | mamba_ssd | ssm | 13.56 | 73.7 | VIABLE |
| 6 | mamba | ssm | 13.81 | 72.4 | VIABLE |
| 9 | delta_net | recurrent | 15.10 | 66.2 | VIABLE |
| 10 | gated_ssm | ssm | 15.35 | 65.1 | VIABLE |

**10 of 34 architectures hit 60 FPS at seq_len=1.**

### Root Cause: Timestep Unrolling

The core problem is **timestep unrolling overhead**. Axon/XLA unrolls recurrent
sequences into individual kernel launches per timestep. Cross-seq_len comparison:

| Architecture | seq=1 | seq=32 | Slowdown Factor | Root Cause |
|-------------|-------|--------|-----------------|------------|
| min_gru | 10.13ms | 84.75ms | 8.5x | 32 kernel launches vs 1 |
| min_lstm | 12.26ms | 74.79ms | 7.7x | Same |
| liquid | 13.15ms | 89.95ms | 8.3x | Same |
| mamba | 13.81ms | 58.01ms | 4.2x | Same |
| lstm | 306.0ms | 502.0ms | 1.6x | Already slow at seq=1 |

Key insight: recurrent models are 4-8x faster at seq_len=1 because they avoid
the unrolling overhead. A fused scan kernel that processes the entire sequence
in a single kernel launch (keeping state in registers/SRAM across timesteps)
would eliminate this gap.

Exception: LSTM/GRU are slow even at seq_len=1 (150-500ms). These need both
the fused scan AND optimized gate computation (cuDNN-style fused LSTM cell).

### What XLA Gives You For Free

XLA already performs automatic kernel fusion:
- **Element-wise fusion**: Chains of pointwise ops (add, mul, sigmoid) fuse into one kernel
- **Reduce fusion**: Reductions after element-wise ops get fused
- **Multi-output fusion**: Multiple consumers of the same data can share a kernel

What XLA **cannot** fuse:
- The loop itself — each iteration is a separate kernel launch for matmul components
- Cross-timestep data flow (parallel scan / register persistence)
- Custom memory hierarchy (SRAM ↔ registers across loop iterations)
- The 3-5x speedup that hand-fused RNN/SSM kernels provide

---

## 2. Existing Infrastructure

### 2.1 ExPhil CUDA Kernels (Already Written)

**`exphil/native/xla_selective_scan/selective_scan_kernel.cu`**
- Mamba selective scan using old-style XLA CustomCall API (not FFI)
- Interface: `void SelectiveScan(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len)`
- Each thread handles one (batch, hidden) pair, scans through sequence sequentially
- State capped at 32 elements per thread (register arrays for h_state and A_diag)
- ScanParams struct packed in opaque data
- Compile: `nvcc -shared -o libxla_selective_scan.so selective_scan_kernel.cu -Xcompiler -fPIC`

**`exphil/native/flash_attention_nif/cuda/flash_attention.cu`**
- FlashAttention-2 forward pass (simplified, based on flash-attention-minimal)
- Uses NIF interface (not XLA custom call) — has host↔device copies (slow!)
- Tiled computation with BLOCK_SIZE=32, online softmax in registers
- Requires Ampere+ GPU (sm_80+)
- NOTE: This should be migrated to XLA FFI to eliminate host↔device copies

### 2.2 EXLA Custom Call Infrastructure (Existing in deps)

EXLA already uses XLA FFI (api_version: 4) for CPU custom calls:

**C++ side** (`deps/exla/c_src/exla/custom_calls/`):
- `qr_f32.cc`, `qr_f64.cc`, `qr_f16.cc`, `qr_bf16.cc`
- `lu_f32.cc`, `lu_f64.cc`, `lu_f16.cc`, `lu_bf16.cc`
- `eigh_f32.cc`, `eigh_f64.cc`

Pattern (from `qr_f32.cc`):
```cpp
#include "qr.h"

ffi::Error qr_cpu_custom_call_f32_impl(ffi::Buffer<ffi::F32> operand,
                                       ffi::ResultBuffer<ffi::F32> q,
                                       ffi::ResultBuffer<ffi::F32> r) {
  return qr_cpu_custom_call_impl<float, ffi::Buffer<ffi::F32>>(operand, q, r);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(qr_cpu_custom_call_f32,
                              qr_cpu_custom_call_f32_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "qr_cpu_custom_call_f32",
                         "Host", qr_cpu_custom_call_f32);
```

**Elixir side** (`deps/exla/lib/exla/mlir/value.ex`):
- `eigh/3`, `qr/3`, `lu/4` emit `stablehlo.custom_call` ops
- Uses `api_version: attr_i32(4)` (typed FFI)
- Only fires on `%EXLA.Client{platform: :host}` — GPU paths use XLA builtins

**Makefile** (`deps/exla/Makefile`):
- Already detects nvcc and compiles `.cu` files
- Wildcard picks up `$(wildcard $(EXLA_DIR)/custom_calls/*.cc)` automatically
- Has CUDA compile flags, link flags, nvcc integration

### 2.3 gpu-custom-calls Branch (Prototype in nx fork)

The `gpu-custom-calls` branch in the nx fork (`/home/dori/git/melee/nx`) contains
a proof-of-concept for GPU custom calls:

**`exla/c_src/exla/custom_calls/gpu_add.cu`** — Prototype vector add kernel:
```cpp
XLA_FFI_DEFINE_HANDLER_SYMBOL(gpu_add, gpu_add_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "exla_gpu_add_f32", "CUDA", gpu_add);
```

**`exla/lib/exla/mlir/value.ex`** — `Value.gpu_add/3` emits `stablehlo.custom_call`
**`exla/lib/exla/gpu_custom_call.ex`** — Documentation module
**`exla/Makefile`** — Modified to compile `.cu` in custom_calls with nvcc

This branch modifies ~64 files, ~1938 insertions, ~1022 deletions. It's a working
prototype but needs cleanup for production use.

### 2.4 XLA FFI Headers (Available)

The FFI headers are already in EXLA's build cache:
`deps/exla/cache/xla_extension/include/xla/ffi/api/ffi.h`

Key types available:
- `ffi::PlatformStream<cudaStream_t>` — Get CUDA stream (line ~1224)
- `ffi::Buffer<ffi::F32>` — Typed input buffer
- `ffi::ResultBuffer<ffi::F32>` — Typed output buffer
- `ffi::ScratchAllocator` — Temporary GPU memory
- `XLA_FFI_DEFINE_HANDLER_SYMBOL` / `XLA_FFI_REGISTER_HANDLER` — Registration macros

### 2.5 Benchmark Script

`edifice/bench/inference_latency.exs` — 4-phase benchmark:
1. GPU warmup (EXLA BFC allocator init)
2. EXLA compilation (JIT all architectures)
3. Quick latency scan (50 iterations)
4. Benchee deep dive (requires MIX_ENV=dev)

Configurable via env vars: `BENCH_EMBED`, `BENCH_SEQ_LEN`, `BENCH_BATCH`, `BENCH_LAYERS`

Usage:
```bash
EXLA_TARGET=cuda XLA_TARGET=cuda12 mix run bench/inference_latency.exs
BENCH_SEQ_LEN=1 EXLA_TARGET=cuda XLA_TARGET=cuda12 mix run bench/inference_latency.exs
```

---

## 3. Implementation Strategy

### Priority Ranking

Based on benchmark data, here are the highest-value kernel fusion targets:

| Priority | Kernel | Architectures Unblocked | Expected Speedup | Effort |
|----------|--------|------------------------|-------------------|--------|
| **P0** | Fused MinGRU scan | min_gru (84.75ms→~12ms) | 7-8x | Medium |
| **P0** | Fused MinLSTM scan | min_lstm (74.79ms→~14ms) | 5-7x | Medium |
| **P1** | Fused Mamba selective scan | mamba, mamba_ssd, mamba_cumsum (58ms→~15ms) | 3-4x | Medium-High |
| **P1** | Fused liquid/LiquidS4 scan | liquid (89.95ms→~15ms) | 5-6x | Medium |
| **P2** | Fused LSTM cell | lstm (502ms→~50ms?) | 5-10x | High (complex gates) |
| **P2** | Fused GRU cell | gru (381ms→~40ms?) | 5-10x | High (complex gates) |
| **P3** | Fused delta_net | delta_net (172ms→~20ms?) | 5-8x | Medium |
| **P3** | Fused xlstm cell | xlstm (91.87ms→~20ms?) | 4-5x | High |
| **Low** | Flash Attention (XLA FFI) | gqa, nystromformer, etc. | 1.5-2x | Medium (port existing) |

### Why MinGRU/MinLSTM First (P0)

1. **Simplest recurrence** — MinGRU has only 2 gates, MinLSTM has 3. No hidden-to-hidden matrix multiply (the bottleneck in classic RNNs).
2. **Highest ROI** — Currently 84ms, target ~12ms. That's a 7x speedup from a relatively simple kernel.
3. **Proves the pipeline** — If we can get a fused scan kernel running through EXLA, the pattern applies to every other recurrent architecture.
4. **Already 60 FPS viable at seq=1** — We know the per-step compute is fast enough; we just need to eliminate kernel launch overhead.

### MinGRU Recurrence (reference)

```
# For each timestep t:
z_t = sigmoid(W_z @ x_t)        # Update gate
h_tilde = W_h @ x_t             # Candidate
h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde  # Interpolation
```

This is a parallel scan candidate:
- The recurrence is linear in h: `h_t = a_t * h_{t-1} + b_t`
- Where `a_t = (1 - z_t)` and `b_t = z_t * h_tilde`
- Parallel prefix scan can process all timesteps in O(log T) parallel steps

### MinLSTM Recurrence (reference)

```
f_t = sigmoid(W_f @ x_t)        # Forget gate
i_t = sigmoid(W_i @ x_t)        # Input gate
h_tilde = W_h @ x_t             # Candidate
# Normalize gates
f_t' = f_t / (f_t + i_t)
i_t' = i_t / (f_t + i_t)
h_t = f_t' * h_{t-1} + i_t' * h_tilde
```

Same parallel scan structure: `h_t = a_t * h_{t-1} + b_t` with
`a_t = f_t'` and `b_t = i_t' * h_tilde`.

---

## 4. Implementation Path: Fork EXLA + XLA FFI

This is the recommended approach. No upstream changes needed.

### Step 1: Set Up Development Environment

```bash
# Clone the nx fork (or use existing)
cd /home/dori/git/melee/nx
git checkout gpu-custom-calls  # Start from the existing prototype

# Or if starting fresh from upstream:
git clone https://github.com/elixir-nx/nx.git
cd nx
git checkout -b gpu-custom-calls

# Ensure CUDA toolkit is available
nvcc --version  # Need CUDA 12.0+
nvidia-smi      # Verify GPU visible

# Set up EXLA
cd exla
mix deps.get
EXLA_TARGET=cuda XLA_TARGET=cuda12 mix compile
```

### Step 2: Write the Fused MinGRU Kernel

Create `exla/c_src/exla/custom_calls/fused_mingru_scan.cu`:

```cuda
#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>

namespace ffi = xla::ffi;

// Fused MinGRU parallel scan kernel
// Inputs:
//   gates: [batch, seq_len, hidden] — pre-computed sigmoid(W_z @ x)
//   candidates: [batch, seq_len, hidden] — pre-computed W_h @ x
//   h0: [batch, hidden] — initial hidden state
// Output:
//   output: [batch, seq_len, hidden] — all hidden states
//
// Strategy: Sequential scan per (batch, hidden) thread.
// For seq_len <= 64, this is faster than parallel prefix scan
// due to lower overhead.

__global__ void fused_mingru_scan_kernel(
    const float* __restrict__ gates,       // [B, T, H] sigmoid values
    const float* __restrict__ candidates,  // [B, T, H] candidate values
    const float* __restrict__ h0,          // [B, H] initial state
    float* __restrict__ output,            // [B, T, H] all hidden states
    int batch, int seq_len, int hidden
) {
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || h >= hidden) return;

    // Load initial state into register
    float h_state = h0[b * hidden + h];

    // Sequential scan through timesteps
    for (int t = 0; t < seq_len; t++) {
        int idx = b * seq_len * hidden + t * hidden + h;
        float z = gates[idx];
        float h_tilde = candidates[idx];

        // MinGRU update: h = (1-z)*h + z*h_tilde
        h_state = (1.0f - z) * h_state + z * h_tilde;

        output[idx] = h_state;
    }
}

// XLA FFI Handler
ffi::Error fused_mingru_scan_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> gates,
    ffi::Buffer<ffi::F32> candidates,
    ffi::Buffer<ffi::F32> h0,
    ffi::ResultBuffer<ffi::F32> output
) {
    auto dims = gates.dimensions();
    int batch = dims[0];
    int seq_len = dims[1];
    int hidden = dims[2];

    int threads_per_block = min(hidden, 256);
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);

    fused_mingru_scan_kernel<<<grid, block, 0, stream>>>(
        (const float*)gates.untyped_data(),
        (const float*)candidates.untyped_data(),
        (const float*)h0.untyped_data(),
        (float*)output->untyped_data(),
        batch, seq_len, hidden
    );

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_mingru_scan, fused_mingru_scan_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // gates
        .Arg<ffi::Buffer<ffi::F32>>()   // candidates
        .Arg<ffi::Buffer<ffi::F32>>()   // h0
        .Ret<ffi::Buffer<ffi::F32>>()   // output
);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
    "exla_fused_mingru_scan_f32", "CUDA", fused_mingru_scan);
```

### Step 3: Register on Elixir Side

Add to `exla/lib/exla/mlir/value.ex` (or a new module):

```elixir
def fused_mingru_scan(func, gates, candidates, h0) do
  # gates: [batch, seq_len, hidden]
  # candidates: [batch, seq_len, hidden]
  # h0: [batch, hidden]
  # output: [batch, seq_len, hidden]
  output_type = MLIR.Type.tensor(MLIR.Value.type(gates))

  op(func, "stablehlo.custom_call", [gates, candidates, h0], [output_type],
    attributes: [
      call_target_name: attr_string("exla_fused_mingru_scan_f32"),
      api_version: attr_i32(4)
    ]
  )
end
```

### Step 4: Wire into Edifice/ExPhil

Option A: Use via `Nx.Defn.Kernel` custom operator
Option B: Conditional dispatch in MinGRU module based on backend

```elixir
# In the MinGRU module, detect EXLA backend and dispatch:
defn fused_scan(gates, candidates, h0) do
  # This would call through to the custom kernel
  custom_call("exla_fused_mingru_scan_f32", [gates, candidates, h0],
    result_shape: Nx.shape(gates))
end
```

### Step 5: Compile and Test

```bash
cd /home/dori/git/melee/nx/exla

# The Makefile should auto-detect .cu files in custom_calls/
EXLA_TARGET=cuda XLA_TARGET=cuda12 mix compile

# Test with a simple benchmark
EXLA_TARGET=cuda XLA_TARGET=cuda12 mix run -e '
  gates = Nx.random_uniform({2, 32, 256})
  candidates = Nx.random_uniform({2, 32, 256})
  h0 = Nx.broadcast(0.0, {2, 256})

  # Call through custom call mechanism
  output = EXLA.fused_mingru_scan(gates, candidates, h0)
  IO.inspect(Nx.shape(output))
'
```

---

## 5. Makefile Changes

The EXLA Makefile already compiles `.cu` files. Verify these sections exist:

```makefile
# In EXLA Makefile, ensure:
# 1. nvcc is detected
NVCC := $(shell which nvcc 2>/dev/null)

# 2. .cu files in custom_calls are compiled
CUDA_CUSTOM_CALL_SRCS := $(wildcard $(EXLA_DIR)/custom_calls/*.cu)
CUDA_CUSTOM_CALL_OBJS := $(CUDA_CUSTOM_CALL_SRCS:.cu=.o)

# 3. nvcc compile rule
%.o: %.cu
	$(NVCC) -c $< -o $@ $(CUDA_FLAGS) --compiler-options '-fPIC'

# 4. Link into the NIF .so
$(NIF_SO): $(OBJS) $(CUDA_CUSTOM_CALL_OBJS)
	$(CXX) -shared -o $@ $^ $(LDFLAGS)
```

If starting from the gpu-custom-calls branch, this is already done. If starting
from upstream EXLA, these modifications are needed.

---

## 6. Kernel Variants to Implement

### 6.1 Fused MinGRU Scan (P0)

**Input**: Pre-computed gates and candidates (matmul done by XLA)
**Kernel**: Sequential scan per (batch, hidden) thread, state in registers
**Why sequential**: For seq_len=32, a sequential scan with register-resident state
is faster than a parallel prefix scan (which has 2x the memory traffic).
**Expected speedup**: 7-8x (84ms → 10-12ms)

### 6.2 Fused MinLSTM Scan (P0)

**Input**: Pre-computed forget/input gates and candidates
**Kernel**: Same pattern as MinGRU but with normalized gate interpolation
**Extra step**: Normalize f/(f+i) and i/(f+i) inside the kernel
**Expected speedup**: 5-7x (74ms → 12-14ms)

### 6.3 Fused Mamba Selective Scan (P1)

**Input**: x, dt, A, B, C tensors
**Kernel**: Already written in `exphil/native/xla_selective_scan/selective_scan_kernel.cu`
**Work needed**: Port from old-style CustomCall API to XLA FFI, integrate with EXLA
**Expected speedup**: 3-4x (58ms → ~15ms)

### 6.4 Fused Liquid / LiquidS4 Scan (P1)

**Input**: Pre-computed gates + ODE step parameters
**Kernel**: Sequential scan with Euler integration step inside
**Expected speedup**: 5-6x (90ms → ~15ms)

### 6.5 Fused LSTM Cell (P2)

**Input**: x_t, h_{t-1}, c_{t-1}, weight matrices
**Kernel**: Fuse all 4 gates (i,f,o,g) + cell update + output in one kernel
**Complexity**: Needs to handle matmul (or split matmul to CUBLAS + fuse the rest)
**Reference**: cuDNN LSTM kernel achieves 3.5-5x over unfused PyTorch
**Expected speedup**: 5-10x (502ms → 50-100ms, still may not hit 16ms)

### 6.6 Flash Attention via XLA FFI (Low)

**Input**: Q, K, V tensors
**Kernel**: Already written in `exphil/native/flash_attention_nif/cuda/flash_attention.cu`
**Work needed**: Port from NIF interface to XLA FFI (eliminate host↔device copies)
**Expected speedup**: 1.5-2x for attention architectures

---

## 7. Alternative Approaches

### 7.1 Device Buffer Pointer Interop (No Fork Required)

EXLA already has `get_buffer_device_pointer` and `create_buffer_from_device_pointer`
(including CUDA IPC handles) in the NIF layer. This means you can:

1. Get a raw GPU pointer from an Nx tensor via EXLA
2. Call a CUDA kernel through a separate NIF or Port
3. Wrap the result back into an Nx tensor

Hacky, but doesn't require forking EXLA. Good for prototyping.

### 7.2 cuDNN RNN Backend

XLA already calls into cuDNN for matmuls. For standard LSTM/GRU, cuDNN has
optimized fused RNN kernels. Investigate whether EXLA can be configured to use
cuDNN's RNN API directly (may require changes to how EXLA compiles while_loop).

### 7.3 Pure Defn Optimization

For some architectures, restructuring the Elixir code to be more XLA-friendly
may help without custom kernels:
- Vectorize where possible (avoid explicit loops)
- Use `Nx.Defn.while` instead of Enum.reduce for recurrence
- Pre-compute all gate matrices as a single batched matmul before the scan

---

## 8. Testing Strategy

### 8.1 Correctness Tests

For each fused kernel, verify against the unfused Elixir implementation:

```elixir
# Generate random inputs
key = Nx.Random.key(42)
{gates, key} = Nx.Random.uniform(key, shape: {2, 32, 256})
{candidates, key} = Nx.Random.uniform(key, shape: {2, 32, 256})
h0 = Nx.broadcast(0.0, {2, 256})

# Run unfused (existing Edifice implementation)
unfused_output = MinGRU.scan_unfused(gates, candidates, h0)

# Run fused (custom CUDA kernel)
fused_output = MinGRU.scan_fused(gates, candidates, h0)

# Compare
diff = Nx.subtract(unfused_output, fused_output) |> Nx.abs() |> Nx.reduce_max()
assert Nx.to_number(diff) < 1.0e-4  # f32 tolerance
```

### 8.2 Latency Tests

Use the existing benchmark script:

```bash
# Before (baseline)
EXLA_TARGET=cuda XLA_TARGET=cuda12 mix run bench/inference_latency.exs > /tmp/bench_before.txt 2>&1

# After (with fused kernels)
EXLA_TARGET=cuda XLA_TARGET=cuda12 mix run bench/inference_latency.exs > /tmp/bench_after.txt 2>&1

# Compare
diff /tmp/bench_before.txt /tmp/bench_after.txt
```

### 8.3 Edge Cases

- seq_len=1 (should be no slower than unfused)
- seq_len=256 (stress test)
- batch=1 vs batch=32 (inference vs training)
- hidden=64 vs hidden=512 (small vs large models)
- NaN/Inf inputs (gates should be clamped to [0,1] by sigmoid)
- Zero initial state
- Very long sequences (seq_len=1024+)

---

## 9. File Layout (Proposed)

```
nx/exla/
├── c_src/exla/custom_calls/
│   ├── fused_mingru_scan.cu      # P0: MinGRU fused scan
│   ├── fused_minlstm_scan.cu     # P0: MinLSTM fused scan
│   ├── fused_selective_scan.cu   # P1: Mamba selective scan (port from ExPhil)
│   ├── fused_liquid_scan.cu      # P1: Liquid/LiquidS4 scan
│   ├── fused_lstm_cell.cu        # P2: Classic LSTM fused cell
│   ├── fused_gru_cell.cu         # P2: Classic GRU fused cell
│   ├── flash_attention.cu        # Low: Port from ExPhil NIF version
│   └── gpu_add.cu                # Existing prototype (keep for reference)
├── lib/exla/
│   ├── gpu_custom_call.ex        # Documentation + dispatch module
│   └── mlir/value.ex             # Add fused_* functions
└── Makefile                      # Already handles .cu compilation

exphil/
├── native/
│   ├── xla_selective_scan/       # Existing (will be superseded)
│   └── flash_attention_nif/      # Existing (will be superseded)
└── bench/
    └── inference_latency.exs     # Existing benchmark (no changes needed)
```

---

## 10. Setup Checklist for CUDA Machine

### Prerequisites

- [ ] NVIDIA GPU (Ampere+ preferred, sm_80+)
- [ ] CUDA 12.0+ toolkit installed
- [ ] cuDNN 8.6+ installed
- [ ] Erlang 27+ / Elixir 1.18+
- [ ] Git access to both nx fork and melee repo

### Environment Setup

```bash
# Verify CUDA
nvcc --version
nvidia-smi

# Clone repos
git clone <melee-repo-url>
cd melee

# Set up nx fork with gpu-custom-calls branch
cd nx
git checkout gpu-custom-calls

# Set up EXLA
cd exla
export EXLA_TARGET=cuda
export XLA_TARGET=cuda12
mix deps.get
mix compile  # This compiles the EXLA NIF with CUDA support

# Run baseline benchmark
cd ../../edifice
EXLA_TARGET=cuda XLA_TARGET=cuda12 mix run bench/inference_latency.exs > /tmp/baseline.txt 2>&1
cat /tmp/baseline.txt
```

### Development Loop

1. Write kernel in `nx/exla/c_src/exla/custom_calls/fused_*.cu`
2. Add Elixir binding in `nx/exla/lib/exla/mlir/value.ex`
3. `cd nx/exla && EXLA_TARGET=cuda XLA_TARGET=cuda12 mix compile`
4. Test correctness against unfused implementation
5. Run latency benchmark, compare with baseline
6. Iterate on kernel (block size, register usage, shared memory)

---

## 11. Success Criteria

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| MinGRU seq=32 | 84.75ms | <16ms | Fused scan kernel |
| MinLSTM seq=32 | 74.79ms | <16ms | Fused scan kernel |
| Mamba seq=32 | 58.01ms | <16ms | Fused selective scan |
| Liquid seq=32 | 89.95ms | <16ms | Fused scan kernel |
| Architectures at 60 FPS (seq=32) | 2/35 | 6+/35 | All P0+P1 kernels |
| LSTM seq=32 | 502ms | <50ms | Fused cell (may not hit 16ms) |

### Stretch Goals

- Parallel prefix scan variant for training (forward + backward)
- bf16/f16 kernel variants for 2x memory bandwidth
- Multi-layer fusion (keep state in registers across layers)
- Triton-style autotune for block sizes

---

## 12. References

- Mamba CUDA kernel: https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan_fwd_kernel.cuh
- FlashAttention-2: https://arxiv.org/abs/2307.08691
- XLA Custom Calls: https://openxla.org/xla/custom_call
- XLA FFI API: https://docs.jax.dev/en/latest/ffi.html
- JAX Custom GPU Ops: https://docs.jax.dev/en/latest/Custom_Operation_for_GPUs.html
- cuDNN RNN optimization: https://developer.nvidia.com/blog/optimizing-recurrent-neural-networks-cudnn-5/
- MinGRU paper (Were RNNs All We Needed?): https://arxiv.org/abs/2410.01201
- Parallel scan for RNNs: https://arxiv.org/abs/2311.06281
- PyTorch fused RNN: https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/
