# ExPhil + Edifice CUDA Integration Plan

## Overview

This document covers two objectives:
1. **Architecture Gap Analysis** — Edifice architectures ExPhil should adopt but hasn't yet
2. **Fused Kernel Roadmap** — What CUDA work is needed per architecture to hit 60 FPS

### Current State

**ExPhil** has 48 valid backbones in `@valid_backbones` (config.ex).
**Edifice** has 234 registered architectures across 26 families.

ExPhil's 60 FPS target = **<16ms inference** at batch=1, embed=256, seq_len=32, layers=2.

**Currently only 2/35 sequence architectures hit 60 FPS at seq_len=32:**
- `gated_ssm` — 12.96ms (77.2 FPS)
- `fnet` — 14.56ms (68.7 FPS)

---

## Part 1: Edifice Architectures ExPhil Should Add

### Currently in ExPhil (48 backbones)

```
mlp, lstm, gru, mamba, mamba_nif, mamba_cumsum, mamba_hillis_steele,
mamba_ssd, gated_ssm, attention, sliding_window, lstm_hybrid, hybrid,
jamba, zamba, griffin, hawk, xlstm, xlstm_slstm, xlstm_mlstm, retnet,
rwkv, gla, hgrn, s5, s4, s4d, h3, performer, deltanet, fnet, perceiver,
ttt, hopfield, ntm, reservoir, snn, bayesian, decision_transformer,
liquid, kan, min_gru, min_lstm, tcn, mamba3, hyena, titans, gated_deltanet
```

### Tier 1: Should Add Now (Sequence Models — Directly Relevant)

These are architectures Edifice has that process sequences and are realistic ExPhil
backbone candidates. They either show competitive inference speed or have interesting
quality/speed tradeoffs.

| Architecture | Family | Why Add | Expected Latency | Notes |
|---|---|---|---|---|
| `native_recurrence` | recurrent | 3 GRU variants (elu_gru, real_gru, diag_linear) from NativeRecurrence paper. Simple, fast. | ~15-25ms (est.) | Similar to min_gru, worth benchmarking |
| `longhorn` | ssm | Drop-in Mamba replacement, no forget gate, from closed-form online recall | ~20-30ms (est.) | Uses same parallel scan as Mamba |
| `samba` | ssm | Hybrid Mamba+SWA+MLP, beats Transformers on short+long context | ~25-40ms (est.) | First hybrid to beat transformers across context lengths |
| `hymba` | ssm | Hymba hybrid architecture | ~25-40ms (est.) | SSM+attention hybrid |
| `gss` | ssm | Gated State Space model | ~15-25ms (est.) | Simpler SSM variant |
| `delta_product` | recurrent | Multi-step DeltaNet via Householder products | ~30-60ms (est.) | Extends existing DeltaNet |
| `huginn` | recurrent | Depth-recurrent transformer with adaptive iteration | ~30-50ms (est.) | Latent reasoning capability |
| `deep_res_lstm` | recurrent | Deep residual LSTM | ~200-400ms (est.) | Same LSTM bottleneck but worth having |
| `gla_v2` | attention | Gated Linear Attention v2 | ~20-35ms (est.) | Improved GLA |
| `hgrn_v2` | attention | HGRN v2 | ~20-35ms (est.) | Improved HGRN |
| `ttt_e2e` | recurrent | End-to-end TTT | ~150-300ms (est.) | May benefit from kernel fusion |
| `gsa` | attention | Gated Slot Attention — linear time, fixed slots | ~20-35ms (est.) | Linear complexity |
| `rla` | attention | Residual Linear Attention | ~20-35ms (est.) | Corrects linear attention errors |
| `nha` | attention | Native Hybrid Attention | ~20-35ms (est.) | Per-layer linear vs full selection |
| `fox` | attention | Forgetting Transformer — learnable forget on softmax | ~18-30ms (est.) | Bounded memory in transformers |
| `log_linear` | attention | O(log T) space attention | ~18-30ms (est.) | Memory-quality tradeoff |
| `laser` | attention | exp(V) attention for larger gradients | ~18-30ms (est.) | Low complexity |
| `moba` | attention | Mixture of Block Attention | ~18-30ms (est.) | Production-proven (Kimi) |
| `tnn` | attention | Toeplitz Neural Network | ~15-25ms (est.) | O(n log n), good extrapolation |
| `coconut` | meta | Continuous chain of thought (latent reasoning) | ~20-35ms (est.) | BFS reasoning without text gen |
| `miras` | recurrent | Memory variants (Moneta, Yaad, Memora) | ~30-60ms (est.) | Multiple memory mechanisms |
| `mixture_of_mamba` | ssm | Modality-aware Mamba sparsity | ~25-40ms (est.) | Per-modality SSM routing |

**Total: 22 architectures to add to ExPhil's `@valid_backbones`.**

### Tier 2: Consider Later (Interesting but Lower Priority)

| Architecture | Family | Why Wait |
|---|---|---|
| `decoder_only` | transformer | Standard transformer, unlikely to hit 16ms at seq=32 |
| `conformer` | attention | Audio-oriented, may not suit Melee state |
| `mega` | attention | Older architecture, superseded by newer variants |
| `based` | attention | Worth testing but less proven |
| `bimamba` | ssm | Bidirectional — not ideal for causal inference |
| `striped_hyena` | ssm | Complex hybrid, may be slow |
| `ss_transformer` | ssm | SSM-Transformer hybrid |
| `hyena_v2` | attention | Hyena improvement, test if faster |
| `xlstm_v2` | recurrent | If xlstm works, try v2 |
| `mixture_of_recursions` | meta | Depth routing — interesting for adaptive compute |
| `mixture_of_expert_depths` | meta | Depth routing variant |
| `medusa` | inference | Speculative decoding — useful post-training |

### Tier 3: Not Applicable to ExPhil

Vision, graph, audio, detection, generative, interpretability, scientific, robotics
families are not backbone candidates for the Melee sequence modeling task.

---

## Part 2: Fused CUDA Kernel Roadmap

### Current Implementation Strategy

The plan is two-phase:

**Phase A: NIF Bridge (Current)**
- CUDA kernels compiled as shared libraries (`libedifice_cuda_kernels.so`)
- Called via Erlang NIF from Elixir
- Device pointer extraction from EXLA tensors → kernel launch → wrap result back
- GC-tracked memory management
- **Status**: Working for MinGRU + MinLSTM (P0 complete)
- **Limitation**: Requires `cudaDeviceSynchronize` before returning (blocks until done)

**Phase B: EXLA Fork (Next)**
- Fork `elixir-nx/nx`, add GPU custom calls via XLA FFI
- Kernels compiled directly into EXLA's NIF .so
- Zero-copy: XLA manages all memory, kernels run on XLA's CUDA stream
- No synchronization overhead (kernel stays on GPU timeline)
- Prototype exists on `gpu-custom-calls` branch

### Per-Architecture Kernel Analysis

#### Already Done

| Architecture | Kernel | Status | Latency Before | Latency After |
|---|---|---|---|---|
| `min_gru` | `fused_mingru_scan.cu` | NIF working | 84.75ms | ~12ms (est.) |
| `min_lstm` | `fused_minlstm_scan.cu` | NIF working | 74.79ms | ~14ms (est.) |

#### P0: Highest Value — Simple Scans (NIF first, then EXLA)

**3. NativeRecurrence variants** (elu_gru, real_gru, diag_linear)
- **What**: Three simple GRU-like sequential scans
- **Recurrence**:
  - elu_gru: `h = (1-z)*h + z*(1+elu(c))`
  - real_gru: `h = (1-z)*h + z*c` (identical to MinGRU)
  - diag_linear: `h = sigmoid(a)*h + b`
- **Kernel effort**: ~100 lines each, near-copy of MinGRU kernel
- **Expected speedup**: 5-8x
- **Why P0**: Trivial to implement given existing MinGRU template

**4. Liquid / LiquidS4**
- **What**: Liquid Time-Constant network — ODE-based recurrence
- **Recurrence**: `h_t = h_{t-1} + dt * (-h_{t-1}/tau + f(x_t))` (Euler step)
- **Inputs**: x_t projections, tau parameters, dt
- **Kernel effort**: ~150 lines (slightly more complex gate math)
- **Expected speedup**: 5-6x (89.95ms → ~15ms)
- **Why P0**: Currently 90ms, simple Euler integration inside scan loop

#### P1: Medium Value — More Complex Scans

**5. Mamba Selective Scan**
- **What**: Port existing `selective_scan_kernel.cu` from old CustomCall API to XLA FFI
- **Recurrence**: `h_t = A_bar*h_{t-1} + B*x_t` with input-dependent discretization
- **Inputs**: x, dt, A, B, C tensors
- **Kernel effort**: ~200 lines (already written, needs API migration)
- **Expected speedup**: 3-4x (58ms → ~15ms)
- **Complication**: State dimension per thread (32 elements), register pressure

**6. DeltaNet**
- **What**: Linear attention with delta rule memory updates
- **Recurrence**: `S_t = S_{t-1} + beta*(v - S_{t-1}@k) @ k^T`, `o_t = S_t@q_t`
- **Inputs**: Q, K, V, beta [batch, seq, num_heads, head_dim]
- **State**: S [batch, num_heads, head_dim, head_dim] — matrix per head
- **Kernel effort**: ~250 lines
- **Expected speedup**: 3-5x
- **Complication**: Matrix state means O(head_dim^2) registers per thread per head.
  Need to either: (a) use shared memory, or (b) split heads across blocks.

**7. Gated DeltaNet**
- **What**: DeltaNet with gated output
- **Recurrence**: Same as DeltaNet + output gate
- **Kernel effort**: ~270 lines (DeltaNet + gate)
- **Expected speedup**: 3-5x
- **Complication**: Same as DeltaNet

**8. DeltaProduct**
- **What**: Multi-step DeltaNet via products of Householder transformations
- **Recurrence**: Chain of Householder updates to memory matrix
- **Kernel effort**: ~300 lines
- **Expected speedup**: 3-5x
- **Complication**: Householder products are more compute-intensive

#### P2: Lower Value — Complex Cells

**9. LSTM (fused cell)**
- **What**: Fuse all 4 gates (i, f, o, g) + cell update + output into one kernel per timestep
- **Recurrence**:
  ```
  i,f,o,g = split(W_x@x + W_h@h + b)
  c = sigmoid(f)*c + sigmoid(i)*tanh(g)
  h = sigmoid(o)*tanh(c)
  ```
- **Kernel effort**: ~350 lines
- **Expected speedup**: 5-10x (502ms → 50-100ms) — **still won't hit 16ms**
- **Complication**: Hidden-to-hidden matmul (`W_h@h`) inside the loop is the real
  bottleneck. Options: (a) use cuBLAS for matmul + fuse the rest, (b) implement
  matmul in shared memory (feasible for hidden≤256). This is why cuDNN exists.
- **Alternative**: Investigate XLA cuDNN RNN integration

**10. GRU (fused cell)**
- **What**: Same approach as LSTM, 3 gates instead of 4
- **Kernel effort**: ~300 lines
- **Expected speedup**: 5-10x (381ms → 40-80ms) — **still won't hit 16ms**
- **Complication**: Same hidden-to-hidden matmul problem

**11. sLSTM (xLSTM variant)**
- **What**: Exponential gating with log-domain stabilization
- **Recurrence**: i,f,o gates with exp() activation, max-tracking for stability
- **Kernel effort**: ~300 lines
- **Expected speedup**: 2-4x
- **Complication**: Log-domain math (exp, log, max) adds numerical edge cases

**12. TTT (Test-Time Training)**
- **What**: Inner model weight updates per timestep
- **Recurrence**:
  ```
  pred = LayerNorm(W@k)
  grad = eta * (pred - v) @ k^T
  W = W - grad
  o = W@q
  ```
- **Kernel effort**: ~350 lines
- **Expected speedup**: 2-4x (160ms → 40-80ms)
- **Complication**: W is [inner_size, inner_size] per batch — massive register/shared
  memory requirement. May need to tile W into shared memory blocks.

#### P3: Parallel Scan Architectures (EXLA-side only)

These already use parallel/Blelloch scan in Elixir. Fused kernels would help but
the gain is smaller since the scan structure is already efficient.

**13. Mamba variants** (mamba_cumsum, mamba_hillis_steele, mamba_ssd)
- Already ~20-25ms. Fused kernel could push to ~12-15ms.
- Mamba's parallel scan is already well-optimized in EXLA.
- **Strategy**: Port selective scan kernel to EXLA FFI (same as #5 above).

**14. Longhorn**
- Uses same parallel scan infra as Mamba
- Same story: already decent, fused kernel is incremental

**15. S4/S4D/S5/H3**
- These use FFT-based convolution (not sequential scan)
- Fusion opportunity is in the FFT → gating → output pipeline
- Lower priority — the FFT is already handled by cuFFT via XLA

#### Not Candidates for Fused Kernels

| Architecture | Why Not |
|---|---|
| `gated_ssm` | Already 12.96ms — fast enough via element-wise ops |
| `fnet` | Uses FFT, already 14.56ms — handled by cuFFT |
| `attention`, `gqa`, etc. | Matmul-dominated, FlashAttention is the right optimization |
| `reservoir` | No learned recurrence (random fixed weights) |
| `mlp`, `kan` | No temporal processing |
| `snn` | Spike-based, different optimization domain |
| `mLSTM` (xLSTM) | Uses D-matrix cumsum formulation, not sequential scan |
| `huginn` | Iterates over depth, not timesteps — transformer blocks inside |
| `samba` | Hybrid — fuse the Mamba component (#5), rest is parallel |
| `jamba`, `zamba` | Hybrid — same story, fuse inner Mamba block |

---

## Part 3: Implementation Phases

### Phase A: NIF Bridge Iteration (Current → Near-term)

Extend the existing NIF bridge pattern to more architectures.

```
edifice/
├── native/cuda/
│   ├── fused_mingru_scan.cu      ✓ Done
│   ├── fused_minlstm_scan.cu     ✓ Done
│   ├── fused_native_rec_scan.cu  ← 3 variants (elu_gru, real_gru, diag_linear)
│   ├── fused_liquid_scan.cu      ← Euler-step ODE scan
│   ├── fused_deltanet_scan.cu    ← Matrix-state linear attention
│   ├── test_kernels.cu           ✓ Done
│   └── bench_kernels.cu          ✓ Done
├── c_src/
│   └── edifice_cuda_nif.c        ← Add new NIF functions
└── lib/edifice/cuda/
    ├── nif.ex                    ← Add new NIF bindings
    └── fused_scan.ex             ← Add dispatch for new kernels
```

**NIF Bridge development loop:**
1. Write kernel `.cu` file (copy from MinGRU template)
2. Add test case in `test_kernels.cu`
3. Add bench case in `bench_kernels.cu`
4. Build with `make` in `native/cuda/`
5. Run standalone GPU tests: `./build/test_kernels`
6. Add NIF function in `edifice_cuda_nif.c`
7. Add Elixir binding in `nif.ex`
8. Add dispatch in `fused_scan.ex`
9. Wire into architecture module (add fused path)
10. Run Elixir tests to verify correctness against unfused

**Order of implementation:**
1. NativeRecurrence (elu_gru, real_gru, diag_linear) — trivial, proves pattern scales
2. Liquid — simple ODE scan, high value (90ms → ~15ms)
3. DeltaNet — first matrix-state kernel, proves the harder pattern
4. GatedDeltaNet — extend DeltaNet kernel

### Phase B: EXLA Fork (Medium-term)

Once NIF kernels are validated, port them to EXLA for zero-copy integration.

**Steps:**
1. Fork `elixir-nx/nx` (or rebase `gpu-custom-calls` branch)
2. Clean up the prototype `gpu_add.cu` → production-quality patterns
3. Port each validated NIF kernel to XLA FFI format:
   - Add `.cu` to `exla/c_src/exla/custom_calls/`
   - Add `Value.fused_*` function to `exla/lib/exla/mlir/value.ex`
   - Register via `XLA_FFI_REGISTER_HANDLER` with `"CUDA"` platform
4. Wire into Nx/Defn custom call mechanism
5. Update Edifice dispatch to detect EXLA fork and use XLA FFI path
6. Benchmark: compare NIF path vs EXLA FFI path (expect ~10-20% improvement from
   eliminating cudaDeviceSynchronize + stream synchronization)

**EXLA FFI kernel template:**
```cuda
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;

__global__ void my_kernel(...) { /* GPU code */ }

ffi::Error my_impl(cudaStream_t stream, ffi::Buffer<ffi::F32> input,
                   ffi::ResultBuffer<ffi::F32> output) {
    // Launch kernel on XLA's stream — no sync needed
    my_kernel<<<grid, block, 0, stream>>>(...);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(my_handler, my_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>());

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_my_kernel_f32", "CUDA", my_handler);
```

**EXLA fork file layout:**
```
nx/exla/c_src/exla/custom_calls/
├── fused_mingru_scan.cu
├── fused_minlstm_scan.cu
├── fused_native_rec_scan.cu
├── fused_liquid_scan.cu
├── fused_selective_scan.cu      (Mamba, ported from ExPhil)
├── fused_deltanet_scan.cu
├── fused_lstm_cell.cu           (P2, if needed)
├── flash_attention.cu           (ported from ExPhil NIF)
└── gpu_add.cu                   (existing prototype)
```

### Phase C: Stretch Goals

1. **bf16/f16 kernel variants** — 2x memory bandwidth, critical for larger models
2. **Backward pass kernels** — Training-time fused backward scans
3. **Multi-layer fusion** — Keep state in registers across layers (avoid global memory round-trip)
4. **Triton-style autotune** — Try multiple block sizes, pick fastest
5. **PR to upstream EXLA** — If clean enough, contribute GPU custom call infra back to `elixir-nx/nx`

---

## Part 4: Priority Summary

### What to do next (ordered)

1. **Add 22 new backbones to ExPhil** (Tier 1 from Part 1)
   - Add to `@valid_backbones` in config.ex
   - Wire up in backbone builder
   - Run inference benchmark on all new backbones

2. **NIF kernels for NativeRecurrence** (P0, ~1-2 days)
   - Near-copy of MinGRU kernel
   - Three variants: elu_gru, real_gru, diag_linear
   - Validates that the pattern generalizes

3. **NIF kernel for Liquid** (P0, ~2-3 days)
   - ODE Euler-step scan
   - High value: 90ms → ~15ms

4. **NIF kernel for DeltaNet** (P1, ~3-5 days)
   - First matrix-state kernel
   - Validates the harder pattern (matrix-vector products in scan)

5. **Port Mamba selective scan to NIF** (P1, ~2-3 days)
   - Already written, just needs NIF wrapper
   - Brings Mamba from 58ms → ~15ms

6. **Set up EXLA fork** (P1, ~3-5 days)
   - Rebase gpu-custom-calls branch
   - Port MinGRU/MinLSTM kernels
   - Validate zero-copy path works end-to-end

7. **Port all NIF kernels to EXLA FFI** (P1-P2, ongoing)
   - Each kernel: ~1 day to port + test

8. **LSTM/GRU fused cells** (P2, ~5-7 days each)
   - Complex: hidden-to-hidden matmul
   - May never hit 16ms — consider cuDNN integration instead

### Success Metrics

| Metric | Current | Target | Kernel Needed |
|---|---|---|---|
| Architectures at 60 FPS (seq=32) | 2/35 | 10+/35 | P0 + P1 |
| MinGRU | 84.75ms | <16ms | fused_mingru_scan ✓ |
| MinLSTM | 74.79ms | <16ms | fused_minlstm_scan ✓ |
| NativeRecurrence | ~50-80ms (est.) | <16ms | fused_native_rec_scan |
| Liquid | 89.95ms | <16ms | fused_liquid_scan |
| DeltaNet | 172ms (est.) | <20ms | fused_deltanet_scan |
| Mamba | 58.01ms | <16ms | fused_selective_scan |
| GatedDeltaNet | ~180ms (est.) | <20ms | fused_gated_deltanet_scan |
| LSTM | 502ms | <50ms | fused_lstm_cell (won't hit 16ms) |

---

## Appendix: Architecture → Kernel Mapping

Quick reference for every ExPhil-relevant architecture and its kernel status.

| Architecture | Scan Type | Kernel Approach | Priority | Status |
|---|---|---|---|---|
| min_gru | Sequential, register | fused_mingru_scan.cu | P0 | Done (NIF) |
| min_lstm | Sequential, register | fused_minlstm_scan.cu | P0 | Done (NIF) |
| native_recurrence | Sequential, register | fused_native_rec_scan.cu | P0 | Planned |
| liquid | Sequential, register | fused_liquid_scan.cu | P0 | Planned |
| mamba | Parallel + selective | fused_selective_scan.cu | P1 | Port from ExPhil |
| delta_net | Sequential, matrix state | fused_deltanet_scan.cu | P1 | Planned |
| gated_delta_net | Sequential, matrix state | fused_gated_deltanet_scan.cu | P1 | Planned |
| delta_product | Sequential, Householder | fused_delta_product_scan.cu | P1 | Planned |
| lstm | Sequential, matmul | fused_lstm_cell.cu | P2 | Planned |
| gru | Sequential, matmul | fused_gru_cell.cu | P2 | Planned |
| slstm | Sequential, log-domain | fused_slstm_scan.cu | P2 | Planned |
| ttt | Sequential, matrix W | fused_ttt_scan.cu | P2 | Planned |
| ttt_e2e | Sequential, matrix W | fused_ttt_e2e_scan.cu | P2 | Planned |
| mamba_ssd | Parallel | Same as mamba | P1 | — |
| mamba_cumsum | Parallel | Same as mamba | P1 | — |
| mamba_hillis_steele | Parallel | Same as mamba | P1 | — |
| mamba3 | Parallel | Same as mamba | P1 | — |
| longhorn | Parallel | Similar to mamba | P3 | — |
| gated_ssm | Element-wise | Not needed | — | Already fast |
| fnet | FFT | Not needed | — | Already fast |
| attention / gqa | Matmul | FlashAttention | Low | Port from ExPhil |
| s4 / s4d / s5 / h3 | FFT conv | cuFFT handles it | Low | — |
| hyena | FFT conv | cuFFT handles it | Low | — |
| reservoir | None (fixed) | Not needed | — | Already fast |
| mlp / kan | None | Not needed | — | No temporal |
| retnet / rwkv | Linear attention | Consider after DeltaNet | P2 | — |
| gla / gla_v2 | Linear attention | Consider after DeltaNet | P2 | — |
| hgrn / hgrn_v2 | Gated recurrence | Consider after NativeRec | P2 | — |
| performer | Linear attention | Kernel approximation | Low | — |
| titans | Recurrent + attention | Multiple kernels needed | P2 | — |
| xlstm (sLSTM) | Sequential, log-domain | fused_slstm_scan.cu | P2 | — |
| xlstm (mLSTM) | D-matrix cumsum | Not a good kernel target | — | Use EXLA |
| huginn | Iterative depth | Not sequential scan | — | Use EXLA |
| samba | Hybrid (Mamba inside) | Fuse Mamba component | P1 | — |
| jamba / zamba | Hybrid (Mamba inside) | Fuse Mamba component | P1 | — |
| decision_transformer | Attention | FlashAttention | Low | — |
| snn | Spike-based | Different domain | — | — |
| hopfield / ntm | Associative memory | Special kernels | P3 | — |
| coconut | Iterative | Not sequential scan | — | Use EXLA |
| fox / log_linear / nha | Attention variants | FlashAttention variant | Low | — |
| laser / moba / mta | Attention variants | FlashAttention variant | Low | — |
| gsa / rla / tnn | Linear attention | Consider after DeltaNet | P2 | — |
| miras | Memory variants | Per-variant kernels | P2 | — |
