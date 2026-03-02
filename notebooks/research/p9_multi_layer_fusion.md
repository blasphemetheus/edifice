# P9 — Multi-Layer Block Scan Fusion

## Overview

Multi-layer block scan kernels keep hidden state in registers across consecutive
layers, eliminating global memory round-trips between layers. The inter-layer
dense projections (LayerNorm + GEMV) run cooperatively via shared memory.

For a 4-layer MinGRU at H=256, the per-timestep output is written to global
memory once (final layer) instead of 4 times.

## What Already Exists

- `native/cuda/fused_mingru_block_scan.cu` — 4-layer MinGRU, full 3-tier dispatch
- `native/cuda/fused_minlstm_block_scan.cu` — 4-layer MinLSTM, full 3-tier dispatch
- Weight packing: `FusedScan.pack_block_weights/3` for MinGRU/MinLSTM
- EXLA custom calls: `Value.fused_mingru_block_scan/4`, `Value.fused_minlstm_block_scan/4`
- Architecture integration: `min_gru.ex` and `min_lstm.ex` have `build_fused_block` paths

### Existing Block Kernel Pattern

```
Grid: (batch, 1), Block: (H, 1, 1)    — one block per batch, one thread per hidden dim
Shared memory: 3×H floats             — input_shared, normed_shared, reduce_buf
Registers: float h_state[MAX_LAYERS]   — up to 16 layers of hidden state

for each timestep t:
    load x_val from global memory
    for each layer l:
        LayerNorm via parallel reduction (shared memory)
        GEMV: gate = W_z @ normed + b_z  (each thread reads all of normed_shared)
        GEMV: cand = W_h @ normed + b_h
        scan update: h_state[l] = f(h_state[l], gate, cand)
        residual: x_val += h_state[l]
    store x_val to global memory (once per timestep)
```

Constraint: H ≤ 256 (single CUDA block must cover entire hidden dim for
LayerNorm reduction + GEMV).

## Feasibility Analysis

### Feasible (scalar/small state per hidden dim)

| Scan Kernel | Per-Thread State | Architectures That Stack | Default Layers |
|---|---|---|---|
| MinGRU | 1 float (h) | MinGRU | 4 |
| MinLSTM | 1 float (h) | MinLSTM | 4 |
| **linear_scan** | 1 float (h) | Griffin RG-LRU, MEGA EMA, SSTransformer, HybridBuilder, GSS, MambaVision | 2-6 |
| **LSTM** | 2 floats (h+c) | DeepResLSTM, standard stacked LSTM | 2-3 |
| **GRU** | 1 float (h) | Standard stacked GRU | 2-3 |

### NOT Feasible (matrix state, too large for registers)

| Scan Kernel | State Size | Why Not |
|---|---|---|
| delta_net_scan | [D,D] per head | 4096 floats for D=64 |
| gated_delta_net_scan | [D,D] per head | Same |
| kda_scan | [D,D] per head | Same |
| rla_scan | 2×[D,D] per head | Double matrix state |
| ttt_scan | [D,D] per batch | SGD weight matrix |
| selective_scan | [hidden, state_size] | Multi-dimensional state |
| slstm_scan | 2 floats BUT R@h needs shared mem for 4*H | Might work but complex |

## Memory Budget

### Register Usage

At H=256, each SM has 65,536 registers. One block = one batch element = 256 threads.

Per thread: `h_state[16]` = 16 registers + ~20 temporaries = ~36 registers total.
Total per block: 256 × 36 = 9,216 registers. Well within SM limits.

LSTM adds `c_state[16]` = 16 more registers → 52 per thread, 13,312 per block. Still fine.

### Shared Memory

- Linear/MinGRU/GRU blocks: 3 × H × 4 bytes = 3 × 256 × 4 = 3,072 bytes (3 KB)
- LSTM/GRU blocks: 4 × H × 4 bytes = 4,096 bytes (4 KB) — extra buffer for h_shared
- SM limit: 48 KB (Turing) or 100 KB (Ampere). No issue.

### Global Memory (Weight Packing)

| Kernel | Per-Layer Stride | 4 Layers @ H=256 |
|---|---|---|
| linear_block | 2H²+4H = 132,096+1024 = 133,120 | 532,480 B = 520 KB |
| lstm_block | 8H²+6H = 524,288+1536 = 525,824 | 2,103,296 B = 2 MB |
| gru_block | 6H²+5H = 393,216+1280 = 394,496 | 1,577,984 B = 1.5 MB |

All fit in L2 cache (typical 4-6 MB on consumer GPUs).

## Implementation Phases

### Phase 1 — Linear Scan Block Kernel

The linear scan (`h = a*h + b`) is the simplest recurrence and covers 6
architectures. No in-kernel nonlinearities — just multiply-accumulate.

**Per-layer weight layout:** `[W_a(H×H) | b_a(H) | W_b(H×H) | b_b(H) | γ(H) | β(H)]`
- Layer stride: `2*H*H + 4*H` floats

**Architecture coverage:**
- Griffin/Hawk (RG-LRU layers)
- MEGA (EMA layers)
- SSTransformer (EMA layers)
- HybridBuilder (EMA layers)
- GSS (SSM layers)
- MambaVision (SSM layers)

### Phase 2 — LSTM Block Kernel

Standard LSTM with inter-layer dense projections. Per-layer: 4 gates from
`W_x @ x + bias`, recurrent `R @ h` via shared memory, h+c state in registers.

**Per-layer weight layout:** `[W_x(H×4H) | b_x(4H) | R(H×4H) | γ(H) | β(H)]`
- Layer stride: `8*H*H + 6*H`

**Architecture coverage:**
- DeepResLSTM (2-3 layer stacks)
- Standard stacked LSTM

### Phase 3 — GRU Block Kernel

Standard GRU with 3 gates (r, z, n) and recurrent R@h. Simpler than LSTM.

**Per-layer weight layout:** `[W_x(H×3H) | b_x(3H) | R(H×3H) | γ(H) | β(H)]`
- Layer stride: `6*H*H + 5*H`

### Phase 4 — Benchmarks & Architecture Wiring

- `bench/block_scan_sweep.exs` — single-layer vs block comparison
- Wire Griffin, DeepResLSTM, and other stacking architectures
- Document H ≤ 256 constraint

## Performance Expectations

For H=256, T=32, 4 layers:

| Metric | Single-Layer (4 launches) | Block (1 launch) |
|---|---|---|
| Global mem writes | 4 × B×T×H | 1 × B×T×H |
| Global mem reads | 4 × B×T×H (input) | 1 × B×T×H (input) |
| Kernel launches | 4 | 1 |
| Inter-layer data | Goes through HBM | Stays in registers/smem |

Expected speedup: 15-30% on element-wise scan architectures (MinGRU/MinLSTM
already show this). Bigger gains at smaller batch sizes where kernel launch
overhead dominates.

## Verification

1. `cd native/cuda && make` — kernels compile (f32 + bf16)
2. `mix compile` — clean Elixir compilation
3. Numerical validation: block output matches sequential single-layer output
   (`assert_all_close atol: 1e-5`)
4. `mix test test/edifice/cuda/` — all CUDA tests pass
5. Benchmark: `mix run bench/block_scan_sweep.exs` shows improvement
