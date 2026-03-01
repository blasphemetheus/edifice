# CUDA NIF Bridge: Results & Lessons Learned

## Summary

Phase 1 of the CUDA kernel integration is complete. Fused scan kernels for
6 recurrent operations are wired through a NIF bridge into the Axon inference
pipeline. The kernels themselves are extremely fast, but the full model path
is slower than expected due to XLA graph fragmentation.

## Architecture

```
Axon Dense layers     ──→  XLA graph (compiled, fast)
                              │
                        ── graph break ──  (expensive: ~97ms)
                              │
Fused scan kernel     ──→  Nx.to_pointer → NIF → Nx.from_pointer
                              │
                        ── graph break ──  (expensive)
                              │
Next Axon layer       ──→  New XLA graph
```

Each `Axon.layer` callback forces XLA to finalize and execute the pending
graph, hand control to the BEAM, run the callback, then start a new graph.

## Benchmark Results (NVIDIA T400, B=1 T=32 H=256)

### Raw Scan Speedup (gates pre-computed, no Axon overhead)

| Config | MinGRU | MinLSTM |
|--------|--------|---------|
| B=1 T=8 H=256 | 34x | 31x |
| B=1 T=32 H=256 | 141x | 88x |
| B=1 T=64 H=256 | 335x | 225x |
| B=1 T=128 H=256 | 237x | 228x |
| B=4 T=64 H=256 | 320x | 289x |
| B=1 T=32 H=1024 | 155x | 94x |

Fused kernel median: **0.44ms** (MinGRU) / **0.65ms** (MinLSTM)
Elixir scan median: **62ms** / **57ms**

### NIF Bridge Overhead Breakdown

| Component | Time |
|-----------|------|
| `Nx.to_pointer` | ~0 ms (instant — just reads the device address) |
| NIF kernel call + cudaDeviceSynchronize | 0.068 ms |
| `Nx.from_pointer` | 0.002 ms |
| Nx.sigmoid (XLA) | 1.2 ms |
| **Full FusedScan.mingru** | **1.6 ms** |

The NIF bridge adds negligible overhead on top of the raw kernel.

### Model Layer Breakdown (MinGRU, 2 layers)

| Component | Median |
|-----------|--------|
| LayerNorm (Axon) | 4.2 ms |
| Dense 256→256 (Axon) | 1.7 ms |
| LN + Two Dense (gate+cand) | 8.7 ms |
| Dense+Dense+FusedScan via `Axon.layer` | **105.8 ms** |
| Full 1-layer MinGRU | 137 ms |
| Full 2-layer MinGRU | 247 ms |

### Where the Time Goes

The ~97ms gap between "LN + Two Dense" (8.7ms) and "Dense+Dense+FusedScan"
(105.8ms) is the **XLA graph break penalty**. Each `Axon.layer` callback:

1. Forces XLA to compile and execute the pending subgraph
2. Transfers control from XLA runtime to the BEAM VM
3. Runs the Elixir callback (pointer extraction + NIF — only ~1.6ms)
4. Returns, starting a new XLA compilation scope

With 2 MinGRU layers, there are 2 graph breaks, each costing ~97ms.

## Key Insight

The fused CUDA kernels are **not the bottleneck**. The bottleneck is the
`Axon.layer` callback mechanism breaking XLA's computation graph. This is
a fundamental limitation of the NIF bridge approach — any time you leave the
XLA graph and return, you pay the graph-break penalty.

This penalty exists regardless of whether the callback runs a fused CUDA
kernel or the Elixir `Enum.reduce` scan. The NIF bridge makes the callback
itself 88-335x faster, but it can't eliminate the graph break overhead.

## What Phase 2 (EXLA Custom Calls) Fixes

Phase 2 registers the CUDA kernels as XLA custom calls via the FFI API.
This means the fused scan stays **inside the XLA graph** — no graph breaks,
no BEAM handoff, no pointer extraction. XLA sees it as just another op
and handles buffer management, stream synchronization, and graph optimization
around it automatically.

Expected full-model time with custom calls: LN + Dense + Dense + FusedScan
should be close to the sum of parts (~8.7ms + ~1.6ms ≈ 10ms per layer),
yielding a 2-layer MinGRU at **~25ms** instead of the current 247ms.

## Files

| File | Purpose |
|------|---------|
| `native/cuda/fused_mingru_scan.cu` | MinGRU kernel (standalone + EXLA FFI) |
| `native/cuda/fused_minlstm_scan.cu` | MinLSTM kernel |
| `native/cuda/fused_native_rec_scan.cu` | ELU-GRU, Real-GRU, Diag-Linear kernels |
| `native/cuda/fused_liquid_scan.cu` | Liquid (LTC) exact solver kernel |
| `c_src/edifice_cuda_nif.c` | NIF bridge — dlopen, GC-tracked cudaMalloc |
| `c_src/Makefile` | NIF compilation (gcc, links -ldl) |
| `lib/edifice/cuda/nif.ex` | Elixir NIF stubs |
| `lib/edifice/cuda/fused_scan.ex` | High-level dispatch + CPU fallback |
| `priv/cuda/libedifice_cuda_kernels.so` | Compiled CUDA kernels (~1MB) |
| `priv/libedifice_cuda_nif.so` | Compiled NIF bridge (~17KB) |
| `bench/nif_overhead_profile.exs` | NIF overhead profiler |
| `bench/model_breakdown.exs` | Model layer-by-layer timing |
| `bench/fused_scan_latency.exs` | Raw scan speedup benchmark |
