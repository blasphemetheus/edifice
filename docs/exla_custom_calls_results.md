# Phase 2: EXLA Custom Calls — Results

## Summary

Fused CUDA scan kernels (MinGRU, MinLSTM) are now registered as XLA custom calls
inside an EXLA fork. When called from a single `defn` compilation unit, the
kernels stay inside the XLA computation graph with zero graph breaks. XLA manages
buffers, streams, and optimization around them.

**Key result:** 2-layer MinGRU drops from ~349ms (Axon) to **2.6ms** (single defn) — a 135x speedup. The scan kernel itself runs in 0.3ms.

## Benchmark Results (NVIDIA T400, B=1 T=32 H=256)

### Single defn (custom call working, zero graph breaks)

| Configuration | Min | Median |
|---|---|---|
| Scan alone (direct defn) | 0.28ms | 0.36ms |
| Dense+Dense+Scan (1 layer, single defn) | 1.35ms | 2.51ms |
| 2-layer MinGRU (single defn) | 1.46ms | 2.59ms |

### Through Axon.layer (graph breaks at each layer boundary)

| Configuration | Min | Median |
|---|---|---|
| Dense+Dense+Scan (1 layer, Axon.layer) | 48ms | 62ms |
| Full MinGRU (1 layer, Axon) | 126ms | 172ms |
| Full MinGRU (2 layers, Axon) | 275ms | 349ms |

### What the overhead is

| Source | Cost |
|---|---|
| Axon per-layer JIT boundary | ~25x overhead (2.5ms → 62ms) |
| Scan kernel execution | 0.3ms |
| Dense projections (XLA fused) | ~2ms |

## Why Axon is slow

Axon compiles each `Axon.layer` callback as a **separate XLA computation**. For a
2-layer MinGRU with LayerNorm, this means 6+ independent compilations:

```
[LN₁] → break → [Dense_gate₁ + Dense_cand₁] → break → [Scan₁] → break →
[LN₂] → break → [Dense_gate₂ + Dense_cand₂] → break → [Scan₂] → break
```

Each break requires:
1. Finishing the current XLA computation
2. Transferring results back to the Elixir runtime
3. Starting a new XLA computation with the results as inputs
4. Re-uploading parameters

This overhead (~25ms per break on T400) dominates the actual computation.

## The custom calls work correctly

The `Nx.Shared.optional` → EXLA `cached_recur_operator` → `stablehlo.custom_call`
pipeline works exactly as designed:

- `custom_call_available?()` detects the EXLA fork at compile time
- In defn context, `Nx.Shared.optional(:fused_mingru_scan, ...)` creates an
  `:optional` Expr node
- EXLA's `cached_recur_operator` pattern-matches on `:fused_mingru_scan` +
  `platform: :cuda` and emits a `stablehlo.custom_call`
- The CUDA kernel runs inside the XLA graph with no breaks
- On non-CUDA platforms, the Elixir fallback runs normally

Verification: `Nx.Defn.jit(fn g, c -> FusedScan.mingru(g, c) end, compiler: EXLA)`
produces 0.3ms execution — identical to raw kernel launch overhead.

## Next step: Axon graph fusion

The custom calls eliminate graph breaks **within** the scan, but Axon still
creates graph breaks **between** layers. To realize the full 135x speedup in
Axon models, Axon needs to compile multiple layers into a single XLA computation.

See `docs/axon_graph_fusion.md` for analysis of potential Axon fork approaches.

## Build notes

### EXLA fork changes (`/home/nixos/nx/exla/`)

- `Makefile`: Added `-DEXLA_FFI` to NVCCFLAGS
- `c_src/exla/custom_calls/fused_mingru_scan.cu`: Copied from Edifice
- `c_src/exla/custom_calls/fused_minlstm_scan.cu`: Copied from Edifice
- `lib/exla/mlir/value.ex`: `fused_mingru_scan/4`, `fused_minlstm_scan/5`
- `lib/exla/defn.ex`: Two `cached_recur_operator(:optional)` clauses for `:cuda`

### Edifice changes

- `mix.exs`: Points nx + exla at fork path deps
- `lib/edifice/cuda/fused_scan.ex`: 3-tier dispatch (custom call → NIF → Elixir)
- `native/cuda/fused_*.cu`: Removed `ffi_api.h` include (broke nvcc build)
- `shell.nix`: Added `libstdc++` to fix segfault from missing shared lib

### Build gotcha: `xla/ffi/ffi_api.h`

Do NOT include `"xla/ffi/ffi_api.h"` in `.cu` files compiled by EXLA's Makefile.
It pulls in `call_frame.h` → `xla/types.h` → `Eigen/Core` → `cuda_fp16.h` →
`<nv/target>` which doesn't exist on the nvcc include path in nix-shell.

Only `"xla/ffi/api/ffi.h"` is needed — it includes `api/api.h` which has
`XLA_FFI_REGISTER_HANDLER` and all required macros.
