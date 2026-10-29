# Phase 2: EXLA Custom Calls — Results

> **Historical (fork-era, NVIDIA T400).** These numbers predate EXLA 1.0's
> `EXLA.load_dylib/1` path and the RTX 5090. For current tier-vs-tier results
> see [kernel_tier_benchmarks.md](kernel_tier_benchmarks.md) (2026-10-26):
> on the 5090 the compiled XLA fallback beats the GRU kernel outright, and
> the selective-scan kernel wins only at batch.

## Summary

Fused CUDA scan kernels (MinGRU, MinLSTM) are now registered as XLA custom calls
inside an EXLA fork. When called with `compiler: EXLA`, the kernels stay inside
the XLA computation graph with zero graph breaks and the compiled graph is cached
across calls.

**Key result:** 2-layer MinGRU drops from ~361ms to **3.7ms** with
`Axon.build(model, compiler: EXLA)` — a **97x speedup**. The scan kernel
itself runs in 0.3ms.

## Benchmark Results (NVIDIA T400, B=1 T=32 H=256)

### Direct defn baselines (no Axon)

| Configuration | Min | Median |
|---|---|---|
| Scan alone (custom call, direct defn) | 0.24ms | 0.29ms |
| Dense+Dense+Scan (1 layer, single defn) | 0.54ms | 2.58ms |

### Axon WITHOUT `compiler: EXLA` (re-traces every call)

| Configuration | Min | Median |
|---|---|---|
| MinGRU 1L | 138ms | 208ms |
| MinGRU 2L | 274ms | 361ms |

### Axon WITH `compiler: EXLA` (graph cached)

| Configuration | Min | Median |
|---|---|---|
| MinGRU 1L | 1.59ms | 3.64ms |
| MinGRU 2L | 2.51ms | 3.66ms |

## Root cause of the Axon overhead

Without `compiler: EXLA`, Axon's `predict_fn` **re-invokes every layer callback
on every call**. This means:

1. The `FusedScan.mingru` dispatch runs again (checking `custom_call_available?`)
2. `Nx.Shared.optional` builds a new `:optional` Expr node
3. EXLA recompiles/re-matches the XLA graph from scratch

We confirmed this with a counter inside the callback — it incremented on every
`predict_fn.(params, input)` call, not just the first.

With `compiler: EXLA`, `Axon.build` wraps `predict_fn` in `Nx.Defn.jit(...,
compiler: EXLA)`. EXLA caches the compiled XLA executable keyed by the expression
tree shape, so subsequent calls skip graph building entirely and dispatch directly
to the cached GPU executable.

## The fix: `compiler: EXLA`

```elixir
# SLOW — re-traces every call (~360ms for 2-layer MinGRU)
{init_fn, predict_fn} = Axon.build(model)

# FAST — compiles once, caches (~3.7ms for 2-layer MinGRU)
{init_fn, predict_fn} = Axon.build(model, compiler: EXLA)
```

This is a standard Axon option — no fork needed. The `compiler:` option tells
Axon to JIT-compile the predict_fn with that compiler, enabling caching.

## Custom call pipeline (verified working)

The `Nx.Shared.optional` → EXLA `cached_recur_operator` → `stablehlo.custom_call`
pipeline works exactly as designed:

- `custom_call_available?()` detects the EXLA fork at graph-build time
- In defn context, `Nx.Shared.optional(:fused_mingru_scan, ...)` creates an
  `:optional` Expr node
- EXLA's `cached_recur_operator` pattern-matches on `:fused_mingru_scan` +
  `platform: :cuda` and emits a `stablehlo.custom_call`
- The CUDA kernel runs inside the XLA graph with no breaks
- On non-CUDA platforms, the Elixir fallback runs normally

## Build notes

### EXLA fork activation

The fork is gated behind an env var in `mix.exs`:

```elixir
# Set EDIFICE_LOCAL_NX=1 to use local nx/exla forks
local_nx? = System.get_env("EDIFICE_LOCAL_NX") == "1"
```

Without it, Edifice uses hex versions and custom calls are unavailable (falls
through to NIF or Elixir fallback — still correct, just slower).

### EXLA fork changes (`/home/nixos/nx/exla/`)

- `Makefile`: Added `-DEXLA_FFI` to NVCCFLAGS
- `c_src/exla/custom_calls/fused_*.cu`: All Edifice kernels copied, `ffi_api.h` removed
- `lib/exla/mlir/value.ex`: `fused_mingru_scan/4`, `fused_minlstm_scan/5`, + 7 more
- `lib/exla/defn.ex`: `cached_recur_operator(:optional)` clauses for `:cuda`

### Build gotcha: `xla/ffi/ffi_api.h`

Do NOT include `"xla/ffi/ffi_api.h"` in `.cu` files compiled by EXLA's Makefile.
It pulls in `call_frame.h` → `xla/types.h` → `Eigen/Core` → `cuda_fp16.h` →
`<nv/target>` which doesn't exist on the nvcc include path in nix-shell.

Only `"xla/ffi/api/ffi.h"` is needed — it includes `api/api.h` which has
`XLA_FFI_REGISTER_HANDLER` and all required macros.
