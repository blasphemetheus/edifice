# Axon Graph Fusion — Analysis & Resolution

## The Problem (now solved)

Custom calls eliminate graph breaks **within** the fused scan kernel (0.3ms
execution). But when used through `Axon.layer` without `compiler: EXLA`, the
2-layer MinGRU took ~361ms instead of ~3.7ms.

## Root Cause: Missing `compiler: EXLA` option

Without `compiler: EXLA`, Axon's `predict_fn` **re-invokes every layer callback
on every call**, rebuilding the entire expression graph and recompiling it through
EXLA each time. This is not a bug in Axon — it's the expected behavior when no
compiler is specified (the predict_fn runs in eager mode).

### Evidence

We placed a counter inside the `Axon.layer` callback:

```elixir
Axon.layer(fn g, c, _opts ->
  :counters.add(call_count, 1, 1)
  FusedScan.mingru(g, c)
end, [gates, cands])
```

After 3 calls to `predict_fn`, the counter was at 4 (1 from `Axon.build` graph
construction + 3 from execution). Every call re-traces the callback.

### The Fix

```elixir
# SLOW — re-traces every call (~361ms for 2-layer MinGRU)
{init_fn, predict_fn} = Axon.build(model)

# FAST — compiles once, caches (~3.7ms for 2-layer MinGRU)
{init_fn, predict_fn} = Axon.build(model, compiler: EXLA)
```

This is a standard Axon option. With `compiler: EXLA`:
1. Axon wraps `predict_fn` in `Nx.Defn.jit(..., compiler: EXLA)`
2. First call traces the callbacks and compiles the XLA graph
3. Subsequent calls use the cached executable — no re-tracing

## Final Benchmark (NVIDIA T400, B=1 T=32 H=256)

| Configuration | Min | Median | Speedup |
|---|---|---|---|
| Direct defn (scan only) | 0.24ms | 0.29ms | baseline |
| Direct defn (dense+scan) | 0.54ms | 2.58ms | — |
| Axon 2L (no compiler) | 274ms | 361ms | 1x |
| **Axon 2L (compiler: EXLA)** | **2.51ms** | **3.66ms** | **97x** |

The ~1ms gap between direct defn (2.58ms) and Axon (3.66ms) is Axon's parameter
map threading overhead — totally acceptable.

## Implications for Edifice

### No Axon fork needed

The original hypothesis was that Axon creates per-layer graph breaks requiring
a fork to fix. This was wrong. Axon compiles the entire `predict_fn` as one
`Nx.Defn.jit()` call when given a compiler. The overhead was simply from
re-tracing without caching.

### Where to use `compiler: EXLA`

- **Benchmarks**: Always pass `compiler: EXLA` for accurate GPU timing
- **Inference entry points**: Performance-critical code should pass the compiler
- **Tests**: Keep backend-agnostic (no compiler option) so tests run on BinaryBackend
- **Training loops**: `Axon.Loop.trainer` handles this internally via its `:compiler` option

### EDIFICE_LOCAL_NX env var

The EXLA fork (with custom call kernels) is only active when `EDIFICE_LOCAL_NX=1`.
Without it, Edifice uses hex versions and the three-tier dispatch falls through
to NIF bridge or Elixir fallback — still correct, just without the in-graph
custom call optimization.

## Files

- `docs/history/exla_custom_calls_results.md` — full benchmark data
- `lib/edifice/cuda/fused_scan.ex` — three-tier dispatch implementation
- `bench/model_breakdown.exs` — per-layer benchmarking script
