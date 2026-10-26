# Exact-stream stochastic sampling kernels

`Edifice.Utils.SamplingRandom.uniform/2` expands Threefry2x32-20 rounds during
tracing. EXLA can fuse these straight-line operations instead of executing
Nx.Random's nested RNG loops. This is a portable Nx implementation; no custom
CUDA NIF is needed. It specializes f32 uniforms in `[0, 1)` and preserves the
reference values, split order, and returned key, including odd tensor sizes.

## Isolated uniform draw

RTX 5090, CUDA through EXLA, 2026-10-26. Each measurement includes transferring
both values and next key to the host. Compilation is excluded; 200 warm calls
per shape. Four seeds per shape must match the reference bit-for-bit before
timing. Raw results: [cuda.json](cuda.json).

| Shape | Nx.Random median / p95 (us) | Unrolled median / p95 (us) |
|---|---:|---:|
| 1 × 8 | 363 / 596 | 58 / 106 |
| 1 × 17 | 347 / 536 | 62 / 103 |
| 8 × 17 | 356 / 601 | 74 / 135 |
| 128 × 17 | 356 / 578 | 59 / 101 |

Reproduce from the repository root:

```sh
devenv shell -- mix test test/edifice/utils/sampling_random_test.exs
devenv shell -- mix run bench/sampling_random.exs cuda /tmp/sampling-cuda.json
devenv shell -- mix run bench/sampling_random.exs host /tmp/sampling-host.json
```

## Controller integration

ExPhil uses the helper for stochastic button and categorical draws, including
autoregressive heads, event/duration heads, and mode-of-N. Its paired benchmark
replaces only the RNG implementation for the reference variant. Source and input
hashes are recorded, and complete seeded priming-controller streams must match.
The companion results live in ExPhil at
`eval_runs/sampling_kernels_20261026/RESULTS.md`; the driver is
`scripts/profile_sampling_kernels.py`.

Three Edifice tests cover exact values/key transitions and invalid keys. ExPhil's
targeted suite passes on host and CUDA: 34 tests, 0 failures, 1 excluded on each.
The optimization increases the traced graph size. These latency results exclude
compilation and do not establish a cold-compilation improvement. Keep Nx.Random
for other types, intervals, distributions, or large training tensors. Exact
compatibility tests should be rerun when upgrading Nx.
