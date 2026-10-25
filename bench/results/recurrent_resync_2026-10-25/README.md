# Recurrent resync profile — 2026-10-25

Replay history assembly and replaying unaffected rows were the two measured
resync costs. The new path stacks chronological history inside the compiled
executable, gathers due rows, replays them and scatters the resulting carry.
A count mask preserves padded rows exactly. ExPhil pads selection sizes to powers
of two (capped at batch size) and warms those shapes during batch setup. When all
rows are due it uses full replay directly, without gather/scatter.

## Backbone resync measurements

Synthetic representative two-layer weights, input width 264, batch 32, highest
precision, native custom calls disabled. Hidden/window: MinGRU 256/80,
GRU 512/60, Mamba 512/80, Mamba-2 256/60. Inputs and weights resident. Each sample
synchronizes final state; first compilation is excluded. These are complete
resync-operation medians, including history assembly and row merging, but exclude
observation embedding, action heads, fresh initial-state creation and IPC.

| Backbone, one row due | CPU before → after | CUDA before → after |
|---|---:|---:|
| MinGRU | 6.814 → 1.272 ms | 7.397 → 1.820 ms |
| GRU | 16.321 → 5.355 ms | 5.174 → 2.371 ms |
| Mamba | 62.348 → 12.754 ms | 12.410 → 5.054 ms |
| Mamba-2 / SSD | 26.118 → 3.981 ms | 9.846 → 2.932 ms |

The JSON also covers three due rows padded to four, four due rows, and all rows.
All selected outputs match full replay within atol=rtol=1e-4; unmodified rows
are bitwise unchanged. Tests separately exercise nested state, convolution
history, count zero, changed row identities and nonzero starting carry.

Eager history assembly alone measured 2.5–3.9 ms on CUDA and 1.5–2.6 ms on CPU.
Stage timings synchronize independently and are not additive. GPU kernel shape
selection and shared CPU load cause variation: do not infer identical speedups
for every batch or due-row count. The CPU ran alongside unrelated repository
checks. Thirty CUDA samples and twenty CPU samples per operation; GPU was free
at launch, and no other GPU compute process was observed at completion of these
backbone runs.

## Full application validation

ExPhil's `scripts/bench_batched_resync.exs` measures trained autoregressive
policies at batch 8, including embedding, heads and output transfer. It restores
a fixed history before each sample, and compares controllers, features and carry
against baseline fixtures. Results live in ExPhil's
`eval_runs/recurrent_resync_profile_20261025/`.

The full-Agent profile also found batched embedding doing many eager GPU
operations, unlike the existing single-agent path. ExPhil now performs the small
normalization/one-hot assembly on BinaryBackend and transfers the finished batch
once. This is separate from Edifice's replay optimization.

Batch setup now precompiles both tensor and tuple history and bounded selection
shapes on throwaway carry. This deliberately adds startup work to avoid a first
reset/resync JIT spike; it does not advance live history or change the resync
schedule, policy weights or deployment defaults.

## Reproduce

```sh
mix run scripts/profile_recurrent_resync.exs --client cuda --batch 32 --iterations 30 --out /tmp/resync-cuda.json
mix run scripts/profile_recurrent_resync.exs --client host --batch 32 --iterations 20 --out /tmp/resync-host.json
```

`current_total` is the old eager history stack + full replay + partial-row merge;
`optimized_total` is tuple-history selected replay, or full tuple replay when
all rows are due. `fused_full` and `fused_history_total` isolate removal of eager
work without row selection. Run with an otherwise idle device for clean timings.
