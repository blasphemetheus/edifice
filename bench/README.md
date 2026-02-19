# Edifice Benchmarks

## Quick Start

All benchmarks require EXLA. For GPU:

```bash
EXLA_TARGET=cuda XLA_TARGET=cuda12 mix run bench/<script>.exs
```

## Scripts

| Script | What it measures | Benchee? | Approx time |
|---|---|---|---|
| `inference_latency.exs` | Per-architecture inference time, 60 FPS viability | Optional (Phase 3 only) | ~10 min |
| `scaling_profile.exs` | How latency scales with seq_len and embed_dim | No | ~15 min |
| `training_throughput.exs` | Forward + backward pass throughput (samples/sec) | No | ~10 min |
| `memory_profile.exs` | Parameter count, size, GPU memory delta | No | ~5 min |
| `full_sweep.exs` | All 80+ architectures, build/compile/inference | No | ~15 min |
| `architecture_profile.exs` | 12 representative archs with Benchee stats | **Yes (required)** | ~5 min |

## Benchee Dependency

Benchee is a `:dev` dependency. Two scripts use it:

- **`architecture_profile.exs`** — requires Benchee, will crash without it
- **`inference_latency.exs`** — optional, falls back to manual timing

To enable Benchee on GPU:

```bash
MIX_ENV=dev EXLA_TARGET=cuda XLA_TARGET=cuda12 mix run bench/inference_latency.exs
```

All other scripts use `:timer.tc` and work in any environment.

## Environment Variables

All scripts support configuration via env vars:

| Variable | Default | Description |
|---|---|---|
| `BENCH_EMBED` | 256 | Embedding dimension |
| `BENCH_SEQ_LEN` | 32 | Sequence length / frame context |
| `BENCH_BATCH` | 1 (inference) / 4 (training) | Batch size |
| `BENCH_LAYERS` | 2 | Number of layers |
| `BENCH_ITERS` | varies | Timing iterations per data point |
| `BENCH_TIME` | 5 | Benchee time per scenario (seconds) |

## Recommended Run Order

1. `inference_latency.exs` — find which backbones can hit 60 FPS
2. `memory_profile.exs` — check parameter counts fit your GPU
3. `scaling_profile.exs` — understand how top candidates scale
4. `training_throughput.exs` — verify training is feasible
5. `full_sweep.exs` — comprehensive sweep of all architectures
