# Kernel dispatch tiers on EXLA 1.0 — 2026-10-26

NVIDIA GeForce RTX 5090 (driver 595.104.02), nx 1.0.0, exla 1.0.0, axon 0.9.0,
Elixir 1.18 / OTP 27. Produced by `bench/kernel_tier_bench.exs`; the generated
tables and raw JSON are in `bench/results/kernel_tiers_2026-10-26/`.

`Edifice.CUDA.FusedScan` can run a scan three ways. This benchmark puts all
three next to each other for the two ops that have a handler in every tier,
GRU scan and Mamba selective scan, both at the op level and inside the
2-layer models built on them.

| Tier | What runs |
|---|---|
| `xla` | the pure-Nx fallback (an unrolled `Enum.reduce`) compiled by EXLA inside `Nx.Defn.jit`, default precision (TF32 matmuls) |
| `xla_hp` | the same at `precision: :highest`, exact f32 like the kernels |
| `ffi` | the CUDA kernel as an XLA custom call in the same graph (`EDIFICE_FUSED_CUSTOM_CALL=1`, `libedifice_xla.so`) |
| `nif` | the same kernel through the C NIF on eager `EXLA.Backend` tensors (forward only) |
| `eager_xla` | the fallback evaluated op by op on eager tensors under `Nx.Defn.Evaluator` |
| `*_grad` | forward plus gradient of a sum-of-squares loss |

Every tier's dispatch arm is asserted through `Edifice.CUDA.AutoTuneProfiler`,
so a row cannot silently measure the wrong path. Each cell is the median of
30 timed iterations after 5 warmups (8 for `eager_xla`); timing includes a
one-element device read so EXLA's asynchronous execution has completed.
"First call" is trace plus XLA compile.

## Results (median ms)

### GRU scan, op level

| config | xla | xla_hp | ffi | nif | eager_xla | xla_grad | xla_hp_grad | ffi_grad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B=1 T=32 H=256 | 0.183 | 0.205 | 0.647 | 0.969 | 24.2 | 0.552 | 0.561 | 4.14 |
| B=1 T=128 H=512 | 0.578 | 0.588 | 5.22 | 6.16 | 101 | 2.14 | 2.13 | 55.7 |
| B=32 T=128 H=512 | 2.30 | 3.28 | 7.03 | 12.5 | 110 | 10.5 | 15.2 | 84.3 |
| B=32 T=128 H=1024 | 4.80 | 5.55 | 22.8 | 36.6 | 117 | 22.0 | 24.9 | 330 |

### Selective scan, op level (state 16)

| config | xla | xla_hp | ffi | nif | eager_xla | xla_grad | xla_hp_grad | ffi_grad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B=1 T=32 H=256 | 0.069 | 0.067 | 0.099 | 0.190 | 7.45 | 0.282 | 0.290 | 0.498 |
| B=1 T=128 H=512 | 0.160 | 0.158 | 0.250 | 0.456 | 29.1 | 1.18 | 1.19 | 1.73 |
| B=32 T=128 H=512 | 2.16 | 2.23 | 1.85 | 5.20 | 125 | 7.62 | 7.97 | 5.70 |
| B=32 T=128 H=1024 | 4.21 | 4.42 | 3.40 | 10.2 | 210 | 14.9 | 14.8 | 10.4 |

### Models (2 layers, `Axon.build(compiler: EXLA)`)

For GRU the no-kernel path is the unrolled fallback in-graph. For Mamba it is
the Blelloch parallel scan, which is what the model runs without the kernel.

| config | xla | xla_hp | ffi | xla_grad | xla_hp_grad | ffi_grad |
|---|---:|---:|---:|---:|---:|---:|
| gru B=1 T=60 H=512 | 0.524 | 0.541 | 4.86 | 4.27 | 4.18 | 55.5 |
| gru B=32 T=60 H=512 | 0.936 | 1.25 | 5.12 | 6.59 | 10.8 | 79.5 |
| gru B=32 T=128 H=1024 | 3.67 | 7.22 | 38.8 | 31.0 | 35.2 | 661 |
| mamba B=1 T=60 H=512 | 0.200 | 0.267 | 0.278 | 4.49 | 4.83 | 5.65 |
| mamba B=32 T=60 H=512 | 1.76 | 1.90 | 0.536 | 13.7 | 14.0 | 9.11 |
| mamba B=32 T=128 H=1024 | 11.4 | 12.2 | 2.52 | 64.1 | 65.7 | 32.8 |

### Speedups, kernel over exact-f32 XLA (median over median)

| config | GRU fwd | GRU grad | SSM fwd | SSM grad |
|---|---:|---:|---:|---:|
| op B=1 T=32 H=256 | 0.3x | 0.1x | 0.7x | 0.6x |
| op B=1 T=128 H=512 | 0.1x | 0.0x | 0.6x | 0.7x |
| op B=32 T=128 H=512 | 0.5x | 0.2x | 1.2x | 1.4x |
| op B=32 T=128 H=1024 | 0.2x | 0.1x | 1.3x | 1.4x |
| model B=1 T=60 H=512 | 0.1x | 0.1x | 1.0x | 0.9x |
| model B=32 T=60 H=512 | 0.2x | 0.1x | 3.5x | 1.5x |
| model B=32 T=128 H=1024 | 0.2x | 0.1x | 4.9x | 2.0x |

### First call (trace + XLA compile, ms)

| | xla fwd | xla grad | ffi fwd | ffi grad |
|---|---:|---:|---:|---:|
| gru op B=32 T=128 H=1024 | 1 879 | 21 389 | 32 | 726 |
| gru model B=32 T=128 H=1024 | 4 216 | 36 460 | 604 | 2 445 |
| ssm op B=32 T=128 H=1024 | 1 090 | 4 567 | 10 | 41 |
| mamba model B=32 T=128 H=1024 | 2 390 | 4 948 | 455 | 1 647 |

## Findings

1. **The GRU kernel loses to XLA's compiled fallback at every size.** Forward
   is 2 to 10 times slower, backward 5 to 50 times slower, and the 2-layer
   model with the kernel is 5 to 10 times slower forward and 8 to 20 times
   slower for gradients. The kernel runs one block per batch row with the
   whole hidden state in one block and recomputes R@h from global memory at
   every step; at batch 1 that is one SM of the 5090's 170. XLA's fallback
   hands the per-step recurrent matmul to cuBLAS and fuses the gate math, so
   it uses the whole GPU. The backward kernel adds per-thread arrays sized
   `MAX_SEQ_LEN` on top. The original comparison, made on an NVIDIA T400
   against eager (non-jitted) execution, no longer holds.

2. **The selective-scan kernel wins only at batch.** Op level it is 0.6 to
   0.7x at batch 1 and 1.2 to 1.4x at batch 32. Inside Mamba, where the
   alternative is the Blelloch parallel scan, it is 3.5 to 4.9x faster
   forward and 1.5 to 2x faster for gradients at batch 32, and a wash at
   batch 1. One thread per (batch, hidden) with state in registers scales
   with batch the way the GRU kernel cannot.

3. **The NIF bridge never wins.** It is 1.3 to 3 times slower than the same
   kernel as a custom call (pointer readiness reads, a device sync before
   every launch, `cudaMalloc` outside XLA's pool) and 3 to 10 times slower
   than XLA. Its only win is over op-by-op eager evaluation, 16 to 64 times,
   which jit removes anyway. Gradients cannot run through it at all.

4. **Compile time is the custom calls' one universal win.** The unrolled
   fallbacks cost 1 to 36 seconds of XLA compile per shape; the custom-call
   graphs compile in well under a second at the op level and about half a
   second to 2.5 seconds inside the models.

5. **EXLA's default precision is not a correctness reference.** Default
   precision uses TF32 matmuls. Over 128 GRU steps at batch 32 that drifts
   0.12 from exact f32 at H=512 and gives unrelated outputs at H=1024. Kernel
   parity has to be measured against `precision: :highest`. At H=1024 T=128
   the recurrence is also chaotic in f32: against an f64 reference the kernel
   drifts 0.46 and exact XLA 0.20, both growing about 4x every 16 steps, so
   the remaining gap at that shape is rounding order, not a kernel bug.

## A bug this exposed: custom-call layouts

Inside the real 2-layer GRU model the kernel's output was off by about 7 at
batch 32 while batch 1, and every parameter-fed op-level test, matched the
fallback to 1e-5. The HLO dump showed why: EXLA emits `stablehlo.custom_call`
with no layout constraints, and XLA's layout assignment gave the scan result
a time-major layout (`f32[B,T,H]{2,0,1}`) because its consumers are a reduce
over H and a last-step slice. The kernel writes row-major and cannot tell.
At batch 1 a permutation of the batch axis is byte-identical, so the
corruption only appears at batch > 1.

The `Edifice.Block.FusedOp` custom-call implementation now pins row-major
`operand_layouts` and `result_layouts` on all four native ops, which makes
XLA insert the transposes instead. `kernel_matrix_gpu_test.exs` runs the
2-layer GRU and Mamba models at batch 4 through the tier against the fallback
(this fails by 4.4 without the pin), and `library_test.exs` asserts the
attributes on CPU. The same hazard applies to any custom call registered
through `EXLA.CustomCall`; see the upstream notes below.

## What this means for the kernel work

Both structural changes below landed on 2026-10-26, right after this run.

- Keep the FFI tier for selective scan and route the GRU scan to the XLA
  fallback. Done: the GRU handler is retired; `Edifice.Block.native_ops/0`
  lists only the selective-scan pair. The GRU backward kernel was not ported
  to FFI; it would have made training slower.
- Delete the NIF tier and the C bridge. Done: nothing it ran was faster than
  the jitted fallback, and gradients never reached it. The sources are on
  the `archive/cuda-nif-kernels` branch, and the `nif` column in the tables
  above can no longer be reproduced from `main`.
- The remaining custom-call value is compile time. If that matters for the
  development loop, a `while`-loop fallback (bounded graph size) is the
  cheaper way to get it than more kernels; the Stateful replay path already
  does this.
- Two upstream EXLA items on the fork `blasphemetheus/nx`: (a)
  `get_buffer_device_pointer` should await the buffer's ready future before
  `UnsafeBufferPointer` (the readiness race behind the NIF's device sync);
  (b) custom calls lowered through `EXLA.CustomCall` should default to
  row-major operand and result layouts, or the `Spec` should document that
  handlers must pin them.

## Reproduce

```bash
scripts/check_cuda_kernels.sh                 # builds the FFI library (the NIF tier is gone)
EDIFICE_FUSED_CUSTOM_CALL=1 EDIFICE_AUTOTUNE=0 \
  BENCH_OUT=bench/results/kernel_tiers_$(date +%F) \
  mix run bench/kernel_tier_bench.exs          # ~25 min; BENCH_QUICK=1 for ~3
```
