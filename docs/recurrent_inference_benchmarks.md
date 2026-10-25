# Recurrent inference benchmarks for ExPhil

## Scope and model selection

Measured on October 24, 2026 on a Ryzen 9 9950X3D and RTX 5090. These are
**Edifice backbone microbenchmarks**, not recorded game-loop or complete policy
latencies. Weights are seeded synthetic weights, not exported trained policies.
The purpose is to identify which execution paths warrant ExPhil integration.

Read-only inspection of ExPhil's `checkpoints/**/*config*.json` found configurations
for MinGRU (101 files), GRU (61), Mamba (46), MLP (23), and GatedDeltaNet (1).
Those counts include resumes and experiments; they are not counts of distinct
trained policies. Mamba-2 also has exported checkpoints (`bakeoff_mamba2.bin` and
`mewtwo_combo_newera_mamba2_screen_*`). LSTM and causal GatedSSM are supplemental
comparisons. The much larger supported architecture catalog is not evidence that
every architecture is deployed.

The representative sizes follow the saved configurations and ExPhil's trainer
mapping `hidden_size: hd(hidden_sizes)`: GRU/Mamba/GatedDeltaNet use hidden 512,
MinGRU uses 256, and the small MLP uses [64,64]. All have two layers. Mamba and
MinGRU use windows of 80; other recurrent models use 60. Backbone input width is
516: raw width 264 with two action and two character IDs expanded to 64 dimensions
each. This is a controlled representative width, not a claim that every checkpoint
uses it. Enhanced Nana, previous-action, style and stage features need separate
shape measurements. Use `--embed` to override it.

## Stateful implementation follow-up

Subsequent real-checkpoint/replay validation lives in ExPhil's
`scripts/validate_recurrent_carry.exs`, `docs/guides/STATEFUL_MINGRU.md`, and
`eval_runs/recurrent_carry_20261024/`. MinGRU, GRU, Mamba and Mamba-2 passed
selected full-prefix comparisons and exact snapshot replay on CPU. MinGRU
also passed on CUDA with highest precision (max feature error 6.26e-7), as
did Mamba (7.63e-6) and GRU (6.88e-6).
This validates the recurrence, not indefinite carry as a behavior-preserving
replacement for a sliding window. In the first 320-frame probe, MinGRU's
full-controller agreement with windowed inference was 98.34%, or 99.59% with
a 30-frame resync counterfactual. In a 1,280-frame midgame clip, MinGRU's
agreement was 96.42% versus 99.58% with resync. The BPTT GRU checkpoint requires continuous
carry; truncation is the counterfactual for that model. These actual exports
use raw input widths 264/288 rather than the 516-wide benchmark above.

The validation also exposed an ExPhil batch-row reset bug, now fixed: only
reset rows receive cold-start padding; continuing games retain their carry.
Compiling a chunk replay to reduce cold-start/resync launches, supporting
batched resync, and wiring Mamba-2 into ExPhil's Agent remain useful next steps.

MinGRU stepping is now enabled in ExPhil's Agent via the existing opt-in
`--stateful-step` flag. The adapter includes input projection, final normalization,
weight-derived state shapes, independent/autoregressive heads, batched state,
per-row resets and snapshot/restore. Synthetic Agent and policy-output tests pass;
no trained policy was promoted or deployed.

Edifice now implements `init_state/2` and `step/3` for `:gated_delta_net` and
`:mamba_ssd`. GatedDeltaNet carries its per-head associative memory and optional
causal convolution history. SSD carries its SSM/conv state, plus cumulative log
decay for `structured_mask: true`. Chunked inference/training modes, pointwise
convolution, parameter gradients and serialized rollback have regression coverage.

New measurements at the same batch/width/hidden sizes, median ms (p95):

| Backbone | CPU full | CPU JIT step | CPU direct step | GPU full | GPU JIT step | GPU direct step |
|---|---:|---:|---:|---:|---:|---:|
| GatedDeltaNet | 8.018 (8.930) | 0.211 (0.287) | 0.224 (0.274) | 0.615 (0.848) | 0.116 (0.260) | 0.117 (0.274) |
| Mamba-2/SSD | 13.688 (14.823) | 0.193 (0.243) | 0.171 (0.205) | 0.248 (0.420) | 0.135 (0.247) | 0.132 (0.216) |

This is about 38× CPU / 5.3× GPU for GatedDeltaNet and 80× CPU / 1.9× GPU for
SSD using the faster measured calling path. Small JIT/direct differences vary
between runs; direct precompilation is not consistently faster for every model.
[New CPU results](../bench/results/recurrent_new_steps_host_2026-10-24.json),
[new GPU results](../bench/results/recurrent_new_steps_cuda_2026-10-24.json).

**SSD compatibility correction:** its matmul transfer matrix incorrectly included
the source timestep's decay, computing `prod(a[s..t])` instead of
`prod(a[s+1..t])`. This has been corrected to agree with the sequential scan.
Previously the one-token, short-matmul and longer chunked paths did not share the
same recurrence. Old checkpoints that used the affected matmul path need
numerical and behavioral revalidation; parameter names/shapes are unchanged.

Validation: the regression run passed 106 tests and 2 properties; the final
focused recurrence suite passed 14 tests, including every-prefix parity across
chunk boundaries and parameter-gradient comparisons. Both new full-sized GPU
benchmarks pass strict parity and exact replay at highest precision.

## Baseline results before this follow-up

Times are **median milliseconds (p95)**. Step columns use the direct precompiled
executable. A dash means no Stateful implementation, not zero cost. MLP is already
a single-frame network. Different hidden sizes reflect usage, not a quality-matched
architecture comparison.

| Backbone | CPU full | CPU step | GPU full | GPU step |
|---|---:|---:|---:|---:|
| GRU, hidden 512 | 4.641 (5.138) | 0.114 (0.146) | 1.775 (2.721) | 0.108 (0.169) |
| Mamba, hidden 512 | 17.693 (19.331) | 0.167 (0.190) | 0.283 (0.474) | 0.116 (0.167) |
| MinGRU, hidden 256 | 0.325 (0.405) | 0.034 (0.043) | 0.189 (0.328) | 0.090 (0.184) |
| GatedDeltaNet, hidden 512 | 8.136 (8.941) | — | 0.581 (0.700) | — |
| MLP, [64,64] | 0.020 (0.026) | — | 0.070 (0.123) | — |
| Mamba-2/SSD, hidden 512 | 13.605 (15.095) | — | 0.249 (0.340) | — |
| LSTM, hidden 512 (supplemental) | 5.997 (6.704) | 0.145 (0.174) | 3.452 (4.417) | 0.127 (0.211) |
| Causal GatedSSM, hidden 512 (supplemental) | 2.964 (3.256) | 0.244 (0.347) | 0.216 (0.326) | 0.129 (0.266) |

The existing GPU JIT wrapper's step medians are 0.123 ms GRU, 0.124 ms Mamba,
and 0.096 ms MinGRU. Precompilation reduces these to 0.108, 0.116 and 0.090 ms.
This is a modest measured difference, not a promise of the same percentage
improvement in the full application.

At batch 1, MinGRU and the small MLP are faster on this CPU. Mamba's CPU
full-window prediction exceeds the 16.67 ms frame budget at this size, while its
step path has ample backbone headroom. GRU/LSTM benefit most from removing the
repeated sequential window. MinGRU's measured CPU opportunity is 0.325 to
0.034 ms (9.6×).

All five supported step configurations pass prefix-output and exact replay checks
on both devices with highest precision. Raw results include eager timings,
first-prediction costs and numerical differences:
[CPU](../bench/results/recurrent_host_2026-10-24.json),
[GPU](../bench/results/recurrent_cuda_2026-10-24.json).

A separate native Mamba scan run passed parity and replay, with the CUDA library
confirmed loaded. It measured 0.415 ms full prediction and 0.114 ms precompiled
step. The step uses the ordinary one-frame recurrence, so enabling the native
sequence scan does not make it a new native step implementation. These results
do not show a native scan advantage at this batch/window size.
[Native run](../bench/results/recurrent_native_mamba_2026-10-24.json).
A standalone portable rerun measured 0.303 ms full prediction and 0.137 ms step,
reinforcing the absence of a native full-window win here and the variability of
small step timings. [Portable rerun](../bench/results/recurrent_portable_mamba_2026-10-24.json).

## Method

Run each backend alone with no simultaneous benchmark/training job:

```sh
devenv shell -- mix run scripts/bench_recurrent_inference.exs --client host --aot --output /tmp/recurrent-host.json
devenv shell -- mix run scripts/bench_recurrent_inference.exs --client cuda --aot --output /tmp/recurrent-cuda.json
EDIFICE_FUSED_CUSTOM_CALL=1 devenv shell -- mix run scripts/bench_recurrent_inference.exs --client cuda --arches mamba --aot --native --output /tmp/recurrent-native.json
```

The script accepts `--arches gru,mamba,min_gru`, `--embed N` and `--iterations N`.
It uses batch 1, f32 tensors, explicit EXLA `precision: :highest`, resident weights
and inputs, ten warmup calls and 200 timed calls. Every call copies the output to
the host to synchronize execution; compilation and warmup are excluded from
steady-state timing. Native custom-call scan kernels are disabled so the baseline
is the portable current implementation; `--native` explicitly bypasses autotuning
and requests the existing opt-in custom-call path for the separate native run.
It requires an already built library and does not rebuild EXLA.
This is not an exhaustive kernel-tuning
search or an upper bound on possible performance.

- **Full:** recompute the window and return its last output. MLP only uses one frame.
- **JIT step:** one frame through the existing cached `Stateful.jit_step` API.
- **AOT step:** use `Nx.Defn.compile/3` once for known parameter/state/frame shapes,
  then call the resulting executable directly. Same model and parameters.
- **Eager step:** execute the recurrence without compiling the whole step.

Before reporting a step speedup, the script checks that stepping through the full
prefix matches its full-model output at `atol=rtol=1e-4` and that snapshot replay
is bitwise identical. Failure rows contain errors, not a usable speed claim.
Only the final prefix output is checked here; the existing stateful test suite
checks every prefix for smaller configurations.

## Correctness and deployment constraints

Default GPU matmul precision failed this benchmark's parity gate for Mamba
(max absolute difference 0.000517), MinGRU (0.001539), and causal GatedSSM
(0.000470). Tolerances were not relaxed. The report uses explicit highest
precision; f32 tensor storage alone does not specify matmul compute precision.

A carried recurrence matches processing the entire prefix from its initial state.
It is **not** equivalent to repeatedly resetting and processing only the latest
window. ExPhil already distinguishes carried BPTT policies from window-trained
policies and has cold-start padding and periodic state resynchronization. Preserve
those contracts. Cold start, resync, rollback bursts, embedding work, action heads,
sampling and emulator IPC are additional costs outside these steady-state numbers.

The GatedSSM row uses `scan_mode: :causal`; it is not a speed replacement for a
legacy GatedSSM checkpoint. The baseline table predates the new GatedDeltaNet and
Mamba-2 step paths; see the follow-up table above for their current measurements.

## Integration priorities

1. **Validate MinGRU on actual checkpoints and game traces.** The Agent adapter
   is implemented and tested. Compare carried-state behavior, resync and complete
   action latency before changing the deployment recipe for a window-trained bot.
2. **Keep weights and state resident and compile the whole recurrence.** ExPhil
   already copies loaded trunk weights to the device and uses `Stateful.jit_step`.
   Its source records a previous GRU regression from 0.21 to 12.4 ms when weights
   were transferred every step. The new benchmark does not remeasure that old result.
3. **Integrate GatedDeltaNet/Mamba-2 into ExPhil when needed.** The Edifice stateful
   contracts now exist. ExPhil's adapters still need parameter extraction, reset,
   batching and head parity tests for these two architectures. Revalidate SSD's
   corrected matmul semantics against the chosen checkpoint first.
4. **Measure full policy latency with actual observations/checkpoints.** Fuse
   compatible embedding, trunk and head work into fewer executable calls, and
   measure p95 including cold starts/resync. A fast backbone does not guarantee
   fast autoregressive sampling or end-to-end controller delivery.
5. **Keep sparse-attention work separate from recurrent speedups.** NSA/MoBA
   selected-block gathering remains useful for attention models, but is not the
   immediate bottleneck for the recurrent architectures identified here.

The baseline pass only inspected ExPhil; the follow-up implements its MinGRU
adapter. Checkpointing remains a training memory project and does not directly
accelerate these inference paths. NSA/MoBA causality and real selected-block
execution remain the next Edifice implementation priorities.

For shape-specialized inference, the existing API is sufficient:

```elixir
module = Edifice.module_for(architecture)
step = Nx.Defn.compile(&module.step/3, [params, state, frame],
  compiler: EXLA, client: :host, precision: :highest)
{features, next_state} = step.(params, state, frame)
```

Keep the executable outside serialized state, reuse it for matching shapes/types,
and leave parameters resident on the selected backend.
