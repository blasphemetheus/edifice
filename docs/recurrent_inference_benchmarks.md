# Recurrent inference benchmarks for ExPhil

## Scope and model selection

### Resync profiling and optimization (2026-10-25)

Profiling isolated eager history assembly (2.5–3.9 ms on CUDA at batch 32) and
replay of unaffected rows as substantial costs. `jit_replay/3` now accepts a
chronological tuple of frames and stacks them inside the executable.
`jit_replay_rows/3` fuses row gathering, replay and scatter, including a runtime
count mask for padded rows. ExPhil uses power-of-two selection sizes and warms
them during batch setup; all-due batches bypass selection. Resync scheduling,
initialization and retained-history semantics are unchanged.

For one of 32 rows due, complete CUDA resync medians fell from 7.397 to 1.820 ms
for MinGRU, 5.174 to 2.371 ms for GRU, 12.410 to 5.054 ms for Mamba, and 9.846
to 2.932 ms for SSD. These are backbone resync costs, not complete Agent calls.
CPU results, method, caveats and reproduction commands are in the
[resync profiling report](../bench/results/recurrent_resync_2026-10-25/README.md).

Full-Agent profiling also identified batched embedding as a larger ordinary-frame
cost: ExPhil now mirrors its single-frame path by assembling observations on the
host and transferring the completed batch once. Trained-checkpoint comparisons
and timing artifacts are in ExPhil `eval_runs/recurrent_resync_profile_20261025/`.

### Compiled replay follow-up (2026-10-25)

`Edifice.Stateful.jit_replay/3` now returns a cached function that consumes a
`[batch, time, features]` chunk and returns final carry. An Nx `while` loop
keeps compilation size bounded and runs the replay in one executable call.
The loop retains convolution history and supports nonzero starting state.
It does not change the recurrence or the resync schedule.

CPU replay-core medians, batch 1, two layers, input width 264, highest
precision, seeded synthetic weights, 40 samples after warmup:

| Architecture | Hidden/window | Cold: separate steps | Cold: compiled replay | Resync: separate steps | Resync: compiled replay |
|---|---|---:|---:|---:|---:|
| MinGRU | 256/80 | 10.945 ms | 1.007 ms | 10.609 ms | 0.979 ms |
| GRU | 512/60 | 12.892 ms | 4.603 ms | 14.675 ms | 4.917 ms |
| Mamba | 512/80 | 27.641 ms | 9.255 ms | 23.975 ms | 9.187 ms |
| Mamba-2/SSD | 256/60 | 11.137 ms | 2.756 ms | 11.385 ms | 2.770 ms |

Inputs/weights were resident and final carry was synchronized. These timings
exclude input assembly, embedding, action heads, first compilation and IPC.
The overnight ExPhil league was running concurrently, so these are indicative
CPU measurements rather than an idle-machine baseline. Results:
`bench/results/recurrent_replay_host_2026-10-25.json`; reproduce with
`mix run scripts/bench_recurrent_replay.exs --client host --out /tmp/replay.json`.

ExPhil integration was merged onto main after the live evaluation queue finished
on October 25, and the application compiled successfully. It uses replay for
single-agent cold starts, observe-only
cold starts, single-agent resync and batched cold/reset rows. Warmup compiles
replay on throwaway carry. Mamba-2 aliases `:mamba_2` and `:mamba_ssd` now have
Agent adapters there. Snapshot v2 also stores resync history/counters, frame
debounce, controller queue and stick-run state; this fixes deterministic replay
across resync boundaries. Per-row batched resync is now implemented in that
integration, with independent history lengths and counters, partial resets, and
observation-only warmup matching single-agent behavior. GPU validation is complete.
The press-window feature branch is also on main: the queue's earlier attempted
merge had failed, so it was integrated explicitly before activating this update.

CUDA replay-core cold medians (separate steps → compiled replay): MinGRU
16.808 → 1.733 ms, GRU 13.954 → 2.327 ms, Mamba 20.376 → 3.811 ms,
SSD 17.648 → 2.510 ms. Every carry tensor passes sequential/compiled parity.
The device was free at launch, but another CUDA test started near the end;
these are indicative measurements, not an isolated performance ceiling.
Artifact: `bench/results/recurrent_replay_cuda_2026-10-25.json`.

Actual ExPhil checkpoint Agent calls (including embedding, heads, dispatch and
output transfer) have the following medians in milliseconds:

| Backbone | CPU window / carry / cold / resync | CUDA window / carry / cold / resync |
|---|---|---|
| MinGRU | 2.457 / 0.661 / 1.617 / 2.562 | 9.896 / 6.548 / 9.086 / 16.835 |
| Mamba | 30.569 / 1.278 / 14.830 / 17.653 | 3.959 / 1.146 / 6.177 / 8.501 |
| Mamba-2 | 7.632 / 0.581 / 3.078 / 3.586 | 3.365 / 1.690 / 4.139 / 6.463 |

The CUDA full-Agent timings above were contended by another test process; do not use
them to rank CPU versus GPU or claim peak throughput. Cold/resync have only
3–6 samples. All three checkpoints pass cold-action parity, warmup state
preservation and exact rollback across resync on CUDA at highest precision.
Timing artifacts use portable kernels. A separate fused-on Mamba run confirms
the handler library loaded and the same Agent checks pass. The loader now
loads OTP crypto before the global CUDA library to prevent its lazy NIF crash;
a fresh Erlang subprocess regression exercises that load order directly.

A final CUDA pass used the compiled main checkout after all queue/test processes
finished. No competing GPU compute process was observed before or after each run.
These supersede the contended CUDA numbers above as the performance baseline:

| Backbone | Window median (p95) | Carry median (p95) | Cold median | Resync median (p95) |
|---|---:|---:|---:|---:|
| MinGRU | 2.838 (3.382) ms | 1.039 (1.548) ms | 2.753 ms | 4.474 (4.840) ms |
| Mamba | 2.591 (3.457) ms | 0.754 (1.115) ms | 4.644 ms | 6.649 (7.465) ms |
| Mamba-2 | 2.224 (2.753) ms | 1.070 (1.642) ms | 3.780 ms | 4.481 (5.022) ms |

Portable kernels, highest precision, resident parameters; 282–348 ordinary
samples, only 3 cold and 3–6 resync samples. First JIT and emulator IPC excluded.
Artifacts: ExPhil `eval_runs/recurrent_replay_20261025/*_agent_cuda_idle_timing.json`.

ExPhil's 32 seeded closed-loop runs compare windowed, carried, resync30 and
resync60 modes. MinGRU versus an active windowed opponent had self-destructs/min
of 3.125 / 5.0 / 4.0 / 4.25 respectively. Resync30 improves over indefinite
carry but does not recover windowed behavior. Mamba and Mamba-2 idle screens
are insufficient for promotion. Keep defaults unchanged; next validation is
larger active-opponent trials and more cold/resync samples.
Artifacts and protocol: ExPhil `eval_runs/recurrent_resync_20261025/RESULTS.md`
and `eval_runs/recurrent_replay_20261025/` on main.

### Original benchmark scope

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
Compiled chunk replay, batched resync and Mamba-2 Agent wiring are now
implemented (see the October 25 follow-up above). Initial full-policy GPU timing
and closed-loop screens are complete, with contention and sample limits above.

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
3. **Integrate GatedDeltaNet into ExPhil when needed.** Its Edifice stateful
   contract exists; the ExPhil adapter remains pending. Mamba-2 parameter
   extraction, reset, batching and head parity are implemented and validated on
   main after integration of `feat/compiled-replay`.
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
