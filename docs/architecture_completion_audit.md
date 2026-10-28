# Architecture completion audit for ExPhil and Melee

## Canonical Mamba-2 — October 28, 2026

**AI-generated.** A separate `:mamba2` architecture now implements the standard
full-SSM mixer, with learned per-head A/D, joint x/B/C convolution, grouped
states and gated RMS normalization. It supports carried stepping and compiled
replay. Full-block reference output and gradient checks pass on portable/EXLA
host paths. See [scope, usage and reference limits](mamba2.md), including the
authorship poem. This is a portable implementation, not fused GPU training or
whole-language-model checkpoint certification. Existing `:mamba_ssd` and ExPhil
adapters retain their previous behavior.

## Current follow-up — October 27, 2026

**Reference certification follow-up:** [GRU, SSD, Mamba and MinGRU validation](recurrent_reference_validation.md)
adds 14 reference tests, with 52 targeted tests passing overall on portable/EXLA
host paths. GRU checks use the actual LibTorch operator; SSD/selective scan use
pinned Rust ports of upstream equations. The audit found and fixed the portable
selective scan ignoring its A argument. It also establishes that `MambaSSD` is
an Edifice Mamba variant with an SSD algorithm, not canonical Mamba-2; Mamba's
builder has fixed-A/no-D restrictions; MinGRU uses signed candidates. These
limits prevent a blanket claim of upstream checkpoint compatibility.

**AI-generated.** The October 26 review below is historical. Since then,
staggered resync became a supported ExPhil opt-in and the carried-state
investigation reached a bounded controller-choice check: no deterministic
changes in 4,096 matching-replay decisions per policy; shared-key sampling
changed zero Fox decisions and one Mewtwo decision. See the
[current recurrent report](recurrent_inference_benchmarks.md). Keep f32 inference;
longer closed-loop quality validation remains separate from numerical checks.

The next completed Edifice gap is streaming KV-cache support. Both
`Generate.generate_stream/3` and `Generate.token_stream/3` now accept the existing
`cache:` option and share normal generation's sampling, stop handling and exact
window-refill behavior. The cache avoids recomputing previous tokens until full;
at capacity it still rebuilds the retained window to preserve semantics.

This also fixes callback generation at `max_tokens: 0` and `1`, oversized
prompts, and prompts already ending in a stop token. Lazy enumeration no longer
predicts an extra unused token. Streaming is explicitly single-prompt; batched
generation remains available through `generate/3`. Callback generation includes
a newly generated stop token; lazy token streams omit it, as before.

Validation: 33 focused serving/cache tests pass, including seeded greedy/stochastic parity,
cache overflow, compiled EXLA host decoding, callback halt, lazy prediction-call
counts, and invalid batch/capacity rejection. This is a reusable serving
improvement, not a measured speedup for ExPhil's recurrent bots.

```sh
devenv shell -- mix test test/edifice/serving/decoder_cache_test.exs \
  test/edifice/serving/generate_test.exs --include exla_only
```

This machine's default dependency directory is stale. Validation used the existing
upstream-pinned build with `MIX_DEPS_PATH=/tmp/edifice-upstream-deps` and
`MIX_BUILD_PATH=/tmp/edifice-upstream-build` prefixed to that command.

Remaining priorities: deployed-checkpoint quality and scheduling validation in
ExPhil; architecture reference fidelity for the documented approximations below;
and targeted optimization when a deployed workload profile identifies a cost.
Activation checkpointing still waits for the user's Nx work. No new recurrent
architecture or sparse CUDA kernel is justified merely by its presence on a list.

**AI-generated poem**

> Remember the tokens already known;
> Compute the next when its need is shown.

## Earlier ExPhil-focused completion review — October 26, 2026

**AI-generated review.** Source and archived evidence were checked against
ExPhil main `2f457fb3` and Edifice through `297b1f6`. This was read-only in ExPhil;
its unrelated work and active GPU process were left alone. No new gameplay or
GPU timing run was performed. The newer October 25–26 reports below supersede
the September handoff's task status and the older benchmark recommendations.

The immediate bottleneck is deployment validation and scheduling, rather than
another recurrent architecture implementation. The basic step, replay, batching,
and stochastic-sampling paths already exist and have measured benefits.

| Area | What is complete | What remains |
| --- | --- | --- |
| Recurrent stepping | Edifice paths and ExPhil Agent adapters for GRU/LSTM, MinGRU, Mamba and Mamba-2/SSD; compiled history replay, selected-row replay, resets and rollback checks | Confirm the exact deployed checkpoint/runtime combination; keep each checkpoint's training-state contract |
| Stochastic controllers | ExPhil uses `SamplingRandom` at all eight uniform-draw call sites; seeded stream parity and host/CUDA sampling tests | Revalidate on dependency changes; no further sampler rewrite is justified by the current evidence |
| Multiple bots | CPU and CUDA actor screen complete: eight profiles, 96 phases, verified raw samples | Longer runs with actual simulator/emulator and training contention; current phases last three seconds and use a synthetic matmul competitor |
| Mamba resync spreading | Two ten-second rounds per condition, CPU/CUDA, independent/batched, idle/loaded | Supported phase-assignment API, reset/snapshot tests, and gameplay validation; current scheduling changes are confined to a benchmark helper |
| Numerical compatibility | Original vision fixtures pass; Whisper upstream erf fix adopted; strict f64 encoder checks and recorded speech pass; bounded one-file transcription has exact reference token parity | Whisper is parked here; no dictation application, new training, or long-form transcription is needed for ExPhil |
| Architecture corrections | NSA/MoBA gathered execution, RetNet head normalization, and actual reservoir spectral scaling implemented | Paper-fidelity limits remain documented; these are not current recurrent-bot bottlenecks |
| Activation checkpointing | Existing boundary/documentation work | Real recomputation waits for the user's upstream Nx work; do not count pass-through helpers as memory savings |

### What the measurements actually recommend

The [full-controller sampling report](../../exphil/eval_runs/sampling_kernels_20261026/RESULTS.md)
records GPU ordinary-call speedups of 2.17–3.76x with identical seeded controller
streams. Eight-row calls take about 1.08–1.33 ms in those fixtures. CPU effects
are mixed, so the CUDA improvement should not be advertised as a universal win.

The [actor throughput screen](../../exphil/eval_runs/actor_throughput_20261026/RESULTS.md)
is **complete**, not a task to resume. Same-checkpoint GPU batches deliver about
6,375–7,812 actions/s idle and 3,403–3,773 under the controlled load. All four
batch profiles meet every deadline in their short paced phases. Independent
Agents can miss deadlines despite sufficient average throughput. This points
to batching and scheduling, not merely faster arithmetic. The experiment does
not establish heterogeneous-checkpoint batching or long-run training throughput.

The [resync spread experiment](../../exphil/eval_runs/resync_spread_20261026/RESULTS.md)
assigns different bots different replay frames. It does **not** divide one bot's
replay over multiple frames. On idle GPU, independent Mamba misses fall from
462/464 per 4,800 calls to zero in both rounds. CPU batch misses fall from
29/24 per 600 calls to 1/0. GPU batches already miss no deadlines when aligned;
spreading increases their ordinary p95 from about 3.6 ms to 7.2–7.6 ms. Therefore
phase assignment should be explicit, with separate choices for each backend
and deployment mode, rather than a blanket new default.

There is a correctness qualification: the spread harness reports batch/single
carry differences beyond its original elementwise `1e-4` test in **both**
schedules (135/133 CPU comparisons and 18/19 GPU comparisons). Timing passed
finite-state and counter checks, not full numerical parity. Do not silently
relax the test or describe this as a proved scheduling bug. First isolate
single versus batch step/replay using the same history, carry, weights and
precision; compare an independent/high-precision recurrence if needed. Record
whether differences alter controller choices or rollback behavior.

### Recommended order and completion criteria

1. **Establish the runtime baseline and isolate Mamba batch/single discrepancies.**
   ExPhil's devenv selects sibling Nx/EXLA with `EDIFICE_LOCAL_NX=1`. Its existing
   application metadata says Nx/EXLA 1.0.0 and Axon 0.8.0, while its Hex lock has
   Nx 0.12.1 / EXLA 0.12.0. These version strings do not prove that it uses
   Edifice's newly pinned XLA 0.11 build. Capture the loaded build, then reproduce
   the carry comparison without changing a running job's libraries. Completion
   means a reproduced cause, an appropriate regression, and a documented
   controller/state contract—not merely passing a wider numerical threshold.
2. **Make resync phases a supported opt-in integration.** The current
   `scripts/lib/resync_spread.exs` changes private counters with `:sys.replace_state`.
   A real option must preserve cold start, partial row resets, snapshot restore,
   observation-only updates and the maximum replay interval. Keep BPTT-trained
   policies exempt from periodic window resync. Run sustained CPU-batch and
   independent-GPU deadline checks; retain aligned GPU batching as the baseline.
3. **Validate at the checkpoint's trained delay in the real deployment harness.**
   The [96-trial delay screen](../../exphil/eval_runs/recurrent_delays_20261025/RESULTS.md)
   is already complete. It used delay-zero-trained checkpoints at delay 3/4,
   and found mixed quality, including regressions at delay 4. Repeat neither
   that exact screen nor a claim that resync always preserves play quality.
   Use appropriate trained checkpoints, multiple opponents/stages, paired seeds,
   controller delivery latency, and actual game outcomes before promotion.

Existing work to reuse: `/home/blewf/git/exphil-nx-upgrade`, branch
`chore/framework-baseline` at `af856e27`, contains build fingerprinting, saved-step
resume checks, and a bounded 32-update / eight-game framework comparison. It is
not merged into ExPhil main. Its guide says longer and closed-loop adoption gates
remain; some orchestration is old Python and should be replaced with Elixir if
resumed. This is reusable migration evidence, not proof for the new dependency
pin or an instruction to restart an old agent or training job.

Lower priority: GatedDeltaNet has an Edifice step implementation but is absent
from ExPhil's Agent stateful allowlist. Add that adapter when a real candidate
checkpoint makes it useful. Mamba-2's adapter is already integrated; its archived
leaked-label checkpoint is a speed fixture, not a clean policy-quality candidate.
Fusing recurrence and sampling, or adding sparse CUDA kernels, should follow a
profile showing a remaining cost in an intended deployment workload.

**AI-generated poem**

> The clocks are measured; paths are known.
> Now test the bot where it is flown.

## Earlier audit and implementation inventory

Audited October 24, 2026. Scope: source inspection of approximation markers,
bot-relevant stateful and generation paths, small cross-framework fixtures, and
the existing pretrained numerical suite. This is not a paper-by-paper certification
of every architecture. The subsequent recurrent follow-up enables MinGRU
stepping in ExPhil (see the linked benchmark report). Real activation
checkpointing remains deferred until the upstream Nx checkpoint work merges.

## Completed in this pass

- **PPO correctness:** inference-mode policy evaluation for both collection and
  differentiation; correct next-observation bootstrapping; environment continuity
  across rollouts; singleton minibatch updates; global gradient clipping before
  Adam. Prediction and the complete optimizer update can compile with EXLA.
  This trainer supports stateless categorical PolicyValue models. ExPhil's own
  recurrent training loop still needs a separate integration review.
- **GQA KV decoding:** `DecoderOnly.Cache` reuses trained backbone parameters and
  projects only new frames. Supports RoPE, NoPE, interleaved RoPE, input projection,
  optional final normalization, chunked prefill and rollback.
  `Generate.build_cache/1` integrates token embedding and the LM head in the same
  executable. Sampling and context-overflow behavior match full prediction.
- **Numerical validation executed:** eight small random-weight fixtures pass.
  The earlier Whisper failure is resolved by the upstream math fix and separate
  f32/f64 validation. All six current pretrained tests pass. See
  [results and commands](plans/pretrained-validation-fixes.md).

## Next priorities

The recurrent microbenchmark is now complete; see
[measured CPU/GPU timings and ExPhil integration gaps](recurrent_inference_benchmarks.md).
Actual-checkpoint MinGRU/Mamba complete-policy latency and state/controller parity
were validated October 25 in ExPhil. Full-update Mamba batch-8 CUDA medians are
1.056 ms ordinary and 5.456 ms for one-row resync; see ExPhil's
`eval_runs/recurrent_resync_profile_20261025/RESULTS.md`.
The follow-up adds GatedDeltaNet and Mamba-2 step paths in Edifice and enables
MinGRU's adapter in ExPhil. SSD's matmul source-decay bug was corrected; old
checkpoints using that path need revalidation (details in the benchmark report).

| Priority | Finding and evidence | Useful next action |
|---|---|---|
| 1 | Actual-checkpoint MinGRU/Mamba replay and host embedding now pass CUDA baseline parity | Extend full-policy profiling to other deployed architectures and evaluate closed-loop quality separately |
| Completed October 25 | NSA/MoBA causal routing, actual selected-block execution and partial lengths fixed; local NSA attention uses gathers | [CPU/CUDA measurements and remaining paper-fidelity limits](../bench/results/sparse_attention_2026-10-25/README.md). Short contexts need not be faster; no fused sparse CUDA kernel yet |
| Completed October 26 | [RetNetV2](../lib/edifice/attention/retnet_v2.ex) now uses per-head normalization | Preserve CPU/CUDA normalization and prefix tests |
| Completed October 26 | [Reservoir](../lib/edifice/recurrent/reservoir.ex) now scales using actual nonsymmetric eigenvalues | Preserve radius tests; initialization cost is separate from inference |
| Completed October 26 | ViT attention heads, position table and pretrained layout corrected; ViT and ConvNeXt pass original numerical bounds | Keep numerical and focused regressions |
| Completed October 26 | DETR positional encoding, projection order, backbone padding and query-weight loading corrected | Class logits and boxes pass original numerical bounds |
| Completed bounded audio path | Whisper CPU encoder parity and short English transcription validated | [Details and limits](../tools/whisper_reference/README.md); return focus to ExPhil |

## Other approximation findings

- **GatedSSM** is its own gated/cumulative approximation, not interchangeable with
  Mamba. **SSTransformer** uses mean-dt gating and cumulative averaging rather
  than a full selective state recurrence. Label and benchmark these as distinct
  models. Mamba's per-channel dt fix and stateful parity tests do not establish
  compatibility with every external Mamba checkpoint.
- **Hymba** actually calls SSM discretization and a Blelloch scan despite a
  “simplified SSM” comment. That comment alone does not prove a placeholder;
  complete topology fidelity still needs reference validation.
- **MAE** uses an MLP-mixer-style encoder without transformer self-attention.
  **JEPA** operates on flat vectors with single-token attention. **EfficientNet**
  explicitly implements a dense 1D variant. None should be treated as a drop-in
  canonical image/video backbone solely from its name.
- **TRELLIS** now has a real voxel density codec, but its attention still derives
  Q/K/V from the same input without independent learned projections.
  **CogVideoX** uses pooling/averaging substitutions in its convolutional path.
  **GaussianSplat** uses only the DC spherical-harmonic component.
- **VAR** uses sinusoidal positions despite a learnable-position comment;
  **FreeTransformer** substitutes a dense latent encoder for bidirectional
  cross-attention; **DPO** constructs deterministic embedding values despite a
  learned-embedding comment. These are lower priorities for frame-vector bots.

Approximation is not automatically a defect: MinGRU/MinLSTM simplification and
S4D's diagonal structure are part of their definitions. Performer/Nystrom and
similar approximate attention methods need algorithm-specific reference checks.
Keyword searches alone cannot establish missing functionality.

## Using the cache for frame-vector policies

```elixir
opts = [embed_dim: 64, hidden_size: 64, num_heads: 4, num_kv_heads: 2,
        num_layers: 2, seq_len: 60, dropout: 0.0]
model = Edifice.Transformer.DecoderOnly.build(opts)
# Train/load params using this model's normal Axon parameter structure.
cache = Edifice.Transformer.DecoderOnly.Cache.build(opts ++ [compiler: EXLA])
state = cache.init.(1)
{features, state} = cache.predict.(params, frames, state)
# frames: {1, new_frame_count, 64}; features: {1, new_frame_count, 64}
saved = Edifice.Stateful.snapshot(state)
# Restore saved to replay from the same position after rollback.
```

For token generation, build the LM normally, then pass
`cache: Edifice.Serving.Generate.build_cache(model_opts ++ [compiler: EXLA])`
to `Generate.generate/3`, `Generate.generate_stream/3`, or `Generate.token_stream/3`
with the same `seq_len`. Validation currently covers f32 GQA without YaRN; other attention
types and YaRN are rejected.

The cache has a fixed capacity. Raw frame callers must reset and prefill the
retained window when it fills; generation does this automatically. Dropping only
old K/V entries would retain higher-layer activations derived from evicted
context and would not match a freshly recomputed sliding window. Consequently,
the exact sliding-window path returns to full prefill on every step after
capacity. Indefinite fast attention inference requires an explicit changed
context contract or a different architecture, not unnoticed cache eviction.

## Validation and benchmark

Targeted RL, generation, decoder, RoPE and stateful tests, including EXLA tests:
**152 tests and 2 properties, zero failures**. These cover terminal and truncated
rollout bootstrapping, parameter updates, rollback, chunk/token parity, sampling,
external embeddings, stop handling and context overflow.

```sh
devenv shell -- mix run scripts/bench_decoder_cache.exs
```

The CPU benchmark checks parity and measures a warmed single-frame decode against
the full last-timestep output at batch 1, window 60, hidden 64, two layers, EXLA
host. It includes output synchronization and the cache bounds check, but excludes
compilation and prefill. It is not a GPU or end-to-end ExPhil measurement.

A standalone run measured full prediction at **340 µs median / 420 µs p95**
versus cached decode at **78 µs median / 85 µs p95** (4.36× median speedup).
Earlier runs varied with host load; measure on the deployment machine.
