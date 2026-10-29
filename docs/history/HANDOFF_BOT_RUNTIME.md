# Handoff: edifice bot-runtime improvements (2026-07-15)

You are working in `~/git/edifice-dev`, a **git worktree** of `~/git/edifice`
on branch `feat/bot-runtime`. Read this whole file before touching anything.
This doc is self-contained — your task list is NOT shared with the session
that wrote this.

## WHY YOU ARE IN A WORKTREE — safety rules (read first)

A GPU training run (exphil "Round 8", possibly later rounds) is live on this
machine, expected to run into the evening and possibly overnight. It was
compiled against the **main checkouts** and its loop script re-invokes
`mix run` per stage, recompiling any changed `.ex` files it sees. Three
distinct hazards, three rules:

1. **NEVER edit files in, or run `mix` in, `~/git/edifice`, `~/git/exphil`,
   or `~/git/nx`.** The training loop compiles `~/git/exphil` with
   `../edifice` and `../nx/*` as path deps; edits there get injected into
   its next stage (a mid-loop compile error killed two training rounds on
   2026-07-15), and any `mix` invocation there can replace the EXLA NIF
   `.so` out from under the running BEAM (SIGSEGV). Reading those trees is
   fine; writing/compiling is not. Work only in `~/git/edifice-dev`.
2. **Do NOT set `EDIFICE_LOCAL_NX`.** With it set, this worktree's mix would
   use `../nx/nx` + `../nx/exla` path deps and compile EXLA's NIF **in the
   shared `~/git/nx` source tree** — same crash. Leave it unset so hex deps
   (`nx ~> 0.11`, `exla ~> 0.11` optional) vendor into this worktree's own
   `deps/` + `_build/`, fully isolated.
3. **Stay off the GPU.** The training run holds ~31.2 of 32.6 GB on the
   RTX 5090. Run tests on `Nx.BinaryBackend` (edifice's default test
   backend); if you must exercise EXLA, force the host client
   (`EXLA_TARGET=host`, CPU). Do not benchmark on GPU — latency numbers
   under contention are garbage anyway. Mark any perf-sensitive work
   "re-measure on idle GPU" and move on.

Environment: `devenv shell` from this directory (devenv files copied in;
if direnv/devenv complains about the worktree, `cd ~/git/edifice-dev &&
devenv shell` explicitly). `mix deps.get && mix test` should pass before
you change anything — establish that baseline first.

Git: commit early and often ON THIS BRANCH (`feat/bot-runtime`). Push the
branch to `origin` freely. Do NOT merge to main, do NOT push main, do NOT
touch other branches. The main session reviews and merges after training
ends.

## CONTEXT — what this is for

Edifice is a 232-architecture Nx/Axon library; exphil (`~/git/exphil`,
READ-ONLY to you) is a Melee AI lab that consumes it. The goal is a
competitive low-tier Melee bot (Mewtwo). Two constraints drive everything:

- **16.67 ms/frame** (60 FPS) inference budget, batch=1, on-GPU eventually.
- **Netplay rollback** (Slippi online): frames get re-simulated when remote
  inputs arrive late, so a stateful policy must snapshot/restore its
  recurrent state or it desyncs from the game it thinks it's playing.
  Bot's netplay debut target: August 2026.

Current inference re-encodes a window of frames every frame — O(window)
work for O(1) new information. The recurrent-family backbones (Mamba, GRU,
GatedSSM, RWKV, xLSTM, S4, Griffin...) all admit O(1) stepping, but edifice
has no uniform API for it (one ad-hoc `step` exists in `ssm/gated_ssm.ex`).

## TASKS, in priority order

### 1. Model manifest — self-describing checkpoints

**Problem observed in the wild (2026-07-15):** SSM shape params passed at
build time weren't threaded through checkpoint load, so inference silently
rebuilt mamba exports with DEFAULT shapes — a silent architecture mismatch
producing garbage outputs (cf. exphil GOTCHAS #29 for the class).

**Design sketch:**
- `Edifice.Spec` struct: `%{arch: atom, build_opts: keyword, edifice_version,
  nx_version, created_at}` (created_at passed in, not Date.now-anything).
- Capture at the central build entrypoint (find it: `lib/edifice.ex` /
  `recipes.ex` / `training.ex` — wherever `Edifice.build(arch, opts)` or
  equivalent dispatches to per-arch builders). If build is per-module with
  no central funnel, add a thin `Edifice.build/2` funnel and capture there.
- `Edifice.Checkpoint.save/load`: embed the spec in the serialized map;
  on load, if a spec is present, **rebuild from spec and validate** the
  param shapes against the stored params — shape mismatch = loud
  descriptive error, not garbage. Absent spec (old checkpoints) = warn once.
- Tests: round-trip with non-default build opts; corrupted/absent spec;
  the mamba-shape-param scenario specifically (non-default d_state or
  similar, load without passing opts, assert it comes back correct).

### 2. Stateful step contract — `init_state/2`, `step/3`, snapshot/restore

The strategic one (latency + rollback correctness).

**Design sketch:**
- A behaviour (e.g. `Edifice.Stateful`): `init_state(model_or_spec, opts)`,
  `step(params_or_predict_fn, state, frame_input) -> {output, new_state}`.
  State must be a plain Nx container (map/tuple of tensors) — that makes
  snapshot/restore free (`Nx.backend_copy/serialize` round-trip) — plus
  document + TEST the snapshot→steps→restore→same-outputs property, since
  that's the rollback primitive.
- Implement for the backbones exphil actually runs first: **Mamba, GRU,
  GatedSSM** (generalize `gated_ssm.ex`'s existing ad-hoc step; check
  `recurrent/recurrent.ex` and `ssm/` for how hidden state is threaded in
  the full-sequence forward).
- **The correctness pin that matters:** for each impl, property-test that
  iterated `step/3` over a sequence reproduces the full-sequence forward
  outputs (same params, same inputs, atol ~1e-5 on BinaryBackend). If that
  equivalence holds, everything downstream is trustworthy.
- Rollback test: run N steps, snapshot at k, run to N, restore snapshot,
  re-run k..N with identical inputs → identical outputs.

### 3. `Profile` step-latency mode

`lib/edifice/profile.ex` has `run/2` + `compare/1` for forward passes. Add
a `:step` mode: batch=1, measure per-`step/3` latency (p50/p95) and state
memory footprint across the backbones implementing task 2's behaviour.
CPU numbers only from this worktree (GPU is occupied) — structure the
report so re-running on idle GPU later is one command. This is the
harness that will pick the bot's backbone; make output comparable across
archs (same table shape as `compare/1`).

### 4. (stretch) Upstream generic interp: Probe / Attribution / Activations

Port the architecture-generic parts of exphil's interp toolkit
(`~/git/exphil/lib/exphil/interp/` — probe.ex, attribution.ex,
activations.ex; READ-ONLY reference) into `lib/edifice/interpretability/`,
following the precedent of LEACE (exphil `Erase` → edifice
`Interpretability.LEACE`, which gained 12 functional tests in the move —
see `test/edifice/interpretability/leace_test.exs` for the house standard:
functional guarantees, not shape checks). Melee-specific parts
(ground_truth.ex, replay_stats.ex) stay in exphil — port the mechanisms,
not the features. Known issue to keep in mind (do not fix blind): probe
training via `Erase.verify`-style paths burned 20+ CPU-min in exphil while
zoo-style probes were fast — unprofiled; if you port Probe, profile its
training loop and note findings.

## KNOWN UPSTREAM NX LANDMINES (avoid, don't fix here)

- `Nx.Defn.while` autodiff is WRONG when the body's Jacobian wrt the
  accumulator depends on the differentiated variable (nx#1747, open).
  Don't introduce `while` in any grad path — unrolled `Enum.reduce` (the
  existing fallback-scan pattern in `cuda/fused_scan.ex`) or custom_grad.
- Vectorized grads through cholesky/triangular_solve/cond fail
  (nx#1729/#1730, open). Don't combine vectorize+grad+LinAlg.
- `Nx.LinAlg.eigh/svd` on EXLA = XLA-compile trap for small matrices —
  BinaryBackend + `Nx.default_backend/1` scoping (edifice CLAUDE.md).
- Loss/objective math in f32 at entry (CLAUDE.md precision policy).
- hex nx here is 0.11.x (pin predates 0.12) — do not depend on 0.12-only
  APIs (`Nx.block`, new padding modes); a version bump happens separately.

## DELIVERABLES

Commits on `feat/bot-runtime`, pushed to origin. For each task: code +
tests green on BinaryBackend (`mix test` in THIS worktree only) + a short
section appended to this file under "## RESULTS" (what shipped, what's
pinned by which test, what needs the idle GPU or the main session).
Priority over completeness: manifest (1) fully done beats four things
half-done. Task 2 > 3 > 4 after that.

## RESULTS

### Task 1 — Model manifest (commit 5d0b03b)

Shipped: `Edifice.Spec` (new `lib/edifice/spec.ex`), `Edifice.build_with_spec/3`
(captures merged opts at the registry funnel; `build/2` refactored onto shared
`resolve!/1`, behavior-identical), `Checkpoint.save(..., spec:)` embedding the
spec as a plain map under `"__edifice_spec__"` ([:safe]-deserializable),
`Checkpoint.fetch_spec/1` (header-only read), `Checkpoint.load_model/2`
(rebuild-from-spec + shape validation, `validate: false` escape hatch),
warn-once on spec-less loads. Safetensors path warns + skips (dep has no
metadata write support). Tuple-returning archs skip shape validation.

Pinned by: `test/edifice/spec_test.exs` (round-trip, [:safe] survival,
garbage tolerance, merged-registry-defaults capture) and
`test/edifice/checkpoint_test.exs` "model manifest" block — including THE
mamba scenario (state_size: 4 saved, `load_model` with no opts reproduces
original outputs at atol 1e-6; lying spec with default state_size raises
naming both shapes).

Baseline note: this worktree's full suite has 17 pre-existing failures + 1
invalid setup_all, ALL attributable to hex nx 0.11 vs the fork main runs
(13× fork-only `Nx.runtime_call/4` in Training.Monitor/Adaptive/Heatmap/
AutoTuneProfiler, 4× nx 0.11 vectorized-grad bugs, 1× ShardingTest
setup_all returns :skip). Failure set verified byte-identical before/after
task 1 (full `--stale` run = whole suite, 212s).

Environment note for whoever works in this worktree next: `direnv allow`
regenerates the gitignored `devenv.lock` with broken newer inputs — the fix
was copying main's known-good lock; `.nvshmem`/`.cuda-compat` are symlinked
from the main checkout (EXLA NIF dlopen needs them).

### Task 2 — Edifice.Stateful step contract (commits 8e0f872, 6108f5f, 9b61375, 8d3dfb3)

Shipped: `Edifice.Stateful` behaviour (`init_state/2`, `step/3`; state =
plain Nx container so `snapshot/serialize/deserialize` — the rollback
primitives — are free), `Edifice.Stateful.Ops` (delegates to
Axon.Layers/Activations so step math matches builders by construction),
registry dispatch via `Edifice.init_state/3`, `Edifice.step/4`,
`Edifice.stateful?/1`. Implementations: **MinGRU** (state `%{h}`), **Mamba**
(`%{h, conv}` — learned depthwise conv ring buffer; step helpers in
`Common` for other mamba variants to adopt), **GatedSSM** (new opt-in
`scan_mode: :causal`; the legacy scan is pointwise seq_len-dependent
weighting, NOT a recurrence, so the old ad-hoc `step/4` never matched the
forward — now `@deprecated` with that documented; **default remains
`:legacy`** so nothing trained on main changes semantics), **GRU/LSTM**
(both graph layouts; Axon layout replicates the glorot-from-stored-key
initial hidden state; fused layout pinned synthetically against the
public FusedScan CPU fallbacks).

Pinned by (`test/edifice/stateful/`): step == full forward at EVERY prefix
length, atol 1e-5 (incl. Mamba seq-40 Blelloch branch at 1e-4, conv
warm-up edges, property tests over seeds/lengths/batches); bitwise
rollback replay through the serialize wire format for all five archs;
GatedSSM `:legacy`-default bitwise regression guard; state-is-pure-tensor
container checks. Note: rwkv/hgrn/liquid/s5 have ad-hoc `init_cache`s that
are future candidates for this behaviour.

### Task 3 — Profile :step mode (commit 290594f)

`Profile.run/compare(mode: :step)`: batch=1 p50/p95/mean/max per-step
latency, `init_ms`, `state_bytes` (rollback snapshot cost), step-specific
table + backend footer. One-command rerun: `mix run bench/step_latency.exs`
(Melee dims); GPU re-measure = `EXLA_TARGET=cuda` prefix ON IDLE GPU.
CPU (BinaryBackend) sanity numbers — latency meaningless on this backend,
state footprints real: min_gru 2KB / gru 2KB / lstm 4KB / gated_ssm 16KB /
mamba 76KB.

### Task 4 + INTERP_AUDIT fixes (commits cc5d6c6, 5ded33a)

Read `~/git/edifice/INTERP_AUDIT_2026-07-15.md`; did the port and the
audit remediation as one piece (per its coordination note).

Audit concrete fixes: `l1_coeff` nil-crash fixed everywhere via `keyword!`
defaults; f32-at-loss-entry policy applied to every interp `loss/4`;
`Crosscoder.loss` made runnable (was Enum.zip-over-lists inside defn —
now stacks and delegates to defn `stacked_loss/4`); quarantine status
warnings on the four broken modules (GatedSAE zero-gradient gate,
MatryoshkaSAE missing nested-prefix loss + phantom `multi_scale_loss/4`
doc, CrossLayerTranscoder acausal/no-loss + stale `verified: true` marker
removed, DASProbe relabeled rank-limited probe with false claims removed).

Port (audit steps 1+3): `Edifice.Interpretability.Probe` (fit_eval with
balanced accuracy, weighted CE from logits, train-only standardization,
-1 masking, single-class guard, majority baseline, shuffled control;
default compiler Evaluator — pass `compiler: EXLA` for real workloads) and
`Edifice.Interpretability.Attribution` (grad×input saliency + group
shares, `:select` for multi-head outputs). Tests are functional
guarantees per the LEACE house standard. **Probe profiling note
resolved**: the exphil 20+ CPU-min mystery was BinaryBackend row gathers
(single-threaded `Nx.take` on {32k,256}), not the training loop —
documented in the Probe moduledoc; `fit_eval` reports `train_ms`.

NOT done (audit items left for the main session): `{reconstruction,
hidden}` container outputs on SAE builds, shared SAE trainer + decoder
unit-norm renorm, BatchTopK inference threshold + AuxK, real JumpReLU/DAS/
CLT/Matryoshka rewrites, activation-capture port (almost entirely
exphil-specific — port the thin tensor plumbing only when the SAE trainer
lands and needs it).
