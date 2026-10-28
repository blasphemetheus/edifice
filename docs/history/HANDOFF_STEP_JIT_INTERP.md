# Handoff 2: step-path JIT + interp fit infrastructure (2026-07-15 evening)

You are the edifice-dev worktree session. Your previous branch
(`feat/bot-runtime`) was reviewed on the local CUDA stack (169 tests +
2 properties, 0 failures — your 17 hex-nx baseline failures all resolved
on the fork stack as you predicted) and **merged to main** (`5b84d62`),
which is pushed to origin along with the interp audit doc your fixes
addressed (`INTERP_AUDIT_2026-07-15.md`, at repo root). Excellent work.
Two new tasks below, both direct follow-ups.

## Setup (changed since last time)

- `git fetch origin && git checkout -b feat/step-jit-interp origin/main`
  — your old branch is merged; start fresh from updated main.
- Your devenv.lock/env notes from last time still apply (don't
  `direnv allow`-regenerate the lock; copy main checkout's if needed).
- **The hex nx pin is now `~> 0.12`** — `mix deps.get` pulls nx/exla
  0.12.x. Your old baseline (17 failures, all nx-0.11 artifacts) is
  stale: REBASELINE first (`mix test` before any change; several of
  those failures — the vectorized-grad ones at least — should vanish
  on 0.12).

## SAFETY — same three rules as last time, still binding

A NEW overnight training run (exphil round 9) is live until ~01:00,
compiled against the main checkouts:
1. Never edit or run `mix` in `~/git/edifice`, `~/git/exphil`,
   `~/git/nx`. Reading is fine.
2. Do NOT set `EDIFICE_LOCAL_NX` (shared NIF hazard — this bit for
   real today: a `fine` version mismatch between consumers' locks broke
   the shared NIF's exports).
3. Stay off the GPU: BinaryBackend or `EXLA_TARGET=host` only. The
   trainer holds ~24 GB (fraction lowered to 0.75, but the headroom is
   not yours tonight).

## TASK A — JIT the Stateful step path (board #21, priority)

Your idle-GPU bench rerun happened: **p50 78–393 ms/step on the 5090**
(min_gru 78.7, gru 214, lstm 286, mamba 279, gated_ssm 393), with
p50≈p95 across the board. That flat profile is eager op-by-op EXLA
dispatch (launch+sync per Nx op), not compute — CPU BinaryBackend
masked it because it's op-by-op anyway. The 60 fps budget is 16.67 ms;
the rollback path for the August netplay debut needs this fixed.

Design intent (adapt as the code dictates):
- Compile each arch's `step/3` once via `Nx.Defn.jit` (or
  `EXLA.jit`-style compiler option) at `init_state/2` time; cache the
  compiled fun keyed by (arch, param shapes, state shape) — in the
  returned state handle or an ETS/persistent_term, your call, BUT the
  state itself must remain a plain Nx container (the snapshot/restore
  rollback property depends on it; keep the compiled fun OUT of the
  serialized state).
- Accept a `compiler:` opt (default Evaluator for tests, EXLA for
  production) mirroring what you did for `Interpretability.Probe`.
- All five existing impls (min_gru, mamba, gru, lstm, gated_ssm) +
  keep the step≡full-forward equivalence and rollback-replay property
  tests green — they are the correctness contract; if JIT breaks one,
  that's a finding, not a test to relax.
- Measure on EXLA host (CPU): report before/after per arch in RESULTS.
  The authoritative GPU rerun (`EXLA_TARGET=cuda mix run
  bench/step_latency.exs`) happens from the main session on idle GPU —
  structure the bench so that's still one command. Success bar there:
  min_gru/gru < 2 ms, mamba < 5 ms.

## TASK B — interp fit infrastructure (board #15, steps 1–2 of the audit)

Read `INTERP_AUDIT_2026-07-15.md` (repo root, now committed). Your
cc5d6c6/5ded33a commits already did the quick fixes + Probe/Attribution.
Remaining, in audit order:

1. **`{reconstruction, hidden}` container outputs** on every SAE +
   transcoder `build/1` (the audit's family-wide blocker: losses need
   hidden activations; models hide them). Keep backward compat if
   cheap (an opt like `output: :reconstruction | :container`) —
   registry consumers currently expect single-output.
2. **Shared SAE trainer**: `fit(model_params_or_build_opts,
   activations_matrix, opts) -> trained_params` — full-batch or
   minibatch defn loop (your Probe trainer is the house pattern),
   f32 at loss entry (already done in the losses), **decoder
   unit-norm renorm each step** (the audit: every L1 objective is
   degenerate without it).
3. **BatchTopKSAE to production**: inference-time fixed threshold
   (running mean of the min selected activation during fit — without
   it feature firing depends on batch composition, which poisons
   ground-truth scoring), AuxK dead-feature loss, and a LEACE-grade
   functional test: plant a sparse dictionary, fit, verify recovery
   (feature-to-planted-atom matching above chance, reconstruction
   beats mean-baseline).
4. If time remains: the quarantined modules stay quarantined (rewrites
   are a later phase, not tonight).

## Deliverables

Commits on `feat/step-jit-interp`, pushed to origin. Tests green on
your rebaselined suite (targeted runs while iterating — full suite is
~9 min). Append RESULTS to THIS file: per-task what shipped, what's
pinned by which test, CPU bench numbers, and anything you had to
decide that the main session should review before merge. Do NOT merge
to main. Task A > Task B if you must choose.

## RESULTS

### Rebaseline (nx 0.12)

nx 0.12.1 / exla 0.12.0 / complex 0.7.0 / fine 0.1.6 resolved. New
baseline: 5323 tests, **4 failures + 10 invalid** — the 13
`Nx.runtime_call/4` failures resolved as predicted, but the 4
VectorizedGradExplorationTest failures PERSIST on 0.12 (nx#1729/#1730
apparently still open), and ShardingTest's `setup_all :skip` bug is
still there. Env note: nothing else in the worktree setup changed; one
self-inflicted detour — I briefly built exla with `XLA_TARGET=cpu`,
which links CUDA objects against the CPU xla_extension and produces an
unloadable NIF (`undefined symbol: cudaGetErrorString`). Cleaned and
rebuilt with the devenv default (cuda12 tarball + `EXLA_TARGET=host`
at runtime). Don't set XLA_TARGET=cpu on this machine.

### Task A — JIT step path (commit 01da43f)

Shipped: `Edifice.step/5` `compiler:` opt (nil = eager, unchanged);
`Edifice.Stateful.jit_step/2` caching `Nx.Defn.jit(&module.step/3)` in
`:persistent_term` per `{module, compiler}` (+ `clear_jit_cache/0`);
`Edifice.init_state/3` `compiler:` warms the jit with one zero-frame
step so compilation lands in init, not mid-game (warns + defers if
`:embed_dim` absent). State stays a plain Nx container; nothing
compiled is serialized — rollback pins run through the jitted path.

All five impls trace cleanly. Pinned by
`test/edifice/stateful/jit_step_test.exs`: the full prefix-equivalence
harness + bitwise rollback under `Nx.Defn.Evaluator` (default runs) and
EXLA-compiled `:exla_only` twins (verified here on the host client).

**EXLA host (CPU) p50 ms/step, embed 287 / hidden 256 / 2 layers:**

| arch | eager (before) | JIT (after) | speedup | init_ms (incl. compile) |
|---|---|---|---|---|
| min_gru | 1.311 | 0.052 | 25x | 72.8 |
| gru | 1.331 | 0.068 | 20x | 206.5 |
| lstm | 1.470 | 0.077 | 19x | 80.5 |
| mamba | 1.820 | 0.092 | 20x | 113.8 |
| gated_ssm | 1.607 | 0.094 | 17x | 81.5 |

GPU rerun is still one command: `EXLA_TARGET=cuda mix run
bench/step_latency.exs` (JIT by default; `BENCH_EAGER=1` for the eager
comparison). Given host CPU is already 180x under the 16.67 ms budget,
the <2 ms / <5 ms GPU bars should clear comfortably.

**Finding worth reviewing — the 5090 numbers were partly a harness
artifact**: `Profile.run_step` sliced each frame eagerly INSIDE the
timed loop, and on EXLA each distinct slice-start compiles its own
executable (~10 ms). The FIRST-profiled arch absorbed all ~210 slice
compilations — that's why min_gru (78.7 ms) looked worse than gru on
the 5090 run, and those numbers mixed real dispatch overhead with
recompilation. Fixed: frames pre-materialized (sliced on BinaryBackend)
before timing. The eager-vs-JIT conclusion stands; per-arch eager GPU
numbers from 2026-07-15 shouldn't be quoted.

### Task B — interp fit infrastructure (commits 79fe8d2, 279c861)

B-step 1 (commit 79fe8d2): all six SAE/transcoder builds gain
`output: :container` → `%{reconstruction, hidden, pre_acts}` (default
single-tensor output bitwise-unchanged, pinned). `pre_acts` = post-ReLU
pre-sparsify, for aux losses/dead-feature logic. Confirmed in passing:
JumpReluSAE's hidden has ZERO exact zeros (the audit's soft-sigmoid
complaint, now demonstrated by test).

B-step 2 (commit 279c861): `Edifice.Interpretability.SAETrainer.fit/3`
— jitted full-batch momentum-SGD train step through the container
build + module `loss/4`, decoder unit-norm row renorm after every
update, `:targets` for transcoders, default Evaluator / `compiler:
EXLA` for real work. Returns `%{params, model, threshold, history,
dead_count, ...}`.

B-step 3 (same commit): BatchTopK production — fit tracks the running
mean of the min selected activation; `BatchTopKSAE.build(
inference_threshold: ...)` swaps batch-global top-k for the fixed
threshold (firing becomes a pure per-sample function — pinned bitwise
solo-vs-in-batch, plus a proof the raw top-k path genuinely differs);
AuxK dead-feature loss (`aux_k`/`aux_coeff`/`dead_eps`) via top-aux_k
dead pre_acts reconstructing the detached residual. LEACE-grade
recovery pin: planted 8-atom dictionary in R^16 → mean best-match
|cos| > 0.5 (chance ~0.2), reconstruction < 0.5x mean-baseline MSE,
decoder rows unit-norm to 1e-4.

Decisions for main-session review:
- Container carries `pre_acts` (3 keys, not the audit's literal 2) —
  needed by AuxK now, JumpReLU-L0/Gated-aux rewrites later.
- Trainer is full-batch only (minibatching = follow-up; slice outside).
- AuxK reconstructs through the decoder kernel WITHOUT bias, residual
  detached (Gao et al. convention).
- Quarantined modules stayed quarantined; GatedSAE/Matryoshka/CLT/DAS
  rewrites untouched per the handoff.
- Activation capture (audit step 1's exphil port) still not ported —
  it needs the SAE trainer's consumer shape settled first; the trainer
  API here is that shape.
