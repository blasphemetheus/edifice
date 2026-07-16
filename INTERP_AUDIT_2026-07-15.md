# Interpretability module audit — 2026-07-15

Audit of all 12 `lib/edifice/interpretability/` modules against the
LEACE standard (`leace_test.exs`: closed-form `fit/2` + tests that plant
a signal and verify the method's mathematical guarantee). Conducted
read-only by three parallel reviewers; consolidated here. This gates the
exphil P5/P6 program (SAEs with ground-truth-scored features).

## Verdict table

| Module | Math vs paper | Fit path | Tests | Verdict |
|---|---|---|---|---|
| LEACE | exact | closed-form `fit/2` | functional guarantees | **production** |
| SparseAutoencoder | OK (vanilla top-k/L1) | loss only | shape + sparsity count | scaffold |
| BatchTopKSAE | selection correct | loss only | 1 real functional test | scaffold (best SAE) |
| JumpReluSAE | soft sigmoid ≠ JumpReLU: no exact zeros, no L0/STE | loss only | near-vacuous | scaffold (mislabeled) |
| GatedSAE | gate has **zero-gradient path**; paper's aux loss absent | gate untrainable | can't detect the dead gate | **broken** |
| MatryoshkaSAE | missing the defining nested-prefix loss; docs cite nonexistent `multi_scale_loss/4` | loss only (wrong objective) | test the wrong objective | **broken** |
| Transcoder | skeleton OK; `loss/4` crashes without `:l1_coeff` (transcoder.ex:110) | loss only | shape-only | scaffold |
| Crosscoder | best of family; sparsity term missing decoder-norm weighting; `loss/4` untested, likely non-runnable (crosscoder.ex:136-152) | closest (shared-name `build_encoder`) | shape-only | scaffold |
| CrossLayerTranscoder | **wrong architecture**: acausal shared code, no per-layer features (:87-124); no loss at all; carries a `verified: true` marker | none | shape-only | **broken** |
| LinearProbe | correct, trivial (~15 lines); softmax baked in loses logit stability | none | shape-only | scaffold |
| DASProbe | **not DAS**: no orthogonal rotation, no interventions, no counterfactual loss; rank-limited linear probe ≤ LinearProbe expressiveness; causal claims unsupported | none | can't distinguish from LinearProbe | **broken** (mislabeled) |
| ConceptBottleneck | faithful minimal CBM skeleton | none; advertised intervention API doesn't exist | shape/range-only | scaffold |

Bottom line: 1 production, 7 scaffolds, 4 broken. The "assume
scaffold-grade" guidance was correct and in four cases generous.

## Family-wide structural gaps (the real blockers)

1. **No fit path anywhere but LEACE.** No trainer, no Axon.Loop recipe,
   no dictionary-learning recipe in `Edifice.Recipes`.
2. **`build/1` hides hidden activations** — every SAE/transcoder loss
   needs `(input, reconstruction, hidden)`, but models output only the
   reconstruction. Losses are unwireable without rebuilding the graph.
3. **No activation-capture utility** for a frozen network. (exphil has
   one: `lib/exphil/interp/activations.ex`, with embedding cache.)
4. **f32 loss policy violated in every `loss/4`** despite this repo's
   own CLAUDE.md naming SAE objectives as the motivating case.
5. **No decoder-norm constraint** anywhere → every L1 objective is
   degenerate (shrink code, grow decoder).
6. **No dead-feature handling** (resampling/AuxK) in any SAE.
7. **Test suites prove nothing**: except LEACE's and one BatchTopK
   check, every test in the directory would pass if the model were
   replaced by a single dense layer.

## Concrete bugs (file:line)

- gated_sae.ex:156-165 + 147-149 — gate reaches forward only through a
  boolean mask (zero gradient); loss lacks the aux term the docstring
  describes. Gate is frozen random noise forever.
- matryoshka_sae.ex:31 — doc directs to `multi_scale_loss/4`; grep: it
  does not exist in the repo. :142-153 — loss is full-width MSE +
  index-weighted L1, not nested prefix reconstruction.
- transcoder.ex:110 — `opts[:l1_coeff]` nil inside defn → crash when
  the documented default is relied on. Use `keyword!` with default.
- crosscoder.ex:136-152 — `loss/4` takes lists of tensors through a
  defn (`Enum.zip` on tracer input), zero test coverage, likely
  non-runnable; same nil `l1_coeff` crash.
- cross_layer_transcoder.ex:5 — `<!-- verified: true -->` unsupported
  by the code below it.
- das_probe.ex:24-31 — "stronger than standard linear probes" is
  mathematically false for a linear composition.

## Remediation plan for P5/P6, in order

1. **Fit infrastructure first** (unblocks everything): activation
   capture (port exphil `activations.ex` mechanisms), `{reconstruction,
   hidden}` container outputs on all SAE/transcoder builds, one shared
   SAE trainer (full-batch or minibatch defn loop in the style of
   exphil `probe.ex:306-338`), f32 cast at every loss entry, decoder
   unit-norm renorm step.
2. **BatchTopKSAE → production**: inference-time fixed threshold
   (running mean of min selected activation — without it, feature
   firing depends on batch composition, which poisons ground-truth
   scoring), AuxK dead-feature loss, then a LEACE-grade planted-
   dictionary recovery test.
3. **LinearProbe → adopt exphil probe machinery** (the 10-item delta
   list from the probe audit: stable class-weighted CE from logits,
   inverse-frequency weights, train-only standardization, label
   masking, balanced accuracy + majority baseline, single-class guard,
   group-wise splits, shuffled-label control, backend hygiene, momentum
   SGD defn loop). Melee-specific feature suites stay in exphil.
4. **Quarantine the broken four** until rewritten: mark GatedSAE,
   MatryoshkaSAE, CrossLayerTranscoder, DASProbe as `@moduledoc` status:
   experimental/incorrect, or rename (das_probe → subspace_probe) so
   nobody trusts the label. CLT needs a redesign (causal ℓ→(ℓ..L)
   decoders, per-layer feature identity), not a patch.
5. JumpReLU: either implement the real thing (Heaviside + STE + L0) or
   rename; the soft-sigmoid version cannot produce the near-binary
   firing P6 scoring assumes.

## Coordination note

The `feat/bot-runtime` worktree session (HANDOFF_BOT_RUNTIME.md, task 4
stretch) may port Probe/Attribution/Activations from exphil — that work
IS step 1+3 of this plan. Review its branch against this doc before
merging; don't double-implement.
