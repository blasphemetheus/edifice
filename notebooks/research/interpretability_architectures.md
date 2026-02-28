# Interpretability Architectures — Research Notes

Research date: 2026-02-27

## Current State

Edifice has 2 interpretability modules:

- **SparseAutoencoder** — TopK + L1 sparsity modes, encoder/decoder with overcomplete dictionary
- **Transcoder** — Cross-layer SAE variant (input_size ≠ output_size), TopK sparsity

Both follow the standard `build/1` + `loss/4` pattern.

---

## Tier 1 — High Value, Straightforward (priority)

### Gated SAE
**Source:** DeepMind (Rajamanoharan et al., NeurIPS 2024)
**What:** Adds a gating network parallel to the encoder. The gate determines *which* features fire (via top-k on gate logits), while the encoder determines *how much* each fires. Decouples feature selection from magnitude estimation.
**Architecture:**
```
Input [batch, d_model]
  ├─ Encoder: dense(dict_size) → magnitudes
  └─ Gate:    dense(dict_size) → gate_logits → top-k mask
  Combined: magnitudes * mask → sparse_activations
  Decoder: dense(d_model)
Output [batch, d_model]
```
**Why:** Significantly reduces feature suppression (dead features). Near-drop-in replacement for standard SAE. ~50% better reconstruction at same sparsity. Used in Gemma Scope.
**Complexity:** Low. Extends existing SAE with one extra linear layer + gating logic.
**Implementation:** Add `:gated` mode to existing `SparseAutoencoder` module, or new `GatedSAE` module.

### JumpReLU SAE
**Source:** DeepMind/Gemma Scope (Rajamanoharan et al., 2024)
**What:** Replaces ReLU with JumpReLU activation: `f(x) = x * H(x - θ)` where H is the Heaviside step function and θ is a learned per-feature threshold. Features only activate when they exceed their threshold.
**Architecture:** Same as standard SAE but with JumpReLU replacing ReLU+TopK. Each dict feature has a learned threshold parameter.
**Why:** Achieves comparable sparsity to TopK without the rigid k constraint. Threshold adapts per-feature — common features get lower thresholds, rare features get higher ones. Used in Gemma Scope production runs.
**Complexity:** Low. New activation function + learnable threshold vector.
**Implementation:** New `JumpReluSAE` module or `:jump_relu` mode on SAE.

### BatchTopK SAE
**Source:** Bussmann et al. (ICLR 2025)
**What:** Instead of per-sample top-k, applies top-k across the entire batch dimension. Each sample can have variable sparsity as long as the batch-level budget is met.
**Architecture:** Same encoder/decoder as SAE. Sparsification: flatten activations across batch, take global top-k, reshape back.
**Why:** More flexible than per-sample TopK. Rare features can fire strongly on the few samples where they're relevant. Outperforms standard TopK SAE on reconstruction metrics.
**Complexity:** Low. Change sparsification from per-row to batch-global.
**Implementation:** Add `:batch_top_k` mode to existing SAE.

### Linear Probe
**Source:** Alain & Bengio (2016), widely used in interpretability
**What:** A single linear layer trained to predict a concept from frozen model activations. The simplest interpretability tool — if a linear probe can decode concept X from layer L, then layer L linearly represents X.
**Architecture:**
```
Input [batch, d_model]  (frozen activations)
  |
Linear: dense(num_classes) or dense(1) for regression
  |
Output [batch, num_classes]
```
**Why:** Foundation of probing-based interpretability. Extremely simple, widely used, good baseline before trying fancier methods. Can probe for syntactic roles, semantic features, factual knowledge.
**Complexity:** Trivial. Single dense layer. The value is in the training/evaluation workflow, not the architecture.
**Implementation:** New `LinearProbe` module with regression/classification modes.

### Crosscoder
**Source:** Anthropic (Lindsey et al., Dec 2024)
**What:** Jointly trains an SAE across multiple model checkpoints or layers simultaneously. Single shared dictionary, but separate encoder weights per source. Finds features that are shared across training stages or layers.
**Architecture:**
```
Inputs [batch, d_model] × N_sources
  |
Per-source encoders: dense(dict_size) each → sum → ReLU → top-k
  |
Shared sparse activations [batch, dict_size]
  |
Per-source decoders: dense(d_model) each
  |
Outputs [batch, d_model] × N_sources
```
**Why:** Finds features shared across model versions (e.g., base vs RLHF). Key tool for understanding what changes during fine-tuning. Published by Anthropic for studying safety-relevant feature changes.
**Complexity:** Medium. Multiple encoder/decoder heads sharing a dictionary. Main challenge is the multi-input/multi-output API.
**Implementation:** New `Crosscoder` module with `num_sources` parameter.

---

## Tier 2 — Moderate Complexity, High Value

### Concept Bottleneck Model (CBM)
**Source:** Koh et al. (ICML 2020)
**What:** Forces a model to predict human-interpretable concepts as an intermediate bottleneck before making final predictions. The concept layer is fully interpretable by design.
**Architecture:**
```
Input [batch, features]
  |
Concept predictor: dense layers → [batch, num_concepts] (sigmoid)
  |
Task predictor: dense(num_classes)
  |
Output [batch, num_classes]
```
**Why:** Inherently interpretable — you can inspect which concepts activated for each prediction. Enables concept-level interventions at test time (override a concept prediction). Used in medical imaging, policy-sensitive domains.
**Complexity:** Low-medium. Two-stage model with concept supervision required during training.
**Implementation:** New `ConceptBottleneck` module.

### DAS Probe (Distributed Alignment Search)
**Source:** Geiger et al. (ICLR 2024)
**What:** Finds linear subspaces (not just directions) in activation space that causally represent concepts. Learns a rotation matrix that aligns a subspace with a target concept, then probes within that subspace.
**Architecture:**
```
Input [batch, d_model]  (frozen activations)
  |
Learned rotation: orthogonal matrix [d_model, d_model]
  |
Project to subspace: first k dimensions of rotated space
  |
Probe: dense(num_classes)
  |
Output [batch, num_classes]
```
**Why:** Stronger than linear probes — finds distributed representations across multiple dimensions. Causal (not just correlational) via interchange intervention training.
**Complexity:** Medium. Needs orthogonal matrix parameterization (e.g., Cayley transform or Householder reflections).
**Implementation:** New `DASProbe` module.

### Matryoshka SAE
**Source:** Bussmann et al. (2025)
**What:** Nested SAE structure where smaller dictionaries are strict subsets of larger ones. Train a single SAE but get multiple granularity levels — use fewer features for coarse understanding, more for fine-grained.
**Architecture:** Standard SAE but with nested loss: features are ordered by importance, and sub-dictionaries at sizes [k1, k2, ...kN] must each independently reconstruct well.
**Why:** One model, multiple sparsity levels. Efficient for exploration — start coarse, drill down. Avoids training separate SAEs at each dictionary size.
**Complexity:** Medium. Multi-scale loss function, ordered feature importance.
**Implementation:** New module or mode on SAE.

### LEACE (LEAst-squares Concept Erasure)
**Source:** Belrose et al. (ICML 2023)
**What:** Linearly erases a concept from activations by projecting onto the orthogonal complement of the concept's subspace. Closed-form solution (no training loop needed).
**Architecture:**
```
Input [batch, d_model]
  |
Eraser: I - P  (projection matrix, computed from data)
  |
Output [batch, d_model]  (concept removed)
```
**Why:** Gold-standard concept erasure. If LEACE can't erase concept X, then X is represented nonlinearly. Useful for fairness (erase protected attributes) and as a diagnostic tool.
**Complexity:** Low. The "architecture" is a single projection matrix. Main work is computing P from data via SVD.
**Implementation:** New `LEACE` module with `fit/2` (compute projector) and `erase/2` (apply).

### Cross-Layer Transcoder (CLT)
**Source:** Anthropic (Dunefsky et al., Feb 2025)
**What:** Extension of Transcoder that maps MLP inputs to MLP outputs *across all layers simultaneously*. One large shared dictionary, with per-layer encoder and decoder weights.
**Architecture:**
```
Per-layer MLP inputs [batch, d_model] × L layers
  |
Per-layer encoders: dense(dict_size) each → sum → ReLU → top-k
  |
Shared features [batch, dict_size]
  |
Per-layer decoders: dense(d_model) each
  |
Per-layer MLP outputs [batch, d_model] × L layers
```
**Why:** Replaces ALL MLPs in a model with a single sparse linear computation. Enables true circuit-level analysis — trace any output back through a sparse set of features. Anthropic's latest interpretability direction.
**Complexity:** Medium-high. Multi-layer input/output, large shared dictionary.
**Implementation:** Extends Crosscoder pattern but with asymmetric input/output (MLP-in → MLP-out).

---

## Tier 3 — Advanced / Niche

### Steering Vectors (CAA — Contrastive Activation Addition)
**Source:** Rimsky et al. (2023), Turner et al. (2024)
**What:** Computes a "steering vector" as the mean activation difference between contrastive prompt pairs (e.g., "honest" vs "deceptive" completions). Adding this vector to activations at inference steers model behavior.
**Architecture:** Not a trainable model — it's a vector computed from data. But could provide a `compute_steering_vector/3` utility and an `apply_steering/3` layer.
**Why:** Simple, effective behavioral control. Key safety tool. No training needed.
**Complexity:** Low (computation), but the value is in the workflow/tooling.

### Conceptor Steering
**Source:** Extended from Jaeger (2014), adapted for LLMs
**What:** Uses soft conceptors (regularized correlation matrices) instead of single vectors for steering. Captures a subspace rather than a direction.
**Complexity:** Medium. Matrix-valued steering instead of vector-valued.

### SAE Feature Steering
**Source:** Templeton et al. (Anthropic, 2024)
**What:** Identify features in a trained SAE, then clamp specific feature activations to steer behavior. E.g., amplify "Golden Gate Bridge" feature to make Claude obsessed with it.
**Why:** More targeted than CAA — steer individual interpretable features rather than coarse directions.
**Complexity:** Low (given an existing SAE). Utility function on top of SAE.

### Edge Pruning / Circuit Discovery
**Source:** Hanna et al. (2024)
**What:** Learns binary masks on edges in the computational graph to find minimal circuits. Differentiable masking with L0 regularization.
**Complexity:** High. Requires hooks into model internals, differentiable binary masks.

### DiscoGP (Distributed Concept Graph Pruning)
**Source:** Extended from ACDC, Conmy et al. (2023)
**What:** Finds circuits using gradient-based pruning on computational graph edges.
**Complexity:** High. Similar to Edge Pruning.

### Knowledge Editing (ROME / MEMIT)
**Source:** Meng et al. (2022, 2023)
**What:** ROME (Rank-One Model Editing) modifies a single factual association by computing a rank-1 update to an MLP weight matrix. MEMIT extends to batch edits.
**Complexity:** Medium-high. Requires causal tracing + constrained optimization. More of a procedure than architecture.

### Beta-VAE
**Source:** Higgins et al. (ICLR 2017)
**What:** VAE with upweighted KL divergence term to encourage disentangled latent representations.
**Complexity:** Low as architecture (just a VAE with β parameter), but disentanglement is the goal, not interpretability per se.

---

## Implementation Priority Order

Based on value-to-effort ratio and alignment with Edifice's architecture-focused approach:

1. **Gated SAE** — Near-drop-in SAE improvement, high impact
2. **JumpReLU SAE** — Alternative SAE sparsity, complements Gated
3. **BatchTopK SAE** — Third SAE variant, rounds out the family
4. **Linear Probe** — Trivial to implement, foundational tool
5. **Crosscoder** — Medium effort, unique multi-source capability
6. **Concept Bottleneck** — Clean architecture, inherently interpretable
7. **DAS Probe** — Stronger probing, moderate complexity
8. **LEACE** — Concept erasure, mostly computation not architecture
9. **Matryoshka SAE** — Multi-scale SAE, moderate complexity
10. **Cross-Layer Transcoder** — Extends existing Transcoder pattern
11. **Steering utilities** — CAA, SAE steering (utility functions, not full architectures)

## Notes

- SAE variants (1-3) share infrastructure with the existing `SparseAutoencoder` — could be modes or sibling modules
- Probing tools (4, 7, 8) are architecturally simple but need good training/evaluation ergonomics
- Crosscoder (5) and CLT (10) share the multi-source encoder pattern
- Circuit discovery (Edge Pruning, DiscoGP) requires model-internal hooks that don't fit Edifice's "build a model" API well — better as separate tooling
- Knowledge editing (ROME/MEMIT) is procedure-heavy, not architecture-heavy — lower priority for Edifice
