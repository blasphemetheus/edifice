# ExPhil Architecture Opportunities — Research Notes

## Overview

ExPhil (a Super Smash Bros. Melee AI) already uses ~70 Edifice backbone
architectures for temporal processing. This doc identifies **architectures
ExPhil doesn't currently use** that could meaningfully improve its game AI
capabilities — either architectures already in Edifice that haven't been wired
up, or new modules Edifice could add.

## Current ExPhil Architecture

### Pipeline

```
Game State Embedding [batch, 288]
        │
        ▼
Temporal Backbone (Mamba, LSTM, Griffin, etc.)
        │
        ▼
  [batch, hidden_dim]
        │
        ▼
Autoregressive Controller Head
  buttons → main_x → main_y → c_x → c_y → shoulder
  [8 Bern]  [17 Cat]  [17 Cat]  [17]   [17]   [5]
```

### What ExPhil Already Has

**~70 backbone architectures** wired through `Policy.Backbone`, including:
- SSMs: Mamba (production default), Mamba-SSD, Mamba-3, S4, S4D, H3, GatedSSM,
  Hyena, Longhorn, Samba, Hymba, GSS, MixtureOfMamba
- Attention: Sliding window, RetNet, RWKV, GLA/GLA-v2, HGRN/HGRN-v2, Performer,
  FNet, Perceiver, GatedAttention, SigmoidAttention, GSA, RLA, NHA, Fox,
  LogLinear, LASER, MoBA, TNN, SPLA
- Recurrent: LSTM, GRU, MinGRU, MinLSTM, xLSTM (sLSTM/mLSTM), DeepResLSTM,
  DeltaNet, GatedDeltaNet, DeltaProduct, NativeRecurrence, TTT, TTT-E2E,
  Titans, MIRAS, Huginn
- Hybrid: Jamba, Zamba, Griffin, Hawk, TransformerLike, HybridBuilder
- Other: MLP, KAN, TCN, Liquid, DecisionTransformer, SNN, Bayesian, Hopfield,
  NTM, Reservoir, Coconut

**World Model** — MLP-based state-action encoder (no temporal backbone), with
ensemble and recurrent variants.

**Autoregressive Controller Head** — 6 component heads with teacher forcing
during training, sequential sampling during inference.

### Production Performance

- **Mamba** is the recommended backbone: 2.97 val loss, 10.8ms inference @ 60 FPS
- **60 FPS constraint** (16.7ms budget) limits viable architectures at training-time
  sequence lengths; single-step inference opens up more options

## Gap Analysis

### Gap 1: Autoregressive Policy with Cross-Component Conditioning

**Problem:** ExPhil's controller head builds each component independently from
the backbone output. The autoregressive structure is handled at sampling time,
but the *logits themselves* don't condition on previously sampled components.
AlphaStar's policy head conditions each action dimension on all previous ones
during both training and inference.

**What exists in Edifice:** `Edifice.Transformer.MultiTokenPrediction` predicts
multiple future tokens but isn't designed for cross-component conditioning within
a single timestep.

**Proposed: `AutoregressiveHead` module**

```
Backbone [batch, hidden]
    │
    ├─→ buttons_logits     (conditioned on: hidden)
    │        │
    ├─→ main_x_logits      (conditioned on: hidden + buttons)
    │        │
    ├─→ main_y_logits      (conditioned on: hidden + buttons + main_x)
    │        │
    ├─→ c_x_logits         (conditioned on: hidden + buttons + main_x + main_y)
    │        │
    ├─→ c_y_logits         (conditioned on: ... + c_x)
    │        │
    └─→ shoulder_logits     (conditioned on: all above)
```

**Architecture:** Small transformer or LSTM that processes action components
sequentially. Each step takes `[hidden; prev_action_embedding]` and outputs
logits for the next component. During training: teacher forcing. During
inference: sample-then-feed.

**Why it matters:** AlphaStar showed that autoregressive action decomposition
significantly improves policy expressiveness for complex action spaces. Melee's
controller has 6 correlated components (e.g., pressing B + specific stick angle
= a particular special move). Independent heads can't learn these correlations.

**Edifice scope:** `Edifice.RL.AutoregressiveHead` — a generic module that takes
a list of action component specs and builds the cross-conditioning chain.

**Difficulty:** Moderate (~200 lines). Reuses existing dense/LSTM layers.

### Gap 2: Pointer Network for Target Selection

**Problem:** In multi-character scenarios (doubles, free-for-all) or when
selecting between multiple game objects (items, projectiles), ExPhil needs to
*point to* one of a variable-length set of entities. Currently, ExPhil hard-codes
the opponent as a fixed input dimension.

**What exists:** Nothing directly. `Perceiver` does cross-attention over
variable-length inputs but isn't designed for discrete selection.

**Proposed: `PointerNetwork` module**

```
Query: [batch, hidden]           (agent's state)
Keys:  [batch, num_entities, dim] (entity embeddings)
                │
                ▼
        Attention(query, keys)
                │
                ▼
        Selection logits [batch, num_entities]
```

**Architecture:** Attention-based pointing (Vinyals et al., 2015). The agent's
hidden state queries over a variable-size set of entity embeddings. Output is a
categorical distribution over entities.

**Why it matters:** AlphaStar uses pointer networks to select *which unit* to
target from thousands of candidates. For Melee: targeting in doubles, selecting
items, choosing between multiple projectiles. Even in 1v1, explicit target
attention over game objects (opponent, platforms, ledges, projectiles) could
replace hand-crafted entity selection.

**Edifice scope:** `Edifice.RL.PointerNetwork` — attention-based selection over
variable-length entity sets.

**Difficulty:** Simple (~100 lines). Just query-key attention + softmax.

### Gap 3: Multi-Timescale Recurrence (FTW-Style)

**Problem:** Game AI needs to reason at multiple timescales simultaneously:
frame-level reactions (4ms), short combos (100-500ms), neutral game strategy
(seconds), match-level adaptation (minutes). ExPhil uses a single backbone
operating at one temporal resolution.

**What exists in Edifice:**
- `MixtureOfRecursions` — routes to different recurrence operators but at the
  same timescale
- `HybridBuilder` — stacks different layer types but all process the same
  sequence at the same rate

**Proposed: `MultiTimescaleRecurrence` module**

```
Input [batch, seq, embed]
        │
        ├─→ Fast  (process every frame,   small hidden)   → h_fast
        ├─→ Med   (process every 4 frames, medium hidden) → h_med
        └─→ Slow  (process every 16 frames, large hidden) → h_slow
                    │
                    ▼
            Merge: concat(h_fast, h_med, h_slow) → hidden
```

**Architecture:** Inspired by FTW (Jaderberg et al., 2019 — DeepMind's
Capture-the-Flag agent). Multiple recurrent cores operating at different
temporal resolutions. The fast core sees every frame; the slow core sees
subsampled/pooled sequences. A merge layer combines all timescales.

**Why it matters:** FTW showed that multi-timescale processing dramatically
improves game AI in environments with hierarchical temporal structure. Melee has
extreme temporal hierarchy: 1-frame tech windows, multi-frame combos,
second-scale neutral exchanges, minute-scale adaptation.

**Edifice scope:** `Edifice.Recurrent.MultiTimescaleRecurrence` — wraps N
backbone instances at different temporal strides with a merge layer.

**Difficulty:** Moderate (~200 lines). Need temporal subsampling + merge.

### Gap 4: Sequence-Level World Model with SSM Dynamics

**Problem:** ExPhil's world model is an MLP that predicts `next_state` from
`(current_state, action)` — no temporal modeling in the dynamics network itself.
The `build_recurrent` variant uses LSTM but doesn't leverage SSM architectures.

**What exists in Edifice:** `Edifice.WorldModel.WorldModel` is a basic
encoder-decoder world model. ExPhil has its own `WorldModel` module.

**Reference: DRAMA (2025)** — Uses Mamba-2 as the dynamics backbone of a world
model, achieving DreamerV3-level performance with 7M params (vs DreamerV3's
200M). Key insight: SSMs are natural dynamics models because state-space
transitions *are* dynamics.

**Proposed enhancement:** Not a new Edifice module — instead, ExPhil should wire
its `build_recurrent` world model variant to use Mamba (or any SSM backbone) via
the same `Backbone.build_temporal_backbone` dispatch. This is an ExPhil change,
not an Edifice change.

**Alternatively, Edifice could add:** `Edifice.WorldModel.SSMWorldModel` that
uses any SSM as the dynamics backbone, with RSSM-style stochastic latent state
(DreamerV3 pattern).

**Difficulty:** Moderate-high (~300 lines for RSSM + SSM dynamics).

### Gap 5: Opponent Modeling / Entity Embedding Module

**Problem:** ExPhil embeds all game state into a flat 288-dim vector. There's no
explicit separation between "self state," "opponent state," and "game state."
The network must learn to disentangle these from the flat embedding.

**Reference:** AlphaStar uses separate entity encoders for each unit, then
aggregates via scatter/attention. Entity-centric processing lets the model
generalize across different entity counts.

**Proposed: `EntityEncoder` module**

```
Entities: [batch, num_entities, entity_dim]
Entity Types: [batch, num_entities] (categorical)
        │
        ├─→ Type Embedding + Feature Projection
        │
        ├─→ Self-Attention over entities
        │
        ├─→ Pool (mean/attention) → [batch, hidden]
        │
        └─→ Per-entity outputs [batch, num_entities, hidden] (for pointer network)
```

**Architecture:** Encode heterogeneous game entities (player, opponent,
projectiles, items, stage features) with type-conditioned embeddings, then
process with self-attention (or set transformer) to produce both a global game
state embedding and per-entity representations.

**Why it matters:** Entity-centric processing is a core pattern in game AI
(AlphaStar, OpenAI Five, DreamerV3). It enables generalization to variable
entity counts and explicit relational reasoning between game objects.

**Edifice scope:** `Edifice.Sets.EntityEncoder` — type-conditioned set encoder
with global pooling and per-entity outputs. Builds on existing `DeepSets` and
`PointNet` patterns.

**Difficulty:** Moderate (~200 lines).

## Edifice Architectures ExPhil Hasn't Wired Up

Beyond new modules, several existing Edifice architectures are potentially
useful for ExPhil but aren't in `Backbone.build_temporal_backbone`:

| Architecture | Family | Why Useful | Priority |
|---|---|---|---|
| `infini_attention` | Attention | Compressive memory for long matches | Medium |
| `mega` | Attention | EMA-gated attention, good for sequences | Low |
| `diff_transformer` | Attention | Differential attention reduces noise | Medium |
| `conformer` | Attention | Conv + attention, good for temporal patterns | Medium |
| `mla` | Attention | Multi-head latent attention (DeepSeek), efficient | Medium |
| `megalodon` | Attention | Chunk-wise attention with exponential decay | Low |
| `lightning_attention` | Attention | O(n) linear attention, very fast | High |
| `flash_linear_attention` | Attention | Hardware-efficient linear attention | High |
| `ssmax` | Attention | State-space inspired softmax | Low |
| `xlstm_v2` | Recurrent | Updated xLSTM architecture | Medium |
| `slstm` (standalone) | Recurrent | Scalar LSTM with exponential gating | Medium |
| `bimamba` | SSM | Bidirectional Mamba (offline analysis) | Low |
| `hyena_v2` | SSM | Improved long convolution | Low |
| `ss_transformer` | SSM | State-space transformer hybrid | Medium |
| `free_transformer` | Transformer | Latent variable transformer | Low |

**Highest value to wire up:** `lightning_attention` and `flash_linear_attention`
— both are O(n) and fast, directly addressing the 60 FPS constraint. Also
`diff_transformer` and `conformer` which have shown strong results on sequential
tasks.

## Research References

### Game AI Systems

1. **AlphaStar** (Vinyals et al., 2019) — StarCraft II
   - Transformer + LSTM backbone, pointer network for unit selection
   - Autoregressive policy head with teacher forcing
   - Entity-centric observation encoding
   - Key insight: large action spaces need autoregressive decomposition

2. **OpenAI Five** (Berner et al., 2019) — Dota 2
   - Single-layer 4096-dim LSTM per hero
   - Pointer network for target selection
   - Max-pooled entity embeddings
   - Key insight: simple architectures scale well with compute

3. **FTW** (Jaderberg et al., 2019) — Capture the Flag
   - Multi-timescale recurrence (fast/slow LSTM cores)
   - Population-based training
   - Key insight: hierarchical temporal processing essential for game AI

4. **DRAMA** (Wu et al., 2025) — Mamba World Model
   - Mamba-2 dynamics backbone: 7M params matches DreamerV3's 200M
   - State-space models are natural dynamics models
   - Key insight: SSM inductive bias perfectly matches physical dynamics

5. **RLBenchNet** (2025) — Architecture Comparison for RL
   - Benchmarked Mamba-2, Transformer, LSTM, minGRU on POMDPs
   - Mamba-2 best overall for partially observable environments
   - minGRU/minLSTM match Mamba with 85% fewer parameters
   - Key insight: SSM inductive bias helps most in partial observability

### Architecture Papers

6. **Pointer Networks** (Vinyals et al., 2015)
   - Attention-based selection from variable-length input sets
   - Used in combinatorial optimization, NLP entity linking, game AI

7. **Set Transformer** (Lee et al., 2019)
   - Self-attention over unordered sets with induced set attention blocks
   - Efficient O(n·m) via inducing points

## Priority Ranking

| # | Gap | Impact | Difficulty | Recommended |
|---|-----|--------|------------|-------------|
| 1 | AutoregressiveHead | High — captures action correlations | Moderate | **Yes** |
| 2 | MultiTimescaleRecurrence | High — hierarchical temporal reasoning | Moderate | **Yes** |
| 3 | EntityEncoder | Medium — better entity representation | Moderate | **Yes** |
| 4 | PointerNetwork | Medium — target selection for doubles/items | Simple | **Yes** |
| 5 | SSM World Model | Medium — better dynamics modeling | Moderate-High | Later |

Recommendations 1-4 are all <250 lines each and address concrete gaps in ExPhil's
architecture. They can be implemented independently.

## Verdict

ExPhil has excellent backbone coverage (~70 architectures) but is missing
**game-AI-specific structural components** that production systems like AlphaStar
and OpenAI Five rely on. The gaps are all in the "how actions are produced" and
"how entities are represented" layers — not in the temporal backbone itself.

The four recommended modules (AutoregressiveHead, MultiTimescaleRecurrence,
EntityEncoder, PointerNetwork) would bring ExPhil's architecture closer to
state-of-the-art game AI systems while staying within Edifice's scope as a
neural architecture library.
