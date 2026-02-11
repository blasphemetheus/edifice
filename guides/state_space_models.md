# State Space Models
> Linear-time sequence modeling through continuous-time dynamical systems with learned discretization.

## Overview

State Space Models (SSMs) bring control theory into deep learning. They model sequences as
continuous-time dynamical systems -- a hidden state evolves according to a linear ODE driven
by the input signal, and the output is a linear readout of that state. When discretized for
digital computation, this yields a recurrence that processes one token at a time with constant
memory, or equivalently a global convolution that processes all tokens in parallel. That dual
nature -- recurrent for inference, convolutional for training -- is the central appeal.

The SSM family in Edifice spans three generations. The first generation (S4, S4D, S5) uses
fixed, time-invariant parameters: the state transition matrix A is initialized once (via
HiPPO theory) and does not change with the input. The second generation (Mamba and variants)
makes these parameters input-dependent ("selective"), allowing the model to decide at each
timestep how much to remember or forget. The third generation (Hybrid, Zamba) recognizes that
pure SSMs still lag behind attention on certain recall-intensive tasks, and interleaves SSM
layers with sparse attention layers for the best of both worlds.

Edifice implements 14 modules in this family, all built on Nx/Axon. They share a common
pattern: input projection, a stack of SSM blocks (each containing a state-space layer, gating,
and a feed-forward network), a final normalization, and last-timestep extraction. The modules
differ in how they parameterize and compute the core SSM recurrence.

## Conceptual Foundation

The continuous-time state space model is defined by four matrices:

```
x'(t) = A x(t) + B u(t)       (state equation)
y(t)  = C x(t) + D u(t)       (output equation)
```

Here x(t) is the hidden state of dimension N, u(t) is the input, and y(t) is the output.
For digital sequences, we discretize using zero-order hold (ZOH):

```
A_bar = exp(dt * A)
B_bar = (A_bar - I) * A^{-1} * B    (simplifies to dt * B for diagonal A)

h[t] = A_bar * h[t-1] + B_bar * u[t]
y[t] = C * h[t]
```

The discretization step dt controls the timescale: small dt means slow state evolution
(long memory), large dt means fast updates (short memory). In Mamba and its descendants,
dt is computed from the input itself, making the model "selective" -- it can choose to
ignore irrelevant tokens (small dt) or integrate important ones (large dt).

The key to efficient training is that this linear recurrence can be computed in parallel.
Three algorithms appear across the Edifice SSM modules:

```
                 Parallel Scan Algorithms
                 ========================

Blelloch (work-efficient):
  O(L) work, O(log L) depth
  Half the elements active per level
  Two passes: up-sweep then down-sweep

                Level 0: [1] [2] [3] [4] [5] [6] [7] [8]
                Level 1: [1] [1+2]  [3] [3+4]  [5] [5+6]  [7] [7+8]
                Level 2: [1] [1+2]  [3] [1-4]  [5] [5+6]  [7] [5-8]
                Level 3: [1] [1+2]  [3] [1-4]  [5] [5+6]  [7] [1-8]
                        (then down-sweep fills in missing prefixes)

Hillis-Steele (max parallelism):
  O(L log L) work, O(log L) depth
  ALL elements active every level
  Better GPU occupancy despite more total ops

                Level 0: [1] [2] [3] [4] [5] [6] [7] [8]
                Level 1: [1] [1+2] [2+3] [3+4] [4+5] [5+6] [6+7] [7+8]
                Level 2: [1] [1+2] [1-3] [1-4] [2-5] [3-6] [4-7] [5-8]
                Level 3: [1] [1+2] [1-3] [1-4] [1-5] [1-6] [1-7] [1-8]

Cumulative Sum (log-space):
  Reformulates the recurrence using cumsum and log-space arithmetic.
  Relies on XLA's cumsum kernel -- performance depends on backend.
```

## Architecture Evolution

```
2021                2022                2023                2024
 |                   |                   |                   |
 S4                  S4D                 Mamba              Mamba-2 (SSD)
 (HiPPO DPLR)       (diagonal A)        (selective SSM)    (chunk matmul)
 |                   |                   |                   |
 |                   S5                  H3 (SSM+conv)      Jamba (Hybrid)
 |                   (MIMO diagonal)     |                   |
 |                                       Hyena (long conv)  Zamba (shared attn)
 |                                       |
 |                                       BiMamba (bidirectional)
 |                                       GatedSSM (simplified)

  Fixed Parameters (LTI)         Input-Dependent (Selective)      Hybrid
  ========================       ============================     =======
  S4, S4D, S5, H3, Hyena        Mamba, MambaSSD, BiMamba,        Hybrid (Jamba)
                                 MambaCumsum, MambaHillisSteele,  Zamba
                                 GatedSSM                         HybridBuilder

  Key transitions:
  S4 -> S4D:   Simplified A from DPLR to pure diagonal (nearly same quality)
  S4D -> S5:   MIMO instead of many SISOs, added D skip connection
  S4D -> Mamba: Made B, C, dt input-dependent ("selection"), added gating
  Mamba -> SSD: Showed SSM = structured attention, exploited matmul (tensor cores)
  Mamba -> Jamba/Zamba: Added periodic attention layers for global recall
```

## When to Use What

```
+-------------------+------------------+------------------+--------------------+
| Requirement       | Best Module      | Runner-up        | Why                |
+-------------------+------------------+------------------+--------------------+
| Long-range deps   | S4, S4D          | S5               | HiPPO init gives   |
| (>4K tokens)      |                  |                  | stable long memory |
+-------------------+------------------+------------------+--------------------+
| General-purpose   | Mamba            | MambaHillisSteele| Selection mechanism|
| sequence modeling |                  |                  | adapts to content  |
+-------------------+------------------+------------------+--------------------+
| Max throughput    | MambaSSD         | MambaCumsum      | Chunk-wise matmul  |
| (GPU training)    |                  |                  | uses tensor cores  |
+-------------------+------------------+------------------+--------------------+
| Bidirectional     | BiMamba          | --               | Forward + backward |
| (offline tasks)   |                  |                  | SSMs combined      |
+-------------------+------------------+------------------+--------------------+
| Quality-critical  | Hybrid (Jamba)   | Zamba            | Attention layers   |
| (LM perplexity)   |                  |                  | for recall tasks   |
+-------------------+------------------+------------------+--------------------+
| Memory-constrained| Zamba            | GatedSSM         | Shared attention = |
| quality           |                  |                  | fewer params       |
+-------------------+------------------+------------------+--------------------+
| Simplest possible | GatedSSM         | S4D              | Gated approx, no  |
| SSM baseline      |                  |                  | parallel scan      |
+-------------------+------------------+------------------+--------------------+
| Attention-free    | Hyena            | H3               | Long convolution   |
| long sequences    |                  |                  | replaces attention |
+-------------------+------------------+------------------+--------------------+
| Ablation /        | MambaCumsum,     | S5               | Same Mamba arch    |
| scan comparison   | MambaHillisSteele|                  | with diff. scans   |
+-------------------+------------------+------------------+--------------------+
| Custom hybrid     | HybridBuilder    | --               | Declarative layer  |
| stacks            |                  |                  | pattern DSL        |
+-------------------+------------------+------------------+--------------------+
```

## Key Concepts

### HiPPO Initialization and Long-Range Memory

The HiPPO (High-order Polynomial Projection Operator) framework provides the mathematical
foundation for SSM initialization. It defines state matrices A that optimally compress
continuous signals into a finite-dimensional polynomial basis. The key insight: a random
matrix A will have eigenvalues that cause the state to either explode or decay to zero over
long sequences. HiPPO matrices have eigenvalues precisely arranged to maintain a useful
compression of the entire input history.

S4 uses the full HiPPO-LegS matrix with DPLR (Diagonal Plus Low-Rank) decomposition.
S4D simplifies to a purely diagonal A with entries a_n = -(n + 1), preserving most of the
long-range benefit. This S4D-Lin initialization is what most modern SSMs build upon.

### The Selection Mechanism (Mamba)

Mamba's key contribution is making SSM parameters input-dependent. In S4/S4D, the matrices
A, B, C are fixed after initialization -- the model processes every token with the same
dynamics. Mamba computes B, C, and the discretization step dt from each input token via
learned projections:

```
dt = softplus(Linear(x))     -- how much to update state
B  = Linear(x)               -- what to write into state
C  = Linear(x)               -- what to read from state
```

This selectivity lets Mamba implement content-based reasoning: it can choose to remember a
token (large dt, strong B), ignore a token (small dt), or selectively read specific state
components (targeted C). The cost is that the parameters now vary across the sequence,
preventing the convolution-mode shortcut. Instead, Mamba relies on parallel associative
scans for efficient training.

### Structured State Space Duality (Mamba-2 / SSD)

Mamba-2 revealed a deep connection: the SSM recurrence and a form of structured linear
attention compute the same thing. Specifically, the SSM output can be written as a
matrix-vector product with a structured (semiseparable) matrix. This "duality" enables
a chunk-wise algorithm:

```
1. Split sequence into chunks of size C
2. Intra-chunk: dense matmul (O(C^2) per chunk, tensor core friendly)
3. Inter-chunk: small sequential scan over chunk boundaries (O(L/C))
4. Combine: merge intra and inter results
```

MambaSSD in Edifice implements this with configurable chunk size and a `training_mode`
flag that switches between the matmul formulation (for training throughput) and the scan
formulation (for inference efficiency).

### Hybrid Architectures: When SSMs Are Not Enough

Pure SSMs struggle with tasks requiring exact recall over long distances -- retrieving a
specific token from thousands of steps ago. Attention excels at this because it can directly
index any past position. The hybrid approach (Jamba, Zamba) interleaves SSM layers (efficient
local processing) with occasional attention layers (global recall):

```
Jamba pattern:  [M, M, A, M, M, A, M, M, A, ...]
                (separate attention weights per layer)

Zamba pattern:  [M, M, M+A, M, M, M+A, M, M, M+A, ...]
                (one shared attention layer, reused at each insertion point)
```

Zamba's insight is that the attention layers primarily need to propagate information globally,
not learn diverse patterns. A single shared-weight attention layer achieves similar quality
with far fewer parameters and a smaller KV cache.

HybridBuilder extends this further with a declarative DSL for arbitrary layer patterns,
supporting Mamba, attention, GLA, RWKV, FFN, and KAN layer types in any combination.

## Complexity Comparison

```
+-------------------+----------+----------+-----------+----------+------------+
| Module            | Training | Inference| Parameters| Causality| Scan Type  |
|                   | (per tok)| (per tok)| (relative)|          |            |
+-------------------+----------+----------+-----------+----------+------------+
| S4                | O(L)     | O(1)     | Base      | Causal   | Cumsum     |
| S4D               | O(L)     | O(1)     | ~S4       | Causal   | Cumsum     |
| S5                | O(L)     | O(1)     | < S4      | Causal   | Cumsum     |
| H3                | O(L)     | O(1)     | ~2x S4    | Causal   | Cumsum     |
| Hyena             | O(L logL)| O(L)     | > S4      | Causal   | FFT/cumsum |
| Mamba             | O(L)     | O(1)     | > S4      | Causal   | Blelloch   |
| MambaSSD          | O(L*C)   | O(1)     | ~Mamba    | Causal   | Chunk+scan |
| MambaCumsum       | O(L)     | O(1)     | ~Mamba    | Causal   | Configurable|
| MambaHillisSteele | O(L logL)| O(1)     | ~Mamba    | Causal   | Hillis-St. |
| BiMamba           | O(L)     | O(L)     | ~2x Mamba | Bidir.   | Cumsum     |
| GatedSSM          | O(L)     | O(1)     | ~Mamba    | Causal   | Gated approx|
| Hybrid (Jamba)    | O(L+L^2) | O(1)+O(W)| > Mamba   | Causal   | Mixed      |
| Zamba             | O(L+L^2) | O(1)+O(W)| < Jamba   | Causal   | Mixed      |
| HybridBuilder     | varies   | varies   | varies    | varies   | Mixed      |
+-------------------+----------+----------+-----------+----------+------------+

Notes:
- L = sequence length, C = chunk size, W = attention window size
- "O(1) inference" means constant work per token with cached hidden state
- Hybrid training cost depends on attention_every ratio and window size
- GatedSSM uses a simplified gated approximation rather than true parallel scan
```

## Module Reference

- `Edifice.SSM.S4` -- Structured State Spaces (HiPPO DPLR initialization, cumsum scan)
- `Edifice.SSM.S4D` -- Diagonal State Spaces (simplified diagonal A, GLU-style FFN)
- `Edifice.SSM.S5` -- Simplified State Space (MIMO diagonal with D skip connection)
- `Edifice.SSM.H3` -- Hungry Hungry Hippos (two SSMs with multiplicative gating and short conv)
- `Edifice.SSM.Hyena` -- Long convolution hierarchy with implicit filters and element-wise gating
- `Edifice.SSM.Mamba` -- Selective SSM with Blelloch parallel associative scan
- `Edifice.SSM.MambaSSD` -- Mamba-2 with chunk-wise matmul/scan duality and training mode toggle
- `Edifice.SSM.MambaCumsum` -- Mamba with configurable scan algorithm (Blelloch, cumsum, log-space)
- `Edifice.SSM.MambaHillisSteele` -- Mamba with Hillis-Steele scan for maximum GPU occupancy
- `Edifice.SSM.BiMamba` -- Bidirectional Mamba with forward and backward SSMs (add or concat merge)
- `Edifice.SSM.GatedSSM` -- Simplified gated SSM with gradient checkpointing support
- `Edifice.SSM.Hybrid` -- Jamba-style Mamba + Attention hybrid with configurable ratio
- `Edifice.SSM.Zamba` -- Mamba + single shared attention layer for minimal parameter overhead
- `Edifice.SSM.HybridBuilder` -- Declarative builder for custom hybrid stacks (Mamba, Attention, GLA, RWKV, FFN, KAN)

## Cross-References

- **[Attention Mechanisms](attention_mechanisms.md)** -- Hybrid and Zamba combine SSM layers with
  attention modules from `Edifice.Attention.MultiHead`. HybridBuilder also integrates GLA, RWKV,
  and Griffin layers.
- **Recurrent Networks** -- SSMs can be viewed as continuous-time RNNs. The discretized recurrence
  h[t] = A_bar * h[t-1] + B_bar * u[t] is a linear RNN with structured transition matrices.
  Griffin's RG-LRU is a gated linear recurrence closely related to SSMs.
- **Building Blocks** -- SSM blocks use RMSNorm/LayerNorm for pre-normalization and SiLU/GELU
  gating in their FFN sub-layers. S4D, S5, Hyena, and BiMamba use SwiGLU-style FFNs.
- **[Generative Models](generative_models.md)** -- Flow matching's ODE formulation connects to the
  probability flow ODE perspective on SSMs as continuous dynamical systems.

## Further Reading

1. Gu, Goel, Re. "Efficiently Modeling Long Sequences with Structured State Spaces." ICLR 2022.
   arXiv:2111.00396 -- The S4 paper that started the SSM revolution.
2. Gu, Gupta, et al. "On the Parameterization and Initialization of Diagonal State Space Models."
   arXiv:2206.11893 -- S4D simplification showing diagonal A is nearly as good as DPLR.
3. Gu, Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752
   -- Input-dependent selection mechanism and hardware-aware parallel scan.
4. Dao, Gu. "Transformers are SSMs: Generalized Models and Efficient Algorithms through
   Structured State Space Duality." arXiv:2405.21060 -- Mamba-2, connecting SSMs to attention.
5. Lieber et al. "Jamba: A Hybrid Transformer-Mamba Language Model." arXiv:2403.19887
   -- Hybrid architecture combining Mamba efficiency with attention quality.
