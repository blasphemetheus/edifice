# Attention Mechanisms
> From quadratic full attention through linear approximations to recurrence-based alternatives that scale to arbitrary sequence lengths.

## Overview

Attention is the mechanism that made modern sequence modeling work at scale. The core idea is
simple: for each position in a sequence, compute a weighted combination of all other positions,
where the weights reflect relevance. This gives every token direct access to every other token,
enabling the kind of long-range dependency modeling that recurrent networks struggle with. The
cost is quadratic: computing all pairwise relevance scores for a length-L sequence requires
O(L^2) time and memory.

The 12 attention modules in Edifice trace three distinct paths away from that quadratic
bottleneck. The first path keeps the full attention formulation but reduces redundancy:
Grouped Query Attention (GQA) shares key-value heads across groups of query heads, cutting
KV cache size without meaningful quality loss. The second path replaces the softmax attention
matrix with a cheaper approximation: LinearTransformer uses kernel feature maps, Performer
uses random orthogonal features (FAVOR+), and Nystromformer uses landmark-based matrix
approximation. The third path abandons the attention matrix entirely in favor of recurrence:
RetNet uses exponential decay retention, RWKV uses weighted key-value aggregation, GLA adds
data-dependent gating to linear attention, HGRN uses hierarchical gating with state expansion,
and Griffin combines a gated linear recurrence with local attention. Two modules stand apart:
FNet replaces attention with an unparameterized Fourier Transform (no learned weights at all),
and Perceiver uses cross-attention to a learned latent array, decoupling compute from input size.

All modules follow a common block pattern: pre-normalization, the attention/mixing mechanism,
a residual connection, then a feed-forward network with another residual. They accept sequence
inputs of shape [batch, seq_len, embed_size] and output [batch, hidden_size] from the last
timestep.

## Conceptual Foundation

The standard attention computation proceeds in three steps. First, project the input into
queries Q, keys K, and values V. Second, compute attention weights as the softmax of scaled
dot products. Third, apply those weights to the values:

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Where:
  Q = X W_Q    [L, d_k]     (what am I looking for?)
  K = X W_K    [L, d_k]     (what do I contain?)
  V = X W_V    [L, d_v]     (what do I provide?)

  Q K^T        [L, L]       (pairwise relevance scores)
  softmax(...)  [L, L]       (normalized attention weights)
  ... * V      [L, d_v]     (weighted combination of values)
```

Multi-head attention runs H independent attention computations in parallel, each with
dimension d_k = d_model / H, then concatenates and projects the results. This lets
different heads attend to different types of relationships.

The three efficiency paths can be understood as different ways to avoid materializing the
L x L attention matrix:

```
                    Efficiency Strategies
                    =====================

Path 1: Reduce Redundancy (keep full attention)
  +------------+     +------------+     +------------+
  | Multi-Head |     |    GQA     |     | Perceiver  |
  | (baseline) | --> | (shared KV)| ... | (latent    |
  |  H KV heads|     | G KV heads |     |  bottleneck|
  +------------+     +------------+     +------------+
  O(L^2)              O(L^2), less KV    O(L*M + M^2)

Path 2: Approximate the Attention Matrix
  +------------+     +------------+     +------------+
  |  Linear    |     | Performer  |     |Nystromformer|
  |Transformer | ... | (FAVOR+)   | ... | (landmarks)|
  | phi(Q)(KV) |     | random feat|     | Nystrom    |
  +------------+     +------------+     +------------+
  O(L*d^2)            O(L*d*m)           O(L*M)

Path 3: Replace with Recurrence
  +--------+     +--------+     +--------+     +--------+     +--------+
  | RetNet |     | RWKV   |     |  GLA   |     | HGRN   |     |Griffin |
  | (decay)|     | (WKV)  |     | (gated)|     | (hier.)|     |(RG-LRU)|
  +--------+     +--------+     +--------+     +--------+     +--------+
  O(L)            O(L)            O(L)           O(L)           O(L)

Special:
  +--------+
  |  FNet  |     No learned weights -- pure FFT mixing
  | (FFT)  |     O(L log L)
  +--------+
```

## Architecture Evolution

```
2017                2020              2021              2023              2024
 |                   |                 |                 |                 |
 Multi-Head Attn     Linear            FNet             RetNet            Griffin
 (Vaswani)          Transformer       (no-params FFT)   (decay retention) (RG-LRU+local)
 |                   |                 |                 |                 |
 |                  Performer          Perceiver        RWKV-7            HGRN-2
 |                  (FAVOR+)          (latent array)    (linear RNN)      (state expansion)
 |                   |                                   |
 |                  Nystromformer                        GLA
 |                  (landmarks)                         (gated linear)
 |
 GQA (2023, Llama 2)

  Full Quadratic          Subquadratic Approx.          Linear Recurrent
  ================        =====================         ================
  MultiHead, GQA          LinearTransformer,            RetNet, RWKV, GLA,
  Perceiver               Performer, Nystromformer,     HGRN, Griffin
                          FNet
```

## When to Use What

```
+-------------------+------------------+------------------+--------------------+
| Requirement       | Best Module      | Runner-up        | Why                |
+-------------------+------------------+------------------+--------------------+
| Maximum quality   | MultiHead        | GQA              | Full O(L^2) attn   |
| (short sequences) |                  |                  | captures everything|
+-------------------+------------------+------------------+--------------------+
| Inference speed   | GQA              | RetNet           | Fewer KV heads =   |
| (KV cache bound)  |                  |                  | smaller cache      |
+-------------------+------------------+------------------+--------------------+
| Streaming /       | RetNet, RWKV     | GLA              | O(1) per-step      |
| real-time         |                  |                  | recurrent mode     |
+-------------------+------------------+------------------+--------------------+
| Very long seqs    | Performer        | Nystromformer    | O(L) attention via |
| (>8K tokens)      |                  |                  | random features    |
+-------------------+------------------+------------------+--------------------+
| Multimodal /      | Perceiver        | --               | Handles arbitrary  |
| variable input    |                  |                  | input modalities   |
+-------------------+------------------+------------------+--------------------+
| Fastest training  | FNet             | LinearTransformer| Zero attention     |
| (quality tradeoff)|                  |                  | params (FFT only)  |
+-------------------+------------------+------------------+--------------------+
| Short seqs with   | GLA              | Griffin          | Data-dependent     |
| hardware efficiency|                 |                  | gating, native ops |
+-------------------+------------------+------------------+--------------------+
| Recurrence +      | Griffin          | HGRN             | RG-LRU layers +   |
| local attention   |                  |                  | periodic local attn|
+-------------------+------------------+------------------+--------------------+
| On-device /       | RWKV             | GLA              | O(1) inference,    |
| memory-limited    |                  |                  | fixed-size state   |
+-------------------+------------------+------------------+--------------------+
| Stable long-range | HGRN             | RetNet           | State expansion    |
| with O(1) memory  |                  |                  | enriches recurrence|
+-------------------+------------------+------------------+--------------------+
```

## Key Concepts

### Grouped Query Attention and KV Cache Efficiency

In standard multi-head attention with H heads, each head maintains its own key and value
projections. During autoregressive inference, the KV cache stores all past K and V tensors
for each head, consuming O(H * L * d) memory. GQA reduces this by grouping G query heads
to share a single KV head. With H=8 query heads and G=2 KV heads, each KV head serves 4
query heads, cutting KV cache by 4x:

```
MHA:  Q1-K1-V1  Q2-K2-V2  Q3-K3-V3  Q4-K4-V4   (4 KV heads, full cache)
GQA:  Q1-K1-V1  Q2-K1-V1  Q3-K2-V2  Q4-K2-V2   (2 KV heads, half cache)
MQA:  Q1-K1-V1  Q2-K1-V1  Q3-K1-V1  Q4-K1-V1   (1 KV head, minimal cache)
```

GQA is used in Llama 2 (70B), Mistral, and Gemma. The quality loss from KV sharing is
minimal because the key-value representations across heads are highly correlated in practice.

### Kernel Approximations: From Quadratic to Linear

The softmax attention matrix softmax(QK^T / sqrt(d)) can be decomposed using kernel methods.
If we find a feature map phi such that exp(q^T k) is approximately equal to phi(q)^T phi(k),
we can rewrite attention as:

```
Standard:   output = softmax(Q K^T) V            -- must form L x L matrix
Linear:     output = phi(Q) (phi(K)^T V)          -- form d x d matrix first
```

By computing phi(K)^T V first (a d x d matrix regardless of sequence length), we avoid the
L x L bottleneck entirely. Different modules use different feature maps:

- **LinearTransformer**: phi(x) = ELU(x) + 1 (simple, deterministic)
- **Performer**: phi(x) uses orthogonal random features to approximate the exponential
  kernel (FAVOR+ mechanism), with better approximation quality
- **Nystromformer**: takes a different approach entirely, sampling M landmark points via
  average pooling and reconstructing the attention matrix as F1 * pinv(F2) * F3

### The Retention / Recurrence Path

RetNet, RWKV, GLA, HGRN, and Griffin share a design philosophy: replace the attention matrix
with a recurrent state that can be updated incrementally. This gives O(1) per-token inference
(constant regardless of sequence length) while still allowing parallel training via scan-like
formulations.

**RetNet** uses multi-scale exponential decay: each head has a decay factor gamma_h, and the
retention mechanism computes s_n = gamma * s_{n-1} + K_n^T * V_n. Different heads use
different decay rates, capturing patterns at multiple timescales. RetNet supports three
computation modes with the same weights: parallel (training), recurrent (inference), and
chunkwise (long sequences).

**RWKV** (version 7, "Goose") uses a generalized delta rule with separate time-mixing and
channel-mixing sub-blocks. The time-mixing block computes weighted key-value aggregation
with learned per-channel decay. The channel-mixing block acts as a gated FFN. RWKV achieves
O(1) per-step inference and has been deployed on 1.5 billion Windows devices.

**GLA** adds data-dependent gating to linear attention. Rather than using fixed feature maps,
it computes gates from the input that modulate information flow. This makes GLA particularly
effective on short sequences (under 2K tokens) where it can outperform FlashAttention-2 in
wall-clock time.

**HGRN-2** uses hierarchical gating with state expansion: the hidden state is expanded to a
higher dimension during recurrence (enriching the representation), then contracted back for
output. The forget and input gates create a hierarchical structure where different layers
operate at different timescales.

**Griffin** combines a Real-Gated Linear Recurrent Unit (RG-LRU) with local sliding-window
attention. The RG-LRU is simpler than Mamba's SSM -- just gated decay with a
norm-preserving sqrt(1 - a^2) input scaling. The local attention layers appear periodically
(e.g., every 3rd layer) for short-range precise recall. The Hawk variant uses only RG-LRU
without any attention.

### FNet and Perceiver: Structural Alternatives

**FNet** demonstrates that the token mixing provided by attention can be partially replaced
by a parameter-free operation. Applying the Fast Fourier Transform along the sequence axis
gives global mixing (every token influences every other through frequency-domain
multiplication) at O(L log L) cost with zero learned parameters. FNet achieves 92-97% of
BERT's quality on standard benchmarks while training roughly 7x faster.

**Perceiver** solves a different problem: handling inputs of arbitrary size and modality.
Instead of self-attending over the full input (O(N^2) for N input elements), Perceiver
cross-attends the input to a small learned latent array of M vectors (M << N). Subsequent
processing happens on the latent array at O(M^2) cost. The total cost O(N*M + M^2) scales
linearly with input size for fixed M, making Perceiver applicable to point clouds, audio,
video, and other modalities that would be infeasible with standard attention.

## Complexity Comparison

```
+-------------------+----------+----------+-----------+---------+-------------+
| Module            | Training | Inference| KV Cache  | Learned | Recurrent   |
|                   | (per tok)| (per tok)| Size      | Params  | Mode?       |
+-------------------+----------+----------+-----------+---------+-------------+
| MultiHead         | O(L^2)   | O(L)     | O(H*L*d)  | QKV+out | No          |
| GQA               | O(L^2)   | O(L)     | O(G*L*d)  | QKV+out | No          |
| Perceiver         | O(L*M)   | O(M^2)   | O(M*d)    | QKV+lat | No          |
| FNet              | O(L logL)| O(L logL)| None      | FFN only| No          |
| LinearTransformer | O(L*d^2) | O(d^2)   | O(d^2)    | QKV+out | Yes         |
| Nystromformer     | O(L*M)   | O(L*M)   | O(M*d)    | QKV+out | No          |
| Performer         | O(L*d*m) | O(d*m)   | O(d*m)    | QKV+out | Yes         |
| RetNet            | O(L^2)*  | O(1)     | O(d^2)    | QKV+gate| Yes (triple)|
| RWKV              | O(L)     | O(1)     | O(d)      | WKV+ch  | Yes         |
| GLA               | O(L)     | O(1)     | O(d^2)    | QKV+gate| Yes         |
| HGRN              | O(L)     | O(1)     | O(E*d)    | Gates   | Yes         |
| Griffin            | O(L)+**  | O(1)+**  | O(d)+**   | RG-LRU  | Yes (hybrid)|
+-------------------+----------+----------+-----------+---------+-------------+

Notes:
- L = sequence length, d = head dimension, H = num heads, G = num KV groups
- M = num landmarks (Nystrom) or num latents (Perceiver)
- m = num random features (Performer)
- E = state expansion factor (HGRN)
- * RetNet parallel mode is O(L^2) but supports O(1) recurrent inference
- ** Griffin adds O(W^2) for local attention layers (W = local window size)
- "Recurrent Mode" = supports O(1) per-step incremental inference
```

## Module Reference

- `Edifice.Attention.MultiHead` -- Standard multi-head attention with optional sliding window and QK LayerNorm
- `Edifice.Attention.GQA` -- Grouped Query Attention with configurable num_kv_heads for KV cache reduction
- `Edifice.Attention.Perceiver` -- Cross-attention to learned latent array for input-agnostic processing
- `Edifice.Attention.FNet` -- Fourier Transform token mixing with zero attention parameters
- `Edifice.Attention.LinearTransformer` -- Kernel-based linear attention using ELU+1 feature maps
- `Edifice.Attention.Nystromformer` -- Nystrom method approximation with configurable landmark points
- `Edifice.Attention.Performer` -- FAVOR+ random orthogonal feature attention for O(N) complexity
- `Edifice.Attention.RetNet` -- Multi-scale exponential decay retention with parallel/recurrent/chunkwise modes
- `Edifice.Attention.RWKV` -- RWKV-7 "Goose" with time-mixing (WKV attention) and channel-mixing (gated FFN)
- `Edifice.Attention.GLA` -- Gated Linear Attention with data-dependent decay and hardware-efficient training
- `Edifice.Attention.HGRN` -- HGRN-2 with hierarchical gating, state expansion, and contraction
- `Edifice.Attention.Griffin` -- Griffin (RG-LRU + local attention) and Hawk (RG-LRU only) variants

## Cross-References

- **[State Space Models](state_space_models.md)** -- SSM hybrid architectures (Jamba, Zamba,
  HybridBuilder) integrate MultiHead attention as periodic layers within SSM stacks. Griffin
  bridges this family with its RG-LRU, which is closely related to a diagonal SSM.
- **Building Blocks** -- RoPE (Rotary Position Embedding) and ALiBi (Attention with Linear Biases)
  provide position information to attention mechanisms. RMSNorm and LayerNorm are used for
  pre-normalization in all attention blocks.
- **Recurrent Networks** -- HGRN and Griffin straddle the boundary between attention and recurrence.
  Their gated recurrences are functionally similar to GRU/LSTM cells but with linear (not
  saturating) dynamics that enable parallel scan training.
- **[Generative Models](generative_models.md)** -- DiT (Diffusion Transformer) uses MultiHead
  attention as its backbone, with AdaptiveNorm for conditioning.

## Further Reading

1. Vaswani et al. "Attention Is All You Need." NeurIPS 2017. -- The transformer paper that
   established multi-head attention as the dominant sequence modeling primitive.
2. Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head
   Checkpoints." arXiv:2305.13245 -- Interpolating between MHA and MQA for inference efficiency.
3. Katharopoulos et al. "Transformers are RNNs: Fast Autoregressive Transformers with Linear
   Attention." ICML 2020. -- The kernel trick that reduces attention from O(L^2) to O(L).
4. Sun et al. "Retentive Network: A Successor to Transformer for Large Language Models."
   arXiv:2307.08621 -- Multi-scale retention with parallel/recurrent/chunkwise triple paradigm.
5. De et al. "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient
   Language Models." arXiv:2402.19427 -- RG-LRU combined with local attention for a practical
   hybrid architecture.
