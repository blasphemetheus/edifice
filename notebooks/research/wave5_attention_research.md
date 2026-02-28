# Wave 5 Attention Architecture Research

> Research findings for Wave 5 attention implementations in Edifice.
> Compiled 2026-02-28.

---

## Status Overview

| Architecture | Type | Paper Venue | Status | Notes |
|-------------|------|-------------|--------|-------|
| LASER | Softmax variant | ICML 2025 | TODO | exp(V) + log-sum-exp output |
| MoBA | Sparse attention | Moonshot Tech Report 2025 | TODO | Block-sparse MoE-style routing |
| Multi-Token Attention | Conv-augmented | Meta 2025 | TODO | 2D conv on attention weights |
| DeltaProduct | Linear RNN | NeurIPS 2025 | TODO | Multi-step Householder products |
| Gated Slot Attention | Linear attention | NeurIPS 2024 | TODO | Two-layer GLA with slot memory |
| Residual Linear Attention | Linear attention | Preprint 2025 | TODO | Error-correcting auxiliary state |

---

## 1. LASER — Attention with Exponential Transformation

**Paper:** Duvvuri, Dhillon. "LASER: Attention with Exponential Transformation." ICML 2025.
[arXiv:2411.03493](https://arxiv.org/abs/2411.03493)

### Key Innovation

LASER (LogArithm of Summed Exponentials of Representations) applies exp() to V before standard attention and log() to the output. This changes the gradient landscape: standard softmax attention Jacobian scales as `a_j(1 - a_j)` which vanishes when attention probabilities are very small or large (80%+ of probs are < 1e-3 in practice). LASER's Jacobian scales as `(1 - a_j)`, providing much larger gradients for learning pre-attention parameters (W_Q, W_K, W_V).

### Architecture

```
Input: Q [B, N, d], K [B, N, d], V [B, N, d]
         |
         v
  +---------------------------------+
  |  1. m = max(V, axis=seq)        |  column-wise max: [B, 1, d]
  |     m = stop_gradient(m)        |
  |                                  |
  |  2. V_hat = V - m               |  shift for stability: [B, N, d]
  |                                  |
  |  3. exp_V = exp(V_hat)          |  [B, N, d]
  |                                  |
  |  4. A = softmax(Q K^T / sqrt(d))|  standard attention: [B, N, N]
  |                                  |
  |  5. O' = A @ exp_V              |  weighted sum of exp(V): [B, N, d]
  |                                  |
  |  6. O = log(O') + m             |  recover log-domain output: [B, N, d]
  +---------------------------------+
         |
         v
Output: O [B, N, d]
```

### Key Equations

Standard attention:
```
attn(X) = softmax(QK^T / sqrt(d)) V
```

LASER attention:
```
laser(X) = log(softmax(QK^T / sqrt(d)) exp(V))
```

Numerically stable LWSE (Log-Weighted-Sum-Exp) trick:
```
m_j     = max_i(V_{ij})                          -- column-wise max
V_hat   = V - m                                    -- shift values
O'      = softmax(QK^T / sqrt(d)) @ exp(V_hat)    -- attention on shifted exp(V)
O_{ij}  = log(O'_{ij}) + m_j                       -- recover with max
```

Gradient advantage (N=2 case):
```
Standard: dO/da_tilde = (v1 - v2) * sigma(a) * (1 - sigma(a))   -- vanishes at extremes
LASER:    dO/da_tilde ~ (1 - sigma(a))                           -- only vanishes at one extreme
```

Temperature variant:
```
laser_temp(X) = log(softmax(QK^T / (tau * sqrt(d))) exp(V))
```
where tau is a scalar or per-dimension learnable temperature.

### Input/Output Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| Q, K, V | `{batch, seq_len, embed_dim}` | Standard projections from input |
| Attention logits | `{batch, seq_len, seq_len}` | QK^T / sqrt(d) |
| Attention weights | `{batch, seq_len, seq_len}` | After softmax |
| m (max) | `{batch, 1, embed_dim}` | Column-wise max of V, stop-gradient |
| exp(V_hat) | `{batch, seq_len, embed_dim}` | Numerically stable exp |
| Output | `{batch, seq_len, embed_dim}` | Same shape as V |

Multi-head form: `{batch, heads, seq_len, head_dim}` for all Q/K/V/O.

### Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `embed_dim` | required | Model dimension |
| `num_heads` | 8 | Number of attention heads |
| `head_dim` | embed_dim / num_heads | Per-head dimension |
| `temperature` | `nil` | Optional scalar or per-dim learnable temp |
| `dropout` | 0.0 | Attention dropout |
| `causal` | `false` | Whether to apply causal mask |

### Implementation Notes for Axon

1. **Drop-in replacement**: LASER wraps standard attention. Compute attention normally but on `exp(V_hat)` instead of `V`, then apply `log() + m` to the output. This can be implemented as a thin wrapper around existing `Primitives.scaled_dot_product_attention/4`.

2. **stop_gradient on m**: Use `Nx.stop_grad(m)` to prevent gradients flowing through the max computation. This is critical for stable training.

3. **Numerical safety**: After `softmax @ exp(V_hat)`, the result `O'` should be strictly positive since softmax produces positive weights and exp produces positive values. But for safety, clamp before log: `Nx.log(Nx.max(o_prime, 1.0e-8))`.

4. **No new parameters**: LASER adds zero parameters beyond standard attention (unless using learnable temperature). Only modifies the forward pass.

5. **Axon implementation**: Use `Axon.layer` to wrap the custom forward function. The layer takes the same Q/K/V inputs as standard attention.

```elixir
# Pseudocode for LASER attention layer
def laser_attention(query, key, value, _opts) do
  m = Nx.reduce_max(value, axes: [-2], keep_axes: true) |> Nx.stop_grad()
  v_hat = Nx.subtract(value, m)
  exp_v = Nx.exp(v_hat)

  # Standard scaled dot-product attention on exp(V_hat)
  scale = Nx.rsqrt(Nx.tensor(head_dim, type: Nx.type(query)))
  logits = Nx.dot(query, [-1], key, [-1]) |> Nx.multiply(scale)
  weights = Axon.Activations.softmax(logits, axis: -1)
  o_prime = Nx.dot(weights, [-1], exp_v, [-2])

  Nx.log(Nx.max(o_prime, Nx.tensor(1.0e-8))) |> Nx.add(m)
end
```

---

## 2. MoBA — Mixture of Block Attention

**Paper:** Lu, Jiang, Liu et al. (Moonshot AI / Tsinghua / Zhejiang). "MoBA: Mixture of Block Attention for Long-Context LLMs." 2025.
[arXiv:2502.13189](https://arxiv.org/abs/2502.13189).
GitHub: [MoonshotAI/MoBA](https://github.com/MoonshotAI/MoBA)

### Key Innovation

MoBA applies Mixture-of-Experts principles to attention: KV cache is partitioned into fixed-size blocks, a parameter-free gating function scores each block's relevance per query, and only the top-k most relevant blocks are attended to. This achieves sparse attention without imposing fixed structural patterns (like sliding window or sink tokens), letting the model learn where to attend.

### Architecture

```
Input: Q [B, N, H, d], K [B, N, H, d], V [B, N, H, d]
         |
         v
  +-----------------------------------------------------+
  |  1. Partition KV into n blocks of size B             |
  |     K_blocks = [K_1, K_2, ..., K_n]                 |
  |     V_blocks = [V_1, V_2, ..., V_n]                 |
  |                                                       |
  |  2. Compute block affinity scores per query           |
  |     K_bar_i = mean_pool(K_i)          [n, H, d]     |
  |     s_i = <q, K_bar_i>               [N, H, n]      |
  |                                                       |
  |  3. Apply causal mask (zero future blocks)            |
  |     s_i = -inf if pos(q) < i * B                     |
  |                                                       |
  |  4. Top-k selection per query                         |
  |     G = topk(s, k)                   [N, H, k]      |
  |                                                       |
  |  5. Self-block attention (causal)                     |
  |     O_self = FlashAttn(Q_b, K_b, V_b, causal=True)  |
  |                                                       |
  |  6. Selected-block attention (non-causal)             |
  |     O_moba = FlashAttn(Q, K[G], V[G], causal=False)  |
  |                                                       |
  |  7. Combine via online softmax                        |
  |     O = combine(O_self, O_moba)                       |
  +-----------------------------------------------------+
         |
         v
Output: O [B, N, H, d]
```

### Key Equations

Block partitioning (n = N / B blocks):
```
I_i = [(i-1)*B + 1, i*B]           -- index range of block i
K_bar_i = (1/B) sum_{j in I_i} K_j -- mean-pooled key per block
```

Gating function (parameter-free):
```
s_i = q^T K_bar_i                   -- dot product affinity
g_i = 1 if s_i in TopK({s_j}, k)    -- binary gate
      0 otherwise
```

Causal constraint:
```
s_i = -inf  if pos(q) < i * B       -- cannot attend to future blocks
```

Selected attention:
```
I = union_{g_i > 0} I_i             -- selected KV indices
MoBA(q, K, V) = softmax(q K[I]^T / sqrt(d)) V[I]
```

Online softmax combination (for merging self-block and MoBA outputs):
```
O = (exp(lse_self) * O_self + exp(lse_moba) * O_moba) /
    (exp(lse_self) + exp(lse_moba))
```
where `lse_*` are the log-sum-exp values from each attention computation.

### Input/Output Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| Q, K, V | `{batch, seq_len, heads, head_dim}` | Standard MHA projections |
| Block-pooled keys K_bar | `{batch, n_blocks, heads, head_dim}` | n_blocks = seq_len / block_size |
| Affinity scores S | `{batch, seq_len, heads, n_blocks}` | Per-query block scores |
| Gate indices G | `{batch, seq_len, heads, top_k}` | Selected block indices |
| Output O | `{batch, seq_len, heads, head_dim}` | Same as input |

### Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `block_size` | 512 | KV block size; larger = fewer blocks, coarser routing |
| `top_k` | 3 | Number of blocks selected per query |
| `num_heads` | 8 | Standard MHA heads |
| `head_dim` | 64 | Per-head dimension |
| `causal` | `true` | Whether to mask future blocks |

Sparsity ratio: `1 - (top_k * block_size) / seq_len`. With top_k=3, B=512, N=8192: 81.25% sparse.

### Implementation Notes for Axon

1. **Static graph challenge**: MoBA's top-k routing produces data-dependent indexing (which blocks are selected varies per query). In Axon's static graph, this requires `Nx.take` with dynamic indices. The top-k operation itself is available via `Nx.top_k/2`.

2. **Block mean pooling**: Use `Nx.reshape` to group the seq dimension into `{n_blocks, block_size}`, then `Nx.mean(axis: block_axis)`. Requires seq_len divisible by block_size.

3. **Online softmax combination**: When combining self-block and MoBA attention outputs, need to track the log-sum-exp from each component and merge using the numerically stable online softmax formula.

4. **Simplified Edifice version**: For a static-graph implementation, consider a simplified variant:
   - Compute full block affinity scores
   - Use a soft mask instead of hard top-k: multiply attention logits by sigmoid(affinity) to approximate sparse selection
   - This avoids dynamic indexing while preserving the block-relevance signal

5. **No new parameters**: MoBA is parameter-free beyond standard Q/K/V projections. The gating is purely based on dot-product affinity with mean-pooled keys.

6. **seq_len must be divisible by block_size**. Enforce this in `build/1` options validation.

---

## 3. Multi-Token Attention (MTA)

**Paper:** Golovneva, Wang, Weston, Sukhbaatar (Meta AI). "Multi-Token Attention." 2025.
[arXiv:2504.00927](https://arxiv.org/abs/2504.00927)

### Key Innovation

Standard attention computes each weight `a_{ij}` from a single (q_i, k_j) pair. MTA applies learned 2D convolution over the attention logit matrix so each weight is influenced by neighboring queries AND keys. It also mixes across head groups via 1D head convolution. This adds only 0.001% parameters but significantly improves long-context tasks.

### Architecture

```
Input: Q [B, H, N, d], K [B, H, N, d], V [B, H, N, d]
         |
         v
  +----------------------------------------------------+
  |  1. Compute raw attention logits                    |
  |     A_hat = Q K^T / sqrt(d)      [B, H, N, N]     |
  |                                                     |
  |  2. Causal zero-mask (pre-conv)                     |
  |     A_hat = Mask_0(A_hat)         [B, H, N, N]     |
  |                                                     |
  |  3. Key-Query 2D convolution (every 4th layer)      |
  |     A_hat = Conv2d_theta(A_hat)   [B, H, N, N]     |
  |     kernel: [c_q, c_k] per head                     |
  |     c_q lookback on queries, c_k centered on keys   |
  |                                                     |
  |  4. Causal -inf mask (post-conv)                    |
  |     A_hat = Mask_{-inf}(A_hat)    [B, H, N, N]     |
  |                                                     |
  |  5. Group normalization + exp scaling               |
  |     A_hat = GroupNorm(A_hat)      [B, H, N, N]     |
  |                                                     |
  |  6. Softmax over key dimension                      |
  |     A = softmax(A_hat, axis=-1)   [B, H, N, N]     |
  |                                                     |
  |  7. Head convolution (every layer)                  |
  |     A = HeadConv1d(A)             [B, H, N, N]     |
  |     kernel: [c_h] across head groups                |
  |                                                     |
  |  8. Value aggregation                               |
  |     O = A @ V                     [B, H, N, d]     |
  +----------------------------------------------------+
         |
         v
Output: O [B, H, N, d]
```

### Key Equations

Standard attention weight:
```
a_{ij} = softmax_j(q_i^T k_j / sqrt(d))
```

MTA pre-softmax 2D convolution (Eq. 4 from paper):
```
a_{ij} = softmax_j( sum_{i'=0}^{c_q-1} sum_{j'=-floor(c_k/2)}^{ceil(c_k/2)-1}
           1_{i >= j-j'} * theta_{i',j'} * q_{i-i'}^T k_{j-j'} / sqrt(d) )
```

Practical implementation with double masking (Eq. 5):
```
A = softmax(Mask_{-inf}(Conv2d_theta(Mask_0(A_hat))))
```

Head mixing convolution (post-softmax, within groups of c_h heads):
```
A_new^h = sum_{h'=1}^{c_h} w_{h,h'} * A^{h'}
```

Combined as single 3D convolution when both are pre-softmax:
```
Conv3d with kernel [c_q, c_k, c_h] over [query_dim, key_dim, head_dim]
```

### Input/Output Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| Q, K, V | `{batch, heads, seq_len, head_dim}` | Standard MHA |
| Attention logits A_hat | `{batch, heads, seq_len, seq_len}` | Before conv |
| Conv2d kernel theta | `{heads, 1, c_q, c_k}` | Depthwise per head |
| Head conv kernel | `{heads, c_h, 1, 1}` | Grouped 1x1 conv over heads |
| Attention weights A | `{batch, heads, seq_len, seq_len}` | After softmax + head conv |
| Output O | `{batch, heads, seq_len, head_dim}` | Final |

### Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `c_q` | 6 | Query conv kernel size (causal lookback) |
| `c_k` | 11 | Key conv kernel size (centered, bidirectional) |
| `c_h` | 2 | Head mixing group size |
| `kq_conv_every` | 4 | Apply key-query conv every N-th layer |
| `head_conv_every` | 1 | Apply head conv every layer |
| `num_heads` | 16 | Standard MHA heads |
| `head_dim` | 96 | Per-head dimension |
| `kernel_init` | identity | Initialize conv kernel as identity for warm start |

Parameter overhead: ~9.4M for 880M model (0.001%).

### Implementation Notes for Axon

1. **Double masking is critical**: Before convolution, zero out future positions in the logit matrix (causal zero-mask). After convolution, apply standard -inf causal mask. The zero-mask prevents information leaking through conv kernel edges.

2. **Depthwise 2D conv on attention matrix**: The attention logit matrix `{B, H, N, N}` is treated as a 4D tensor where the conv operates over the last two dims (query, key) independently per head. Use `Axon.conv` with groups = num_heads.

3. **Asymmetric kernel**: c_q is causal (only past queries, kernel is one-sided) while c_k is centered (past and future keys within conv window). Implement via asymmetric padding: `pad: {c_q - 1, 0, floor(c_k/2), floor(c_k/2)}`.

4. **Head convolution**: After softmax, apply 1D conv across the head dimension within groups. Reshape `{B, H, N, N}` to `{B*N*N, H, 1}` and use grouped 1D conv with groups = H/c_h.

5. **Identity kernel init**: Initialize the 2D conv kernel so that only `theta[0, 0] = 1` (center of key kernel, current query). This makes MTA behave as standard attention at initialization, enabling stable warm-up.

6. **Group normalization**: Apply per-head group norm with exponential depth scaling to the conv output before softmax. This is needed for training stability.

7. **Selective application**: Apply KQ conv only every 4th layer (configurable) to balance quality and compute. Head conv is cheap and applied every layer.

8. **Memory concern**: Materializing the full `{B, H, N, N}` attention matrix is required (no FlashAttention compatibility). For Edifice's typical sequence lengths (<2K) this is fine, but document the O(N^2) memory requirement.

---

## 4. DeltaProduct — Multi-Step DeltaNet with Householder Products

**Paper:** Sieber et al. "DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products." NeurIPS 2025.
[arXiv:2502.10297](https://arxiv.org/abs/2502.10297)

### Key Innovation

DeltaNet interprets its recurrence as one step of online gradient descent per token on an associative recall objective. DeltaProduct takes n_h gradient steps per token, producing state-transition matrices that are products of n_h generalized Householder reflections. This gives a tunable expressivity-efficiency tradeoff: more steps = richer state transitions (diagonal + rank-n_h), with the Cartan-Dieudonne theorem guaranteeing that any orthogonal matrix can be decomposed into at most d Householder reflections.

### Architecture

```
Input: x [B, T, d_model]
         |
         v
  +-------------------------------------------------------+
  |  For each token x_i:                                   |
  |                                                         |
  |  1. Generate n_h key-value-beta triplets:               |
  |     k_{i,j} = normalize(W_j x_i)    [d_state]         |
  |     v_{i,j} = V_j x_i               [d_state]         |
  |     beta_{i,j} = sigmoid(U_j x_i)   scalar            |
  |     for j = 1..n_h                                     |
  |                                                         |
  |  2. Apply n_h sequential Householder updates:           |
  |     H_{i,0} = H_{i-1}                                  |
  |     H_{i,j} = (I - beta * k k^T) H_{i,j-1}            |
  |              + beta * k v^T                             |
  |     for j = 1..n_h                                     |
  |                                                         |
  |  3. Read output via query:                              |
  |     q_i = W_Q x_i                   [d_state]          |
  |     y_i = H_{i,n_h}^T q_i           [d_state]          |
  |                                                         |
  |  4. Project to output:                                  |
  |     o_i = W_O y_i                    [d_model]          |
  +-------------------------------------------------------+
         |
         v
Output: o [B, T, d_model]
```

### Key Equations

Single Householder update (DeltaNet, n_h = 1):
```
H_i = (I - beta_i k_i k_i^T) H_{i-1} + beta_i k_i v_i^T
```

DeltaProduct multi-step update (n_h steps per token):
```
H_{i,j} = (I - beta_{i,j} k_{i,j} k_{i,j}^T) H_{i,j-1} + beta_{i,j} k_{i,j} v_{i,j}^T
```
for j = 1..n_h, with H_{i,0} = H_{i-1,n_h}.

State transition matrix (product of Householder reflections):
```
A(x_i) = product_{j=1}^{n_h} (I - beta_{i,j} k_{i,j} k_{i,j}^T)
```

Input matrix (accumulated across steps):
```
B(x_i) = sum_{j=1}^{n_h} [product_{k=j+1}^{n_h} (I - beta_{i,k} k_{i,k} k_{i,k}^T)]
          * beta_{i,j} k_{i,j} v_{i,j}^T
```

Combined update:
```
H_i = A(x_i) H_{i-1} + B(x_i)
y_i = H_i^T q_i
```

Key generation (normalized):
```
k_{i,j} = W_j x_i / ||W_j x_i||_2
```

Beta range:
```
beta_{i,j} = sigmoid(U_j x_i)       -- eigenvalues in [0, 1]
beta_{i,j} = 2 * sigmoid(U_j x_i)   -- eigenvalues in [-1, 1] (allows reflections)
```

Norm stability: `||A(x_i)|| <= product_j ||I - beta_{i,j} k k^T|| <= 1` when beta in [0,2] and ||k||=1.

### Input/Output Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| Input x_i | `{batch, d_model}` | Per-token input |
| k_{i,j} | `{batch, n_heads, d_state}` | L2-normalized key per step |
| v_{i,j} | `{batch, n_heads, d_state}` | Value per step |
| beta_{i,j} | `{batch, n_heads}` | Scalar gate per head per step |
| H_i | `{batch, n_heads, d_state, d_state}` | Hidden state matrix |
| q_i | `{batch, n_heads, d_state}` | Query for readout |
| y_i | `{batch, n_heads, d_state}` | Per-head output |
| Output o_i | `{batch, d_model}` | After concat + output proj |
| Sequence input | `{batch, seq_len, d_model}` | Full sequence |
| Sequence output | `{batch, seq_len, d_model}` | Full sequence |

### Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `n_householder` | 2 | Number of Householder steps per token (n_h). 1 = DeltaNet |
| `d_state` | 128 | Hidden state dimension per head |
| `num_heads` | 8 | Number of parallel heads |
| `beta_range` | `[0, 1]` | `:unit` for [0,1] sigmoid, `:symmetric` for [-1,1] (2*sigmoid) |
| `d_model` | required | Model dimension |
| `num_layers` | 6 | Stacked layers |

Compute cost: n_h times DeltaNet's recurrence cost. Expressivity: can represent permutation groups of up to n_h + 1 elements per layer.

### Implementation Notes for Axon

1. **Interleaving trick for chunk-wise training**: DeltaProduct's training trick is to interleave the n_h sub-steps as if they were separate tokens in the sequence. Generate `[k_{1,1}, ..., k_{1,n_h}, k_{2,1}, ..., k_{2,n_h}, ...]`, run a single DeltaNet-style chunk-wise recurrence on the expanded sequence, then downsample the output by keeping every n_h-th element. This reuses existing linear attention kernels.

2. **L2 normalization of keys**: Use `Nx.divide(k, Nx.LinAlg.norm(k, axes: [-1], keep_axes: true))` after projection. Critical for norm stability of the Householder product.

3. **State matrix is d_state x d_state**: Unlike DeltaNet which stores d_state vectors, DeltaProduct stores a full matrix. Memory per layer per head: `d_state^2`. With d_state=128, that is 16KB in f32 per head.

4. **Recurrent inference**: At inference time, iterate the Householder updates sequentially. Each step is `H = (I - beta * outer(k, k)) @ H + beta * outer(k, v)`. The outer product is `Nx.outer(k, k)`.

5. **Relation to existing Edifice modules**: Edifice already has DeltaNet-style recurrences in the linear attention family. DeltaProduct extends this with the n_h parameter. Consider implementing as a `:variant` option on an existing DeltaNet module, or as a standalone `Edifice.Attention.DeltaProduct`.

6. **Projection matrices**: Need n_h separate W_k, W_v, W_beta projection matrices per layer. These can be batched as a single projection of size `d_model -> n_h * d_state` and reshaped.

---

## 5. Gated Slot Attention (GSA)

**Paper:** Zhang, Yang et al. "Gated Slot Attention for Efficient Linear-Time Sequence Modeling." NeurIPS 2024.
[arXiv:2409.07146](https://arxiv.org/abs/2409.07146)

### Key Innovation

GSA is a two-layer GLA (Gated Linear Attention) architecture linked via softmax. It maintains fixed-size memory "slots" (m slots, each of dimension d) -- much smaller than the d x d state in standard linear attention. The first GLA pass writes to slots; the softmax intermediate step provides context-aware focusing; the second GLA pass reads from slots to produce output. Adaptive forgetting gates enable explicit memory management.

### Architecture

```
Input: x [B, T, d_model]
         |
         v
  +------------------------------------------------------------+
  |  Projections:                                               |
  |    q = silu(W_Q x)              [B, T, d]                  |
  |    k = silu(W_K x)              [B, T, d]                  |
  |    v = W_V x                    [B, T, d]                  |
  |    alpha = sigmoid(W_alpha x)^(1/tau)   [B, T, m]          |
  |                                                              |
  |  Pass 1: Write to slots (GLA)                               |
  |    o'_t = GLA_1(q_t, k_t, alpha_t)     [B, T, m]           |
  |    Recurrence:                                               |
  |      K_tilde_t = diag(alpha_t) K_tilde_{t-1}               |
  |                  + (1 - alpha_t) outer k_t                   |
  |      o'_t = K_tilde_t^T q_t            [m]                  |
  |                                                              |
  |  Softmax focusing:                                           |
  |    p_t = softmax(o'_t)                  [B, T, m]           |
  |                                                              |
  |  Pass 2: Read from slots (GLA)                               |
  |    o_t = GLA_2(p_t, alpha_t, v_t)      [B, T, d]           |
  |    Recurrence:                                               |
  |      V_tilde_t = diag(alpha_t) V_tilde_{t-1}               |
  |                  + (1 - alpha_t) outer v_t                   |
  |      o_t = V_tilde_t^T p_t             [d]                  |
  |                                                              |
  |  Output projection:                                          |
  |    o = W_O o_t                          [B, T, d_model]     |
  +------------------------------------------------------------+
         |
         v
Output: o [B, T, d_model]
```

### Key Equations

Adaptive forgetting gate:
```
alpha_i^h = sigmoid(W_alpha^h x_i)^{1/tau}
```
where tau = 8 (damping factor pushing gates toward 1 for slow forgetting).

Pass 1 -- Slot key accumulation (write):
```
K_tilde_t[s] = alpha_t[s] * K_tilde_{t-1}[s] + (1 - alpha_t[s]) * k_t     for each slot s
o'_t = K_tilde_t^T q_t     -- [m] dimensional slot scores
```

Intermediate softmax:
```
p_t = softmax(o'_t)        -- [m] slot attention distribution
```

Pass 2 -- Slot value readout (read):
```
V_tilde_t[s] = alpha_t[s] * V_tilde_{t-1}[s] + (1 - alpha_t[s]) * v_t     for each slot s
o_t = V_tilde_t^T p_t      -- [d] weighted slot readout
```

Matrix form for full recurrence:
```
K_tilde_t = diag(alpha_t) K_tilde_{t-1} + diag(1 - alpha_t) 1 k_t^T     [m x d]
V_tilde_t = diag(alpha_t) V_tilde_{t-1} + diag(1 - alpha_t) 1 v_t^T     [m x d]
```

### Input/Output Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| Input x | `{batch, seq_len, d_model}` | Sequence input |
| q, k | `{batch, seq_len, d}` | After projection + SiLU |
| v | `{batch, seq_len, d}` | Value projection |
| alpha | `{batch, seq_len, m}` | Per-slot forget gates |
| K_tilde | `{batch, m, d}` | Slot key memory (recurrent state) |
| V_tilde | `{batch, m, d}` | Slot value memory (recurrent state) |
| o' (intermediate) | `{batch, seq_len, m}` | Slot scores (pre-softmax) |
| p (slot weights) | `{batch, seq_len, m}` | Softmax slot distribution |
| Output o | `{batch, seq_len, d_model}` | After output projection |

Per-head shapes: divide d by num_heads; K_tilde and V_tilde are per-head.

### Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_slots` | 64 | Number of memory slots (m). Tested: 32, 64, 128 |
| `num_heads` | 4 | Attention heads |
| `head_dim` | 64 | Per-head dimension |
| `damping` | 8 | Tau: damping factor for forget gate (higher = slower forget) |
| `activation` | `:silu` | Applied to Q, K projections |
| `d_model` | required | Model dimension |
| `num_layers` | 6 | Stacked GSA layers |

State size: `m * d` per head per layer (vs `d * d` for standard linear attention). With m=64, d=64: 4K elements vs 4K -- but m is decoupled from d, allowing independent tuning.

### Implementation Notes for Axon

1. **Two sequential GLA passes**: GSA is structurally two GLA layers with softmax between them. If Edifice already has `GLA.layer/2`, GSA can compose two calls. The first GLA maps `{q, k} -> m` dimensional slot scores; the second maps `{softmax(scores), v} -> d` dimensional output.

2. **Slot memory is small**: The recurrent state K_tilde and V_tilde are `{m, d}` matrices per head. With m=64 and d=64, this is 4096 elements -- much smaller than the `{d, d}` = 4096 elements of standard GLA. The advantage is that m and d are independently tunable.

3. **Chunk-wise parallel form**: Both GLA passes support chunk-wise parallelism. Each chunk processes `C` tokens in parallel, accumulating slot states across chunks. This matches the existing `FlashLinearAttention` pattern in Edifice.

4. **Gate damping**: The `^{1/tau}` power on the sigmoid gate is critical. With tau=8, `sigmoid(0)^{1/8} = 0.5^{0.125} = 0.917`, so even neutral inputs produce high retention. Implement as `Nx.pow(Nx.sigmoid(gate_logits), 1.0 / tau)`.

5. **SiLU activation on Q, K**: Use `Axon.activation(:silu)` (Axon 0.8 uses `:silu` not `:swish`).

6. **Relation to existing Edifice GLA**: GSA can be built as a composition layer that internally uses two GLA-style recurrences. The key difference is the intermediate softmax that provides a "bottleneck" for slot selection.

---

## 6. Residual Linear Attention (RLA)

**Paper:** "Enhancing Linear Attention with Residual Learning." Preprint 2025.
[arXiv:2509.25223](https://arxiv.org/abs/2509.25223)

### Key Innovation

Standard linear attention decomposes as `o_t = S_{t-1} q_t + (v_t k_t^T) q_t` -- a historical prediction plus a single-token correction. This single-token correction is an expressivity bottleneck. RLA introduces an auxiliary recurrent state R that accumulates prediction errors over time. The residual `r_t = v_t - S_{t-1} k_t` captures how much the base state mis-predicts the current token, and R learns to correct these systematic errors. RDN (Residual Delta Net) extends this with delta-rule updates and residual clipping.

### Architecture

```
Input: x [B, T, d_model]
         |
         v
  +------------------------------------------------------------+
  |  Projections:                                               |
  |    q = L2_norm(silu(W_Q x))        [B, T, d_h]            |
  |    k = L2_norm(silu(W_K x))        [B, T, d_h]            |
  |    v = W_V x                        [B, T, d_h]            |
  |    alpha = sigmoid(W_alpha x)       [B, T, 1]   (decay)   |
  |    beta  = sigmoid(W_beta x)        [B, T, 1]   (base)    |
  |    gamma = sigmoid(W_gamma x)       [B, T, 1]   (residual)|
  |                                                              |
  |  Per-token recurrence:                                       |
  |    r_t = clip(v_t - S_{t-1} k_t, -c, c)  -- residual error |
  |    R_t = alpha_t R_{t-1} + gamma_t r_t k_t^T               |
  |    S_t = alpha_t S_{t-1} + beta_t v_t k_t^T                |
  |    o_t = alpha_t S_{t-1} q_t + gamma_t R_t q_t             |
  |                                                              |
  |  Output projection:                                          |
  |    o = W_O [o_1, ..., o_T]          [B, T, d_model]        |
  +------------------------------------------------------------+
         |
         v
Output: o [B, T, d_model]
```

### Key Equations

Base linear attention decomposition:
```
o_t = S_{t-1} q_t + (v_t k_t^T) q_t
      ^^^^^^^^^^     ^^^^^^^^^^^^^^^^
      historical     single-token correction (bottleneck)
      prediction
```

Residual error:
```
r_t = clip[-c, c](v_t - S_{t-1} k_t)
```
This measures how well the base state S predicts the current value given the current key.

RLA recurrence:
```
r_t = clip[-c, c](v_t - S_{t-1} k_t)       -- compute residual
R_t = alpha_t R_{t-1} + gamma_t r_t k_t^T   -- update residual state
S_t = alpha_t S_{t-1} + beta_t v_t k_t^T    -- update base state
o_t = alpha_t S_{t-1} q_t + gamma_t R_t q_t -- output = base + correction
```

RDN (Residual Delta Net) variant with delta-rule:
```
R_t = alpha_t R_{t-1} (I - gamma_t k_t k_t^T) + gamma_t r_t k_t^T
S_t = alpha_t S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
o_t = alpha_t S_{t-1} q_t + gamma_t R_t q_t
```

Gate parameterization:
```
alpha_t = reparameterize_mamba2(W_alpha x_t)    -- decay (from Mamba-2)
beta_t  = sigmoid(W_beta x_t)                    -- base state learning rate
gamma_t = sigmoid(W_gamma x_t)                   -- residual learning rate
```

### Input/Output Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| Input x | `{batch, seq_len, d_model}` | Sequence input |
| q, k | `{batch, seq_len, head_dim}` | L2-normalized + SiLU |
| v | `{batch, seq_len, head_dim}` | Value projection |
| alpha, beta, gamma | `{batch, seq_len, 1}` | Per-token scalar gates |
| S (base state) | `{batch, head_dim, head_dim}` | Per-head recurrent state |
| R (residual state) | `{batch, head_dim, head_dim}` | Per-head auxiliary state |
| r (residual error) | `{batch, seq_len, head_dim}` | Clipped prediction error |
| Output o | `{batch, seq_len, d_model}` | After output projection |

Per-head: all head_dim tensors are divided by num_heads.

### Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `clip_threshold` | 1.0 | Residual clipping bound c |
| `num_heads` | 8 | Attention heads |
| `head_dim` | 64 | Per-head dimension |
| `d_model` | required | Model dimension |
| `num_layers` | 6 | Stacked layers |
| `variant` | `:rla` | `:rla` for basic, `:rdn` for delta-rule version |
| `activation` | `:silu` | Applied to Q, K before L2 norm |

### Implementation Notes for Axon

1. **Two state matrices**: RLA maintains both S (base) and R (residual) as `{head_dim, head_dim}` matrices per head. Double the state of standard linear attention. For head_dim=64, that is 2 * 4096 = 8192 elements per head.

2. **Residual computation requires base state output**: The residual `r_t = v_t - S_{t-1} k_t` needs the base state's prediction at the current key. When using chunk-wise parallel training, the kernel must return both the attention output and the intermediate `S_{t-1} k_t` values.

3. **Residual clipping**: `r_t = Nx.clip(r_t, -c, c)` with c=1.0. This stabilizes training by preventing large residuals from dominating the auxiliary state.

4. **L2 normalization on Q, K**: Apply `Nx.divide(q, Nx.LinAlg.norm(q, axes: [-1], keep_axes: true))` after SiLU activation. This is a common pattern in delta-rule methods.

5. **Reuse of linear attention kernels**: The paper's key efficiency claim is that existing optimized linear attention chunk-wise kernels can be augmented to also return intermediate values needed for residual computation. In Edifice, if we already have chunk-wise linear attention, we can extend it to return `S_{t-1} k_t` alongside `o_t`.

6. **RDN variant adds delta-rule to both states**: The `(I - beta_t k_t k_t^T)` term in RDN makes both S and R use delta-rule updates (like DeltaNet). This is a rank-1 modification to the decay. Compute as: `S_new = alpha * S - alpha * beta * k outer(k, S @ k) + beta * outer(v, k)`.

7. **Gate decoupling**: alpha (decay) is shared between base and residual states. beta controls base state learning rate, gamma controls residual learning rate. These are separate learned projections, allowing the model to decouple error accumulation speed from base memory speed.

8. **Module structure**: Consider implementing as `Edifice.Attention.ResidualLinearAttention` with a `:variant` option (`:rla` or `:rdn`). The forward pass structure is similar to GLA but with the auxiliary state.

---

## Cross-Architecture Comparison

| Architecture | Type | Complexity | New Params | State Size | Best For |
|-------------|------|-----------|------------|------------|----------|
| LASER | Softmax mod | O(N^2) | 0 | None | Drop-in attention improvement |
| MoBA | Sparse softmax | O(N * k * B) | 0 | None | Long-context sparse attention |
| MTA | Conv-augmented | O(N^2) | ~0.001% | None | Long-context multi-hop reasoning |
| DeltaProduct | Linear RNN | O(T * n_h * d^2) | n_h projections | d^2 per head | State tracking, length extrapolation |
| GSA | Linear attn | O(T * m * d) | slot gates | m * d per head | Efficient recall with small state |
| RLA/RDN | Linear attn | O(T * d^2) | 3 gates | 2 * d^2 per head | Error-corrected linear attention |

## Implementation Priority Recommendation

1. **LASER** -- Zero new parameters, 5-line forward pass modification, immediate benefit. Highest ROI.
2. **GSA** -- Composes existing GLA primitives, small state size, NeurIPS 2024. Good building block.
3. **RLA/RDN** -- Extends linear attention with principled error correction. Two variants from one module.
4. **DeltaProduct** -- Extends DeltaNet family. Interleaving trick reuses existing kernels.
5. **MTA** -- Novel 2D conv on attention matrix. Interesting but requires materializing N x N matrix.
6. **MoBA** -- Dynamic routing is challenging in static graphs. May need soft approximation.

---

## References

1. Duvvuri, Dhillon. [LASER: Attention with Exponential Transformation](https://arxiv.org/abs/2411.03493). ICML 2025.
2. Lu et al. [MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2502.13189). Moonshot AI, 2025.
3. Golovneva et al. [Multi-Token Attention](https://arxiv.org/abs/2504.00927). Meta AI, 2025.
4. Sieber et al. [DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products](https://arxiv.org/abs/2502.10297). NeurIPS 2025.
5. Zhang et al. [Gated Slot Attention for Efficient Linear-Time Sequence Modeling](https://arxiv.org/abs/2409.07146). NeurIPS 2024.
6. [Enhancing Linear Attention with Residual Learning](https://arxiv.org/abs/2509.25223). Preprint, 2025.
