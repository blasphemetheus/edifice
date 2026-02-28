# Wave 5 SSM & Recurrent Architecture Research

> Research findings for Wave 5 implementations: advanced SSM and depth-recurrent architectures.
> Compiled 2026-02-28.

---

## Status Overview

| Architecture | Family | Status | Notes |
|-------------|--------|--------|-------|
| Longhorn | SSM | TODO | Online associative recall SSM (replaces Mamba's scan) |
| Samba | SSM (Hybrid) | TODO | Mamba + SWA + MLP interleaving |
| Huginn | Recurrent | TODO | Depth-recurrent transformer, adaptive iteration |
| Mixture-of-Mamba (MoM) | SSM (Meta) | TODO | Modality-aware sparse Mamba blocks |

---

## 1. Longhorn -- SSM from Online Associative Recall

**Paper:** Bo Liu, Rui Wang, Lemeng Wu, Yihao Feng, Peter Stone, Qiang Liu. "Longhorn: State Space Models are Amortized Online Learners." ICLR 2025. [arXiv:2407.14207](https://arxiv.org/abs/2407.14207).

**Code:** [github.com/Cranial-XIX/Longhorn](https://github.com/Cranial-XIX/Longhorn)

### Key Innovation

Derives the SSM recurrence from the closed-form solution of an online associative recall problem, rather than discretizing a continuous ODE (S4) or adding ad hoc gates (Mamba). The forgetting mechanism emerges naturally from the key vector -- no explicit forget gate or A matrix initialization needed. Achieves 1.8x sample efficiency over Mamba and 16x context extrapolation.

### Architecture

```
Input x_t: {batch, seq_len, d}
      |
  +-----------+
  | Linear W_q |---> q_t in R^m  (query projection)
  | Linear W_k |---> k_t in R^m  (key projection)
  | Linear W_b |---> beta_t = sigmoid(W_b * x_t) in (0,1)^d  (learning rate)
  +-----------+
      |
  +-----------------------------------+
  | State Update (Online Learning)    |
  |                                   |
  | epsilon_t,i = beta_t,i / (1 + beta_t,i * ||k_t||^2)  |
  |                                   |
  | S_t = (1 - epsilon_t (x) k_t^2) * S_{t-1}            |
  |       + (epsilon_t * x_t) (x) k_t                     |
  +-----------------------------------+
      |
  o_t = S_t * q_t    (output: state-query product)
      |
  {batch, seq_len, d}
```

The outer architecture follows Mamba exactly: the Longhorn recurrence replaces only the selective scan SSM block inside each Mamba layer. All other components (input projection, depthwise conv, gating, output projection) remain identical.

```
Full Block (identical to Mamba block structure):

Input [batch, seq_len, embed_dim]
      |
 +----+----+
 |         |
 | Linear  | Linear
 | (expand) | (expand)
 |         |
 | DepthwiseConv+SiLU  |
 |         |
 | Longhorn SSM  SiLU  |
 |         |
 +-- multiply --+
      |
 Linear (project down)
      |
Output [batch, seq_len, embed_dim]
```

### Key Equations

**Online Associative Recall Objective:**

At each timestep, the state S_t solves:

```
S_t = argmin_S { ||S - S_{t-1}||^2_F + ||S*k_t - x_t||^2_diag(beta_t) }
```

This balances staying close to the previous state (memory retention) with accurately recalling the current input x_t given key k_t (new learning).

**Closed-Form Solution (Theorem 3.1):**

```
S_{t,i} = (I - epsilon_{t,i} * k_t * k_t^T) * S_{t-1,i} + epsilon_{t,i} * k_t * x_{t,i}

where:  epsilon_{t,i} = beta_{t,i} / (1 + beta_{t,i} * k_t^T * k_t)
```

**Diagonal Approximation (practical form):**

The full outer product `k_t * k_t^T` is O(m^2). The diagonal approximation replaces it with element-wise squaring:

```
S_t = (1_{d x m} - epsilon_t (x) k_t^{o2}) * S_{t-1} + (epsilon_t * x_t) (x) k_t
```

where:
- `k_t^{o2}` = element-wise squaring of k_t
- `(x)` = outer product (broadcasting d-dim with m-dim)
- `*` = element-wise (Hadamard) product
- `epsilon_t` = `beta_t / (1 + beta_t * ||k_t||^2)` in R^d

**Projections:**

```
q_t = W_q * x_t    in R^m     (query)
k_t = W_k * x_t    in R^m     (key)
beta_t = sigmoid(W_beta * x_t) in (0,1)^d  (per-dimension learning rate)
```

**Output:**

```
o_t = S_t * q_t    in R^d     (state-query product)
```

**Connection to Mamba:**

Mamba's scan: `S_t = exp(-Delta * exp(A)) * S_{t-1} + Delta * B * x_t`

Longhorn:     `S_t = (1 - epsilon * k^2) * S_{t-1} + (epsilon * x) * k`

Key difference: Mamba's A matrix is a learned parameter requiring careful initialization (S4D-Real log-spacing). Longhorn's "forgetting" `(1 - epsilon * k^2)` is derived from the optimization, linking forgetting to the key vector. No separate A matrix or special initialization needed.

### Input/Output Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| Input x_t | `{batch, seq_len, d}` | d = model dimension |
| State S_t | `{batch, d, m}` | d x m state matrix per batch |
| Query q_t | `{batch, seq_len, m}` | m = key/query dimension |
| Key k_t | `{batch, seq_len, m}` | |
| Beta beta_t | `{batch, seq_len, d}` | Per-dimension learning rate |
| Epsilon epsilon_t | `{batch, seq_len, d}` | Derived from beta and k |
| Output o_t | `{batch, seq_len, d}` | Same as input dimension |

### Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `d` (model dim) | 256 | Matches Mamba's hidden_size |
| `m` (key dim) | 16 | Analogous to Mamba's state_size |
| `expand_factor` | 2 | Inner dim = expand * d (Mamba convention) |
| `conv_size` | 4 | Depthwise conv kernel (Mamba convention) |
| `num_layers` | varies | Same stacking as Mamba |
| `dropout` | 0.0 | Standard |

No special initialization required (unlike Mamba's S4D-Real A-matrix init).

### Implementation Plan

**Module:** `Edifice.SSM.Longhorn`

**Strategy:** Reuse `Edifice.SSM.Common` infrastructure (block structure, input projection, depthwise conv, gating, output projection). Replace only the inner SSM scan with Longhorn's recurrence.

**Key implementation steps:**

1. **Projections:** Add W_q, W_k, W_beta linear layers (analogous to Common's B, C, dt projections)
2. **Epsilon computation:** `epsilon = sigmoid(W_beta * x) / (1 + sigmoid(W_beta * x) * Nx.sum(k * k, axes: [-1], keep_axes: true))`
3. **State update (Axon.layer):**
   - Forget: `(1 - epsilon_outer * k_sq_outer)` element-wise multiply with S_{t-1}
   - Update: `(epsilon * x)_outer * k` add to result
   - This is a linear recurrence -- can use parallel associative scan
4. **Output:** `o = Nx.dot(S, q)` -- batched matmul of state with query
5. **Parallel scan:** The recurrence `S_t = A_t * S_{t-1} + B_t` is associative. Use existing `Common` scan infrastructure with custom A_t and B_t derived from epsilon, k, x.

**Axon-specific notes:**
- The state S is `{d, m}` per batch element -- use `Axon.layer` with explicit shape tracking
- `Nx.outer` or broadcasting via `Nx.new_axis` for the d x m outer products
- beta must go through `Nx.sigmoid` then division -- watch for BinaryBackend sigmoid overflow on large negative inputs (apply sigmoid first, then divide)
- Parallel scan form: `A_t = (1 - epsilon_t (x) k_t^2)` (element-wise), `B_t = (epsilon_t * x_t) (x) k_t` -- both are `{d, m}` tensors

**Options:**

```elixir
@type build_opt ::
  {:embed_dim, pos_integer()}
  | {:hidden_size, pos_integer()}     # d
  | {:key_size, pos_integer()}        # m (default: 16)
  | {:expand_factor, pos_integer()}   # default: 2
  | {:conv_size, pos_integer()}       # default: 4
  | {:num_layers, pos_integer()}      # default: 2
  | {:dropout, float()}               # default: 0.0
```

---

## 2. Samba -- Hybrid Mamba + SWA + MLP

**Paper:** Liliang Ren, Yang Liu, Yadong Lu, Yelong Shen, Chen Liang, Weizhu Chen. "Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling." ICLR 2025. [arXiv:2406.07522](https://arxiv.org/abs/2406.07522).

**Code:** [github.com/microsoft/Samba](https://github.com/microsoft/Samba)

### Key Innovation

Layer-wise interleaving of Mamba (recurrent compression), Sliding Window Attention (precise short-range retrieval), and SwiGLU MLP (factual recall). The three components specialize: Mamba captures long-range recurrent structure, SWA retrieves high-fidelity signals from recent context, and MLP stores factual knowledge. Scales to 256K context with perfect memory recall (trained on 4K). 3.73x throughput over grouped-query attention Transformers at 128K.

### Architecture

```
Input tokens: {batch, seq_len}
      |
  Token Embedding + (no positional encoding at this level)
      |
  +==========================================+
  | Samba Block (repeated N/3 times)         |
  |                                          |
  |  x = x + Mamba(RMSNorm(x))              |  <- recurrent compression
  |  x = x + SWA(RMSNorm(x))               |  <- precise local retrieval
  |  x = x + SwiGLU(RMSNorm(x))            |  <- factual recall
  |                                          |
  +==========================================+
      |
  RMSNorm -> LM Head
      |
  {batch, seq_len, vocab_size}
```

Each block is a 3-layer unit: Mamba -> SWA -> MLP, with pre-RMSNorm and residual connections around each sublayer. The pattern repeats N/3 times to reach the total layer count.

### Key Equations

**Mamba Sublayer (Selective S6):**

```
Input expansion:    H = x * W_in          in R^{n x d_e}
Short conv:         U = SiLU(DepthwiseConv(H))  in R^{n x d_e}
Selective gates:    Delta = Softplus(U * W_r * W_q + b)   in R^{n x d_e}
                    B = U * W_b           in R^{n x d_s}
                    C = U * W_c           in R^{n x d_s}

State update:       Z_t = exp(-Delta_t * exp(A)) * Z_{t-1} + Delta_t * (B_t (x) U_t)
Output:             Y_t = Z_t * C_t + D * U_t

Gated output:       O = (Y * SiLU(x * W_g)) * W_out  in R^{n x d_m}
```

Mamba hyperparameters:
- `d_e = 2 * d_m` (expansion)
- `d_s = 16` (state dimension)
- `d_r = d_m / 16` (low-rank for Delta)
- `A` initialized as S4D-Real: `A_{ij} = log(j)` for `1 <= j <= d_s`
- `D` initialized to 1
- Conv kernel size = 4

**Sliding Window Attention (SWA) Sublayer:**

```
Q = x * W_q,  K = x * W_k,  V = x * W_v    in R^{n x d_m}

Apply RoPE to Q, K (within window positions only)

Attention: for each position i, attend only to positions max(0, i-w)..i
  A_{ij} = softmax(Q_i * K_j^T / sqrt(d_k))   for j in [i-w, i]

O = A * V
```

Window size `w = 2048`. Implemented with FlashAttention 2. Complexity: O(n * w) instead of O(n^2). RoPE relative positions are computed within the sliding window.

**SwiGLU MLP Sublayer:**

```
gate = SiLU(x * W_gate)     in R^{n x d_p}
up   = x * W_up             in R^{n x d_p}
out  = (gate * up) * W_down  in R^{n x d_m}
```

### Input/Output Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| Input tokens | `{batch, seq_len}` | Integer token IDs |
| Embeddings | `{batch, seq_len, d_m}` | After token embedding |
| Mamba state Z | `{batch, d_e, d_s}` | Per-layer recurrent state |
| SWA Q, K, V | `{batch, num_heads, seq_len, head_dim}` | Multi-head |
| MLP intermediate | `{batch, seq_len, d_p}` | SwiGLU expansion |
| Output logits | `{batch, seq_len, vocab_size}` | LM output |

### Hyperparameters

| Parameter | 421M | 1.3B | 1.7B | 3.8B |
|-----------|------|------|------|------|
| Total layers (N) | 24 | 36 | 48 | 64 |
| Samba blocks (N/3) | 8 | 12 | 16 | ~21 |
| Model width (d_m) | 1536 | 2304 | 2048 | 2816 |
| MLP intermediate (d_p) | 4096 | 6144 | 8196 | 9984 |
| Query heads | 12 | 18 | 32 | 11 |
| KV heads | 12 | 18 | 4 | 1 |
| SWA window (w) | 2048 | 2048 | 2048 | 2048 |
| SSM state dim (d_s) | 16 | 16 | 16 | 16 |
| Conv kernel | 4 | 4 | 4 | 4 |
| Training length | 4096 | 4096 | 4096 | 4096 |
| Norm | RMSNorm | RMSNorm | RMSNorm | RMSNorm |

Design insight: optimal training sequence length = 2x window size (4096 = 2 * 2048).

### Implementation Plan

**Module:** `Edifice.SSM.Samba`

**Strategy:** Use `Edifice.SSM.HybridBuilder` pattern but with a fixed 3-layer repeating block (Mamba -> SWA -> MLP). Alternatively, build as a standalone module since the pattern is simple and fixed.

**Key implementation steps:**

1. **Block structure:** Build a `samba_block/2` function that chains:
   - `x + Mamba.build_mamba_block(RMSNorm(x), mamba_opts)`
   - `x + swa_layer(RMSNorm(x), attn_opts)` (using `Edifice.Attention.MultiHead` with `:window_size`)
   - `x + SwiGLU(RMSNorm(x))` (using `Edifice.Blocks.FFN`)
2. **SWA layer:** Reuse `Edifice.Attention.MultiHead` with RoPE and a window_size option. The existing `RNoPE_SWA` module or `MultiHead` with `:rope` option handles this. Use `:window_size` to limit attention span.
3. **Stack blocks:** Repeat the 3-sublayer block `num_blocks` times.
4. **Final norm + output projection:** RMSNorm -> Dense (for LM head, or just return hidden states for Edifice's backbone convention).

**Axon-specific notes:**
- The SWA causal mask with window can be built as a static mask tensor via `Edifice.Blocks.CausalMask` with window clipping
- RoPE positions within the window: use `Edifice.Blocks.RoPE` with position indices mod window_size
- GQA (grouped-query attention) for the 1.7B and 3.8B configs: `MultiHead` already supports `:num_kv_heads`

**Options:**

```elixir
@type build_opt ::
  {:embed_dim, pos_integer()}           # input dim (required)
  | {:hidden_size, pos_integer()}       # d_m (default: 256)
  | {:num_blocks, pos_integer()}        # number of Mamba+SWA+MLP blocks (default: 4)
  | {:num_heads, pos_integer()}         # SWA query heads (default: 8)
  | {:num_kv_heads, pos_integer()}      # SWA KV heads for GQA (default: num_heads)
  | {:window_size, pos_integer()}       # SWA window (default: 2048)
  | {:mlp_dim, pos_integer()}           # SwiGLU intermediate (default: 4 * hidden_size)
  | {:state_size, pos_integer()}        # Mamba SSM state dim (default: 16)
  | {:expand_factor, pos_integer()}     # Mamba expansion (default: 2)
  | {:conv_size, pos_integer()}         # Mamba conv kernel (default: 4)
  | {:dropout, float()}                 # default: 0.0
```

---

## 3. Huginn -- Depth-Recurrent Transformer with Adaptive Iteration

**Paper:** Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, Tom Goldstein. "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach." 2025. [arXiv:2502.05171](https://arxiv.org/abs/2502.05171).

**Code:** [github.com/seal-rg/recurrent-pretraining](https://github.com/seal-rg/recurrent-pretraining)
**Model:** [huggingface.co/tomg-group-umd/huginn-0125](https://huggingface.co/tomg-group-umd/huginn-0125)

### Key Innovation

Replaces deep stacking of unique transformer layers with a small set of weight-tied recurrent blocks that iterate r times. A 3.5B parameter model with (2, 4, 2) layer config and r=32 iterations simulates a 132-effective-layer model. Reasoning happens in latent space (not chain-of-thought tokens), requiring no specialized training data. Test-time compute scales by adjusting r per token, with KL-divergence early stopping.

### Architecture

```
Input tokens: {batch, seq_len}
      |
  Token Embedding
      |
  +============================+
  | PRELUDE (l_p layers)       |  <- 2 unique transformer layers
  |   Standard transformer     |
  |   blocks (not weight-tied) |
  +============================+
      |
  e = prelude output: {batch, seq_len, h}
      |
  s_0 ~ N(0, sigma^2 * I)       <- random initial latent state
      |
  +============================+
  | CORE (l_r layers x r iter) |  <- 4 layers, weight-tied, iterated r times
  |                             |
  | for i = 1 to r:            |
  |   input = Adapter([s_{i-1}; e])  <- concatenate state + embedding
  |   s_i = CoreBlock(input)    |    <- 4 transformer layers (shared weights)
  |   s_i = n_o(s_i)           |    <- RMSNorm rescaling (critical)
  +============================+
      |
  s_r: {batch, seq_len, h}      <- final latent state
      |
  +============================+
  | CODA (l_c layers)          |  <- 2 unique transformer layers
  |   Standard transformer     |
  |   blocks (not weight-tied) |
  +============================+
      |
  RMSNorm -> LM Head
      |
  {batch, seq_len, vocab_size}
```

### Key Equations

**Recurrence:**

```
e = P(x)                                  [Prelude: embed input]
s_0 ~ N(0, sigma^2 * I_{n x h})           [Random init, n=seq_len, h=hidden]
s_i = n_o(R(e, s_{i-1}))   for i in {1,...,r}   [Core iteration with RMSNorm]
p = C(s_r)                                [Coda: decode to output]
```

**Adapter (core block entry):**

```
A: R^{2h} -> R^h

input_i = A([s_{i-1} ; e])     [concatenate along hidden dim, then project]
```

At scale, concatenation + projection outperforms simple addition for combining the latent state with the embedding.

**Transformer layers within each block:**

Sandwich normalization (4 norms per layer):

```
x_hat_l = n_2(x_{l-1} + Attn(n_1(x_{l-1})))
x_l     = n_4(x_hat_l + MLP(n_3(x_hat_l)))
```

Attention: standard causal self-attention with RoPE (base 50000), learnable biases on Q and K only.

MLP: Gated SiLU (SwiGLU): `SiLU(x * W_gate) * (x * W_up)` then `W_down`.

**Critical RMSNorm rescaling (n_o):**

After each core iteration, the output is rescaled by RMSNorm. This prevents magnitude drift across iterations and is required for stable training at scale.

**Training: stochastic iteration count:**

```
tau ~ N(log(r_bar) - 0.5 * sigma^2, sigma)    [sigma = 0.5]
r ~ Poisson(exp(tau)) + 1
```

Mean recurrence `r_bar = 32`. The log-normal Poisson distribution samples mostly below r_bar but has a heavy right tail for occasional deep iterations. This trains the model to operate under variable compute budgets.

**Backprop truncation:** Gradient is truncated at k=8 iterations (not full r=32 unroll). This limits memory while maintaining training signal quality.

**Inference: adaptive early stopping:**

```
if KL(p_{i} || p_{i-1}) < threshold:
    stop iterating, emit token

threshold = 5e-4 (default)
```

KL-divergence between successive output distributions. Zero-shot -- no specialized training for early stopping. Different tokens naturally converge at different speeds.

### Input/Output Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| Input tokens | `{batch, seq_len}` | Integer token IDs |
| Embeddings e | `{batch, seq_len, h}` | After prelude |
| Latent state s_i | `{batch, seq_len, h}` | Updated each iteration |
| Adapter input | `{batch, seq_len, 2h}` | Concatenation of s and e |
| Adapter output | `{batch, seq_len, h}` | Projected back to h |
| Attention Q, K, V | `{batch, num_heads, seq_len, head_dim}` | Per layer in core |
| Output logits | `{batch, seq_len, vocab_size}` | After coda + LM head |

### Hyperparameters

| Parameter | Huginn-3.5B | Edifice Default |
|-----------|-------------|-----------------|
| Hidden dimension (h) | 5280 | 256 |
| Attention heads | 55 | 8 |
| Head dimension | 96 | 64 |
| MLP inner dimension | 17920 | 4 * hidden_size |
| Prelude layers (l_p) | 2 | 2 |
| Core layers (l_r) | 4 | 4 |
| Coda layers (l_c) | 2 | 2 |
| Mean recurrences (r_bar) | 32 | 8 |
| Backprop truncation (k) | 8 | 4 |
| KL early-stop threshold | 5e-4 | 5e-4 |
| RMSNorm epsilon | 1e-6 | 1e-6 |
| RoPE base | 50000 | 50000 |
| Vocab size | 65536 | (configurable) |
| Context length | 4096 | (configurable) |
| Norm type | Sandwich RMSNorm | Sandwich RMSNorm |

Parameter breakdown: ~1.5B recurrent (core), ~1.5B non-recurrent (prelude+coda), ~0.5B embedding.

### Implementation Plan

**Module:** `Edifice.Recurrent.Huginn`

**Family:** `:recurrent` (alongside TransformerLike, TTT, DeltaNet, etc.)

**Strategy:** Three-phase architecture with `Axon.layer` for the iterative core. The core block is a standard transformer block (reuse `Edifice.Blocks.TransformerBlock`) with weight-tied iteration.

**Key implementation steps:**

1. **Prelude:** Stack `l_p` standard transformer layers (unique weights). Use `TransformerBlock.layer/3` or build directly with `MultiHead` attention + SwiGLU FFN.

2. **Adapter:** `Axon.concatenate([s, e], axis: -1)` -> `Axon.dense(h)`. This maps `{batch, seq_len, 2h}` -> `{batch, seq_len, h}`.

3. **Core block:** Build a single block of `l_r` transformer layers. Then iterate it `r` times using `Enum.reduce/3` at graph construction time.
   - **Key challenge:** Weight tying in Axon. Since Axon builds a static computation graph, "iterating the same block r times" means unrolling the graph r times but sharing the same layer names/parameters. Axon's name-based parameter sharing handles this: if each core layer has the same name across iterations, the parameters are automatically shared.
   - Use `Axon.layer` for the iteration wrapper with explicit parameter sharing via consistent naming.

4. **Sandwich norms:** Wrap each attention and MLP sublayer with 4 RMSNorm instances (pre-attn, post-attn, pre-mlp, post-mlp) instead of the usual 2.

5. **RMSNorm rescaling (n_o):** Apply RMSNorm after each core iteration. Use the same named norm for weight sharing across iterations.

6. **Random initial state s_0:** At training time, `s_0 ~ N(0, sigma^2 * I)`. In Axon, generate via `Axon.layer` that calls `Nx.Random.normal`. For deterministic inference, use zeros.

7. **Coda:** Stack `l_c` standard transformer layers (unique weights).

8. **Adaptive iteration (inference only):** Not part of the Axon graph (which has fixed topology). Implement as a separate inference utility function that calls `predict_fn` in a loop with early stopping based on KL divergence.

**Axon-specific notes:**
- The main challenge is weight tying across depth iterations. Axon shares parameters by name, so naming core layers identically across iterations achieves this automatically.
- Fixed `r` for the Axon graph (e.g., r=8 default). Adaptive r is an inference-time wrapper, not in the graph.
- Stochastic r during training: sample r, then build/run a graph unrolled to that depth. In practice, build a graph with `max_r` and use `Axon.layer` to conditionally mask later iterations.
- Backprop truncation: Nx's `Nx.Defn.stop_grad/1` can detach the state at truncation boundaries.
- Concatenation adapter: straightforward with `Axon.concatenate` + `Axon.dense`.

**Options:**

```elixir
@type build_opt ::
  {:embed_dim, pos_integer()}           # input dim (required)
  | {:hidden_size, pos_integer()}       # h (default: 256)
  | {:num_heads, pos_integer()}         # attention heads (default: 8)
  | {:head_dim, pos_integer()}          # per-head dim (default: 64)
  | {:mlp_dim, pos_integer()}           # SwiGLU inner dim (default: 4 * hidden_size)
  | {:prelude_layers, pos_integer()}    # l_p (default: 2)
  | {:core_layers, pos_integer()}       # l_r per iteration (default: 4)
  | {:coda_layers, pos_integer()}       # l_c (default: 2)
  | {:num_iterations, pos_integer()}    # r (default: 8, unrolled in graph)
  | {:rope_base, pos_integer()}         # RoPE frequency base (default: 50000)
  | {:dropout, float()}                 # default: 0.0
```

**Build output:** `Axon.t()` returning `{batch, seq_len, hidden_size}` (or last timestep per Edifice convention).

---

## 4. Mixture-of-Mamba (MoM) -- Modality-Aware Sparse Mamba

**Paper:** Weixin Liang, Junhong Shen, Genghan Zhang, Ning Dong, Luke Zettlemoyer, Lili Yu. "Mixture-of-Mamba: Enhancing Multi-Modal State-Space Models with Modality-Aware Sparsity." ICLR 2025. [arXiv:2501.16295](https://arxiv.org/abs/2501.16295).

**Code:** [github.com/Weixin-Liang/Mixture-of-Mamba](https://github.com/Weixin-Liang/Mixture-of-Mamba)

### Key Innovation

Introduces modality-specific parameterization within Mamba blocks for multimodal sequences. Each input/output projection (in_proj, x_proj, dt_proj, out_proj) has separate weights per modality (text, image, speech), while Conv1D and state transition A remain shared. A modality mask routes each token to the correct projection set. Achieves equivalent performance at 35-65% of baseline FLOPs across Transfusion, Chameleon, and 3-modality settings.

### Architecture

```
Input: mixed-modality sequence {batch, seq_len, d_model}
  + Modality mask M: {batch, seq_len} (integer: 0=text, 1=image, 2=speech, ...)
      |
  +================================================+
  | MoM Block (repeated N times)                   |
  |                                                |
  | For each token at position t:                  |
  |   m = M[t]  (modality index)                   |
  |                                                |
  |   +-- Modality-Specific Projections --+        |
  |   |                                   |        |
  |   | in_proj:  H = x * W_in^{(m)}     |        |
  |   | x_proj:   B,C = U * W_x^{(m)}    |        |
  |   | dt_proj:  Delta = U * W_dt^{(m)}  |        |
  |   | out_proj: O = Y * W_out^{(m)}     |        |
  |   +-----------------------------------+        |
  |                                                |
  |   +-- Shared Components --+                    |
  |   |                       |                    |
  |   | DepthwiseConv1D       |  <- local patterns |
  |   | SSM state transitions |  <- A matrix       |
  |   | SSM recurrence        |  <- shared scan    |
  |   +-----------------------+                    |
  |                                                |
  +================================================+
      |
  Output: {batch, seq_len, d_model}
```

Detailed Mamba block with MoM modifications:

```
Input x: {batch, seq_len, d_model}

1. Modality-specific input projection:
   [H; Z] = M(x, W_in, b_in; m)    in R^{2 * d_inner}
   (H is SSM path, Z is gate path)

2. Shared depthwise conv + activation:
   U = SiLU(DepthwiseConv1d(H))     in R^{d_inner}

3. Modality-specific SSM projections:
   B = M(U, W_x_B; m)              in R^{d_state}
   C = M(U, W_x_C; m)              in R^{d_state}
   Delta = Softplus(M(U, W_dt; m))  in R^{d_inner}

4. Shared SSM recurrence (same A for all modalities):
   A_bar = exp(-Delta * exp(A))
   B_bar = Delta * B
   h_t = A_bar * h_{t-1} + B_bar * u_t
   Y_t = C * h_t + D * u_t

5. Gating:
   Y = Y * SiLU(Z)

6. Modality-specific output projection:
   O = M(Y, W_out, b_out; m)       in R^{d_model}
```

### Key Equations

**Modality-specific linear transformation:**

```
M(X, W, b; m) = X * W^{(m)} + b^{(m)}
```

Where `W^{(m)}` and `b^{(m)}` are the weight and bias for modality m. Each modality has its own copy of the projection weights. During forward pass, the modality mask selects which weights to apply to each token.

**Standard Mamba SSM (shared across modalities):**

```
Discretization:
  A_bar_t = exp(-Delta_t * exp(A))    in R^{d_inner x d_state}
  B_bar_t = Delta_t * B_t            in R^{d_inner x d_state}

Recurrence:
  h_t = A_bar_t * h_{t-1} + B_bar_t * u_t    in R^{d_inner x d_state}

Output:
  y_t = C_t^T * h_t + D * u_t                in R^{d_inner}
```

**Why shared A and Conv1D:**
- Conv1D operates across time on feature channels -- it mixes features from adjacent timesteps regardless of modality boundaries
- A governs the state transition dynamics -- shared because the temporal evolution structure is modality-agnostic (how fast to forget vs. retain is a sequence property, not modality property)
- Both operate on aggregated features where per-token modality is not well-defined

**FLOP accounting:**

Modality-specific projections: each token only activates 1 of K projection sets, so per-token FLOP cost is unchanged. The total parameter count increases by K (number of modalities), but active parameters per token stay constant. FLOP savings come from better optimization -- the model converges faster (to the same loss) because each projection specializes.

### Input/Output Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| Input x | `{batch, seq_len, d_model}` | Mixed-modality tokens |
| Modality mask M | `{batch, seq_len}` | Integer modality index per token |
| W_in^{(m)} | `{d_model, 2 * d_inner}` | One per modality |
| W_x^{(m)} | `{d_inner, d_state * 2}` | B and C projections, one per modality |
| W_dt^{(m)} | `{d_inner, d_inner}` | (or low-rank) one per modality |
| W_out^{(m)} | `{d_inner, d_model}` | One per modality |
| SSM state h | `{batch, d_inner, d_state}` | Shared across modalities |
| Conv1D kernel | `{d_inner, 1, d_conv}` | Shared |
| A | `{d_inner, d_state}` | Shared |

### Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `d_model` | 768 | Model dimension |
| `d_state` | 16 | SSM state expansion |
| `d_conv` | 4 | Local convolution width |
| `expand` | 2 | Block expansion factor (`d_inner = expand * d_model`) |
| `num_modalities` | 2 | Number of modality-specific projection sets |
| `num_layers` | varies | Number of MoM blocks |
| `split_in_proj` | true | Modality-specific input projection |
| `split_x_proj` | true | Modality-specific B/C projection |
| `split_dt_proj` | true | Modality-specific Delta projection |
| `split_out_proj` | true | Modality-specific output projection |

Parameter count: ~`3 * expand * d_model^2` per layer (standard Mamba), multiplied by `num_modalities` for split projections.

**Ablation results (which splits matter most):**
- All 4 splits together: best performance (synergistic effect)
- Removing any single split degrades performance
- Joint decoupling yields greater gains than individual modifications
- in_proj and out_proj splits contribute the most individually

### Implementation Plan

**Module:** `Edifice.SSM.MixtureOfMamba` (or `Edifice.Meta.MixtureOfMamba`)

**Family:** Could be `:ssm` or `:meta` -- the modality routing is a meta-pattern applied to Mamba. Recommend `:ssm` since the core is a Mamba variant.

**Strategy:** Extend `Edifice.SSM.Common` block structure. Add a second input (`"modality_mask"`) and replace the four projection layers with modality-dispatched versions.

**Key implementation steps:**

1. **Dual inputs:** `Axon.input("sequence", shape: {nil, nil, d_model})` and `Axon.input("modality_mask", shape: {nil, nil})`.

2. **Modality-specific projection layer:** Build a custom `Axon.layer` that:
   - Takes input tensor and modality mask
   - Contains K weight matrices (one per modality) as parameters
   - For each token, selects the weight matrix based on `modality_mask[t]`
   - Implementation: gather/scatter pattern or `Nx.take` to index into stacked weights

   Efficient implementation:
   ```
   # Stack all modality weights: {num_modalities, in_dim, out_dim}
   W_stacked = parameter("W_stacked", shape: {K, in_dim, out_dim})

   # Gather weights per token: {batch, seq_len} -> index into {K, in, out}
   W_per_token = Nx.take(W_stacked, modality_mask, axis: 0)
   # W_per_token: {batch, seq_len, in_dim, out_dim}

   # Batched matmul: {batch, seq, 1, in} @ {batch, seq, in, out} -> {batch, seq, 1, out}
   output = Nx.dot(Nx.new_axis(x, -2), [-1], [0, 1], W_per_token, [-2], [0, 1])
   output = Nx.squeeze(output, axes: [-2])
   ```

3. **Shared components:** Reuse `Common.build_depthwise_conv1d/4` and `Common.build_ssm_projections/2` (for A matrix), and the parallel scan from `Mamba.build_selective_ssm_parallel/2`.

4. **Block assembly:** Wire the modality-specific projections into the standard Mamba block flow, keeping Conv1D and A shared.

5. **Configuration flags:** Support `split_in_proj`, `split_x_proj`, `split_dt_proj`, `split_out_proj` booleans for ablation. When false, use a single shared projection (standard Mamba behavior).

**Axon-specific notes:**
- The modality-dispatch pattern requires `Nx.take` on parameters indexed by the mask -- this is compatible with Axon's static graph since the indexing is data-dependent but the parameter shapes are fixed.
- Batched matmul with per-token weights: use `Nx.dot` with explicit batch axes `[0, 1]` (batch and seq_len).
- The modality mask must be integer type for `Nx.take` indexing.
- For 2 modalities, an alternative is `Nx.select(mask, W_text_result, W_image_result)` which avoids gather overhead.

**Options:**

```elixir
@type build_opt ::
  {:embed_dim, pos_integer()}           # input dim (required)
  | {:hidden_size, pos_integer()}       # d_model (default: 256)
  | {:state_size, pos_integer()}        # d_state (default: 16)
  | {:expand_factor, pos_integer()}     # default: 2
  | {:conv_size, pos_integer()}         # default: 4
  | {:num_layers, pos_integer()}        # default: 2
  | {:num_modalities, pos_integer()}    # K (default: 2)
  | {:split_in_proj, boolean()}         # default: true
  | {:split_x_proj, boolean()}          # default: true
  | {:split_dt_proj, boolean()}         # default: true
  | {:split_out_proj, boolean()}        # default: true
  | {:dropout, float()}                 # default: 0.0
```

**Build output:** Returns `Axon.t()` from dual inputs `"sequence"` and `"modality_mask"`.

---

## Cross-Architecture Comparison

| | Longhorn | Samba | Huginn | MoM |
|---|---------|-------|--------|-----|
| **Core idea** | SSM from online learning | Hybrid SSM+Attn+MLP | Depth-recurrent transformer | Modality-aware SSM |
| **Family** | SSM | SSM (Hybrid) | Recurrent | SSM (Meta) |
| **Complexity** | O(L) | O(L*w) | O(L^2 * r) | O(L) |
| **New inputs** | None | None | None | Modality mask |
| **Builds on** | Mamba block | Mamba + MultiHead + FFN | TransformerBlock | Mamba block |
| **Key Axon challenge** | Custom scan (like Mamba) | Layer interleaving | Weight-tied iteration | Per-token weight dispatch |
| **Reuses from Edifice** | SSM.Common (everything except scan) | SSM.Mamba, Attention.MultiHead, Blocks.FFN, Blocks.RoPE | Blocks.TransformerBlock, Attention.MultiHead, Blocks.FFN | SSM.Common (conv, A, scan) |
| **Estimated LOC** | ~150 (scan replacement) | ~200 (composition) | ~250 (3-phase + adapter) | ~250 (modality dispatch) |
| **Test pattern** | Shape + finite check | Shape + finite check | Shape + finite check | Shape + multi-modality mask check |

## Implementation Priority

1. **Longhorn** -- Lowest risk. Drop-in replacement for Mamba's scan. Reuses almost all of SSM.Common. Good test of the scan abstraction.
2. **Samba** -- Medium complexity. Composes existing modules (Mamba blocks, MultiHead attention, FFN). Good test of the HybridBuilder pattern. May use HybridBuilder or be standalone.
3. **MoM** -- Medium complexity. Extends Mamba with modality routing. The `Nx.take`-based weight dispatch is the main novel pattern. Tests multi-input model support.
4. **Huginn** -- Highest complexity. Weight-tied iteration, adapter mechanism, and sandwich norms are all new patterns. But it's architecturally important as the first depth-recurrent model in Edifice.

## References

- Bo Liu et al. "Longhorn: State Space Models are Amortized Online Learners." ICLR 2025. [arXiv:2407.14207](https://arxiv.org/abs/2407.14207)
- Liliang Ren et al. "Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling." ICLR 2025. [arXiv:2406.07522](https://arxiv.org/abs/2406.07522)
- Jonas Geiping et al. "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach." 2025. [arXiv:2502.05171](https://arxiv.org/abs/2502.05171)
- Weixin Liang et al. "Mixture-of-Mamba: Enhancing Multi-Modal State-Space Models with Modality-Aware Sparsity." ICLR 2025. [arXiv:2501.16295](https://arxiv.org/abs/2501.16295)
