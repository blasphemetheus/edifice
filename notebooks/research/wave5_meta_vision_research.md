# Wave 5 Architecture Research -- Meta, Adaptation, Vision, and Operators

> Research findings for Wave 5 implementations.
> Compiled 2026-02-28.

---

## Status Overview

| Architecture | Family | Target Module | Notes |
|-------------|--------|---------------|-------|
| VeRA | Meta (Adaptation) | `Edifice.Meta.VeRA` | 10x fewer params than LoRA |
| Kron-LoRA | Meta (Adaptation) | `Edifice.Meta.KronLoRA` | 4x fewer params than LoRA |
| Mixture of Transformers | Meta (Multi-Modal) | `Edifice.Meta.MixtureOfTransformers` | Modality-specific sparse transformer |
| Vision KAN | Vision | `Edifice.Vision.VisionKAN` | Attention-free hierarchical backbone |
| Temporal Neural Operator | Operator | `Edifice.Operator.TNO` | Time-dependent PDE solver |

---

## 1. VeRA -- Vector-based Random Matrix Adaptation

**Paper:** Kopiczko, Blankevoort, Asano. "VeRA: Vector-based Random Matrix Adaptation." ICLR 2024. [arXiv:2310.11454](https://arxiv.org/abs/2310.11454).

**Core idea:** Share a single pair of frozen random low-rank matrices across ALL adapted layers. Train only two small scaling vectors per layer. 10x fewer trainable parameters than LoRA with comparable performance.

### Architecture Overview

```
Input x: {B, d_in}
     |
     +----> W_0 * x  (frozen pretrained)          {B, d_out}
     |            |
     +----> Lambda_b * B * Lambda_d * A * x        {B, d_out}
     |      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     |      Trainable    Frozen   Trainable  Frozen
     |      {d_out}     {d_out,r} {r}       {r,d_in}
     |            |
     |            v
     +------> Sum ---> Output: {B, d_out}
```

Key distinction from LoRA:
- **LoRA:** Per-layer trainable A and B matrices. Params = L * r * (d_in + d_out).
- **VeRA:** Frozen shared A, B (generated from PRNG seed). Per-layer trainable scaling vectors lambda_d and lambda_b. Params = L * (d_out + r) + seed.

### Key Equations

**LoRA recap:**
```
h = W_0 * x + (alpha / r) * B * A * x
```
where A: {r, d_in}, B: {d_out, r} are trainable per layer.

**VeRA forward pass:**
```
h = W_0 * x + Lambda_b * B * Lambda_d * A * x
```

Where:
- `A`: {r, d_in} -- frozen random matrix, shared across all layers
- `B`: {d_out, r} -- frozen random matrix, shared across all layers
- `Lambda_d`: diag(d) where d: {r} -- trainable per layer
- `Lambda_b`: diag(b) where b: {d_out} -- trainable per layer

**Step-by-step forward pass:**
```
1. z = A * x             {r}        -- project down via frozen A
2. z = Lambda_d * z      {r}        -- scale by trainable d vector
3. z = B * z             {d_out}    -- project up via frozen B
4. z = Lambda_b * z      {d_out}    -- scale by trainable b vector
5. h = W_0 * x + z      {d_out}    -- add to frozen output
```

**Trainable parameter count per model:**
```
|Theta| = L_tuned * (d_model + r)
```
where L_tuned is the number of adapted layers.

**Initialization:**
- A, B: Kaiming uniform initialization from shared PRNG seed
- lambda_d: initialized to small value (default 0.1) -- controls initial magnitude
- lambda_b: initialized to small value (default 0.1)
- Shared matrices are sliced per layer to handle different d_in/d_out shapes: the global A has shape {r, max(d_in)}, global B has shape {max(d_out), r}, and submatrices are extracted per layer

### Input/Output Shapes

| Tensor | Shape | Trainable |
|--------|-------|-----------|
| Input x | {batch, d_in} | -- |
| W_0 | {d_out, d_in} | Frozen |
| A (shared) | {r, max_d_in} | Frozen |
| B (shared) | {max_d_out, r} | Frozen |
| lambda_d | {r} per layer | Yes |
| lambda_b | {d_out} per layer | Yes |
| Output h | {batch, d_out} | -- |

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r` (rank) | 256 | VeRA rank dimension (use higher than LoRA since params are much cheaper) |
| `d_initial` | 0.1 | Initial value for lambda_d vector |
| `b_initial` | 0.1 | Initial value for lambda_b vector |
| `projection_prng_key` | 0 | PRNG seed for generating frozen A, B |
| `dropout` | 0.0 | Dropout on VeRA path |
| `save_projection` | true | Whether to save A, B in checkpoint (vs regenerate from seed) |

### Implementation Notes for Axon

**Module:** `Edifice.Meta.VeRA`

**API pattern:** Match existing `Edifice.Meta.LoRA` with `build/1` and `wrap/3`.

**Key considerations:**

1. **Shared frozen matrices:** In Axon's static graph, the shared A and B matrices are best represented as constant parameters initialized once. Use `Axon.param("vera_A", {r, max_d_in}, initializer: kaiming_from_seed)` as a frozen (non-trainable) parameter. Axon does not natively support freezing individual parameters, so the frozen matrices must be injected via `Axon.constant/2` or via a custom layer that uses `Nx.Random.key/1` to regenerate them deterministically.

2. **Diagonal scaling:** `Lambda_d * z` is element-wise multiplication: `Nx.multiply(d, z)` where d is a trainable {r} vector. No need for actual diagonal matrix construction. Similarly `Lambda_b * z` is `Nx.multiply(b, z)` with b: {d_out}.

3. **Matrix slicing for variable shapes:** When adapting layers with different d_in/d_out, slice from the shared matrices: `Nx.slice(vera_A, [0, 0], [r, d_in])`. This is a compile-time shape operation in Axon.

4. **PRNG reproducibility:** Use `Nx.Random.key(projection_prng_key)` to generate deterministic A, B. Store the seed rather than the full matrices for checkpoint efficiency.

5. **Parameter comparison:**
   - LoRA rank-8 on GPT-2 (12 layers, d=768): 12 * 8 * (768 + 768) = 147,456 params
   - VeRA rank-256 on GPT-2: 12 * (768 + 256) = 12,288 params (12x fewer)

---

## 2. Kron-LoRA -- Hybrid Kronecker-LoRA Adapters

**Paper:** Shen. "Kron-LoRA: Hybrid Kronecker-LoRA Adapters for Scalable, Sustainable Fine-tuning." arXiv preprint, 2025. [arXiv:2508.01961](https://arxiv.org/abs/2508.01961).

**Core idea:** Factorize weight updates as a Kronecker product of two smaller matrices, then further compress one factor via LoRA decomposition. Leverages `rank(A kron B) = rank(A) * rank(B)` to achieve high effective rank with 4x fewer parameters than standard LoRA.

### Architecture Overview

```
Input x: {B, d_in}
     |
     +----> W_0 * x  (frozen pretrained)          {B, d_out}
     |            |
     +----> delta_W * x                            {B, d_out}
     |      where delta_W = A_kron kron (B_1 * B_2)
     |
     |      A_kron: {d_A2, d_A1}  -- small Kronecker factor
     |      B_1:    {d_B2, r}     -- LoRA down-project
     |      B_2:    {r, d_B1}     -- LoRA up-project
     |            |
     |            v
     +------> (alpha/r) * Sum ---> Output: {B, d_out}

Dimension constraints:
  d_out = d_A2 * d_B2    (Kronecker row dimension)
  d_in  = d_A1 * d_B1    (Kronecker column dimension)
```

### Key Equations

**Kronecker factorization of weight update:**
```
delta_W = A_kron kron B
```
where A_kron: {d_A2, d_A1}, B: {d_B2, d_B1}, and the full delta_W: {d_A2*d_B2, d_A1*d_B1} = {d_out, d_in}.

**LoRA compression of the B factor:**
```
B ~= B_1 * B_2,    B_1: {d_B2, r},  B_2: {r, d_B1}
```

**Combined update:**
```
delta_W = A_kron kron (B_1 * B_2)
```

**Efficient forward pass (avoids materializing full delta_W):**
```
1. Reshape x: {B, d_in} -> {B, d_A1, d_B1}
2. Y_1 = B_2 * x_reshaped    {B, d_A1, r}      -- LoRA down-project
3. Y_2 = Y_1 * A_kron^T      {B, d_A2, r}      -- Kronecker mixing
4. Y_3 = B_1 * Y_2           {B, d_A2, d_B2}   -- LoRA up-project
5. Reshape Y_3: {B, d_out}
6. h = W_0 * x + (alpha/r) * Y_3
```

Three matrix operations (two matmul + one linear) vs LoRA's two linear. 5-8% speed overhead.

**Kronecker rank identity:**
```
rank(A_kron kron (B_1 * B_2)) = rank(A_kron) * rank(B_1 * B_2) = rank(A_kron) * min(r, d_B1, d_B2)
```
With rank-8 LoRA on B and full-rank A_kron (rank = min(d_A1, d_A2)), the effective rank can be 8 * 2 = 16 for typical configurations.

**Parameter count:**
```
|Theta| = |A_kron| + |B_1| + |B_2|
        = d_A1 * d_A2 + r * (d_B2 + d_B1)
```

For d_in = d_out = 4096, d_A1 = 2, d_A2 = 20 (d_B1 = 2048, d_B2 = ~205), r = 8:
```
Kron-LoRA: 2*20 + 8*(205 + 2048) = 40 + 18,024 = 18,064
LoRA r=8: 8*(4096+4096) = 65,536   --> 3.6x fewer
```

### Input/Output Shapes

| Tensor | Shape | Trainable |
|--------|-------|-----------|
| Input x | {batch, d_in} | -- |
| W_0 | {d_out, d_in} | Frozen |
| A_kron | {d_A2, d_A1} | Yes |
| B_1 | {d_B2, r} | Yes |
| B_2 | {r, d_B1} | Yes |
| Output h | {batch, d_out} | -- |

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rank` (r) | 8 | LoRA rank for B factor compression |
| `alpha` | 32 | Scaling factor |
| `dropout` | 0.1 | Dropout rate on adapter path |
| `d_A1` | 2 | Kronecker factor A column dimension |
| `block_size` | ~200 | Target d_B2 = d_out / d_A2; d_A2 chosen to achieve this |
| `lr` | 3e-4 | Learning rate |

### Implementation Notes for Axon

**Module:** `Edifice.Meta.KronLoRA`

**API pattern:** Match `Edifice.Meta.LoRA` with `build/1` and `wrap/3`.

**Key considerations:**

1. **Reshape for Kronecker structure:** The key implementation detail is reshaping the input to separate the Kronecker block dimensions. `Nx.reshape(x, {batch, d_A1, d_B1})` then contract along the correct axes.

2. **Three-step matmul:** The forward pass uses three contractions:
   - `Nx.dot(b2, [1], [0], x_reshaped, [2], [0])` -- contract d_B1, batch on dim 0
   - `Nx.dot(y1, [1], [0], Nx.transpose(a_kron), [0], [0])` -- contract d_A1, batch on dim 0
   - `Nx.dot(b1, [1], [0], y2, [2], [0])` -- contract r, batch on dim 0

3. **Dimension auto-computation:** Given d_in, d_out, and d_A1, compute: d_B1 = d_in / d_A1, d_A2 = d_out / block_size (rounded), d_B2 = d_out / d_A2. Validate that d_in and d_out are evenly divisible.

4. **Initialization:** A_kron with Kaiming uniform, B_2 with Kaiming uniform, B_1 initialized to zeros (like LoRA's B matrix) so delta_W starts at zero.

5. **Vocabulary projection special case:** For vocab projection layers, use d_A1 = 1 and d_A2 = half the standard value. Can expose this as a `:vocab_projection` boolean option.

---

## 3. Mixture of Transformers (MoT) -- Modality-Sparse Multi-Modal Architecture

**Paper:** Liang, Yu, Luo, Iyer, Dong, Zhou, Ghosh, Lewis, Yih, Zettlemoyer, Lin. "Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models." TMLR 2025. [arXiv:2411.04996](https://arxiv.org/abs/2411.04996). Code: [github.com/facebookresearch/Mixture-of-Transformers](https://github.com/facebookresearch/Mixture-of-Transformers).

**Core idea:** Decouple ALL non-embedding transformer parameters by modality. Each modality (text, image, speech) gets its own FFN, attention projections (Q/K/V/O), and layer normalization. Only the core attention computation (softmax(QK^T/sqrt(d))V) is shared across modalities via global self-attention over the full interleaved sequence. Matches dense baseline quality at 55.8% FLOPs (text+image) or 37.2% FLOPs (text+image+speech).

### Architecture Overview

```
Multi-modal input sequence: [text_1, img_1, img_2, text_2, speech_1, ...]
                             |
Modality masks: M_text, M_img, M_speech  (binary, mutually exclusive)
                             |
     +=======================+========================+
     |                       |                        |
   Text tokens            Image tokens            Speech tokens
     |                       |                        |
  LN_text(x)             LN_img(x)               LN_speech(x)
     |                       |                        |
  Q_text = W_q^text * x   Q_img = W_q^img * x    Q_speech = W_q^speech * x
  K_text = W_k^text * x   K_img = W_k^img * x    K_speech = W_k^speech * x
  V_text = W_v^text * x   V_img = W_v^img * x    V_speech = W_v^speech * x
     |                       |                        |
     +===========Scatter by mask into full Q, K, V====+
                             |
              Global Self-Attention (SHARED)
              Attn = softmax(Q * K^T / sqrt(d)) * V
                             |
     +===========Gather by mask from full output======+
     |                       |                        |
  O_text = W_o^text * a   O_img = W_o^img * a    O_speech = W_o^speech * a
     |                       |                        |
  + residual              + residual               + residual
     |                       |                        |
  LN2_text(x)            LN2_img(x)              LN2_speech(x)
     |                       |                        |
  FFN_text(x)             FFN_img(x)              FFN_speech(x)
     |                       |                        |
  + residual              + residual               + residual
     |                       |                        |
     +===========Scatter back into sequence============+
                             |
                    Output: {B, S, d_model}
```

### Key Equations

**Modality-specific attention projections:**
```
For modality m in {text, img, speech}:
  Q_m = W_q^m * LN^m(x_m)      W_q^m: {d_model, d_model}
  K_m = W_k^m * LN^m(x_m)      W_k^m: {d_model, d_model}
  V_m = W_v^m * LN^m(x_m)      W_v^m: {d_model, d_model}
```

**Global self-attention (shared computation):**
```
Q = scatter(Q_text, Q_img, Q_speech, masks)    {B, S, d_model}
K = scatter(K_text, K_img, K_speech, masks)    {B, S, d_model}
V = scatter(V_text, V_img, V_speech, masks)    {B, S, d_model}

A = softmax(Q * K^T / sqrt(d_head)) * V       {B, S, d_model}
```

**Modality-specific output + FFN:**
```
For modality m:
  a_m = gather(A, mask_m)              {B, S_m, d_model}
  o_m = W_o^m * a_m                    {B, S_m, d_model}
  x_m = x_m + o_m                      (residual)
  x_m = x_m + FFN_m(LN2^m(x_m))       (residual + modality FFN)
```

**Modality FFN (standard SwiGLU or GELU):**
```
FFN_m(x) = W_2^m * act(W_1^m * x)     W_1^m: {d_model, d_ffn}, W_2^m: {d_ffn, d_model}
```

**FLOPs savings:** FFN accounts for ~67% of transformer params. With M modalities, each token only activates 1/M of the FFN + attention projection parameters, while the shared attention (33% of params) processes all tokens together.

### Input/Output Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| Input tokens | {batch, seq_len} | Interleaved multi-modal sequence |
| Modality masks | {num_modalities, batch, seq_len} | Binary, mutually exclusive |
| Per-modality QKV | {batch, S_m, d_model} per modality | S_m = tokens of modality m |
| Global QKV | {batch, seq_len, d_model} | After scatter |
| Attention output | {batch, seq_len, d_model} | Shared computation |
| Per-modality FFN output | {batch, S_m, d_model} per modality | After gather |
| Final output | {batch, seq_len, d_model} | After scatter back |

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 4096 | Model hidden dimension |
| `num_heads` | 32 | Attention heads (shared across modalities) |
| `num_layers` | 32 | Transformer layers |
| `d_ffn` | 11008 | FFN intermediate dimension (per modality) |
| `num_modalities` | 2 | Number of modalities (text+image, or text+image+speech) |
| `modality_names` | [:text, :image] | Named modalities for parameter routing |
| `activation` | :silu | FFN activation (SwiGLU in Chameleon) |
| `vocab_sizes` | per modality | Token vocabulary size per modality |

### Implementation Notes for Axon

**Module:** `Edifice.Meta.MixtureOfTransformers`

**Key considerations:**

1. **Modality mask routing:** The scatter/gather pattern is the central challenge. In Axon's static graph, use `Axon.layer` with a custom function that takes the full sequence + modality masks and routes tokens. Implementation: `Nx.indexed_put` and `Nx.take` with mask-derived indices. Pre-compute per-modality indices from the binary masks.

2. **Per-modality parameter sets:** Create separate `Axon.dense` layers for each modality's Q, K, V, O, FFN_1, FFN_2, and `Axon.layer_norm` for each modality's pre/post-attention norms. Name them with modality prefix: `"text_wq"`, `"img_wq"`, etc.

3. **Global attention is standard:** After scattering per-modality QKV into the full sequence, the attention computation itself is identical to standard multi-head attention. Can reuse `Edifice.Attention.MultiHead` or `Edifice.Blocks.TransformerBlock` internals.

4. **MoT vs MoE:** This is NOT mixture-of-experts. Routing is deterministic (based on known modality of each token), not learned. No load balancing loss needed. More similar to conditional computation.

5. **Masking implementation options:**
   - **Dense path (simpler):** Run all modality-specific projections on all tokens, then zero out wrong-modality outputs. Wastes FLOPs but simplifies graph.
   - **Sparse path (efficient):** Gather modality-specific tokens, run smaller projections, scatter back. Saves FLOPs but requires dynamic indexing.
   - Recommendation: Start with dense path for correctness, add sparse optimization as a `:sparse` option.

6. **Layer structure:** Each MoT layer wraps a `TransformerBlock` with the attention/FFN replaced by modality-routed versions. Build as a composable layer function: `MoT.layer(input, masks, opts)`.

---

## 4. Vision KAN -- Hierarchical RBFKAN Vision Backbone

**Paper:** "Vision KAN: Towards an Attention-Free Backbone for Vision with Kolmogorov-Arnold Networks." arXiv preprint, January 2026. [arXiv:2601.21541](https://arxiv.org/abs/2601.21541).

**Core idea:** Replace attention entirely in a hierarchical vision backbone with Kolmogorov-Arnold Networks using Radial Basis Function (RBF) activations. Each stage uses MultiPatch-RBFKAN blocks combining: (1) patch-wise RBF nonlinear modeling, (2) axis-wise separable depthwise convolution for local mixing, and (3) low-rank global path for long-range dependencies. Attention-free, linear complexity in sequence length.

### Architecture Overview

```
Input image: {B, 3, H, W}
       |
  Patch Embed (4x4 conv, stride 4) -> {B, C_1, H/4, W/4}
       |
  +-----------+
  | Stage 1   |  N_1 ViK blocks, channels C_1
  +-----------+
       |
  Downsample (2x2 conv, stride 2) -> {B, C_2, H/8, W/8}
       |
  +-----------+
  | Stage 2   |  N_2 ViK blocks, channels C_2
  +-----------+
       |
  Downsample -> {B, C_3, H/16, W/16}
       |
  +-----------+
  | Stage 3   |  N_3 ViK blocks, channels C_3
  +-----------+
       |
  Downsample -> {B, C_4, H/32, W/32}
       |
  +-----------+
  | Stage 4   |  N_4 ViK blocks, channels C_4
  +-----------+
       |
  Global Average Pool -> {B, C_4}
       |
  Classification Head -> {B, num_classes}


Each ViK Block:
  +-----------------------------------------+
  | Input x: {B, C, H_s, W_s}              |
  |     |                                   |
  | LayerNorm                               |
  |     |                                   |
  | MultiPatch-RBFKAN                       |
  |   +-- Patch Grouping (P x P patches)   |
  |   +-- RBFKAN per patch                  |
  |   +-- Axis-wise Depthwise Conv          |
  |   +-- Low-rank Global Path              |
  |     |                                   |
  | + residual                              |
  |     |                                   |
  | LayerNorm                               |
  |     |                                   |
  | FFN (Conv 1x1 -> GELU -> Conv 1x1)     |
  |     |                                   |
  | + residual                              |
  +-----------------------------------------+
```

### Key Equations

**RBFKAN activation (per edge in KAN):**
```
phi(x) = w_b * silu(x) + w_s * sum_{i=1}^{G} w_i * exp(-||x - c_i||^2 / (2 * sigma_i^2))
```
Where:
- `c_i`: {G} learnable RBF center positions
- `sigma_i`: {G} learnable RBF widths
- `w_i`: {G} learnable RBF weights
- `w_b`: base weight (linear component)
- `w_s`: spline weight (RBF component)
- `G`: number of RBF grid points (default: 5)

**KAN layer (replacing MLP layer):**
```
KAN: R^{d_in} -> R^{d_out}
y_j = sum_{i=1}^{d_in} phi_{i,j}(x_i)    for j = 1..d_out
```
Each edge (i,j) has its own learnable activation function phi_{i,j}.

**MultiPatch-RBFKAN module:**

1. **Patch grouping:** Reshape {B, C, H, W} -> {B * n_patches, C, P, P} where P is patch size.

2. **Per-patch RBFKAN:**
```
z = RBFKAN(x_patch)    per {C, P, P} patch independently
```

3. **Axis-wise separable depthwise convolution:**
```
z_h = DWConv_horizontal(z, kernel=(1, K))    -- horizontal mixing
z_v = DWConv_vertical(z, kernel=(K, 1))      -- vertical mixing
z_local = z_h + z_v                          -- combine
```
Direction-sensitive: captures horizontal and vertical structure separately.

4. **Low-rank global path:**
```
z_global = W_up * GELU(W_down * GlobalAvgPool(x))    -- bottleneck MLP
z_global = broadcast(z_global, spatial_dims)          -- expand to spatial
```
W_down: {C, C/reduction}, W_up: {C/reduction, C}. Provides cross-patch interaction.

5. **Combined output:**
```
y = z_rbfkan + z_local + z_global
```

**Ablation results:** Removing axis-wise mixing drops accuracy from 76.5% to 74.6%. Removing global path drops to 73.9%. Both components are essential.

### Input/Output Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| Input image | {batch, 3, H, W} | Default H=W=224 |
| After patch embed | {batch, C_1, H/4, W/4} | Stage 1 input |
| After stage 1 | {batch, C_1, H/4, W/4} | |
| After stage 2 | {batch, C_2, H/8, W/8} | |
| After stage 3 | {batch, C_3, H/16, W/16} | |
| After stage 4 | {batch, C_4, H/32, W/32} | |
| Classification output | {batch, num_classes} | After GAP + linear head |

### Hyperparameters

| Parameter | Default (Small) | Description |
|-----------|----------------|-------------|
| `image_size` | 224 | Input resolution |
| `patch_size` | 4 | Initial patch embedding stride |
| `channels` | [64, 128, 256, 512] | Per-stage channel dimensions |
| `depths` | [2, 2, 6, 2] | ViK blocks per stage |
| `kan_patch_size` | 7 | Patch size P for grouping in MultiPatch-RBFKAN |
| `num_rbf_centers` (G) | 5 | RBF grid points per activation |
| `dw_kernel_size` | 7 | Depthwise conv kernel size K |
| `global_reduction` | 4 | Reduction ratio for low-rank global path |
| `num_classes` | 1000 | Classification head output |
| `drop_path` | 0.1 | Stochastic depth rate |

### Implementation Notes for Axon

**Module:** `Edifice.Vision.VisionKAN`

**Key considerations:**

1. **RBFKAN as custom Axon layer:** The core RBF activation is a custom `Axon.layer` with learnable centers, widths, and weights. Each "edge" in the KAN has G learnable parameters for centers + G for widths + G for weights + 2 for base/spline weights = 3G + 2 params.

2. **Patch grouping:** `Nx.reshape` the spatial feature map into patches. For {B, C, H, W} with patch size P: reshape to {B * (H/P) * (W/P), C, P, P}. After RBFKAN processing, reshape back.

3. **Axis-wise depthwise conv:** Use `Axon.depthwise_conv` twice: once with kernel {1, K} (horizontal) and once with {K, 1} (vertical). Sum the outputs.

4. **Low-rank global path:** `Axon.global_avg_pool` -> `Axon.dense(C/r)` -> `:gelu` -> `Axon.dense(C)` -> broadcast to spatial dims via `Nx.broadcast`.

5. **Hierarchical stages:** Build as a sequence of `stage/3` functions, each containing N ViK blocks + a downsample layer (strided conv). Similar structure to `Edifice.Vision.ConvNeXt` if it exists.

6. **Parameter count concern:** Full KAN layers have d_in * d_out * (3G + 2) parameters per layer. Patch grouping is essential -- without it, a KAN layer on 3136 spatial positions would be prohibitive. The patch grouping reduces this to (P*P) * C * (3G + 2) per block.

7. **Channels-last vs channels-first:** Nx/Axon convention is channels-last {B, H, W, C}. The paper uses channels-first. Adapt all conv operations accordingly.

---

## 5. Temporal Neural Operator (TNO) -- Temporal Branch for DeepONet

**Paper:** Diab, Al-Kobaisi. "Temporal Neural Operator for Modeling Time-Dependent Physical Phenomena." Nature Scientific Reports, 2025. [arXiv:2504.20249](https://arxiv.org/abs/2504.20249).

**Core idea:** Extend the DeepONet branch-trunk operator learning framework with a dedicated temporal branch that processes solution history. The three branches (spatial/branch, temporal, trunk) combine via Hadamard product, enabling long-range temporal extrapolation, resolution invariance, and robustness to error accumulation for time-dependent PDEs.

### Architecture Overview

```
Inputs:
  v(t, .)        -- input function (forcing, boundary conditions)   {H, W}
  U_hist(t)      -- solution history (L past time steps)            {L, H, W}
  (x, t)         -- spatio-temporal query coordinates               {H, W, 2+1}

                    +-----------+
  v(t, .) -------->| Branch    |
                   | Network   |----> U_b: {p, H, W}
                   |  (U-Net)  |
                   +-----------+
                        |
                        |  (Hadamard product)
                        v
                    +--------+
                    | Merge  |----> G(U_b * U_tb * t_i) ----> Output: {K, H, W}
                    +--------+      |
                        ^           |
                        |       +-------+
                        |       | MLP G |  (decoder: R^p -> R^K)
  U_hist(t) ---------> |       +-------+
                   +-----------+
                   | Temporal  |
                   | Branch    |----> U_tb: {p, H, W}
                   |  (U-Net)  |
                   +-----------+
                        ^
                        |
  (x, t) ------------->|
                   +-----------+
                   |  Trunk    |
                   | Network   |----> t_i: {p, H, W}
                   |  (MLP)    |
                   +-----------+


Branch U-Net:                    Temporal-Branch U-Net:
  Input: v(t,.) {H, W}            Input: U_hist {L, H, W}
       |                                |
  P_b: linear encoding {p}        P_tb: linear encoding {p}
       |                                |
  AdaptiveAvgPool2d {p,S,S}       AdaptiveAvgPool2d {p,S,S}
       |                                |
  U-Net (enc-bottleneck-dec)       U-Net (enc-bottleneck-dec)
       |                                |
  Upsample to {p,H,W}             Upsample to {p,H,W}
       |                                |
  U_b: {p, H, W}                  U_tb: {p, H, W}


Trunk MLP:
  Input: (x, t) {m+1}    (m=2 for 2D problems)
       |
  FC layers with tanh activation
       |
  t_i: {p, H, W}         (broadcast/reshape to spatial grid)
```

### Key Equations

**TNO output prediction (L past states -> K future states):**
```
G_theta^{L->K}(U_hist(t))(x) = G( U_b(x,t) * U_tb(x,t) * t_i(x,t) )
```
Where `*` denotes element-wise (Hadamard) product and G: R^p -> R^K is an MLP decoder.

**Branch network (processes input function v):**
```
h_b = P_b(v(t_i, .))                  -- linear encoding, h_b in R^p
q_b = AdaptiveAvgPool2d(h_b)          -- {p, S, S}
U_b = Upsample(UNet_b(q_b))           -- {p, H, W}
```

**Temporal-branch network (processes solution history):**
```
U_hist(t) = {u(t - (l-1)*dt, .)}_{l=1}^{L}    -- L past snapshots
h_tb = P_tb(U_hist(t))                          -- linear encoding, h_tb in R^p
q_tb = AdaptiveAvgPool2d(h_tb)                   -- {p, S, S}
U_tb = Upsample(UNet_tb(q_tb))                   -- {p, H, W}
```

**Trunk network (processes spatio-temporal coordinates):**
```
t_i(x, t) = MLP_trunk(x, t)           -- {p, H, W} via feedforward
```
Activation: tanh (trunk), SiLU (branch/t-branch U-Nets).

**Temporal bundling (predict K future steps in one forward pass):**
```
U_fut(t) = {u(t + k*dt, .)}_{k=1}^{K}
```
The decoder G maps the p-dimensional fused representation to K outputs simultaneously.

**Autoregressive inference (Markov assumption, L=1):**
```
Step n: u(t + (n+1)*dt) = TNO(u(t + n*dt), v, (x, t + n*dt))
```
For L > 1, condition on L most recent predictions:
```
U_fut(t) = G_theta^{L->K}(t, U_hist(t))
```

**Training strategies:**
1. **Teacher forcing:** Use ground truth as input during initial training (60 epochs typical)
2. **Fine-tuning:** Switch to model's own predictions as input (40 epochs typical)
3. **Temporal bundling:** Predict K steps simultaneously to improve temporal learning
4. **Markov (L=1):** Simplest -- condition only on current state

### Input/Output Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| Input function v | {batch, H, W} | Forcing/boundary conditions |
| Solution history | {batch, L, H, W} | L past time steps |
| Query coordinates | {batch, H, W, m+1} | Spatial (m dims) + temporal |
| Branch output | {batch, p, H, W} | Latent dim p |
| T-branch output | {batch, p, H, W} | Latent dim p |
| Trunk output | {batch, p, H, W} | Latent dim p |
| Fused (Hadamard) | {batch, p, H, W} | Element-wise product |
| Prediction | {batch, K, H, W} | K future time steps |

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` (p) | 20 | Latent representation dimension |
| `history_steps` (L) | 1 | Number of past states to condition on |
| `bundle_size` (K) | 4 | Number of future steps to predict |
| `pooling_resolution` (S) | 8 | Adaptive pooling target resolution |
| `spatial_dim` (m) | 2 | Spatial dimensionality (1D, 2D, 3D) |
| `unet_channels` | [32, 64, 128] | U-Net encoder channel progression |
| `trunk_layers` | [64, 64, 64] | Trunk MLP layer sizes |
| `trunk_activation` | :tanh | Trunk network activation |
| `branch_activation` | :silu | Branch/T-branch U-Net activation |
| `lr` | 1e-3 | Learning rate (Adam) |
| `weight_decay` | 1e-3 | L2 regularization |
| `teacher_forcing_epochs` | 60 | Epochs with ground truth inputs |
| `finetune_epochs` | 40 | Epochs with model's own predictions |

### Implementation Notes for Axon

**Module:** `Edifice.Operator.TNO`

This would be the first "Operator" family in Edifice (neural operators for PDE solving). Could alternatively go in a new family or under `Edifice.Scientific.TNO`.

**Key considerations:**

1. **Three-branch architecture:** Build branch, temporal-branch, and trunk as three separate Axon subgraphs. Combine via `Axon.layer` that performs element-wise multiplication (Hadamard product) + MLP decoder.

2. **Multi-input model:** Use `Axon.input("input_function", shape: {nil, h, w})`, `Axon.input("solution_history", shape: {nil, l, h, w})`, `Axon.input("coordinates", shape: {nil, h, w, m_plus_1})`. Three named inputs, similar to `Edifice.Detection.DETR` pattern.

3. **U-Net submodule:** The branch and temporal-branch both use small U-Nets. Implement as a shared helper `unet/2` with encoder (strided convs + batch norm + activation) -> bottleneck (1x1 conv) -> decoder (transposed convs + skip connections). Use Xavier initialization.

4. **Adaptive pooling:** `Nx.window_reduce` or a custom layer that pools to fixed {S, S} regardless of input resolution. This is key for resolution invariance -- the model trains at one resolution and infers at another.

5. **Container output:** Return `Axon.container(%{prediction: ..., branch_features: ..., temporal_features: ...})` to allow inspection of intermediate representations.

6. **Training loop note:** The teacher-forcing -> fine-tuning transition is a training schedule concern, not an architecture concern. The Axon model itself just takes inputs and produces outputs. Document the two-phase training strategy in the moduledoc.

7. **Autoregressive rollout:** For inference beyond K steps, the caller loops: feed prediction back as new history. This is outside the model graph -- document as a utility function `rollout/4`.

---

## Cross-Cutting Implementation Notes

### Family Organization

| Architecture | Family | Rationale |
|-------------|--------|-----------|
| VeRA | `:meta` | Parameter-efficient adaptation, alongside LoRA/DoRA |
| Kron-LoRA | `:meta` | Parameter-efficient adaptation, alongside LoRA/DoRA |
| MoT | `:meta` | Multi-modal architecture pattern, alongside MoE/MoD |
| Vision KAN | `:vision` | Hierarchical vision backbone, alongside ConvNeXt/PoolFormer |
| TNO | `:operator` (new) | Neural operator for PDEs -- new family |

### Shared Patterns

1. **VeRA and Kron-LoRA** both extend the existing `LoRA.build/1` + `LoRA.wrap/3` API pattern. They should expose the same interface for easy swapping: `VeRA.wrap(input, original, rank: 256)` and `KronLoRA.wrap(input, original, rank: 8)`.

2. **VeRA's frozen shared matrices** require a new Axon pattern: parameters that are initialized but never updated. Options:
   - Use `Axon.constant` with pre-generated tensors
   - Use `Axon.param` with a custom initializer + mark as frozen in training loop
   - Pre-generate and pass as additional model input (simplest for Axon's static graph)

3. **MoT's modality routing** is similar to the existing `Edifice.Meta.MoE` expert routing but deterministic. Could share the scatter/gather utilities.

4. **Vision KAN's RBFKAN** is a new primitive that could be useful beyond this architecture. Consider implementing `Edifice.Blocks.RBFKAN` as a reusable block, similar to how `Edifice.Blocks.RoPE` serves multiple attention modules.

### Priority Recommendation

1. **VeRA** (high priority) -- Direct complement to existing LoRA/DoRA. Minimal new patterns needed. Immediately useful for ExPhil model adaptation.
2. **Kron-LoRA** (high priority) -- Same rationale as VeRA. Novel factorization pattern.
3. **Vision KAN** (medium priority) -- Introduces KAN primitives to Edifice. RBFKAN block is reusable.
4. **MoT** (medium priority) -- Multi-modal capability. Requires modality routing infrastructure.
5. **TNO** (lower priority) -- New family (operators). Interesting for scientific computing but less directly relevant to ExPhil/Melee AI.

### Test Patterns

All architectures should follow the standard Edifice test pattern:
```elixir
test "builds and runs forward pass" do
  model = Module.build(opts)
  {init_fn, predict_fn} = Axon.build(model)
  params = init_fn.(input_template, %{})
  output = predict_fn.(params, input)
  assert Nx.shape(output) == expected_shape
  assert Nx.all(Nx.is_finite(output)) |> Nx.to_number() == 1
end
```

For VeRA and Kron-LoRA, also test:
- `wrap/3` produces correct output shape
- Parameter count is significantly less than equivalent LoRA
- Output matches expected dimensionality

For MoT, test with synthetic modality masks ensuring correct routing.

For Vision KAN, test each stage independently + full backbone.

For TNO, test with synthetic PDE data (e.g., heat equation on unit square).
