# Wave 3 Architecture Research — New Families & Gap Fills

> Research findings for 2026 Wave 3 implementations.
> Compiled 2026-02-26.

---

## Status Overview

| Architecture | Family | Status | Notes |
|-------------|--------|--------|-------|
| DETR | Detection | Done | Set-based object detection |
| RT-DETR | Detection | Done | Real-time detection |
| SAM 2 | Detection | Done | Promptable segmentation |
| Sigmoid Attention | Attention | Done | Drop-in softmax replacement |
| Whisper | Audio | Done | Encoder-decoder ASR |
| Decision Transformer | RL | Done | Offline RL as sequence modeling |
| Mercury/MDLM | Generative | TODO | Discrete diffusion LM |
| Rectified Flow | Generative | TODO | Straight-trajectory flow matching |
| DINOv3 | Vision | TODO | 7B self-supervised backbone |
| EAGLE-3 | Meta | TODO | Multi-level speculative draft head |
| ReMoE | Meta | TODO | Differentiable MoE routing |
| mHC | Meta | TODO | Manifold hyper-connections |
| DimeNet | Graph | TODO | Directional message passing |
| SE(3)-Transformer | Graph | TODO | Equivariant attention |

---

## Completed Architectures (Summary)

### DETR — DEtection TRansformer
**Paper:** Carion et al. (2020). "End-to-End Object Detection with Transformers."
- CNN backbone -> transformer encoder-decoder -> learned object queries
- Set-based Hungarian matching loss (no NMS/anchors)
- Returns `%{class_logits: ..., bbox_pred: ...}` via `Axon.container`

### RT-DETR — Real-Time DETR
**Paper:** Zhao et al. (Baidu, 2024). Hybrid CNN+transformer encoder, anchor-free.
- 53-55% AP at 108 FPS
- Returns `%{class_logits: ..., bbox_pred: ...}` via `Axon.container`

### SAM 2 — Segment Anything Model 2
**Paper:** Meta AI (2024). Promptable segmentation for images + video.
- Image encoder + prompt encoder + mask decoder + memory attention
- Returns `%{masks: ..., iou_scores: ...}` via `Axon.container`

### Sigmoid Self-Attention
**Paper:** ICLR 2025. Properly normalized sigmoid replacing softmax.
- FlashSigmoid: 17% kernel speedup over FlashAttention2 on H100
- Eliminates token competition (each position attends independently)
- Distinct from Gated Attention's post-SDPA sigmoid gate

### Whisper
**Paper:** Radford et al. (OpenAI, 2023). Robust speech recognition via large-scale weak supervision.
- Log-mel spectrogram frontend + transformer encoder-decoder
- Multitask: transcription, translation, timestamps, language ID

### Decision Transformer
**Paper:** Chen et al. (2021). "Decision Transformer: Reinforcement Learning via Sequence Modeling."
- Frames offline RL as conditional sequence generation
- Interleaves (return-to-go, state, action) triples into causal GPT
- Inputs: returns `{B,K}`, states `{B,K,state_dim}`, actions `{B,K,action_dim}`, timesteps `{B,K}`
- Output: `{B,K,action_dim}` — next action predictions
- Directly relevant to ExPhil imitation learning pipeline

---

## TODO Architectures (Detailed Research)

### 1. MDLM / Mercury — Discrete Diffusion Language Model

**Paper:** Sahoo et al., "Simple and Effective Masked Diffusion Language Models," NeurIPS 2024. [arXiv:2406.07524](https://arxiv.org/abs/2406.07524). Mercury (Inception Labs, 2025): [arXiv:2506.17298](https://arxiv.org/abs/2506.17298). Related: SEDD (Lou et al., ICML 2024 Best Paper): [arXiv:2310.16834](https://arxiv.org/abs/2310.16834).

#### Key Innovation

Parallel token denoising instead of autoregressive generation. Progressively masks tokens (forward process) and learns to unmask them (reverse process). All tokens generated simultaneously through iterative refinement. 10x decoding speedup over AR models.

#### Architecture

Encoder-only transformer (BERT/DiT-style) with timestep conditioning and RoPE.

```
Input: [MASK, MASK, ..., MASK]  (fully masked sequence)
         |
   + timestep embedding t
         |
   Transformer encoder (N layers)
         |
   Logits: {batch, seq_len, vocab_size}
         |
   Sample unmasked tokens (reverse process)
         |
   Repeat T steps: t=1 -> t=0
         |
   Output: [token_1, token_2, ..., token_L]
```

#### Key Equations

Forward marginal (absorbing state diffusion):
```
q(z_t | x) = Cat(z_t; alpha_t * x + (1 - alpha_t) * m)
```
- `alpha_t`: monotonically decreasing schedule (1 -> 0)
- `m`: one-hot mask token (K-th category)

SUBS reverse parameterization:
```
p_theta(z_s | z_t) = Cat(z_s; [(1-alpha_s)*m + (alpha_s-alpha_t)*x_theta(z_t,t)] / (1-alpha_t))
```
Unmasked positions copied through unchanged.

Training loss (reduces to weighted MLM):
```
L = E_q integral_0^1 [alpha'_t / (1-alpha_t)] * sum_l log <x_theta^l(z_t,t), x^l> dt
```
Mask token logit must be `-inf` to enforce `<x_theta, m> = 0`.

#### Implementation Plan

**Module:** `Edifice.Generative.MDLM`

**Options:**
- `:vocab_size` — vocabulary size (required)
- `:seq_len` — sequence length (required)
- `:hidden_size` — transformer hidden dim (default: 256)
- `:num_layers` — transformer blocks (default: 6)
- `:num_heads` — attention heads (default: 8)
- `:noise_schedule` — `:cosine` | `:loglinear` (default: `:cosine`)
- `:num_diffusion_steps` — sampling steps (default: 100)

**Inputs:** `"tokens"` `{batch, seq_len}` + `"timestep"` `{batch}` scalar
**Output:** `{batch, seq_len, vocab_size}` logits

**Key detail:** This is an encoder-only (bidirectional) transformer, NOT causal. No causal mask. Timestep conditioning via FiLM or additive embedding.

---

### 2. Rectified Flow

**Paper:** Liu, Gong, Liu, "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow," ICLR 2023 Spotlight. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003).

#### Key Innovation

ODE transport between noise and data along straight-line trajectories. Enables accurate generation with as few as 1 Euler step. Used by Stable Diffusion 3 and Flux.

#### Architecture

Velocity network (U-Net or DiT) predicts direction from noise to data at any interpolation point.

```
Noise z_0 ~ N(0, I)
     |
     v
z_t = t * x_1 + (1-t) * z_0    (training: linear interpolation)
     |
v_theta(z_t, t) -> predicts (x_1 - z_0)
     |
z_1 = z_0 + v_theta(z_0, 0)    (1-step generation after rectification)
```

#### Key Equations

ODE:
```
dZ_t = v_theta(Z_t, t) dt,  t in [0,1],  Z_0 ~ N(0,I)
```

Training loss (conditional flow matching):
```
L = integral_0^1 E[||(X_1 - X_0) - v_theta(X_t, t)||^2] dt
```
where `X_t = t * X_1 + (1-t) * X_0`.

Rectification (reflow) — straightens trajectories:
```
Z^(k+1) = Rectflow(Z_0^k, Z_1^k)
```
Each iteration reduces straightness measure: `S(Z^k) = O(1/k)`.

One-step generation (post-rectification):
```
Z_1 = Z_0 + v_theta(Z_0, 0)
```

#### Implementation Plan

Could be a variant/option on existing `Edifice.Generative.FlowMatching` or standalone module.

**Key difference from FlowMatching:** FlowMatching already does conditional flow matching. Rectified Flow adds: (1) explicit reflow procedure to straighten trajectories, (2) few-step sampling (1-10 steps vs 50-100). The backbone network is identical — the distinction is in the training/sampling procedure.

**Recommendation:** Add `:rectified` option to existing FlowMatching, or create `Edifice.Generative.RectifiedFlow` with the reflow training utility as a separate function.

---

### 3. DINOv3

**Paper:** Meta AI, August 2025. [arXiv:2508.10104](https://arxiv.org/abs/2508.10104).

#### Key Innovation

7B parameter self-supervised vision model. Student-teacher framework with DINO (image-level) + iBOT (patch-level) + Koleo losses. **Gram anchoring** prevents dense feature degradation during long training. Axial RoPE for resolution flexibility.

#### Architecture

```
Input: {B, 3, H, W}
     |
Patch embed (16x16 patches) -> {B, (H/16)*(W/16), 4096}
     |
+ 4 register tokens -> {B, P+4, 4096}
     |
40 transformer blocks (SwiGLU FFN, axial RoPE, 32 heads)
     |
Separate LayerNorm for local/global crops
     |
CLS token: {B, 4096} (global representation)
Patch tokens: {B, P, 4096} (dense features)
```

#### Key Details

- **ViT-7B:** 40 blocks, dim 4096, 32 heads (128 dim/head), SwiGLU FFN (8192 inner)
- **Patch size 16** (vs 14 in DINOv2)
- **4 register tokens** to mitigate high-norm patch outliers
- **Axial RoPE:** coordinates normalized to [-1,1] per patch, with RoPE-box jittering
- **Multi-crop:** 2 global (256x256) + 8 local (112x112) crops

Training losses:
```
L_pre = L_DINO + L_iBOT + 0.1 * L_DKoleo
```

Gram anchoring (refinement phase):
```
L_Gram = ||X_S * X_S^T - X_G * X_G^T||_F^2
```
Frobenius norm of Gram matrix difference — preserves patch-patch similarity structure.

#### Implementation Plan

**Module:** `Edifice.Vision.DINOv3`

For Edifice, implement the backbone architecture (forward pass only, not the self-supervised training):
- ViT backbone with axial RoPE, SwiGLU FFN, register tokens
- Configurable model sizes (T/S/B/L/G/7B)
- Output: CLS token + patch features

**Key difference from DINOv2:** Axial RoPE (resolution flexible), SwiGLU FFN, register tokens, 6x larger model capacity.

---

### 4. EAGLE-3

**Paper:** Li et al., "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test," NeurIPS 2025. [arXiv:2503.01840](https://arxiv.org/abs/2503.01840).

#### Key Innovation

Multi-level feature fusion from target LLM (low/mid/high layers) into a lightweight single-decoder-layer draft head. Training-time testing simulates autoregressive draft generation conditions. 4-6x speedup, 1.4x over EAGLE-2.

#### Architecture

```
Target LLM (frozen):
  Layer L_low  -> f^low   {B, S, H}
  Layer L_mid  -> f^mid   {B, S, H}
  Layer L_high -> f^high  {B, S, H}
         |
  Concat + Project: W_fuse * [f^low ; f^mid ; f^high]  -> {B, S, H}
         |
  + Token embedding of previous sampled token
  Concat + Project: W_in * [g_t ; e(token)]  -> {B, S, H}
         |
  Single Transformer Decoder Layer
         |
  Target LLM's LM Head (shared, frozen)  -> {B, S, V}
         |
  Tree-based speculative verification
```

#### Key Equations

Multi-layer fusion:
```
g_t = W_fuse * [f_t^low ; f_t^mid ; f_t^high]     // {H, 3H} projection
```

Input construction:
```
input_t = W_in * [g_t ; e(token_{t-1})]            // {H, 2H} projection
```

#### Implementation Plan

**Module:** `Edifice.Meta.Eagle3` or `Edifice.Inference.Eagle3`

**Options:**
- `:hidden_size` — matches target model hidden dim (required)
- `:num_heads` — attention heads for draft decoder layer
- `:feature_layers` — list of 3 layer indices `[low, mid, high]`
- `:vocab_size` — vocabulary size (for shared LM head)

**Note:** This is a speculative decoding architecture — the draft head is lightweight (~2-5% of target model params). The target model's LM head and embeddings are reused, not duplicated.

---

### 5. ReMoE — ReLU-Routed Mixture of Experts

**Paper:** Wang et al. (Tsinghua), "ReMoE: Fully Differentiable Mixture-of-Experts with ReLU Routing," ICLR 2025. [arXiv:2412.14711](https://arxiv.org/abs/2412.14711).

#### Key Innovation

Replaces discrete TopK routing with `ReLU(x * W_r)`. ReLU naturally produces sparse outputs (zeros for negative inputs), so no discontinuous TopK needed. Fully differentiable. Variable number of active experts per token.

#### Architecture

```
Input x: {B, S, d}
     |
Router: R(x) = ReLU(x * W_r)   -> {B, S, E}  (non-negative routing weights)
     |
Each expert FFN_e(x) computed only if R(x)_e > 0
     |
Output: y = sum_e R(x)_e * FFN_e(x)
```

#### Key Equations

ReLU routing:
```
R(x) = ReLU(x * W_r)          // W_r: {d, E}
```

Adaptive L1 sparsity regularization:
```
L_reg = (1/(L*T)) * sum_l sum_t ||R(x_t^l)||_1
```

Adaptive coefficient update:
```
lambda_{i+1} = lambda_i * alpha^{sign((1 - k/E) - S_i)}
```
- `S_i`: measured sparsity
- `k/E`: target active fraction
- `alpha > 1`: update multiplier (default 1.2)

Load-balanced regularization:
```
L_reg_lb = (1/(L*T)) * sum_l sum_t sum_e f_{l,e} * R(x_t^l)_e
f_{l,e} = (E / (k*T)) * sum_t 1{R(x_t^l)_e > 0}
```

#### Implementation Plan

**Module:** `Edifice.Meta.ReMoE`

**Options:**
- `:input_size` — input dimension (required)
- `:num_experts` — total experts (default: 8)
- `:target_active` — target active experts per token (default: 2)
- `:hidden_size` — expert FFN hidden size (default: input_size * 4)
- `:lambda_init` — initial regularization coefficient (default: 1.0e-8)
- `:alpha` — adaptive update multiplier (default: 1.2)

**Key difference from existing MoE:** No TopK, no Softmax, no Gumbel. Just ReLU. Sparsity is learned through regularization rather than enforced by hard top-k selection.

---

### 6. mHC — Manifold Hyper-Connections

**Paper:** DeepSeek-AI, "mHC: Manifold-Constrained Hyper-Connections," December 2025. [arXiv:2512.24880](https://arxiv.org/abs/2512.24880).

#### Key Innovation

Expands residual stream width by factor n (e.g., 4x) with n parallel streams. Mixing matrix constrained to Birkhoff Polytope (doubly stochastic) to prevent training instability. Allows multi-rate information flow while maintaining stability.

#### Architecture

```
Standard residual:  x_{l+1} = x_l + F(x_l)

mHC residual (n streams):
  x_l: {B, S, n*C}  (n parallel streams of width C)
       |
  H_pre: aggregate n streams -> 1   (n*C -> C)
       |
  F(x): standard layer (attention/FFN) on C dims
       |
  H_post: distribute 1 -> n streams  (C -> n*C)
       |
  H_res: mix between streams         (n x n doubly stochastic)
       |
  x_{l+1} = H_res * x_l + H_post^T * F(H_pre * x_l)
```

#### Key Equations

Single-layer propagation:
```
x_{l+1} = H_l^res * x_l + (H_l^post)^T * F(H_l^pre * x_l, W_l)
```

Birkhoff Polytope constraint on `H_res`:
- All row sums = 1, all column sums = 1, all entries >= 0
- Spectral norm bounded by 1 (prevents gradient explosion)
- Achieved via Sinkhorn-Knopp projection (20 iterations):
```
M^(0) = exp(H_tilde_res)
M^(t) = T_r(T_c(M^(t-1)))    // alternate row/column normalization
```

#### Implementation Plan

**Module:** `Edifice.Blocks.HyperConnection` or `Edifice.Meta.HyperConnection`

**Options:**
- `:hidden_size` — per-stream width C (required)
- `:expansion` — number of streams n (default: 4)
- `:sinkhorn_iters` — Sinkhorn projection iterations (default: 20)
- `:gate_init` — initial gating factor (default: 0.01)

This is a **block/wrapper** rather than a standalone architecture — it modifies the residual connection in any transformer. Could wrap TransformerBlock.

**Key consideration:** 6.7% training overhead for n=4. The main value is improved training dynamics for large models.

---

### 7. DimeNet — Directional Message Passing

**Paper:** Gasteiger, Gross, Gunnemann, "Directional Message Passing for Molecular Graphs," ICLR 2020. [arXiv:2003.03123](https://arxiv.org/abs/2003.03123). DimeNet++: [arXiv:2011.14115](https://arxiv.org/abs/2011.14115).

#### Key Innovation

First GNN to incorporate angular information via directional message passing. Embeds edges (messages) instead of nodes. Updates messages based on angles between triplets of atoms using spherical Bessel functions and spherical harmonics.

#### Architecture

```
Atoms z: {N}          Positions pos: {N, 3}
     |                      |
     |               Compute distances d_ji, angles angle_kji
     |                      |
EmbeddingBlock: [emb(Z_i) ; emb(Z_j) ; RBF(d_ji)] -> {E, hidden}
     |
InteractionBlock x num_blocks:
  |  x_kj_proj = sigma(W * x_kj) * W_rbf * rbf(d_ji)
  |  x_ji_proj = sigma(W * x_ji)
  |  m_ji = bilinear(sbf(angle,d), x_kj_proj)    // directional messages
  |  h = x_ji_proj + aggregate(m_ji)
  |  h = ResidualLayers(h) + skip
     |
OutputBlock at each level:
  scatter_sum(W_rbf * x_ji, target_atoms) -> per-atom
     |
sum_atoms -> per-molecule prediction
```

#### Key Equations

Radial Basis Functions (Bessel):
```
e_RBF(d) = envelope(d/cutoff) * sin(freq_n * d / cutoff)
```

Spherical Basis Functions:
```
a_SBF(angle, d) = bessel_nl(d) * spherical_harmonic_l(angle)
```

Bilinear directional message:
```
m_ji = einsum('wj,wl,ijl->wi', sbf, x_kj_proj, W_bilinear)
```
`W_bilinear`: `{hidden, num_bilinear, num_spherical * num_radial}`

#### Implementation Plan

**Module:** `Edifice.Graph.DimeNet`

**Options:**
- `:hidden_channels` — edge embedding dimension (default: 128)
- `:out_channels` — output dimension (default: 1)
- `:num_blocks` — interaction blocks (default: 6)
- `:num_bilinear` — bilinear embedding size (default: 8)
- `:num_spherical` — spherical harmonics order (default: 7)
- `:num_radial` — radial basis functions (default: 6)
- `:cutoff` — distance cutoff in Angstroms (default: 5.0)
- `:envelope_exponent` — envelope decay exponent (default: 5)

**Inputs:** `"atomic_numbers"` `{N}`, `"positions"` `{N, 3}`, `"batch"` `{N}`
**Output:** `{num_molecules, out_channels}`

**Nx considerations:** Requires spherical Bessel functions and spherical harmonics — implement as pure Nx operations using recurrence relations. The bilinear einsum is straightforward with `Nx.dot` + reshape. Scatter operations need `Nx.indexed_add` or equivalent.

---

### 8. SE(3)-Transformer — Equivariant Attention

**Paper:** Fuchs et al., "SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks," NeurIPS 2020. [arXiv:2006.10503](https://arxiv.org/abs/2006.10503).

#### Key Innovation

Attention mechanism that maintains SE(3) equivariance: rotating the input rotates the output identically. Uses fiber-typed features (spherical tensors of different degrees), Clebsch-Gordan tensor products with spherical harmonics for equivariant messages, and norm-based invariant attention weights.

#### Architecture

```
Points: {N, 3}    Features: type-0 {N, 1, c0}, type-1 {N, 3, c1}, ...
     |
Build k-NN graph
     |
Compute relative positions r_ij, spherical harmonics Y^l(r_hat_ij)
Precompute Clebsch-Gordan weight basis
     |
SE(3) Attention Layer x num_layers:
  |  Q, K, V via equivariant linear (per-type projection)
  |  Attention weights: softmax(phi(||q||) . phi(||k||) / sqrt(d))  // invariant!
  |  Value messages: tensor product with spherical harmonics       // equivariant!
  |  Aggregate: sum_j alpha_ij * v_ij
     |
Output fiber features per node
```

#### Key Concepts

**Fiber features:** Each node has features organized by type-l:
- Type-0 (scalars): `{1, c_0}` — invariant under rotation
- Type-1 (vectors): `{3, c_1}` — rotate as 3D vectors
- Type-2 (tensors): `{5, c_2}` — rotate as quadrupoles
- Type-l: `{2l+1, c_l}` — transforms under Wigner-D matrix

**Norm-based attention (invariant):**
```
alpha_ij = softmax_j(phi(||q_i^0||, ..., ||q_i^L||) . phi(||k_j^0||, ..., ||k_j^L||))
```
Norms are rotation-invariant since `||D^l(g) f^l|| = ||f^l||`.

**Equivariant value messages (tensor products):**
```
msg^J_ij = sum_{paths(k,l,J)} w * R_J(||r_ij||) * CG_tensor_product(f^k_j, Y^l(r_hat_ij))
```

#### Implementation Plan

**Module:** `Edifice.Graph.SE3Transformer`

**Options:**
- `:num_layers` — transformer layers (default: 4)
- `:num_degrees` — max type l (default: 3, meaning types 0,1,2,3)
- `:num_channels` — channels per type (default: 16)
- `:num_heads` — attention heads (default: 4)
- `:cutoff` — radius for graph construction (optional, alternative to k-NN)
- `:k_neighbors` — k for k-NN graph (default: 16)

**Inputs:** `"positions"` `{N, 3}`, `"features"` `{N, 1, c_0}` (type-0 input), `"batch"` `{N}`
**Output:** fiber features per node

**Nx considerations:** This is the most mathematically complex architecture in Edifice:
- Clebsch-Gordan coefficients: precompute as constant tensors (small, tabulated)
- Spherical harmonics: implement via recurrence relations in Nx
- Wigner-D matrices: not needed at inference (only for proving equivariance)
- Tensor products: careful einsum/dot operations with CG coefficients
- Radial functions: standard MLP on scalar distances

**Complexity warning:** The tensor product operations scale as O(L^6) with max degree L. Keep L small (2-3) for practical use. EGNN (already implemented) is a simpler alternative that handles only type-0 and type-1 features.

---

## Implementation Priority

Based on ExPhil relevance and implementation complexity:

1. **MDLM** — High priority. Novel generative paradigm, relatively straightforward (encoder-only transformer + masked diffusion sampling).
2. **Rectified Flow** — Medium priority. Could extend existing FlowMatching. Minimal new code if done as variant.
3. **ReMoE** — Medium priority. Clean implementation, similar structure to existing MoE modules but simpler routing.
4. **EAGLE-3** — Medium priority. Lightweight architecture, useful for inference optimization.
5. **DimeNet** — Medium priority. Fills molecular modeling gap in graph family. Requires spherical basis functions.
6. **mHC** — Lower priority. Block-level modification, most useful for very large models.
7. **DINOv3** — Lower priority. Massive model (7B), the backbone is "just" a scaled ViT with axial RoPE.
8. **SE(3)-Transformer** — Lower priority. Most complex implementation (CG coefficients, fiber types). EGNN covers simpler equivariant use cases.
