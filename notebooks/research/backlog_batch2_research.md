# Backlog Batch 2 Research Notes

Research findings for SPLA, CausVid, MIRAS, JanusFlow, MoR, and MoED.
Compiled 2026-02-28.

---

## 1. SPLA -- Block Sparse + Linear Attention Hybrid

**Paper:** "SPLA: Block Sparse Plus Linear Attention for Long Context Modeling"
**Authors:** Bailin Wang, Dan Friedman, Tao Lei
**ArXiv:** [2601.22379](https://arxiv.org/abs/2601.22379) (January 2026)
**Code:** No public repository as of February 2026

### Architecture Overview

SPLA combines two complementary attention mechanisms for efficient long-context processing:

1. **Block-sparse exact attention** on selected high-relevance blocks
2. **Residual linear attention (RLA)** on unselected "long tail" blocks

The key insight: instead of discarding unselected blocks (which causes quality degradation at scale), compress them into a compact recurrent state via linear attention.

### Forward Pass

```
Input: Q, K, V [batch, seq_len, num_heads, dim_head]
  |
  v
Block Partitioning (stride=16, window=32)
  |
  v
Taylor Selection Metric (2nd-order)
  score = exp(q^T * k_bar) * (1 + 0.5 * q^T * Cov(k) * q)
  |
  v
+--- Selected Blocks (top-K by score) ---+--- Unselected Blocks ---+
|                                         |                         |
v                                         v                         |
Exact Softmax Attention              Residual Linear Attention      |
(full quadratic on selected)         o_RLA = o_global - o_selected  |
|                                         |                         |
+--- Combine: o_sparse + o_RLA ----------+
  |
  v
Output: [batch, seq_len, num_heads, dim_head]
```

### Key Innovation: 2nd-Order Taylor Selection

Previous methods (InfLLM-v1) use first-order approximation E[exp(q^T * k_bar)] for block scoring. SPLA uses the second-order Taylor expansion:

```
E[exp(q^T k)] ~ exp(q^T k_bar) * (1 + 0.5 * q^T * Cov(k) * q)
```

This is computed without auxiliary training using streaming block statistics (mean and covariance).

### Residual Linear Attention (RLA)

Instead of explicitly computing attention on unselected blocks, SPLA subtracts:

```
o_RLA = o_global - o_selected
```

Where `o_global` is linear attention over ALL tokens (O(n) via kernel trick / recurrent state) and `o_selected` is linear attention over only the selected blocks.

Linear attention uses feature kernel phi (e.g., elu(x)+1):
```
S_t = sum(phi(k_i)^T * v_i^T)     -- recurrent state [dim_head, d_v]
z_t = sum(phi(k_i)^T * 1)          -- normalizer [dim_head]
o_t = (S_t * q_t) / (z_t * q_t)    -- output
```

### Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| block_stride | 16 | Stride between block starts |
| block_window | 32 | Size of each block |
| selection_ratio | 0.25-0.5 | Fraction of blocks selected |
| feature_kernel | elu(x)+1 | For linear attention |
| taylor_order | 2 | Second-order (critical) |

### Performance

- Context length: tested up to 256K tokens
- RULER benchmark: 72.3% (vs 42.6% InfLLM-v2)
- Complexity: O(L * (k + D^2)) where k = selected blocks, D = head dim

### Implementation Notes for Axon

**Challenges:**
- Axon is static graph -- no dynamic top-K at runtime
- Need soft selection (softmax-weighted mask) or Gumbel-softmax instead of hard top-K
- No dynamic gather -- use mask multiplication + reshape
- Recurrent linear attention needs `Axon.fold`/scan for state accumulation

**Module structure:** `lib/edifice/attention/spla.ex`
- Vectorized block statistics (Nx.mean, Nx.variance on reshaped tensors)
- Soft block masking (differentiable)
- Sparse attention branch (full quadratic on selected)
- Linear attention branch (scan with recurrent state)
- Subtraction-based RLA
- Feature kernel: elu(x)+1 for stability

### Verdict: Implementable but Complex

The block partitioning, Taylor metric, and linear attention scan are all doable in Axon but require careful tensor reshaping and the scan/fold pattern. Estimated ~200-250 lines.

---

## 2. CausVid -- Causal Video DiT Distillation

**Paper:** "From Slow Bidirectional to Fast Autoregressive Video Diffusion Models"
**ArXiv:** [2412.07772](https://arxiv.org/abs/2412.07772) (CVPR 2025)
**Authors:** MIT and Adobe Research
**Code:** [github.com/tianweiy/CausVid](https://github.com/tianweiy/CausVid)

### Critical Finding: NOT a Novel Architecture

CausVid is a **training/distillation technique**, not a new model architecture. The causal model uses standard transformer blocks with a modified attention mask pattern.

### What CausVid Actually Does

1. Takes a pretrained **bidirectional** video DiT (e.g., Wan 2.1)
2. Adapts it to **autoregressive** generation via:
   - Block-wise causal attention masking
   - Distribution Matching Distillation (DMD) from bidir teacher to causal student
   - Student initialization via teacher's ODE trajectory pairs

### Attention Pattern (Only Architectural Change)

```
Frames: [0, 1, 2, ..., 11 | 12, 13, 14, ..., 23 | ...]
         <-- Block 0 -->    <-- Block 1 -->

For position p in block B:
  - Bidirectional attention within block B (all positions)
  - Causal attention to all previous blocks (< B)
  - No attention to future blocks (> B)
```

Block size: typically 12 frames (Wan adaptation).

### Training (Two-Stage)

1. **Student Initialization:** Pretrain causal student on ODE solution pairs from bidirectional teacher
2. **Asymmetric Distillation:** DMD loss -- bidirectional teacher supervises causal student

### Performance

- VBench-Long: 84.27 (rank 1)
- Streaming: 9.4 FPS on single GPU
- Latency: 1.3s initial, then streaming
- Steps: 50 teacher -> 4 student (12.5x reduction)

### Verdict: Skip for Edifice

CausVid's block-wise causal attention mask could be added as a utility (similar to CausalMask), but the core innovation is in training procedure. Not worth a dedicated architecture module. Could be noted as a training technique in documentation.

---

## 3. MIRAS -- Google's Associative Memory Framework

**Paper:** "It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization"
**ArXiv:** [2504.13173](https://arxiv.org/abs/2504.13173) (April 2025)
**Blog:** [Google Research: Titans + MIRAS](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)

### Critical Finding: NOT a Standalone Architecture

MIRAS is a **theoretical framework/taxonomy** that unifies sequence models (Transformers, Mamba, RetNet, DeltaNet, Titans) as instances of online optimization over associative memory.

### The Four Design Dimensions

| Dimension | Purpose | Examples |
|-----------|---------|----------|
| **Memory Architecture** | Structure for storing info | Vectors (RNN), matrices (linear RNN), MLPs (Titans) |
| **Attentional Bias** | Internal learning objective | MSE, dot-product, generalized norms, Huber loss, KL-divergence |
| **Retention Gate** | Balance new vs. past knowledge | Weight decay, momentum, softmax normalization |
| **Memory Learning Algo** | Optimization method | GD, GD with momentum |

### Three Concrete Instantiations

**Moneta** -- High-fidelity recall
- Attentional bias: Generalized norms (strict mathematical penalties)
- Retention: Weight decay
- Best for: Noisy/hard-retrieval tasks

**Yaad** -- Robust to outliers
- Attentional bias: Huber loss with input-dependent threshold
- Retention: Adaptive momentum
- Best for: Messy/inconsistent data

**Memora** -- Memory stability
- Attentional bias: KL-divergence
- Retention: Softmax normalization + weight resets
- Best for: Strict probability mapping, controlled updates

### Key Equations

Memory update rule:
```
W_t = A_t / ||A_t||_4^2
A_t = alpha_t * A_{t-1} - eta_t * grad(loss)
```
Where alpha_t is data-dependent forgetting rate, eta_t is data-dependent learning rate.

### Relationship to Titans

- **Titans** is a concrete architecture (already exists as concept in Edifice)
- **MIRAS** is the theoretical framework that explains Titans and others
- Titans = MIRAS with MLP memory + MSE bias + weight decay + GD with momentum

### Performance

- Moneta/Yaad/Memora outperform Transformers and linear RNNs at all scales >= 340M
- Scales to 2M+ tokens effectively
- Fast parallelizable training

### Verdict: Implement Moneta/Yaad/Memora as Recurrent Architectures

MIRAS itself is a framework, not implementable. But the three variants (Moneta, Yaad, Memora) are concrete architectures that could be implemented as recurrent/memory family modules. They extend the Titans memory module pattern with different loss functions and retention strategies. TODO item should be updated to reflect this.

---

## 4. JanusFlow -- AR Text + Rectified Flow Images

**Paper:** "JanusFlow: Harmonizing Autoregression and Rectified Flow"
**ArXiv:** [2411.07975](https://arxiv.org/abs/2411.07975)
**Authors:** DeepSeek AI
**Code:** [github.com/deepseek-ai/Janus](https://github.com/deepseek-ai/Janus)

### Architecture Overview

Unified multimodal model combining autoregressive language modeling (text) with rectified flow (continuous image generation) in a single LLM backbone.

Two modes:
- **Understanding:** Image + text -> autoregressive text (standard VLM)
- **Generation:** Text prompt -> rectified flow ODE solving -> VAE decode -> image

### Component Inventory

```
MultiModalityCausalLM
  |
  |-- vision_und_enc_model      (SigLIP-L)              Understanding encoder
  |-- vision_und_enc_aligner    (Linear: 1024 -> 2048)  Understanding projection
  |-- beg_of_und_embed          (Parameter: [1, 2048])  BOI embedding
  |
  |-- vision_gen_enc_model      (ShallowUViTEncoder)     Generation encoder
  |-- vision_gen_enc_aligner    (Linear: 768 -> 2048)    Gen encoder -> LLM
  |
  |-- vision_gen_dec_model      (ShallowUViTDecoder)     Generation decoder
  |-- vision_gen_dec_aligner_norm (LlamaRMSNorm: 2048)   Decoder input norm
  |-- vision_gen_dec_aligner    (Linear: 2048 -> 768)    LLM -> gen decoder
  |
  |-- language_model            (LlamaForCausalLM)       LLM backbone
```

### Exact Dimensions (JanusFlow-1.3B)

| Component | Parameter | Value |
|-----------|-----------|-------|
| LLM | hidden_size | 2048 |
| LLM | num_layers | 24 |
| LLM | num_heads | 16 |
| LLM | vocab_size | 102400 |
| Understanding Enc | model | SigLIP-Large-Patch16-384 |
| Understanding Enc | output_dim | 1024 |
| Understanding Enc | num_patches | 576 (24x24) |
| Gen Encoder | input_channels | 4 (VAE latent) |
| Gen Encoder | block_out_channels | [768] |
| Gen Encoder | stride/kernel | 2 |
| Gen Encoder | ConvNeXt blocks | 2 |
| Gen Decoder | in_channels | 768 |
| Gen Decoder | out_channels | 4 |
| Latent space | shape | [batch, 4, 48, 48] (SDXL-VAE) |
| Image | resolution | 384x384 |

### ShallowUViTEncoder

```
Input: z_t [batch, 4, 48, 48] + timestep (scalar)
  |
  v
time_proj: Sinusoidal(768) -> [batch, 768]
time_embed: Linear(768->2048) + SiLU + Linear(2048->2048) -> t_emb [batch, 2048]
  |
  v
in_conv: Conv2d(4, 768, kernel=2, stride=2) -> [batch, 768, 24, 24]
  |
  v
mid_block: 2x ConvNeXt blocks (conditioned by t_emb via FiLM)
  |
  v
Returns: (x_emb [batch, 768, 24, 24], t_emb, [skip_connections])
```

### ConvNeXt Block (with FiLM Conditioning)

```
depthwise: Conv2d(ch, ch, kernel=7, padding=3, groups=ch)
norm: RMSNorm(ch)
linear_1: Linear(ch, ch*4)    -- expansion
gelu
global_response_norm: GRN(ch*4)
linear_2: Linear(ch*4, ch)    -- projection
dropout + residual
scale, shift = cond_mapper(silu(t_emb)).chunk(2)   -- FiLM
x = x * (1 + scale) + shift
```

### GlobalResponseNorm (GRN)

```
gx = L2_norm(x, spatial_dims)           -- [batch, 1, 1, dim]
nx = gx / (mean(gx, channel_dim) + eps)
return bias + (weight * nx + 1) * x
```

### Generation Forward Pass (ODE Loop)

```
z = randn(batch, 4, 48, 48)
dt = 1.0 / num_steps  (default: 30 steps)

for step in range(num_steps):
    t = step / num_steps * 1000
    z_enc, t_emb, hs = gen_encoder(z, t)
    z_seq = reshape(z_enc, [B, 576, 768])
    z_seq = gen_enc_aligner(z_seq)          -- [B, 576, 2048]

    llm_input = cat([text_embeds, t_emb.unsqueeze(1), z_seq])
    hidden = llm(llm_input)
    img_hidden = hidden[:, -576:, :]
    img_hidden = gen_dec_aligner(gen_dec_aligner_norm(img_hidden))
    img_spatial = reshape(img_hidden, [B, 768, 24, 24])

    v = gen_decoder(img_spatial, hs, t_emb)  -- velocity [B, 4, 48, 48]

    # CFG
    v = cfg_weight * v_cond - (cfg_weight-1) * v_uncond

    # Euler step
    z = z + dt * v

image = vae.decode(z)  -- external SDXL-VAE
```

### REPA (Training Only)

Representation alignment at layer 6: small MLP projects LLM features to align with understanding encoder features via cosine similarity. No inference cost.

### Key Differences from Janus (Original)

| Aspect | Janus | JanusFlow |
|--------|-------|-----------|
| Image gen | VQ-VAE discrete tokens | Continuous rectified flow |
| Gen encoder | VQ-VAE encoder | ShallowUViTEncoder (ConvNeXt) |
| Gen decoder | VQ-VAE decoder | ShallowUViTDecoder + SDXL-VAE |
| Gen loop | Token-by-token (576 tokens) | ODE solving (30 Euler steps) |
| Loss | Cross-entropy on codes | MSE on velocity vectors |

### Verdict: Very Complex, Many New Primitives Needed

Implementation would require:
- ShallowUViTEncoder/Decoder (ConvNeXt blocks, GRN, PixelShuffle)
- FiLM conditioning throughout
- Multi-mode attention (understanding vs generation)
- ODE sampling loop integration
- External VAE and SigLIP encoder concepts

Estimated: 400+ lines for the velocity network alone. Recommend implementing as a later priority after simpler backlog items.

---

## 5. MoR -- Mixture of Recursions

**Paper:** "Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation"
**ArXiv:** [2507.10524](https://arxiv.org/abs/2507.10524) (NeurIPS 2025)
**Authors:** Bae, Kim, Bayat et al. (Google DeepMind, KAIST AI, Mila)
**Code:** [github.com/raymin0223/mixture_of_recursions](https://github.com/raymin0223/mixture_of_recursions)

**Status: IMPLEMENTED** (commit `dcf6b6c`)

### Architecture

Combines three ideas:
1. **Weight-tied (recursive) transformer** -- shared block reused N_r times
2. **Per-token routing** -- router decides recursion depth per token
3. **Middle-cycle sharing** -- unique first/last layers, shared middle

### Two Routing Variants

**Expert-Choice (preferred):**
- Per-recursion router: `dense -> sigmoid -> alpha*gate -> top-k mask`
- Hierarchical filtering: active token set shrinks monotonically
- Capacity: {N_r/N_r, (N_r-1)/N_r, ..., 1/N_r} of tokens per step

**Token-Choice:**
- Single router at step 1: `dense -> softmax over N_r depths`
- Each token commits to a total recursion depth
- Cumulative gating for intermediate steps

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| num_recursions (N_r) | 3 | Recursion steps |
| routing | :expert_choice | Routing variant |
| alpha | 0.1 | Gate scaling factor |
| hidden_size | 256 | Transformer hidden dim |
| num_heads | 4 | Attention heads |
| num_layers | 6 | Total layers |

### Implementation Notes

The Edifice implementation uses soft gating (all tokens pass through, weighted by router scores) to maintain Axon's static graph compatibility. Weight tying is achieved by using the same layer name "shared_block" across recursion steps.

---

## 6. MoED -- Mixture of Expert Depths

**Paper:** "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models" (Raposo et al., 2024)
**ArXiv:** [2404.02258](https://arxiv.org/abs/2404.02258)
**Also:** "Mixture-of-Experts-and-Depths" by Kodela (SSRN 5403647, 2025)

**Status: IMPLEMENTED** (commit `dcf6b6c`)

### Architecture: Integrated MoDE

Unifies MoE (width) and MoD (depth) routing in a single mechanism:

```
Input [batch, seq, hidden]
  |
  v
Per Layer:
  Attention sublayer (standard)
  |
  v
  Single Router -> softmax over (E experts + 1 no-op)
  |
  +--- Expert_0 (FFN) ---|
  +--- Expert_1 (FFN) ---|--- Weighted combination
  +--- ...               |
  +--- No-op (identity) -|
  |
  v
Gated residual output
```

The "no-op expert" is the key: tokens learn to choose the residual path, effectively skipping the FFN computation at that layer. This was found to be distinctly better than simply reducing expert capacity.

### Two MoDE Variants

**Integrated MoDE (implemented):**
- Single routing decision over E+1 options (E experts + no-op)
- Simpler, better performing
- Natural fit for Axon

**Staged MoDE (not implemented):**
- First MoD routing (top-k selection)
- Then MoE routing on selected tokens
- More complex, marginal gains

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| num_experts | 2 | Expert FFN count |
| hidden_size | 256 | Transformer hidden dim |
| num_heads | 4 | Attention heads |
| num_layers | 6 | Transformer layers |
| expert_hidden_multiplier | 4 | Expert FFN width |

### Implementation Notes

Uses `ModelBuilder.build_sequence_model` with custom block_builder. The `build_combine_fn` pattern handles variable expert count via case statement returning anonymous functions of different arities (since Axon.layer unpacks list elements as positional args).

---

## Summary: Implementation Priorities

| Architecture | Status | Priority | Notes |
|-------------|--------|----------|-------|
| MoR | DONE | -- | Committed dcf6b6c |
| MoED | DONE | -- | Committed dcf6b6c |
| SPLA | DONE | -- | Block-sparse + residual linear attention |
| CausVid | Skip | -- | Training technique, not architecture |
| MIRAS (Moneta/Yaad/Memora) | DONE | -- | Three memory variants implemented |
| InfLLM-V2 | DONE | -- | Dense-sparse switchable attention |
| JanusFlow | Open | Low | Very complex, many new primitives |
