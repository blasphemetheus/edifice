# Wave 5 Generative Architecture Research

> Research findings for next-generation discrete diffusion and flow-based generative
> architectures targeting Edifice implementation.
> Compiled 2026-02-28.

---

## Status Overview

| Architecture | Type | Status | Notes |
|-------------|------|--------|-------|
| LLaDA | Discrete Diffusion LM | TODO | 8B masked diffusion LLM, competitive with LLaMA3 |
| CaDDi | Discrete Diffusion LM | TODO | Non-Markovian causal discrete diffusion |
| DeepFlow | Flow-Based Image Gen | TODO | Deeply supervised flow matching, 8x faster than SiT |
| Meissonic | Masked Image Gen | TODO | VQ + masked transformer T2I, SDXL-quality at 1B |

---

## 1. LLaDA -- Large Language Diffusion with Masking

**Paper:** Nie, Zhu, You, Zhang, Ou, Hu, Zhou, Lin, Wen, Li. "Large Language Diffusion Models." ICLR 2025. [arXiv:2502.09992](https://arxiv.org/abs/2502.09992).

**Code:** [github.com/ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)

**Model weights:** [GSAI-ML/LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base), [GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)

### Key Innovation

First 8B-scale discrete diffusion language model competitive with autoregressive LLMs
(LLaMA3 8B) on standard benchmarks (MMLU, GSM8K). Uses a vanilla bidirectional
transformer as the mask predictor -- no special diffusion-specific architecture, no
time-step embeddings (proved unnecessary by prior work RADD). Trained under the
standard pre-training + SFT paradigm. Inherently avoids the "reversal curse" of AR
models since generation is not left-to-right.

### Architecture

```
Inputs: tokens [batch, seq_len], timestep t (scalar, for masking only)
      |
+--------------------------------------------------------------+
| Forward Process (training only):                              |
|   t ~ U(0, 1)                                                |
|   For each token i: x_t^i = [MASK] if rand_i < t, else x_0^i|
+--------------------------------------------------------------+
      |
+--------------------------------------------------------------+
| Token Embedding (vocab_size x hidden_size)                    |
| + RoPE Positional Encoding                                    |
+--------------------------------------------------------------+
      |
+--------------------------------------------------------------+
| Bidirectional Transformer Block x num_layers                  |
|   Pre-RMSNorm                                                 |
|   Multi-Head Self-Attention (NO causal mask, full bidi)       |
|   + Grouped Query Attention (GQA)                             |
|   + RoPE on Q, K                                              |
|   Residual                                                    |
|   Pre-RMSNorm                                                 |
|   SwiGLU FFN (gate_proj, up_proj, down_proj)                  |
|   Residual                                                    |
+--------------------------------------------------------------+
      |
| RMSNorm + Linear -> vocab_size                                |
      |
Output: logits [batch, seq_len, vocab_size]
```

**Critical difference from MDLM:** LLaDA uses NO timestep conditioning at all.
The transformer receives only the masked token sequence. Prior work (RADD) proved
that mask-based discrete diffusion models are equivalent to any-order autoregressive
models and do not require time embeddings. This makes LLaDA architecturally identical
to a bidirectional LLM (BERT-style) but trained with variable masking ratios.

### Key Equations

**Forward process** (absorbing-state masking):
```
q(x_t | x_0) = Product_i q(x_t^i | x_0^i)

q(x_t^i | x_0^i) = (1-t) * delta(x_t^i, x_0^i) + t * delta(x_t^i, [MASK])
```
Each token is independently masked with probability t, where t ~ U(0,1).

**Mask predictor:**
```
p_theta(x | x_t) = Product_{i : x_t^i = [MASK]} p_theta(x^i | x_t)
```
The model predicts clean token distributions only at masked positions. Unmasked
positions are copied through unchanged.

**Training loss** (weighted cross-entropy on masked positions):
```
L = E_{x_0 ~ data} E_{t ~ U(0,1)} [ (1/t) * sum_{i : x_t^i=[MASK]} -log p_theta(x_0^i | x_t) ]
```
The weight w(t) = 1/t provides importance weighting -- low masking ratios (few masks)
get higher per-token weight. This is an upper bound on negative log-likelihood,
making LLaDA a principled generative model.

**Reverse process** (generation):
```
For step n = 1, ..., N:
  t_n = 1 - (n-1)/N,  t_{n+1} = 1 - n/N
  logits = f_theta(x_{t_n})
  x_hat = sample from softmax(logits) at masked positions
  n_unmask = floor(seq_len * (t_n - t_{n+1}))
  Select n_unmask positions to unmask (lowest confidence or random)
  x_{t_{n+1}} = unmask selected, remask remaining with [MASK]
```

**Low-confidence remasking** (Algorithm 5 in paper):
Instead of random selection, rank predicted tokens by confidence (max softmax
probability). The lowest-confidence tokens are remasked for the next step. This
substantially improves generation quality over random remasking.

### Input/Output Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| tokens (input) | `{batch, seq_len}` | Integer token IDs, [MASK] = 126336 |
| logits (output) | `{batch, seq_len, vocab_size}` | Raw logits over vocabulary |

### Hyperparameters

LLaDA-8B uses a LLaMA-style architecture (the model type is registered as "llada"
but architecturally mirrors LLaMA with bidirectional attention):

| Parameter | LLaDA-8B Value | Edifice Default |
|-----------|---------------|-----------------|
| `hidden_size` | 4096 | 256 |
| `num_layers` | 32 | 6 |
| `num_heads` | 32 | 8 |
| `num_kv_heads` | 8 (GQA) | 8 |
| `intermediate_size` | 14336 | 1024 |
| `vocab_size` | 126337 | required |
| `seq_len` | 2048 | required |
| `mask_token_id` | 126336 (= vocab_size - 1) | vocab_size - 1 |
| `max_position_embeddings` | 2048 | 2048 |
| `rms_norm_eps` | 1e-5 | 1e-5 |
| `rope_theta` | 500000.0 | 10000.0 |
| `num_diffusion_steps` | N/A (= response length) | 64 |
| `learning_rate` | 4e-4 (pre-train) | -- |
| `training_tokens` | 2.3T | -- |

### Implementation Notes for Axon

**Module:** `Edifice.Generative.LLaDA`

1. **No timestep conditioning.** Unlike MDLM which uses AdaLN-Zero modulation from
   timestep embeddings, LLaDA's transformer receives zero information about the
   current diffusion timestep. The architecture is a pure bidirectional transformer.
   This is a major simplification.

2. **Bidirectional attention.** No causal mask. This means standard `Axon.Layers`
   multi-head attention with no mask, or full attention mask of ones. Opposite of
   decoder-only LLMs.

3. **GQA support.** The 8B model uses 32 query heads and 8 key-value heads. Our
   existing `MultiHead` attention with `:num_kv_heads` option handles this.

4. **RoPE.** Standard rotary position embeddings on Q and K. Reuse existing
   `Edifice.Blocks.RoPE`.

5. **SwiGLU FFN.** `output = down_proj(silu(gate_proj(x)) * up_proj(x))`. Already
   implemented in several Edifice modules.

6. **Mask token.** The last token in vocabulary is [MASK]. During training, replace
   tokens with mask_token_id. Logit at mask_token position should be `-inf` to
   enforce p_theta([MASK]) = 0.

7. **Sampling is external.** The `build/1` function returns the denoising network
   only. The forward masking, reverse sampling loop, and low-confidence remasking
   are inference-time procedures, not part of the static Axon graph. Provide helper
   functions or document the sampling loop.

8. **Relationship to MDLM.** LLaDA and MDLM are both discrete diffusion LMs but
   differ significantly:
   - MDLM: uses timestep-conditioned DiT blocks (AdaLN-Zero), cosine noise schedule
   - LLaDA: no timestep conditioning, linear masking, LLaMA-style architecture
   - LLaDA is simpler architecturally but scales to 8B parameters

**Options:**
- `:vocab_size` -- vocabulary size including mask token (required)
- `:seq_len` -- maximum sequence length (required)
- `:hidden_size` -- transformer hidden dimension (default: 256)
- `:num_layers` -- number of transformer blocks (default: 6)
- `:num_heads` -- number of attention heads (default: 8)
- `:num_kv_heads` -- number of KV heads for GQA (default: num_heads)
- `:intermediate_size` -- SwiGLU FFN intermediate dimension (default: hidden_size * 4)
- `:rope_theta` -- RoPE base frequency (default: 10000.0)

---

## 2. CaDDi -- Causal Discrete Diffusion

**Paper:** Zhang, He, Levine, Zhao, Zhang, Rizvi, Zhang, Zappala, Ying, van Dijk. "Non-Markovian Discrete Diffusion with Causal Language Models." NeurIPS 2025. [arXiv:2502.09767](https://arxiv.org/abs/2502.09767).

**Lab:** van Dijk Lab @ Yale.

### Key Innovation

Lifts the Markov assumption in discrete diffusion by conditioning each denoising step
on the entire generative trajectory. Traditional discrete diffusion (MDLM, LLaDA,
D3PM, SEDD) generates x_{t-1} conditioned only on x_t (Markov). CaDDi conditions
on the full history (x_T, x_{T-1}, ..., x_{t+1}, x_t), allowing the model to revisit
and refine previously generated tokens. This is achieved by flattening the 2D
(sequence position x diffusion timestep) trajectory into a 1D sequence and processing
it with a standard causal (left-to-right) transformer.

The key insight: a standard causal LM, when applied to the flattened diffusion
trajectory, naturally implements non-Markovian discrete diffusion. Autoregressive
generation is a special case (T=1 diffusion steps). This means pretrained LLM weights
can be directly reused for diffusion with no architectural changes.

### Architecture

```
Diffusion trajectory (T timesteps, each with L tokens):
  x_T    = [M, M, M, ..., M]           (fully masked)
  x_{T-1}= [w1, M, w3, ..., M]         (partially unmasked)
  ...
  x_1    = [w1, w2, w3, ..., w_L]       (nearly clean)
  x_0    = [w1, w2, w3, ..., w_L]       (clean data)

Flatten into 1D causal sequence (temporal-first or token-first):
  [x_T^1, x_T^2, ..., x_T^L, x_{T-1}^1, ..., x_{T-1}^L, ..., x_0^1, ..., x_0^L]
                |
+--------------------------------------------------------------+
| Token Embedding + Positional Encoding                         |
| (position encodes both sequence position and diffusion step) |
+--------------------------------------------------------------+
                |
+--------------------------------------------------------------+
| Causal Transformer (decoder-only, standard AR LM)             |
|   Causal attention mask (attend only to past = noisier steps) |
|   Block x num_layers                                          |
+--------------------------------------------------------------+
                |
| LM Head -> vocab_size                                         |
                |
Output: next-token predictions at each position
```

**Two variants:**

1. **CaDDi-Block** -- predicts entire blocks of L tokens at each diffusion step.
   The trajectory is organized as T blocks, each of length L. Causal attention
   is applied across blocks (can see all previous diffusion steps) but tokens
   within a block can attend to each other bidirectionally.

2. **CaDDi-AR** -- token-level prediction. Each token in the flattened sequence
   is predicted autoregressively. When T=1, this reduces exactly to a standard
   AR language model. This variant enables direct fine-tuning of pretrained LLMs.

### Key Equations

**Non-Markovian reverse process:**
```
p_theta(x_{t-1} | x_T, x_{T-1}, ..., x_t) = p_theta(x_{t-1} | x_{T:t})
```
Unlike Markov models where p(x_{t-1} | x_t), CaDDi conditions on all noisier states.

**Forward corruption kernel** (absorbing mask, same as MDLM/LLaDA):
```
q(x_t | x_0) = Product_i [(1 - beta_t) * delta(x_t^i, x_0^i) + beta_t * delta(x_t^i, [MASK])]
```
where beta_t is the masking probability at step t.

**Trajectory construction:**
For each training example x_0, sample T noisy versions:
```
x_T, x_{T-1}, ..., x_1  where  x_t ~ q(x_t | x_0)
```
These are sampled independently from the marginal (not sequentially), giving
different masking patterns at each step.

**Training loss** (next-block or next-token prediction):
```
L = -sum_{t=0}^{T-1} sum_{i=1}^{L} log p_theta(x_t^i | x_{T:t+1}, x_t^{<i})
```
For CaDDi-Block, the inner sum is a masked prediction loss (predict unmasked tokens
given the block context). For CaDDi-AR, it is standard next-token cross-entropy.

**Generation** (iterative refinement):
```
1. x_T = all [MASK]
2. For t = T-1, ..., 0:
     x_t = sample from p_theta(x_t | x_{T:t+1})
     (Model sees entire trajectory so far as context)
3. Return x_0
```

### Input/Output Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| trajectory (input) | `{batch, T * seq_len}` | Flattened diffusion trajectory |
| logits (output) | `{batch, T * seq_len, vocab_size}` | Next-token logits |

For CaDDi-Block variant with block-level attention:

| Tensor | Shape | Notes |
|--------|-------|-------|
| blocks (input) | `{batch, T, seq_len}` | T diffusion steps, each seq_len tokens |
| logits (output) | `{batch, T, seq_len, vocab_size}` | Per-block token logits |

### Hyperparameters

| Parameter | Paper Setting | Edifice Default |
|-----------|--------------|-----------------|
| `vocab_size` | dataset-dependent | required |
| `seq_len` | 128 (LM1B) | required |
| `hidden_size` | GPT-2 small (768) | 256 |
| `num_layers` | 12 | 6 |
| `num_heads` | 12 | 8 |
| `num_diffusion_steps` T | 64 | 16 |
| `context_window` | 4 time points | 4 |
| `block_size` | seq_len | seq_len |
| `variant` | `:block` or `:ar` | `:block` |
| `learning_rate` | 1e-4 | -- |

**Evaluation models:** GPT-2, Llama-2-7B, Llama-3.2-3B (used as oracles for
guided perplexity). CaDDi trained from scratch outperforms MDLM, SEDD, D3PM,
and Discrete Flow Matching baselines.

### Implementation Notes for Axon

**Module:** `Edifice.Generative.CaDDi`

1. **Standard causal transformer.** The backbone is a completely standard decoder-only
   transformer with causal attention mask. No architectural modifications needed.
   This is the same as `Edifice.Transformer.DecoderOnly` in principle.

2. **Trajectory flattening.** The key preprocessing step: take T noisy versions of
   the sequence and concatenate them into one long sequence of length T * seq_len.
   The causal mask ensures each diffusion step can only attend to earlier (noisier)
   steps. This flattening is done outside the model.

3. **Position encoding.** Need to encode both (a) position within a block and
   (b) which diffusion step. Options:
   - Separate learned embeddings for position + timestep, summed
   - Single absolute position embedding over the full T * seq_len length
   - RoPE over position, with timestep added as a learned embedding

4. **Block-level variant.** For CaDDi-Block, use a block-causal attention mask:
   within each block of seq_len tokens, attention is bidirectional; across blocks,
   attention is causal (later blocks cannot attend to earlier blocks). This is
   similar to prefix-LM masking but repeated for each block.

5. **AR variant and LLM fine-tuning.** CaDDi-AR with T=1 is exactly a causal LM.
   This means existing pretrained weights can be loaded and fine-tuned by simply
   increasing T > 1 during training. The `build/1` function should support both
   variants through the `:variant` option.

6. **Context window.** For efficiency, CaDDi uses a sliding context window of W
   time points rather than the full trajectory. Default W=4 means each step sees
   only the 4 most recent diffusion states.

7. **Relationship to MDLM and LLaDA.** All three are discrete diffusion LMs:
   - MDLM: encoder-only bidirectional, timestep-conditioned, Markov
   - LLaDA: encoder-only bidirectional, no timestep conditioning, Markov
   - CaDDi: decoder-only causal, no explicit timestep, non-Markov (trajectory)
   CaDDi is the only one using a causal (decoder-only) architecture.

**Options:**
- `:vocab_size` -- vocabulary size including mask token (required)
- `:seq_len` -- tokens per diffusion block (required)
- `:hidden_size` -- transformer hidden dimension (default: 256)
- `:num_layers` -- transformer blocks (default: 6)
- `:num_heads` -- attention heads (default: 8)
- `:num_diffusion_steps` -- T, number of diffusion steps (default: 16)
- `:context_window` -- sliding window over trajectory (default: 4)
- `:variant` -- `:block` (block-causal) or `:ar` (token-level) (default: `:block`)
- `:intermediate_size` -- FFN hidden size (default: hidden_size * 4)

---

## 3. DeepFlow -- Deeply Supervised Flow Matching

**Paper:** Shin, Yang, Chen (ByteDance Seed). "Deeply Supervised Flow-Based Generative Models." ICCV 2025. [arXiv:2503.14494](https://arxiv.org/abs/2503.14494).

**Code:** [github.com/ByteDance-Seed/DeepFlow](https://github.com/ByteDance-Seed/DeepFlow)

**Project page:** [deepflow-project.github.io](https://deepflow-project.github.io/)

### Key Innovation

Standard flow-matching models (SiT, DiT) predict velocity using only the final
transformer layer's output. This underutilizes intermediate representations.
DeepFlow partitions the transformer into K balanced branches, each with its own
velocity prediction head (deep supervision). Between adjacent branches, a lightweight
VeRA (Velocity Refiner with Acceleration) block refines velocity features using
second-order dynamics (acceleration). This achieves 8x faster convergence than SiT
on ImageNet-256 and reduces FID by 2.6 while halving training time.

### Architecture

```
Input: noisy latent z_t [batch, num_patches, hidden]  +  timestep t
      |
+--------------------------------------------------------------+
| Patch Embedding + Positional Encoding                         |
| + AdaLN timestep conditioning                                |
+--------------------------------------------------------------+
      |
+--------------------------------------------------------------+
| Branch 1: Transformer Blocks [0, ..., L/K - 1]               |
|   DiT blocks with AdaLN-Zero modulation                      |
+--------------------------------------------------------------+
      |                    |
      |           +------------------+
      |           | Velocity Head 1  |---> v_1(z_t, t_1)   (deep supervision)
      |           +------------------+
      |
+--------------------------------------------------------------+
| VeRA Block (Velocity Refiner with Acceleration)               |
|   1. ACC MLP: generates acceleration feature a from v_1       |
|   2. Time-Gap Conditioning: AdaLN-Zero(concat(v, a), delta_t)|
|   3. Cross-Space Attention: cross-attn(velocity, spatial)     |
+--------------------------------------------------------------+
      |
+--------------------------------------------------------------+
| Branch 2: Transformer Blocks [L/K, ..., 2L/K - 1]            |
+--------------------------------------------------------------+
      |                    |
      |           +------------------+
      |           | Velocity Head 2  |---> v_2(z_t, t_2)   (deep supervision)
      |           +------------------+
      |
| ... VeRA -> Branch 3 -> VeRA -> ... -> Branch K              |
      |
+--------------------------------------------------------------+
| Final Velocity Head K                                         |
+--------------------------------------------------------------+
      |
Output: v_K(z_t, t)  [batch, num_patches, patch_dim]
```

**VeRA Block detail:**
```
+---------------------------------------------------------------+
| VeRA: Velocity Refiner with Acceleration                       |
|                                                                |
|  Input: velocity feature v_prev [batch, patches, hidden]       |
|         spatial feature s [batch, patches, hidden]             |
|         time gap delta_t (scalar)                              |
|                                                                |
|  1. Acceleration Generation:                                   |
|     a = ACC_MLP(v_prev)              [batch, patches, hidden]  |
|                                                                |
|  2. Time-Gap Modulation:                                       |
|     h = AdaLN_Zero(concat(v_prev, a), delta_t)                |
|                                                                |
|  3. Cross-Space Attention:                                     |
|     Q = Linear_q(h)                                            |
|     K = Linear_k(s)                                            |
|     V = Linear_v(s)                                            |
|     out = softmax(QK^T / sqrt(d)) V                            |
|                                                                |
|  Output: refined velocity feature [batch, patches, hidden]     |
+---------------------------------------------------------------+
```

### Key Equations

**Flow matching ODE** (standard, from SiT):
```
dz_t / dt = v_theta(z_t, t),    t in [0, 1]

z_0 ~ N(0, I),   z_1 = data
```

**Linear interpolation:**
```
z_t = (1 - t) * z_0 + t * z_1
```

**Standard velocity loss:**
```
L_vel = E_{t, z_0, z_1} || v_theta(z_t, t) - (z_1 - z_0) ||^2
```

**Deep supervision loss** (per-branch velocity matching):
```
L_deep = sum_{k=1}^{K} beta_k * E_{t_k} || v_k(z_{t_k}, t_k) - (z_1 - z_0) ||^2
```
Each branch k predicts velocity at a distinct timestep t_k. The timesteps are
deliberately differentiated across branches using a temporal gap upper bound.

**Acceleration loss** (second-order ODE, VeRA training):
```
L_acc = || a_theta(z_t, t) - (dv/dt) ||^2

where dv/dt approx (v(z_{t+dt}, t+dt) - v(z_t, t)) / dt
```
The ACC MLP learns to predict the rate of change of velocity (acceleration),
enabling the VeRA block to refine velocity features using second-order dynamics.

**Total training loss:**
```
L = L_vel_final + sum_{k=1}^{K-1} (beta_k * L_vel_k + gamma_k * L_acc_k)
```

**Timestep sampling for branches:**
During training, branch k receives timestep t_k sampled such that
|t_k - t_{k+1}| <= delta_max (the `--tg-upper-bound` parameter), ensuring
adjacent branches handle nearby but distinct timesteps.

### Input/Output Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| noisy_latent (input) | `{batch, C, H, W}` | e.g. `{B, 4, 32, 32}` for 256px |
| timestep (input) | `{batch}` | scalar t in [0, 1] |
| class_label (input) | `{batch}` | optional class conditioning |
| velocity (output) | `{batch, C, H, W}` | predicted velocity field |

After patchification (patch_size=2): `{batch, num_patches, hidden}` where
num_patches = (H/p) * (W/p) = 256 for 32x32 latent with p=2.

### Hyperparameters

| Parameter | DeepFlow-XL | Edifice Default |
|-----------|-------------|-----------------|
| `input_size` | 32 (latent spatial) | required |
| `patch_size` | 2 | 2 |
| `in_channels` | 4 (VAE latent) | 4 |
| `hidden_size` | 1152 | 256 |
| `num_layers` | 28 | 12 |
| `num_heads` | 16 | 8 |
| `num_branches` K | 4 (7 layers each) | 4 |
| `mlp_ratio` | 4.0 | 4.0 |
| `num_classes` | 1000 (ImageNet) | 0 (unconditional) |
| `beta` (deep supervision weight) | learned/tuned | 1.0 |
| `tg_upper_bound` | configurable | 0.25 |
| `vera_hidden` | hidden_size | hidden_size |
| `learning_rate` | 1e-4 | -- |
| `training_epochs` | 80-400 | -- |

**Model size variants** (following DiT/SiT naming):
- DeepFlow-S: 12 layers, hidden 384, 6 heads
- DeepFlow-B: 12 layers, hidden 768, 12 heads
- DeepFlow-L: 24 layers, hidden 1024, 16 heads
- DeepFlow-XL: 28 layers, hidden 1152, 16 heads

### Implementation Notes for Axon

**Module:** `Edifice.Generative.DeepFlow`

1. **Builds on SiT/DiT pattern.** The core transformer blocks are identical to
   our existing `Edifice.Generative.SiT` -- DiT blocks with AdaLN-Zero modulation.
   DeepFlow adds branching, intermediate velocity heads, and VeRA blocks.

2. **Branch partitioning.** Divide `num_layers` into `num_branches` equal groups.
   Each group is a standard stack of DiT blocks. Insert VeRA blocks between groups.
   If num_layers is not evenly divisible, the last branch gets the remainder.

3. **Velocity heads.** Each branch has a lightweight prediction head:
   `LayerNorm -> Linear(hidden_size, patch_size^2 * in_channels)`. These are
   only used during training for deep supervision loss. At inference, only the
   final velocity head output is used.

4. **VeRA block in Axon.** Three sub-components:
   - ACC MLP: `Linear -> GELU -> Linear` (simple 2-layer MLP)
   - Time-gap conditioning: `AdaLN-Zero` conditioned on delta_t (scalar embedding)
   - Cross-space attention: standard cross-attention between velocity features
     (as Q) and spatial features from the original patch embeddings (as K, V)

5. **Deep supervision as training-only.** The intermediate velocity heads and their
   losses are training concerns. For `build/1`, we can either:
   (a) Return a container output with all K velocity predictions (for training)
   (b) Return only the final velocity (for inference)
   Recommend: return `%{velocity: final, branch_velocities: [v_1, ..., v_K]}` as
   `Axon.container` so users can compute deep supervision loss externally.

6. **Relationship to SiT.** DeepFlow = SiT + branch partitioning + VeRA + deep
   supervision. Could be implemented as a wrapper/extension of SiT, or standalone.
   Recommend standalone to keep SiT clean.

**Options:**
- `:input_size` -- spatial size of input (required, e.g. 32 for 256px)
- `:patch_size` -- patch size for patchification (default: 2)
- `:in_channels` -- input channels (default: 4)
- `:hidden_size` -- transformer hidden dimension (default: 256)
- `:num_layers` -- total transformer blocks (default: 12)
- `:num_heads` -- attention heads (default: 8)
- `:num_branches` -- number of supervised branches K (default: 4)
- `:num_classes` -- class conditioning (default: 0, unconditional)
- `:mlp_ratio` -- FFN expansion ratio (default: 4.0)
- `:tg_upper_bound` -- max temporal gap between branches (default: 0.25)

---

## 4. Meissonic -- Masked Generative Transformer for Images

**Paper:** Bai, Ye, Chow, Song, Li, Dong, Zhu, Yan. "Meissonic: Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image Synthesis." ICLR 2025. [arXiv:2410.08261](https://arxiv.org/abs/2410.08261).

**Code:** [github.com/viiika/Meissonic](https://github.com/viiika/Meissonic)

### Key Innovation

Elevates non-autoregressive masked image modeling (MIM) for text-to-image generation
to parity with SDXL (a diffusion model) while being more efficient. Key advances:
(1) hybrid multi-modal + single-modal transformer layers at 1:2 ratio,
(2) RoPE instead of absolute/sinusoidal position encoding (crucial for high-res),
(3) feature compression layers for VQ token efficiency,
(4) micro-conditions (resolution, crop coords, human preference score) for fine
control, and (5) iterative mask-predict inference (MaskGIT-style with cosine schedule).
Only 1B parameters, runs on 8GB VRAM, generates 1024x1024 images.

### Architecture

```
Text Prompt: "a painting of..."
      |
+--------------------------------------------------------------+
| CLIP Text Encoder (fine-tuned, 1024-dim latents)              |
| -> text_hidden [batch, text_len, 1024]                        |
| -> pooled_text [batch, 1024]                                  |
+--------------------------------------------------------------+
      |                                    |
      |                             +--------------------+
      |                             | Micro-Conditions    |
      |                             |  - resolution (H,W) |
      |                             |  - crop_top_left     |
      |                             |  - aesthetic_score   |
      |                             | -> cond [batch, D]   |
      |                             +--------------------+
      |                                    |
+--------------------------------------------------------------+
| VQ Encoder (frozen VQ-GAN/VQ-VAE)                             |
| Image [B, 3, 1024, 1024]                                      |
|   -> Encode -> Quantize -> token_ids [B, 1024]                |
|   (32x32 spatial grid, each position = codebook index)        |
+--------------------------------------------------------------+
      |
| Masking: replace fraction of tokens with [MASK]               |
      |
+--------------------------------------------------------------+
| Feature Compression Layer                                      |
| Compress VQ token embeddings before transformer                |
+--------------------------------------------------------------+
      |
+--------------------------------------------------------------+
| Multi-Modal Transformer Block x N_mm                           |
|   Joint attention over image tokens + text tokens              |
|   RoPE for image spatial positions                             |
|   Cross-attention (image queries, text keys/values)            |
|   + Self-attention on image tokens                             |
|   Conditioned on micro-conditions + pooled text                |
+--------------------------------------------------------------+
      |
+--------------------------------------------------------------+
| Single-Modal Transformer Block x N_sm  (N_sm = 2 * N_mm)      |
|   Self-attention on image tokens only                          |
|   RoPE for spatial positions                                   |
|   FFN with conditioning                                        |
+--------------------------------------------------------------+
      |
| Feature Decompression + Linear -> codebook_size                |
      |
Output: logits [batch, 1024, codebook_size]
```

**Layer ratio:** Empirically, a 1:2 ratio of multi-modal to single-modal layers
yields optimal training efficiency and quality. The multi-modal layers handle
cross-modal alignment (text-image), while the single-modal layers refine the
image representation. Example: 8 MM layers + 16 SM layers = 24 total.

### Key Equations

**VQ tokenization:**
```
z = Encoder(image)                      # continuous features
q = argmin_k || z_i - e_k ||^2         # quantize to nearest codebook entry
token_ids = q                           # discrete token indices
```
Codebook size is typically 8192 entries with embedding dimension matching the
VQ encoder output.

**Mask-predict training loss:**
```
L = E_{mask_ratio} [ -sum_{i in masked} log p_theta(token_i | x_masked, text) ]
```
Standard cross-entropy over codebook indices at masked positions.

**Cosine masking schedule** (for inference, MaskGIT-style):
```
gamma(r) = cos(pi * r / 2)

n_masked(step) = ceil(N * gamma(step / T))
```
where r = step/T is the fraction of generation completed, N is total tokens,
and T is total sampling steps. At step 0, all tokens are masked. The cosine
schedule unmasks more tokens in later steps (concave curve).

**Inference algorithm** (iterative mask-predict):
```
1. Start: x = all [MASK], N = total tokens (e.g. 1024)
2. For step s = 1, ..., T:
     logits = model(x, text_cond, micro_conds)
     probs = softmax(logits / temperature)
     predicted_tokens = sample from probs at masked positions
     confidence = max(probs, dim=-1) at predicted positions
     n_keep = N - ceil(N * gamma(s / T))      # tokens to unmask this step
     top_k = argsort(confidence, descending)[:n_keep]
     x[top_k] = predicted_tokens[top_k]       # unmask highest confidence
     x[remaining masked positions] = [MASK]    # re-mask the rest
3. Final pass: predict remaining [MASK] tokens
4. Decode: image = VQ_Decoder(x)
```

**Classifier-free guidance (CFG):**
```
logits_guided = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
```
Default CFG scale = 9.0.

**Micro-conditions** (concatenated as conditioning vector):
```
cond = embed(resolution_H, resolution_W, crop_top, crop_left, aesthetic_score)
```
Each micro-condition is embedded via Fourier features or learned embeddings,
then concatenated and projected.

### Input/Output Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| image_tokens (input) | `{batch, num_tokens}` | e.g. 1024 for 32x32 grid |
| text_hidden (input) | `{batch, text_len, text_dim}` | CLIP text features, text_dim=1024 |
| pooled_text (input) | `{batch, text_dim}` | CLS token from CLIP |
| micro_conds (input) | `{batch, cond_dim}` | resolution, crop, aesthetic |
| mask (input) | `{batch, num_tokens}` | boolean mask |
| logits (output) | `{batch, num_tokens, codebook_size}` | logits over VQ codebook |

### Hyperparameters

| Parameter | Meissonic 1B | Edifice Default |
|-----------|-------------|-----------------|
| `codebook_size` | 8192 | required |
| `num_image_tokens` | 1024 (32x32) | required |
| `hidden_size` | ~1536-2048 (est.) | 256 |
| `text_dim` | 1024 (CLIP) | 1024 |
| `num_mm_layers` | 8 (est.) | 4 |
| `num_sm_layers` | 16 (est., 2x mm) | 8 |
| `num_heads` | 16 (est.) | 8 |
| `mlp_ratio` | 4.0 | 4.0 |
| `cfg_scale` | 9.0 | 9.0 |
| `num_sampling_steps` | 48-64 | 48 |
| `temperature` | tuned | 1.0 |
| `image_resolution` | 1024x1024 | 256 |
| Total parameters | ~1B | -- |
| VRAM (inference) | 8-12 GB | -- |

**Note:** Exact layer counts and hidden dimensions are estimated from the paper's
1B parameter budget and the 1:2 multi-modal/single-modal ratio. The paper does not
explicitly publish a full config table. Values marked "(est.)" should be verified
against the official implementation.

### Implementation Notes for Axon

**Module:** `Edifice.Generative.Meissonic`

1. **Multi-input architecture.** Like SAM 2 and RT-DETR in the detection family,
   Meissonic takes multiple inputs: image tokens, text features, pooled text,
   micro-conditions, and mask. Use `Axon.input("name", shape: ...)` for each.

2. **Two transformer block types.** The multi-modal blocks use cross-attention
   between image and text tokens. The single-modal blocks use only self-attention
   on image tokens. Both use RoPE.

   - Multi-modal block: Self-Attention(image) + Cross-Attention(image, text) + FFN
   - Single-modal block: Self-Attention(image) + FFN

3. **RoPE for 2D spatial positions.** Image tokens are arranged in a 2D grid.
   Apply 2D RoPE by splitting the embedding dimension in half and applying separate
   rotary encodings for row and column indices. Reuse `Edifice.Blocks.RoPE` with
   appropriate position computation.

4. **Feature compression.** Linear projection layers that reduce the embedding
   dimension of VQ tokens before the transformer and expand it back after. This
   reduces computational cost since the VQ codebook embedding dim may differ from
   the transformer hidden size.

5. **VQ tokenizer is external.** The VQ-GAN encoder/decoder is NOT part of the
   Edifice module. `build/1` builds only the masked transformer. The VQ tokenizer
   is a separate model (e.g., from diffusers). Inputs are already-quantized
   token indices.

6. **Micro-conditions.** Concatenate embedded micro-conditions (resolution, crop,
   aesthetic score) with pooled text features. Use as AdaLN conditioning or
   additive bias.

7. **Mask-predict inference.** The iterative unmasking loop with cosine schedule
   is an inference-time procedure, not part of the static graph. Provide as a
   helper function or document the algorithm.

8. **Relationship to MDLM/LLaDA.** All three use masked prediction, but:
   - MDLM: text tokens, encoder-only, timestep-conditioned
   - LLaDA: text tokens, encoder-only, no timestep conditioning
   - Meissonic: image VQ tokens, encoder-only, text-conditioned, multi-modal
   Meissonic is the image-domain analog with the additional complexity of
   cross-modal conditioning and VQ tokenization.

**Options:**
- `:codebook_size` -- VQ codebook entries (required)
- `:num_image_tokens` -- number of image tokens in grid (required, e.g. 1024)
- `:hidden_size` -- transformer hidden dim (default: 256)
- `:text_dim` -- text encoder hidden dim (default: 1024)
- `:num_mm_layers` -- multi-modal transformer layers (default: 4)
- `:num_sm_layers` -- single-modal transformer layers (default: 8)
- `:num_heads` -- attention heads (default: 8)
- `:mlp_ratio` -- FFN expansion ratio (default: 4.0)
- `:num_sampling_steps` -- mask-predict inference steps (default: 48)
- `:cfg_scale` -- classifier-free guidance scale (default: 9.0)

---

## Cross-Architecture Comparison

### Discrete Diffusion LM Family

| | MDLM | LLaDA | CaDDi |
|---|------|-------|-------|
| **Attention** | Bidirectional | Bidirectional | Causal |
| **Timestep cond.** | AdaLN-Zero | None | None (implicit via position) |
| **Markov** | Yes | Yes | No (trajectory) |
| **Architecture** | DiT encoder | LLaMA bidi | Standard decoder-only |
| **Noise schedule** | Cosine/loglinear | Linear | Absorbing mask |
| **Loss weight** | alpha'/(1-alpha) | 1/t | Standard NTP |
| **Pretrained LLM** | No | No | Yes (CaDDi-AR) |
| **Scale** | 110M-1.7B | 8B | GPT-2 to Llama-3.2-3B |
| **Generation** | Parallel unmask | Low-conf remask | Iterative trajectory |

### Flow-Based Image Generation

| | SiT (existing) | DeepFlow | Rectified Flow (existing) |
|---|------|----------|------------|
| **ODE** | Linear interp. | Linear interp. | Linear interp. + reflow |
| **Prediction** | Final layer velocity | Multi-branch velocity | Velocity |
| **Deep supervision** | No | Yes (K branches) | No |
| **VeRA** | No | Yes (acceleration) | No |
| **Conv. speed** | 1x baseline | 8x faster | 1x (but fewer steps) |
| **Architecture** | DiT blocks | DiT blocks + VeRA | DiT/U-Net blocks |

### Masked Image Generation

| | MaskGIT (reference) | Meissonic |
|---|------|----------|
| **Tokenizer** | VQ-GAN | VQ-GAN (enhanced) |
| **Conditioning** | Class label | CLIP text + micro-conds |
| **Transformer** | Single-type | Multi-modal + Single-modal |
| **Position enc.** | Absolute | RoPE (2D) |
| **Resolution** | 256x256 | 1024x1024 |
| **Feature comp.** | No | Yes |
| **Scale** | ~300M | ~1B |

---

## Implementation Priority

Based on novelty, Edifice gaps, and implementation complexity:

1. **LLaDA** -- High priority. Architecturally simple (bidi LLaMA, no timestep cond),
   extends discrete diffusion family naturally alongside MDLM. Strong benchmark
   results. Relatively straightforward -- reuse RoPE, SwiGLU, GQA from existing
   modules.

2. **DeepFlow** -- High priority. Novel deep supervision pattern for flow matching.
   Builds directly on existing SiT infrastructure. VeRA block is a clean, isolated
   component. Good demonstration of Edifice composability.

3. **CaDDi** -- Medium priority. Conceptually elegant (unifies AR and diffusion)
   but the architecture is just a standard causal transformer. The innovation is
   in the training/sampling procedure (trajectory construction). The model itself
   is very simple to implement.

4. **Meissonic** -- Medium-lower priority. Most complex due to multi-input,
   multi-modal/single-modal block types, feature compression, and micro-conditions.
   Also depends on external VQ tokenizer and CLIP encoder. But fills the important
   "masked image generation" gap in the generative family.

---

## References

- Nie et al. "Large Language Diffusion Models." ICLR 2025. arXiv:2502.09992.
- Zhang et al. "Non-Markovian Discrete Diffusion with Causal Language Models." NeurIPS 2025. arXiv:2502.09767.
- Shin et al. "Deeply Supervised Flow-Based Generative Models." ICCV 2025. arXiv:2503.14494.
- Bai et al. "Meissonic: Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image Synthesis." ICLR 2025. arXiv:2410.08261.
- Sahoo et al. "Simple and Effective Masked Diffusion Language Models." NeurIPS 2024. arXiv:2406.07524. (MDLM, existing in Edifice)
- Ma et al. "Scalable Interpolant Transformers." 2024. (SiT, existing in Edifice)
- Liu et al. "Flow Straight and Fast: Rectified Flow." ICLR 2023. arXiv:2209.03003. (Rectified Flow, existing in Edifice)
- Chang et al. "MaskGIT: Masked Generative Image Transformer." CVPR 2022. (MaskGIT reference)
