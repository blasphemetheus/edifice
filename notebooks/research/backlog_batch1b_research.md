# Backlog Batch 1b Research Notes

Research findings for Diffusion Policy, F5-TTS, and Show-o implementations.
Compiled 2026-02-28.

---

## 1. Diffusion Policy -- ConditionalUnet1D with FiLM Conditioning

**Paper:** Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (RSS 2023)
**ArXiv:** [2303.04137](https://arxiv.org/abs/2303.04137)
**Code:** [github.com/real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)

**Status: IMPLEMENTED** as `Edifice.Robotics.DiffusionPolicy`

### Architecture

Two variants: ConditionalUnet1D (primary) and TransformerForDiffusion.

**ConditionalUnet1D:**
```
noisy_actions [B, Tp, action_dim]  +  observations [B, To, obs_dim]  +  timestep [B]
  |                                        |                                 |
  v (transpose to [B, Da, Tp])          (flatten)                     SinusPosEmb(256)
  |                                        |                          -> MLP -> [B, 256]
  v                                        v                                |
  Down: ResBlock+FiLM x2 -> Downsample     global_cond = cat[t_emb, obs] --+
  Down: ResBlock+FiLM x2 -> Downsample     [B, 256+To*obs_dim]
  Down: ResBlock+FiLM x2 (no downsample)
  |
  Mid: ResBlock+FiLM x2
  |
  Up: cat(skip) -> ResBlock+FiLM x2 -> Upsample
  Up: cat(skip) -> ResBlock+FiLM x2
  |
  Final: Conv1dBlock -> Conv1d(256, action_dim)
  -> noise prediction [B, Tp, action_dim]
```

**FiLM Conditioning (ConditionalResidualBlock1D):**
- `cond -> Mish -> Linear(cond_dim, out_ch*2) -> split -> scale, bias`
- `out = scale * Conv1dBlock(x) + bias` (channel-wise)
- Plus residual connection

### Key Hyperparameters

| Parameter | U-Net | Transformer |
|-----------|-------|-------------|
| down_dims | [256, 512, 1024] | -- |
| kernel_size | 5 | -- |
| n_groups | 8 | -- |
| n_layer | -- | 8 |
| n_head | -- | 4 |
| n_emb | -- | 256 |
| Tp (prediction) | 16 | 5 |
| To (observation) | 2 | 3 |
| Ta (action) | 8 | 1 |
| num_train_timesteps | 100 | 100 |
| beta_schedule | squaredcos_cap_v2 | squaredcos_cap_v2 |
| prediction_type | epsilon | epsilon |

### Noise Schedule

Cosine schedule (iDDPM) with only 100 timesteps (not 1000 like image diffusion).

### Observation Conditioning

Three modes: global (FiLM, default for U-Net), local (temporal concatenation), or inpainting (trajectory masking). Transformer uses cross-attention.

---

## 2. F5-TTS -- Non-Autoregressive Flow-Matching TTS

**Paper:** Chen et al., "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching"
**ArXiv:** [2410.06885](https://arxiv.org/abs/2410.06885)
**Code:** [github.com/SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)

**Status: IMPLEMENTED** as `Edifice.Audio.F5TTS`

### Architecture

DiT backbone with ConvNeXt V2 text encoder, RoPE, and convolutional position embeddings.

```
Inputs: noisy_mel [B, N, 100], cond_mel [B, N, 100], text [B, T], timestep [B]

Text Processing:
  Embedding(V+1, 512) + SinPosEmbed + ConvNeXtV2 x4 -> [B, N, 512]

Timestep:
  SinPosEmb(256) -> MLP(256->1024->SiLU->1024) -> t_emb [B, 1024]

Input Fusion:
  cat(noisy_mel, cond_mel, text_embed) -> [B, N, 712]
  Linear(712, 1024) + ConvPositionEmbedding -> [B, N, 1024]

DiT Backbone (22 layers):
  AdaLN-Zero (6-param from t_emb via SiLU pre-activation):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
  Self-Attention (16 heads, dim_head=64, RoPE)
  FFN (GELU, ff_mult=2: 1024->2048->1024)

Final: AdaLN_Final(2-param) -> Linear(1024, 100) (zero-init)
Output: velocity [B, N, 100]
```

### Key Components

**ConvNeXt V2 Block:**
- Depthwise Conv1d(512, 512, kernel=7, groups=512)
- LayerNorm -> Linear(512, 1024) -> GELU -> GRN(1024) -> Linear(1024, 512)
- Residual connection

**Global Response Normalization (GRN):**
- `gx = L2_norm(x, spatial_dims)`
- `nx = gx / (mean(gx, channel_dim) + eps)`
- `return gamma * (x * nx) + beta + x`

**Convolutional Position Embedding:**
- `Conv1d(dim, dim, kernel=31, groups=16) -> Mish -> Conv1d -> Mish` (residual)

### Flow Matching (OT-CFM)

- Linear interpolation: `phi_t(x) = (1-t)*x0 + t*x1`
- Target velocity: `u_t = x1 - x0`
- Loss: MSE on velocity prediction
- Inference: Euler ODE solving, 32 steps default (16 fast)
- Sway sampling: `f(u; s=-1) = u + s*(cos(pi*u/2) - 1 + u)` biases toward early steps

### Key Hyperparameters (Base, 335.8M)

| Parameter | Value |
|-----------|-------|
| dim | 1024 |
| depth | 22 |
| heads | 16 |
| dim_head | 64 |
| ff_mult | 2 |
| dropout | 0.1 |
| text_dim | 512 |
| text_num_embeds | 2546 |
| conv_layers | 4 |
| mel_dim | 100 |
| sample_rate | 24000 |
| NFE (inference) | 32 (default), 16 (fast) |
| cfg_strength | 2.0 |

### Comparison to E2-TTS

F5-TTS improves on E2-TTS by:
1. ConvNeXt V2 text refinement (main WER improvement)
2. DiT with AdaLN-Zero instead of flat U-ViT (41% fewer GFLOPs)
3. Sway sampling for better inference quality

---

## 3. Show-o -- Unified AR + Discrete Diffusion

**Paper:** Xie et al., "Show-o: One Single Transformer to Unify Multimodal Understanding and Generation" (ICLR 2025)
**ArXiv:** [2408.12528](https://arxiv.org/abs/2408.12528)
**Code:** [github.com/showlab/Show-o](https://github.com/showlab/Show-o)

**Status: IMPLEMENTED** as `Edifice.Generative.ShowO`

### Architecture

Standard Phi-1.5 transformer with extended vocabulary and omni-attention mask.

```
input_ids [B, seq_len] -- combined text + image discrete tokens
modality_mask [B, seq_len] -- 0=text, 1=image
  |
  v
Token Embedding (extended vocab: 50,295 text + 8,192 image + 10 special = 58,498)
  |
  v
24x Transformer Blocks:
  LayerNorm -> Self-Attention (omni-mask + QK-Norm + partial RoPE) -> Residual
  LayerNorm -> MLP (gelu_new, 8192 intermediate) -> Residual
  |
  v
LM Head -> logits [B, seq_len, 58498]
```

### Omni-Attention Mask

```
mask[i, j] = 1  if j <= i                    (causal baseline)
          OR if image[i] AND image[j]         (bidirectional within image block)
```

Text tokens: causal (standard LLM). Image tokens: bidirectional among themselves, causal to preceding text. Same pattern as Transfusion.

### Discrete Diffusion (MaskGIT-style)

**Training:** Randomly mask image tokens with [MASK], predict original token IDs via cross-entropy.

**Inference (iterative unmasking):**
1. Start: all 256 image positions = [MASK]
2. For T steps (~18):
   - Forward pass -> logits at mask positions
   - CFG: `logit = (1+w)*cond - w*uncond`
   - Sample tokens, score confidence: `log(prob) + temp*gumbel_noise`
   - Keep top-confidence predictions, re-mask rest
   - Unmask ratio: cosine schedule `cos(t*pi/2)`

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| hidden_size | 2048 |
| num_layers | 24 |
| num_heads | 32 |
| intermediate_size | 8192 |
| partial_rotary_factor | 0.5 |
| qk_layernorm | true |
| llm_vocab_size | 50,295 |
| codebook_size | 8,192 |
| total_vocab_size | 58,498 |
| num_vq_tokens | 256 (16x16) |
| denoising_timesteps | 18 |

### Key Differences from Transfusion (Already in Edifice)

| Aspect | Transfusion | Show-o |
|--------|------------|--------|
| Image representation | Continuous patches | Discrete tokens (MAGVIT-v2) |
| Image diffusion | Continuous (MSE loss) | Discrete (CE loss on masked tokens) |
| Output heads | Separate text + image | Single unified LM head |
| Timestep conditioning | Sinusoidal + MLP injection | No explicit timestep (masking ratio encodes it) |
| Architecture | Custom transformer | Standard Phi-1.5 with extended vocab |

The neural network is remarkably simple: a standard LLM transformer with extended vocabulary and mixed attention mask. All the diffusion/generation behavior comes from the masking strategy and inference procedure, not the architecture.
