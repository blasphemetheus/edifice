# Tier 3 Architecture Research

> Research findings for New Tier 3 (2026) implementations.
> Compiled 2026-02-23.

---

## 1. MambaVision

**Paper:** "MambaVision: A Hybrid Mamba-Transformer Vision Backbone" (Hatamizadeh & Kautz, NVIDIA, 2024). CVPR 2025.
**Code:** https://github.com/NVlabs/MambaVision

### Architecture Overview

Hierarchical 4-stage vision backbone. Early stages use CNN, middle/late stages use Mamba SSM + windowed self-attention.

```
Input: (B, 3, 224, 224)
  -> PatchEmbed (2x Conv3x3 stride 2 each = 4x downsample) -> (B, dim, 56, 56)
  -> Stage 1 (ConvBlocks)           -> (B, dim,   56, 56)  -> Downsample
  -> Stage 2 (ConvBlocks)           -> (B, 2*dim, 28, 28)  -> Downsample
  -> Stage 3 (Mamba + Transformer)  -> (B, 4*dim, 14, 14)  -> Downsample
  -> Stage 4 (Mamba + Transformer)  -> (B, 8*dim, 7, 7)
  -> BatchNorm -> AdaptiveAvgPool -> Flatten -> Linear -> (B, num_classes)
```

### Key Components

**PatchEmbed (Stem):**
- Two Conv2d(3x3, stride=2) + BatchNorm + ReLU layers
- Total 4x spatial downsampling

**ConvBlock (Stages 1-2):**
```
input -> Conv2d(3x3) -> BN -> GELU -> Conv2d(3x3) -> BN -> [layer_scale] -> + input
```

**Hybrid Block (Stages 3-4):**
```
x = x + drop_path(gamma_1 * Mixer(LayerNorm(x)))
x = x + drop_path(gamma_2 * MLP(LayerNorm(x)))
```
- First half of blocks: MambaVisionMixer (SSM)
- Second half of blocks: Windowed self-attention

**MambaVisionMixer (Core Innovation):**
- Dual-branch: SSM branch + symmetric non-SSM branch
- SSM operates on half channels, other half goes through Conv1d + SiLU only
- Non-causal convolution (padding='same', not causal)
- Branches concatenated (not gated like original Mamba)
- Parameters: d_state=8, d_conv=3, expand=1

**Attention Block:**
- Standard multi-head self-attention on windowed tokens
- No relative position bias, no shifted windows
- Window sizes make stages 3-4 effectively global attention (window=14 at 14x14, window=7 at 7x7)

**Downsampling:** Conv2d(3x3, stride=2) between stages

**No explicit positional encoding** — CNN stem provides implicit spatial bias

### Model Variants

| Variant | dim | depths | num_heads | Params |
|---------|-----|--------|-----------|--------|
| T | 80 | [1,3,8,4] | [2,4,8,16] | 31.8M |
| S | 96 | [3,3,7,5] | [2,4,8,16] | 50.1M |
| B | 128 | [3,3,10,5] | [2,4,8,16] | 97.7M |
| L | 196 | [3,3,10,5] | [4,8,16,32] | 227.9M |

Channel progression: dim -> 2*dim -> 4*dim -> 8*dim across stages.

---

## 2. KDA (Kimi Delta Attention)

**Paper:** "Kimi Linear: An Expressive, Efficient Attention Architecture" (Moonshot AI). ArXiv: 2510.26692
**Code:** https://github.com/MoonshotAI/Kimi-Linear, https://github.com/fla-org/flash-linear-attention

### Key Innovation: Channel-Wise Gating

The single key difference from Gated DeltaNet: **per-channel (per-dimension) decay gate** instead of scalar per-head.

**Gated DeltaNet:** `alpha_t` is scalar per head — all channels decay at the same rate
**KDA:** `alpha_t` is in R^{d_k} — each channel decays independently

### State Update Equation

```
S_t = (I - beta_t * k_t * k_t^T) * Diag(alpha_t) * S_{t-1} + beta_t * k_t * v_t^T
h_t = S_t^T * q_t
```

Where:
- S_t: R^{d_k x d_v} state matrix
- alpha_t: (0,1)^{d_k} — channel-wise decay vector
- beta_t: [0,1] — scalar update gate per head
- Diag(alpha_t): diagonal matrix from alpha vector

### Alpha Gate Production (Low-Rank MLP)

```
alpha_t = sigmoid(W_up * SiLU(W_down * x_t))
```
- W_down: hidden_size -> bottleneck (head_v_dim)
- SiLU activation
- W_up: bottleneck -> key_dim (num_heads * head_dim)
- Then processed through gate kernel: g_out = -exp(A_log) * softplus(g_in + dt_bias)
- Stored in log-space for numerical stability

### Complete KDA Layer

```
Projections:
  q = q_proj(x)        # hidden -> key_dim
  k = k_proj(x)        # hidden -> key_dim
  v = v_proj(x)        # hidden -> value_dim
  g = f_proj(x)        # MLP: hidden -> bottleneck -> key_dim (alpha gate)
  beta = sigmoid(b_proj(x))  # hidden -> num_heads
  gate = g_proj(x)     # MLP: hidden -> bottleneck -> value_dim (output gate)

Short Convolution (kernel_size=4):
  q, k, v = ShortConv(q), ShortConv(k), ShortConv(v) with SiLU

L2 Normalize: q, k

Core Recurrence: KDA scan

Output: FusedRMSNormGated(output, gate=sigmoid(gate))
        o_proj(output)  # value_dim -> hidden_size
```

### Key Differences from Gated DeltaNet

| Aspect | Gated DeltaNet | KDA |
|--------|---------------|-----|
| Decay gate | Scalar per head | Vector per channel |
| Alpha production | Simple projection | Low-rank MLP + A_log/dt_bias |
| Output gate | SiLU | Sigmoid (validated by ablation) |
| Output norm | Varies | FusedRMSNormGated |
| Short conv | Optional | Default on, kernel=4 |

### Kimi Linear Hybrid Architecture

3:1 ratio: 3 KDA layers then 1 MLA (full attention) layer, repeating.
MLA layers use NoPE (no positional encoding).

---

## 3. Multimodal Fusion Layers

### Five Main Approaches

**1. Cross-Attention (Flamingo-style)**
- Gated cross-attention blocks inserted every 4th layer of frozen LLM
- Visual features (from Perceiver Resampler: 64 tokens) serve as K,V
- LLM hidden states serve as Q
- Gating: `y = x + tanh(alpha) * CrossAttention(x, v_features)`, alpha init=0
- Preserves frozen LLM capabilities

**2. Early Fusion (Fuyu-style)**
- No separate vision encoder
- Raw image patches linearly projected into LLM embedding space
- Special `<image-newline>` tokens for 2D structure
- Single causal transformer processes mixed image+text tokens

**3. Late Fusion (CLIP-style)**
- Separate encoders, fusion only at output via cosine similarity
- Great for retrieval, cannot generate text

**4. Perceiver-Based (BLIP-2 Q-Former)**
- 32 learned query embeddings cross-attend to visual features
- Outputs fixed 32 visual tokens for LLM
- Both encoder and LLM stay frozen

**5. MLP Projection (LLaVA-style) — DOMINANT APPROACH**
- Pre-trained ViT → 2-layer MLP → prepend to LLM input
- `z = Linear_2(GELU(Linear_1(visual_tokens)))`
- 576 visual tokens from CLIP ViT-L/14
- Simplest, most well-documented, state-of-the-art results

### 2025-2026 Production VLMs

| Model | Approach |
|-------|----------|
| Qwen2.5-VL | MLP projection (4:1 token compression) |
| InternVL 2.5 | MLP projection (pixel unshuffle) |
| LLaVA-NeXT | MLP projection |
| PaliGemma 2 | Linear projection |
| LLaMA 3.2 Vision | Cross-attention (Flamingo-style) |
| Gemini | Native early fusion |

### Implementation Plan

Build two approaches:
1. **MLP Projection** (LLaVA-style) — simplest, most practical
2. **Cross-Attention** (Flamingo-style) — more architecturally interesting

---

## 4. RL Environment Integration

### What Exists

- `Edifice.RL.Environment` behaviour with rollout/4
- `Edifice.RL.PolicyValue` actor-critic model

### PPO Algorithm

On-policy actor-critic. Collect rollout → compute GAE advantages → K epochs of clipped surrogate optimization.

**PPO Loss:**
```
ratio = exp(new_log_prob - old_log_prob)
surr1 = ratio * advantages
surr2 = clip(ratio, 1-epsilon, 1+epsilon) * advantages
policy_loss = -mean(min(surr1, surr2))
value_loss = mean((new_values - returns)^2)
entropy_loss = -mean(entropy)
total = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
```

**Key hyperparams:** gamma=0.99, lambda=0.95, epsilon=0.2, K=4 epochs

**GAE (Generalized Advantage Estimation):**
```
for t = T-1 down to 0:
    delta = rewards[t] + gamma * next_value * (1-done) - values[t]
    gae = delta + gamma * lambda * (1-done) * gae
    advantages[t] = gae
returns = advantages + values
```

### CartPole Environment

- State: [x, x_dot, theta, theta_dot]
- Action: Discrete(2) — push left/right
- Reward: +1.0 per step
- Done: |x| > 2.4 or |theta| > 12° or steps >= 500
- Physics: Euler integration with standard cart-pole dynamics

### SAC Algorithm (Future)

Off-policy with replay buffer. 2 Q-networks + targets + actor.
Squashed Gaussian policy with reparameterization trick.
Automatic entropy temperature tuning.

### What to Build

| Module | Purpose |
|--------|---------|
| `RL.Environments.CartPole` | Built-in discrete env |
| `RL.Environments.GridWorld` | Built-in simple env |
| `RL.GAE` | Advantage estimation |
| `RL.PPOTrainer` | PPO training loop |
| `RL.ReplayBuffer` | For SAC (future) |
