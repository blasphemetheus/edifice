# Building Blocks
> Composable primitives -- normalization, position encoding, gating, patching, and cross-attention -- that combine to form complete architectures.

## Overview

The Blocks family contains the fundamental layers that appear repeatedly across transformers, vision models, diffusion systems, and state-space architectures. Rather than reimplementing these primitives inside every architecture, Edifice factors them into standalone, tested modules with a consistent `layer/2` or functional API.

These blocks fall into five categories: **normalization** (RMSNorm, AdaptiveNorm), **position encoding** (SinusoidalPE, RoPE, ALiBi), **activation gating** (SwiGLU), **input tokenization** (PatchEmbed), and **cross-sequence attention** (CrossAttention). Each is designed to slot into an Axon computation graph as a composable node, so higher-level architectures can mix and match without coupling to a specific implementation.

Understanding when to use each block is often more important than understanding their internals. A position encoding choice, for example, determines whether a model can generalize to longer sequences than it saw during training. A normalization choice affects training stability and throughput. This guide provides the conceptual grounding to make those decisions.

## Conceptual Foundation

All blocks in this family serve the same meta-purpose: **constrain or inject structure** into an otherwise generic sequence of linear projections and nonlinearities. Without normalization, activations drift and training diverges. Without position encoding, attention treats sequences as unordered bags. Without gating, feed-forward blocks lack the multiplicative interactions that improve gradient flow.

The key equation underlying most of these blocks is the transformer's attention computation, into which position encodings and normalizations are inserted:

    Attention(Q, K, V) = softmax( (QK^T + bias) / sqrt(d_k) ) V

where `bias` may come from ALiBi slopes, the Q and K may be rotated by RoPE, the inputs may be RMSNorm'd, and the feed-forward that follows may use SwiGLU gating.

## Architecture Evolution

```
2017  Vaswani et al. "Attention Is All You Need"
  |     - LayerNorm, Sinusoidal PE, standard FFN
  |
2019  Zhang & Sennrich
  |     - RMSNorm: drop mean centering, ~50% faster normalization
  |
2020  Shazeer "GLU Variants Improve Transformer"
  |     - SwiGLU/GeGLU: gated FFN with multiplicative interactions
  |
2021  Su et al. "RoFormer"                    Dosovitskiy et al. "ViT"
  |     - RoPE: rotary embeddings,              - PatchEmbed: images
  |       relative position via rotation           as patch sequences
  |
2022  Press et al. "Train Short, Test Long"
  |     - ALiBi: no learned params,
  |       linear bias for extrapolation
  |
2023  Peebles & Xie "DiT"
  |     - AdaptiveNorm (AdaLN-Zero):
  |       condition-dependent scale/shift/gate
  |
2024+ Modern LLMs (LLaMA, Mistral, Mamba-2)
        - RMSNorm + RoPE + SwiGLU as standard stack
```

## When to Use What

### Normalization

| Scenario | Module | Rationale |
|----------|--------|-----------|
| General transformer layers | `RMSNorm` | Faster than LayerNorm, no mean subtraction, standard in modern LLMs |
| Axon built-in suffices | `Axon.layer_norm` | When mean centering matters (e.g., small models, batch norm alternative) |
| Conditional generation (diffusion, class-conditional) | `AdaptiveNorm` | Scale/shift/gate predicted from conditioning signal |
| Diffusion Transformers (DiT) | `AdaptiveNorm` with `:adaln_zero` mode | Zero-initialized gate for stable early training |

### Position Encoding

| Scenario | Module | Rationale |
|----------|--------|-----------|
| Fixed-length, no extrapolation needed | `SinusoidalPE` | Simplest, deterministic, no learned params |
| Long-context LLMs, extrapolation required | `RoPE` | Relative position via rotation, smooth extrapolation with NTK-aware scaling |
| Maximum extrapolation, minimal complexity | `ALiBi` | Zero learned params, linear bias, best length generalization |
| Vision transformers | `SinusoidalPE` or learned | 2D grid positions, extrapolation less critical |
| Encoder-decoder (translation, summarization) | `SinusoidalPE` or `RoPE` | Both work; RoPE preferred for modern systems |

### Feed-Forward / Gating

| Scenario | Module | Rationale |
|----------|--------|-----------|
| Modern transformer FFN | `SwiGLU` | 3 projections (gate, up, down) with SiLU gating; better quality per FLOP |
| Legacy / simple models | `Axon.dense` + activation | Standard 2-projection FFN when parameter budget is tight |
| GeGLU or ReGLU variant needed | `SwiGLU` with `:activation` option | Same module, different gate activation |

### Tokenization and Cross-Attention

| Scenario | Module | Rationale |
|----------|--------|-----------|
| Image input to transformer | `PatchEmbed` | Splits image into non-overlapping patches, projects to embedding dim |
| Multimodal fusion (text + image) | `CrossAttention` | Queries from one modality, keys/values from another |
| Encoder-decoder architectures | `CrossAttention` | Decoder attends to encoder outputs |
| U-Net conditioning (Stable Diffusion) | `CrossAttention` | Inject text conditioning into image generation |

## Key Concepts

### Normalization: LayerNorm vs RMSNorm vs AdaptiveNorm

**LayerNorm** (Axon built-in) normalizes by subtracting the mean and dividing by the standard deviation across the feature dimension, then applies learned scale (gamma) and shift (beta). This two-step process -- centering then scaling -- costs roughly twice the compute of RMSNorm.

**RMSNorm** drops the mean subtraction entirely. It divides by the root mean square of the activations and applies only a learned scale. Empirically, the centering step contributes little to training stability in large models, making RMSNorm the better default. LLaMA, Mistral, Mamba-2, and most 2024+ architectures use RMSNorm exclusively.

    RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * gamma

**AdaptiveNorm** (AdaLN / AdaLN-Zero) replaces fixed gamma and beta with values predicted from a conditioning signal. In diffusion models, the conditioning signal is typically a timestep embedding. AdaLN-Zero additionally predicts a gating factor alpha initialized to zero, which ensures that at initialization each transformer block acts as an identity function -- critical for stable training of deep generative models.

```
                  LayerNorm              RMSNorm              AdaptiveNorm
                  ---------              -------              ------------
Centering:        Yes (subtract mean)    No                   Via predicted beta
Scaling:          Yes (learned gamma)    Yes (learned gamma)  Via predicted gamma
Extra gate:       No                     No                   Optional (AdaLN-Zero)
Parameters:       2 * hidden_size        hidden_size          Predicted from condition
Typical use:      Legacy, small models   Modern LLMs/SSMs     Conditional generation
```

### Position Encoding: Absolute, Relative, and None

Position encoding determines how a model distinguishes token positions. The three approaches in Edifice represent distinct philosophies:

**SinusoidalPE** (absolute) assigns each position a unique vector using sine and cosine waves at geometrically spaced frequencies. It is added to the input embeddings once. The model must learn that these additive signals encode position. Extrapolation beyond the training length is theoretically possible but unreliable in practice.

**RoPE** (relative, multiplicative) rotates query and key vectors by position-dependent angles. Because rotation is multiplicative, the inner product between Q at position m and K at position n depends only on (m - n), making it inherently relative. RoPE handles extrapolation better than sinusoidal PE and is the standard for modern LLMs. The base frequency (default 10,000) can be scaled for longer contexts.

```
Position encoding injection points in a transformer block:

  Input --[+ SinusoidalPE]--> LayerNorm --> Q,K,V projection
                                               |
                                         [RoPE rotates Q,K]
                                               |
                                         [ALiBi biases QK^T scores]
                                               |
                                            softmax --> output
```

**ALiBi** (relative, additive bias) adds no parameters at all. Instead, it adds a linear penalty to attention scores based on the distance between query and key positions. Each head gets a different slope from a geometric schedule, so some heads attend locally (steep slope) and others globally (gentle slope). ALiBi provides the strongest length extrapolation because it never saw position-specific parameters during training -- the linear bias generalizes naturally.

### Activation Gating: SwiGLU

Standard transformer FFN blocks use two linear projections with a nonlinearity:

    FFN(x) = W2 * activation(W1 * x)

SwiGLU adds a third projection that acts as a gate:

    SwiGLU(x) = W2 * (SiLU(V * x) * W1 * x)

The element-wise multiplication between the gate path and the linear path creates richer gradient dynamics. The SiLU (x * sigmoid(x)) activation on the gate provides smooth gating. Despite having 50% more parameters in the projections, SwiGLU achieves better quality per FLOP because the gating improves gradient flow through depth. The default expansion factor (2.667x, rounded to a multiple of 8 for tensor core alignment) is calibrated to match the parameter count of a standard 4x FFN.

### Patch Embedding for Vision

PatchEmbed bridges the gap between spatial image data and the sequence format that transformers expect. It divides an image into a grid of non-overlapping patches, flattens each patch into a vector, and projects it to the model's embedding dimension.

For a 224x224 RGB image with 16x16 patches: 196 patches, each originally 768-dimensional (16 * 16 * 3), projected to the target embedding dimension. This is equivalent to a single convolution with kernel size and stride both equal to the patch size, but Edifice implements it as reshape + dense for clarity and compatibility with the Axon graph.

Smaller patches (8x8, 14x14) give finer spatial resolution at the cost of longer sequences. Larger patches (32x32) are more efficient but lose spatial detail. The 16x16 default from ViT remains the most common choice.

## Complexity Comparison

| Module | Params per Layer | Time Complexity | Space Complexity | Notes |
|--------|-----------------|-----------------|------------------|-------|
| `RMSNorm` | hidden_size | O(n * d) | O(d) | Single learnable scale vector |
| `SinusoidalPE` | 0 | O(n * d) | O(n * d) table | Precomputed, no learned params |
| `RoPE` | 0 | O(n * d) | O(n * d/2) table | Rotation angles precomputed |
| `ALiBi` | 0 | O(n^2 * h) | O(h * n * n) bias | Slopes fixed by geometric schedule |
| `SwiGLU` | 3 * d * d_inner | O(n * d * d_inner) | O(d_inner) | Three projections vs two in standard FFN |
| `PatchEmbed` | P^2 * C * d_embed | O(N_patches * P^2 * C * d) | O(N_patches * d) | One-time input tokenization |
| `AdaptiveNorm` | cond_dim * (2 or 3) * d | O(n * d) | O(d) | Params in conditioning projection |
| `CrossAttention` | 4 * d^2 | O(n_q * n_kv * d) | O(n_q * n_kv) | Standard attention across two sequences |

Where n = sequence length, d = hidden dimension, h = number of heads, P = patch size, C = channels.

## Module Reference

- `Edifice.Blocks.RMSNorm` -- Root mean square normalization without mean centering; standard for LLaMA, Mistral, Mamba-2
- `Edifice.Blocks.SwiGLU` -- Gated feed-forward with SiLU/GELU/ReLU gate; 3-projection FFN used in modern transformers
- `Edifice.Blocks.RoPE` -- Rotary position embedding via pairwise dimension rotation; relative position without learned params
- `Edifice.Blocks.ALiBi` -- Attention bias from head-specific linear slopes; zero learned parameters, best extrapolation
- `Edifice.Blocks.PatchEmbed` -- Image-to-patch tokenization for vision transformers; reshape + linear projection
- `Edifice.Blocks.SinusoidalPE` -- Fixed sinusoidal position encoding from the original transformer; deterministic baseline
- `Edifice.Blocks.AdaptiveNorm` -- Condition-dependent normalization (AdaLN/AdaLN-Zero) for diffusion and conditional generation
- `Edifice.Blocks.CrossAttention` -- Cross-attention between two sequences; used in encoder-decoder and multimodal models

## Cross-References

- **attention_mechanisms.md** -- Position encodings (RoPE, ALiBi, SinusoidalPE) plug directly into attention score computation
- **vision_architectures.md** -- PatchEmbed is the standard input layer for ViT, DeiT, Swin, and MAE; AdaptiveNorm appears in DiT
- **generative_models.md** -- AdaLN-Zero is central to Diffusion Transformers (DiT) for conditional image generation
- **state_space_models.md** -- RMSNorm and SwiGLU appear in Mamba blocks alongside selective state-space layers

## Further Reading

1. Vaswani et al., "Attention Is All You Need" (2017) -- arxiv.org/abs/1706.03762. Introduced sinusoidal PE, LayerNorm placement, and the transformer FFN.
2. Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019) -- arxiv.org/abs/1910.07467. Demonstrates RMSNorm matches LayerNorm quality with less compute.
3. Shazeer, "GLU Variants Improve Transformer" (2020) -- arxiv.org/abs/2002.05202. Systematic comparison of gated FFN variants; SwiGLU emerges as best.
4. Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021) -- arxiv.org/abs/2104.09864. Derives RoPE from rotation matrices in complex space.
5. Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (2022) -- arxiv.org/abs/2108.12409. ALiBi achieves strong extrapolation with no learned position parameters.
