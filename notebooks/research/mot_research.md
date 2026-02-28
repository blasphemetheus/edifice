# Mixture of Transformers (MoT) — Architecture Research

> Research notes for the Edifice implementation of MoT.
> Paper: Liang, Yu, Luo, Iyer, Dong, Zhou, Ghosh, Lewis, Yih, Zettlemoyer, Lin.
> "Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal
> Foundation Models." TMLR 2025. [arXiv:2411.04996](https://arxiv.org/abs/2411.04996)
> Code: [github.com/facebookresearch/Mixture-of-Transformers](https://github.com/facebookresearch/Mixture-of-Transformers)

---

## Core Idea

MoT decouples **all non-embedding transformer parameters** by modality. Each
modality (text, image, speech) gets its own:
- Attention projections (Q, K, V, O)
- Layer normalization
- Feed-forward network (FFN)

The **only shared computation** is the core attention mechanism:
`softmax(QK^T / sqrt(d)) * V`. This is applied globally over the full
interleaved multi-modal sequence, so all modalities can attend to each other.

### Why This Matters

In a standard transformer:
- FFN accounts for ~67% of parameters
- Attention projections account for ~33%

With M modalities, each token only activates 1/M of the FFN + projection
parameters. The paper reports:
- **55.8% FLOPs** with text + image (2 modalities)
- **37.2% FLOPs** with text + image + speech (3 modalities)

...while matching dense baseline quality.

### MoT vs MoE

MoT is **not** Mixture-of-Experts:
- **Routing is deterministic** — based on known modality of each token
- **No load balancing loss** needed
- More similar to conditional computation / modality-specific parameter sets
- Each modality always uses exactly its own expert — no top-k routing

---

## Architecture

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
                    Output: [B, S, d_model]
```

---

## Key Equations

### Modality-Specific Attention Projections

For each modality m in {text, img, speech}:
```
Q_m = W_q^m * LN^m(x_m)      W_q^m: [d_model, d_model]
K_m = W_k^m * LN^m(x_m)      W_k^m: [d_model, d_model]
V_m = W_v^m * LN^m(x_m)      W_v^m: [d_model, d_model]
```

### Global Self-Attention (Shared)

```
Q = scatter(Q_text, Q_img, Q_speech, masks)    [B, S, d_model]
K = scatter(K_text, K_img, K_speech, masks)    [B, S, d_model]
V = scatter(V_text, V_img, V_speech, masks)    [B, S, d_model]

A = softmax(Q * K^T / sqrt(d_head)) * V       [B, S, d_model]
```

### Modality-Specific Output + FFN

```
For modality m:
  a_m = gather(A, mask_m)              [B, S_m, d_model]
  o_m = W_o^m * a_m                    [B, S_m, d_model]
  x_m = x_m + o_m                      (residual)
  x_m = x_m + FFN_m(LN2^m(x_m))       (residual + modality FFN)
```

### Modality FFN

Standard SwiGLU or GELU per modality:
```
FFN_m(x) = W_2^m * act(W_1^m * x)     W_1^m: [d_model, d_ffn], W_2^m: [d_ffn, d_model]
```

---

## Tensor Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| Input tokens | `[batch, seq_len]` | Interleaved multi-modal sequence |
| Modality masks | `[batch, seq_len, num_modalities]` | Binary, mutually exclusive, one-hot |
| Per-modality QKV | `[batch, seq_len, d_model]` per modality | Dense path: computed on all tokens |
| Combined QKV | `[batch, seq_len, d_model]` | After mask-combine (sum of masked projections) |
| Attention output | `[batch, seq_len, d_model]` | Shared computation |
| Per-modality FFN | `[batch, seq_len, d_model]` per modality | After mask-combine |
| Final output | `[batch, seq_len, vocab_size]` | After final norm + LM head |

---

## Hyperparameters

| Parameter | Paper Default | Edifice Default | Description |
|-----------|--------------|-----------------|-------------|
| `d_model` / `hidden_size` | 4096 | 256 | Model hidden dimension |
| `num_heads` | 32 | 8 | Attention heads (shared across modalities) |
| `num_layers` | 32 | 6 | Number of MoT layers |
| `d_ffn` / `intermediate_size` | 11008 | hidden_size * 4 | FFN intermediate dim per modality |
| `num_modalities` | 2 | 2 | Number of modalities |
| `activation` | SiLU | SiLU | FFN activation |
| `norm` | RMSNorm | RMSNorm | Layer normalization |

---

## Implementation Notes

### Edifice Module: `Edifice.Meta.MixtureOfTransformers`

**Inputs:**
- `"tokens"` — `[batch, seq_len]` integer token IDs
- `"modality_mask"` — `[batch, seq_len, num_modalities]` one-hot float mask

**Output:** `[batch, seq_len, vocab_size]` logits

### Dense Path (Current Implementation)

The Edifice implementation uses the "dense path" for mask routing:

1. Run all M modality-specific projections on ALL tokens
2. Mask-multiply each projection by its modality mask slice
3. Sum across modalities to get the combined result

```elixir
# For modality m, mask its output:
mask_m = Nx.slice_along_axis(mask, m, 1, axis: 2)   # [batch, seq, 1]
masked_m = Nx.multiply(projection_m, mask_m)          # zeros out wrong tokens

# Sum all masked projections:
combined = sum(masked_0, masked_1, ..., masked_{M-1})
```

**Trade-off:** This wastes some FLOPs (each projection runs on all tokens
instead of just its modality's tokens), but avoids dynamic gather/scatter
indexing which is awkward in Axon's static computation graph.

### Sparse Path (Future Optimization)

For production efficiency, a sparse path would:
1. `Nx.take` / gather only modality-m tokens from the sequence
2. Run smaller projections on the gathered subset
3. `Nx.indexed_put` / scatter results back into the full sequence

This requires dynamic shapes (varying number of tokens per modality per batch),
which is better handled with EXLA JIT compilation.

### Per-Modality Parameters

Each MoT layer creates **M copies** of:
- Pre-attention RMSNorm
- Q, K, V projections (each `[hidden_size, hidden_size]`)
- O projection (`[hidden_size, hidden_size]`)
- Pre-FFN RMSNorm
- FFN up projection (`[hidden_size, intermediate_size]`)
- FFN down projection (`[intermediate_size, hidden_size]`)

Total per-modality parameters per layer:
```
norms: 0 (RMSNorm has no learnable params in our impl)
Q + K + V + O: 4 * hidden_size^2
FFN up + down: hidden_size * intermediate_size * 2
```

With M=2, hidden=4096, ffn=11008: ~134M params per layer per modality.
Dense baseline: ~67M params per layer. MoT total: ~134M but each token
only activates ~67M (same as dense but with modality specialization).

---

## Related Work

### vs Dense Transformer
Dense transformers apply the same FFN and projections to all tokens regardless
of modality. MoT saves FLOPs proportional to modality count while maintaining
quality through shared attention.

### vs Mixture-of-Experts (MoE)
MoE uses learned routing with top-k selection and load balancing losses.
MoT routing is deterministic (modality labels are known). No router network,
no auxiliary losses, no token dropping.

### vs Chameleon (Meta)
The paper uses Chameleon (Meta's multimodal model) as the dense baseline.
MoT achieves Chameleon-quality results at ~56% of the FLOPs for text+image.

### vs Multimodal Adapters
Adapter approaches (like LLaVA) freeze the LM and train lightweight adapters.
MoT jointly trains all modality-specific parameters end-to-end, allowing
deeper modality specialization.

---

## Training Details (from paper)

- Pre-trained on 2.9T tokens (text) + image data
- Vocabulary: 65K text tokens + 8192 image tokens (via VQ-VAE)
- Sequence length: 4096
- Batch size: 4M tokens
- Optimizer: AdamW, lr=1e-4, warmup 2000 steps
- Mixed precision (bf16)

---

## Evaluation Results (from paper)

| Model | Text (PPL) | Image (FID) | FLOPs |
|-------|-----------|-------------|-------|
| Chameleon 7B (dense) | 8.2 | 12.4 | 100% |
| MoT 7B (text+image) | 8.3 | 12.1 | 55.8% |
| MoT 7B (text+image+speech) | 8.4 | 12.3 | 37.2% |

---

## Edifice Test Configuration

Small config for BinaryBackend testing:
```elixir
@small_opts [
  vocab_size: 32,
  hidden_size: 32,
  num_heads: 4,
  num_layers: 2,
  intermediate_size: 64,
  num_modalities: 2,
  seq_len: 8
]
```

Tests verify: build/shape/finite output, 3-modality variant, single layer,
larger vocab, uniform modality mask, and `output_size/1`.
