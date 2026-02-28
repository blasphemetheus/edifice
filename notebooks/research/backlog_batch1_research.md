# Backlog Batch 1 Research Notes

Research findings for DeepONet, GPS, PointNet++, and Wav2Vec 2.0 implementations.

## DeepONet — Branch-Trunk Operator Learning

**Paper:** Lu et al., "Learning nonlinear operators via DeepONet" (Nature Machine Intelligence 2021)
**Reference:** https://arxiv.org/abs/1910.03193

### Architecture

DeepONet learns nonlinear operators G: u -> G(u)(y) mapping input functions to output
functions. Two sub-networks evaluate at query points:

```
Input function u(x) sampled at sensors [x1..xm]
        |
  Branch Net (MLP)
        |
  [batch, output_dim * latent_dim]     Branch output
        |
        +-- dot product --+
        |                 |
  Trunk Net (MLP)         |
        |                 |
  [batch, queries, output_dim * latent_dim]  Trunk output
        |
  [batch, queries, output_dim]   Operator evaluation G(u)(y)
```

### Key Details

- **Branch network**: MLP processing sensor values, outputs `[batch, output_dim * latent_dim]`
- **Trunk network**: MLP processing query coordinates, outputs `[batch, num_queries, output_dim * latent_dim]`
- **Combination**: Reshaped dot product over latent dimension + optional bias
- **Universal approximation**: Can approximate any continuous operator to arbitrary accuracy
- **Applications**: PDE solving, climate modeling, material science

### Defaults

| Parameter | Default |
|-----------|---------|
| latent_dim | 40 |
| output_dim | 1 |
| branch_hidden | [128, 128, 128] |
| trunk_hidden | [128, 128, 128] |
| activation | :tanh |
| use_bias | true |

### Implementation Notes

- Returns single `Axon.t()` (not tuple) with two inputs: "sensors" and "query_points"
- Axon.layer arity gotcha: splitting combine into `combine_with_bias/4` and `combine_no_bias/3`
  to match the unpacked argument count from `Axon.layer(&fun/N, [inputs], opts)`
- Registered as `:deep_onet` in scientific family

---

## GPS — General, Powerful, Scalable Graph Transformer

**Paper:** Rampasek et al., "Recipe for a General, Powerful, Scalable Graph Transformer" (NeurIPS 2022)
**Reference:** https://arxiv.org/abs/2205.12454

### Architecture

Dual-branch hybrid: local GIN message passing + global multi-head attention in parallel,
combined by summation, followed by FFN. Each GPS layer:

```
Input X^l [batch, N, hidden_size], Adjacency A [batch, N, N]
        |
   +----+----+
   |         |
 GIN(X,A)   MHA(X)         (parallel branches)
   |         |
 +res+norm  +res+norm       (independent residuals)
   |         |
   +-- sum --+              (combine)
        |
   FFN + res + norm
        |
Output X^{l+1}
```

### Key Design Choices

- **Post-norm** residual pattern (norm after residual addition)
- **LayerNorm** (our impl; paper uses BatchNorm by default)
- **GIN MPNN** provides WL-test expressivity for local structure
- **Global attention** without adjacency masking for long-range information flow
- **FFN** with 2x hidden multiplier and ReLU activation
- **RWSE** (Random Walk Structural Encoding) as positional information

### RWSE Details

Computes `diag((D^{-1}A)^k)` for k = 1..walk_length:
- Row-normalize adjacency: `rw_matrix = D^{-1} * A`
- Iteratively compute powers and extract diagonals
- Each node gets walk_length-dimensional structural encoding
- Projected via LayerNorm + Dense to pe_dim, concatenated with node features

### Defaults

| Parameter | Default |
|-----------|---------|
| hidden_size | 64 |
| num_heads | 4 |
| num_layers | 5 |
| dropout | 0.0 |
| ffn_multiplier | 2 |
| activation | :relu |
| pe_dim | 8 |
| rwse_walk_length | 16 |

### Implementation Notes

- Reuses `GIN.gin_layer/4` for the local MPNN branch
- Custom `global_attention_impl/4` for full multi-head attention without adjacency masking
- Two inputs: "nodes" `[batch, N, input_dim]`, "adjacency" `[batch, N, N]`
- Optional `:pool` (`:mean` or `:sum`) and `:num_classes` for graph-level tasks
- Registered as `:gps` in graph family

### Bugs Encountered

1. `Nx.select(degree > 0, ...)` — boolean not tensor, fixed with `Nx.greater` + float cast
2. Elixir `-` operator on Nx tensors in softmax, fixed with `Nx.subtract`
3. `:math.sqrt(head_dim)` for attention scale (can't use Nx.tensor in traced context)

---

## PointNet++ — Hierarchical Point Cloud Processing

**Paper:** Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space" (NeurIPS 2017)
**Reference:** https://arxiv.org/abs/1706.02413

### Architecture

Extends PointNet with hierarchical Set Abstraction (SA) layers for multi-scale local feature
learning. Each SA layer: FPS downsample -> ball query neighbors -> mini-PointNet per group.

```
Point Cloud [batch, N, 3]
      |
SA Layer 1 (N -> N1): FPS -> Ball Query -> MLP -> Max Pool
      |
SA Layer 2 (N1 -> N2): FPS -> Ball Query -> MLP -> Max Pool
      |
Global SA (N2 -> 1): Group all -> MLP -> Global Max Pool
      |
FC Head: Dense -> LayerNorm -> ReLU -> Dropout -> Dense(num_classes)
```

### Key Algorithms

**Farthest Point Sampling (FPS):**
- Iteratively select the point farthest from all previously selected points
- Provides maximal spatial coverage of the point cloud
- Sequential (each step depends on previous), implemented via `Enum.reduce`

**Ball Query:**
- For each centroid, find up to K neighbors within radius r
- Pairwise distance matrix + radius masking + argsort + slice
- Out-of-radius points masked with large distance (1e10)

**Mini-PointNet:**
- Shared MLP (Dense + LayerNorm + activation) applied per group
- Max pool over K neighbors dimension
- Centroid-relative coordinates for translation invariance

### Defaults

| Parameter | Default |
|-----------|---------|
| input_dim | 3 (xyz) |
| activation | :relu |
| dropout | 0.3 |
| SA configs | [{32, 0.2, 16, [32,32,64]}, {16, 0.4, 16, [64,64,128]}] |
| global_mlp | [128, 256, 512] |
| fc_dims | [256, 128] |

### Implementation Notes

- Single input: "points" `[batch, N, input_dim]`
- FPS + ball query + grouping combined in one `Axon.layer` call
- Centroid extraction in separate `Axon.layer` (recomputes FPS — same deterministic result)
- `batch_gather` and `batch_gather_neighbors` helpers for efficient index-based gathering
- `put_column` helper for FPS state updates (one-hot mask + multiply pattern)
- Registered as `:pointnet_pp` in sets family

### Bugs Encountered

- Unused `input_feat_dim` and `activation` params caused warnings-as-errors
- Fixed by prefixing unused with `_` and using activation in global_sa

---

## Wav2Vec 2.0 — Self-Supervised Speech Representation Learning

**Paper:** Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" (NeurIPS 2020)
**Reference:** https://arxiv.org/abs/2006.11477

### Architecture

Processes raw audio waveforms through CNN feature encoder, convolutional positional encoding,
and Transformer encoder. Separate product quantization module for pre-training contrastive targets.

```
Raw waveform [batch, samples]
      |
7-layer CNN Feature Encoder (stride 320 total -> 50 Hz frame rate)
      |
Feature Projection (Linear + LayerNorm)
      |
+ Convolutional Positional Encoding (kernel=128, groups=16, GELU)
      |
Transformer Encoder x N (pre-norm self-attention + FFN)
      |
[batch, T, hidden_dim]

(Parallel path for pre-training:)
CNN output -> Product Quantization -> Contrastive targets
```

### CNN Feature Encoder

7-layer 1D conv stack, all 512 channels (configurable via :cnn_channels for testing):

| Layer | Kernel | Stride | Norm |
|-------|--------|--------|------|
| 0 | 10 | 5 | GroupNorm(32) |
| 1-6 | 3,3,3,3,2,2 | 2,2,2,2,2,2 | None (BASE) |

All layers use GELU activation. Total stride = 5*2^6 = 320 samples.

### Product Quantization

- Projects CNN output to `[batch, T, G * V]` logits (G=2 groups, V=320 entries)
- Gumbel softmax for differentiable discrete selection (hard argmax at inference)
- Soft codes projected to codevector_dim (256)

### Variants

| Variant | hidden_dim | layers | heads | FFN dim |
|---------|-----------|--------|-------|---------|
| BASE | 768 | 12 | 8 | 3072 |
| LARGE | 1024 | 24 | 16 | 4096 |

### Defaults

| Parameter | Default |
|-----------|---------|
| hidden_dim | 768 |
| encoder_layers | 12 |
| num_heads | 8 |
| ffn_dim | 3072 |
| dropout | 0.1 |
| conv_pos_kernel | 128 |
| conv_pos_groups | 16 |
| num_codebook_groups | 2 |
| codebook_entries | 320 |
| codevector_dim | 256 |

### Implementation Notes

- Returns `{encoder, quantizer}` tuple (two separate Axon models)
- Uses `TransformerBlock.stack` with custom SDPA-based `attention_fn` callback
- Uses `SDPA.compute/5` from shared blocks
- `:cnn_channels` option (default 512) enables small configs for BinaryBackend testing
- Registered as `:wav2vec2` in audio family

### Bugs Encountered

1. Missing `attention_fn` for TransformerBlock — added SDPA-based self_attention callback
2. Elixir `if` scoping — `if dropout > 0.0 do x = ...` doesn't update x outside block
3. 512-channel CNN timeout on BinaryBackend — added configurable cnn_channels option
4. Line too long in credo — reformatted multi-line function call
