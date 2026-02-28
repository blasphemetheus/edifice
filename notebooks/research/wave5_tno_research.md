# Wave 5: Toeplitz Neural Network (TNN) / Toeplitz Neural Operator (TNO) Research

## Paper Reference

**"Toeplitz Neural Network for Sequence Modeling"** by Zhen Qin, Xiaodong Han, Weixuan Sun, Bowen He, et al. Published at **ICLR 2023** (Spotlight). arXiv: [2305.04749](https://arxiv.org/abs/2305.04749). Code: [OpenNLPLab/Tnn](https://github.com/OpenNLPLab/Tnn).

Follow-up: **"Accelerating Toeplitz Neural Network with Constant-time Inference Complexity"** (EMNLP 2023) — exact conversion to SSM for O(1) per-step inference via [ETSC algorithm](https://github.com/OpenNLPLab/ETSC-Exact-Toeplitz-to-SSM-Conversion).

## Core Mathematical Idea

A **Toeplitz matrix** T of size n x n has constant values along each diagonal: `T[i,j] = t[i - j]`. Only **2n - 1 unique parameters** define the entire n x n matrix (vs n^2 for dense). Each entry depends only on **relative position** `i - j`.

For sequence modeling: the "interaction strength" between positions depends only on their distance, giving:
1. **O(n log n)** via FFT-based Toeplitz-vector multiplication (vs O(n^2) for attention)
2. **O(n) parameters** for the interaction pattern
3. **Length extrapolation**: trained on length 512, generalizes to 14K+

## Architecture Components

### RPE (Relative Position Encoder)

Small MLP that maps scalar relative position indices to Toeplitz kernel coefficients:

```
Input: position index (scalar, shape [n, 1])
  -> Linear(1, rpe_dim)
  -> [Norm -> Act -> Linear(rpe_dim, rpe_dim)] x rpe_layers
  -> Norm -> Act -> Linear(rpe_dim, heads * dim_per_head)
Output: shape [heads, n, dim_per_head]
```

- `rpe_dim`: internal dimension (recommended: `max(embed_dim // 8, 32)`)
- `rpe_layers`: number of hidden layers (default: 3, range 1-6)
- `rpe_act`: activation (default: relu)
- Uses SimpleRMSNorm between layers

### TNO (Toeplitz Neural Operator) — Token Mixing

Constructs and applies the Toeplitz matrix via FFT:

1. Generate position indices: `pos = [1..n-1]`, `neg = [-(n-1)..-1]` (bidirectional) or zeros (causal)
2. Feed through RPE to get learned coefficients: `a_pos = RPE(pos)`, `a_neg = RPE(neg)`, `a_zero = RPE(0)`
3. Apply exponential decay: `a_pos[k] = gamma^k * a_pos[k]`
4. Assemble circulant-embedded kernel:
   - **Causal**: `a = [a_zero, a_pos, zeros(n)]` (length 2n)
   - **Bidirectional**: `a = [a_zero, a_pos, a_zero, a_neg]` (length 2n)
5. FFT convolution:

```
y = rfft(x, n=2*seq_len)     # FFT of input
v = rfft(a, n=2*seq_len)     # FFT of kernel
output = irfft(v * y)[:n]    # IFFT, truncate to seq_len
```

Per-dimension operation: separate Toeplitz matrix per feature dimension per head.

### GTU (Gated Toeplitz Unit) — Token + Channel Mixing

Wraps TNO in a gated architecture:

```
u = act(W_u * x)                    # gate branch [batch, seq, expand_dim]
v = act(W_v * x)                    # value branch [batch, seq, expand_dim]
v = reshape(v, [batch, heads, seq, dim_per_head])
v = TNO(v)                          # token mixing via Toeplitz
v = reshape(v, [batch, seq, expand_dim])
output = W_out * Norm(u * v)        # gating + project back
```

- `expand_ratio`: controls `expand_dim = embed_dim * expand_ratio` (default: 3)
- `num_heads`: independent Toeplitz heads (default: 1)

### GLU (Gated Linear Unit) — Channel Mixing

Standard GLU for feature mixing (no sequence interaction):

```
o1 = act(W_1 * x)     # gate
o2 = W_2 * x          # value
output = W_3 * (o1 * o2)
```

### Full TNN Block

```
x = x + GTU(Norm(x))      # token mixing with residual
x = x + GLU(Norm(x))      # channel mixing with residual
```

Analogous to Transformer block: GTU replaces multi-head attention, GLU replaces FFN.

## Key Hyperparameters

| Parameter | Description | Default |
|---|---|---|
| `embed_dim` | Model hidden dimension | Task-dependent |
| `num_heads` | Toeplitz heads in GTU | 1 |
| `rpe_dim` | RPE MLP internal dim | `max(embed_dim // 8, 32)` |
| `rpe_layers` | RPE hidden layers | 3 |
| `rpe_activation` | RPE activation | `:relu` |
| `expand_ratio` | GTU expansion factor | 3 |
| `activation` | GTU/GLU gate activation | `:silu` |
| `causal` | Causal masking | `false` |
| `use_decay` | Exponential decay | `true` |
| `gamma` | Decay rate | 0.99 |
| `num_layers` | Number of TNN blocks | Task-dependent |
| `dropout` | Dropout rate | 0.0 |

## Comparison with Other Architectures

| Architecture | Training | Inference/step | Mechanism | Length Extrapolation |
|---|---|---|---|---|
| Transformer | O(n^2 d) | O(n^2 d) | Content-based attention | Poor |
| S4 | O(n d) | O(d) | Fixed structured SSM kernel | Good |
| Mamba | O(n d) | O(d) | Input-selective SSM | Good |
| TNN | O(n d log n) | O(n d log n) | Learned relative-position Toeplitz | Excellent (512->14K) |
| TNN + ETSC | O(n d log n) train | O(d) inference | Convert to SSM at inference | Excellent |

## Implementation Plan for Edifice

### Module: `Edifice.Attention.TNN`

Place in attention family — it's a token-mixing mechanism like attention but position-based.

### Build options
```elixir
@type build_opt ::
  {:embed_dim, pos_integer()}
  | {:hidden_size, pos_integer()}
  | {:num_heads, pos_integer()}
  | {:num_layers, pos_integer()}
  | {:expand_ratio, pos_integer()}
  | {:rpe_dim, pos_integer()}
  | {:rpe_layers, pos_integer()}
  | {:rpe_activation, atom()}
  | {:activation, atom()}
  | {:causal, boolean()}
  | {:use_decay, boolean()}
  | {:gamma, float()}
  | {:dropout, float()}
  | {:seq_len, pos_integer()}
```

### Key implementation details

1. **FFT**: Use `Nx.fft/2` and `Nx.ifft/2` for the Toeplitz convolution
2. **RPE**: Small Axon subgraph (Dense -> [Norm -> Act -> Dense] x L)
3. **Exponential decay**: `Nx.pow(gamma, Nx.iota({n-1}) + 1)` — gamma learnable via sigmoid
4. **SimpleRMSNorm**: `x / sqrt(mean(x^2) + eps)` — no learnable scale
5. **Causal**: Zero out negative-index kernel values
6. **No positional embedding needed**: Toeplitz structure encodes relative position

### Block structure
```
Input [batch, seq, embed_dim]
  -> N x TNN Block:
       -> LayerNorm
       -> GTU (2 linear projs -> SiLU -> TNO on value -> gate -> Norm -> project out)
       -> Residual add
       -> LayerNorm
       -> GLU (2 linear projs -> SiLU gate -> project out)
       -> Residual add
  -> Extract last timestep [batch, hidden_size]
```

## References

- [Toeplitz Neural Network for Sequence Modeling (ICLR 2023)](https://arxiv.org/abs/2305.04749)
- [OpenNLPLab/Tnn - Official Implementation](https://github.com/OpenNLPLab/Tnn)
- [Accelerating Toeplitz Neural Network (EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.750/)
- [SKI to go Faster: Accelerating TNO via Asymmetric Kernels](https://arxiv.org/abs/2305.09028)
