# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-19

### Added

- 113 registered architectures across 19 families (up from 92/16)
- **Transformer**: Decoder-Only (GPT-style with GQA, RoPE, SwiGLU, RMSNorm)
- **Feedforward**: KAT (Kolmogorov-Arnold Transformer), BitNet (ternary/binary weight quantization)
- **SSM**: Mamba-3 (complex state dynamics, trapezoidal discretization, MIMO rank-r), StripedHyena (gated conv + Hyena hybrid)
- **Attention**: Based (Taylor expansion linear attention), InfiniAttention (compressive memory + local attention), Conformer (conv + transformer for audio), Mega (EMA + single-head gated attention), RingAttention (chunked ring-distributed attention), MLA (Multi-Head Latent Attention with KV compression and decoupled RoPE), DiffTransformer (dual softmax noise-cancelling attention)
- **Vision**: FocalNet (focal modulation), PoolFormer (pooling-based MetaFormer), NeRF (positional encoding + MLP for radiance fields)
- **Graph**: GINv2 (GIN with edge features)
- **Meta**: MixtureOfDepths (dynamic per-token compute allocation), MixtureOfAgents (multi-proposer + aggregator routing), RLHFHead (reward model and preference heads)
- **Contrastive**: JEPA (Joint Embedding Predictive Architecture with context encoder + predictor)
- **Blocks**: CausalMask (unified mask creation), DepthwiseConv (1D depthwise separable convolution)
- `ARCHITECTURE_ROADMAP.md` tracking remaining architectures by priority tier

### Enhanced

- MultiHead and GQA attention: added `:rope` option for built-in RoPE integration
- TTT (Test-Time Training): added `:variant` option for `:linear` and `:mlp` inner models
- TransformerBlock: added `:custom_ffn` callback for non-standard feed-forward networks
- xLSTM: added `:mlstm` registry alias (`Edifice.build(:mlstm, opts)`)
- sLSTM: log-domain stabilization (m_t state), recurrent connections (R*h_{t-1}), proper normalization (max(|n_t|, 1))

### Changed

- Removed unnecessary `require Axon` from 104 modules (Axon has zero macros)
- BitNet `bitlinear_impl` comment clarified: STE is implicit via Axon's param/callback architecture

### Fixed

- MessagePassing aggregate: added batch axes to `Nx.dot` for correct batched matrix multiplication
- RetNet: corrected `recurrent_retention_step` batching
- TTT: paper-faithful initialization for numerical stability
- FNet: replaced `Nx.fft` with real DFT matrix multiply for compatibility
- RWKV: fixed seq_len=1 compile failure

## [0.1.1] - 2026-02-14

### Fixed

- **MoE top-k routing**: `top_k_forward` now uses `Nx.top_k` indices with one-hot mask for correct expert selection (was ignoring indices and averaging first k experts)
- **MoE hash routing**: `hash_forward` now properly selects expert by hash (was always returning first expert)
- **SwitchMoE routing**: Replaced soft weighted average with hard top-1 selection via straight-through estimator, restoring the sparsity that defines Switch Transformer
- **SchNet filter generation**: Added learned 2-layer filter-generating network (RBF -> Dense -> SiLU -> Dense) replacing naive mean aggregation
- **ConvNeXt layer scale**: Changed from frozen constant to learnable `Axon.param`, matching Liu et al. 2022
- **MessagePassing aggregate**: Added batch axes to `Nx.dot` for correct batched matrix multiplication
- **SNN docstring**: Corrected reset mechanism description from hard reset to soft reset (subtract threshold)

### Changed

- **KAN default basis**: Changed from `:sine` (Fourier features) to `:bspline` (cubic B-spline via Cox-de Boor), faithful to Liu et al. 2024. Previous bases (`:sine`, `:chebyshev`, `:fourier`, `:rbf`) remain available as options
- **TTT W_0 initialization**: Changed from `0.01 * Identity` to `:glorot_uniform` per Sun et al. 2024
- **TTT output RMS norm**: Made optional via `:output_rms_norm` option (default: `false`), was unconditionally applied

### Removed

- Unused `_x` and `_dt` parameters from Liquid `integrate_ode`

## [0.1.0] - 2026-02-14

### Added

- 92 registered architectures across 16 families
- Unified interface: `Edifice.build(:name, opts)` and `Edifice.list_architectures()`
- **Feedforward**: MLP, KAN (Kolmogorov-Arnold Networks), TabNet
- **Convolutional**: Conv1D/2D, ResNet, DenseNet, TCN, MobileNet, EfficientNet
- **Recurrent**: LSTM, GRU, xLSTM, MinGRU, MinLSTM, DeltaNet, TTT, Titans, Reservoir (ESN)
- **State Space Models**: Mamba (parallel scan), Mamba-2 (SSD), MambaCumsum, MambaHillisSteele, S4, S4D, S5, H3, Hyena, BiMamba, GatedSSM, Jamba, Zamba
- **Attention**: Multi-Head (sliding window, hybrid), GQA, Perceiver, FNet, LinearTransformer, Nystromformer, Performer, RetNet, RWKV-7, GLA, HGRN-2, Griffin/Hawk
- **Vision**: ViT, DeiT, Swin Transformer, U-Net, ConvNeXt, MLP-Mixer
- **Generative**: VAE, VQ-VAE, GAN (WGAN-GP), DDPM Diffusion, DDIM, DiT, Latent Diffusion, Consistency Model, Score SDE, Flow Matching, Normalizing Flows
- **Contrastive**: SimCLR, BYOL, Barlow Twins, MAE (Masked Autoencoder), VICReg
- **Graph**: GCN, GAT, GIN, GraphSAGE, Graph Transformer, PNA, SchNet
- **Sets**: DeepSets, PointNet
- **Energy**: EBM (contrastive divergence), Modern Hopfield Networks, Neural ODE
- **Probabilistic**: Bayesian Neural Networks, MC Dropout, Evidential Neural Networks
- **Memory**: Neural Turing Machine, Memory Networks
- **Meta**: MoE, Switch MoE, Soft MoE, LoRA, Adapter, Hypernetworks, Capsule Networks
- **Liquid**: Liquid Neural Networks (continuous-time ODE)
- **Neuromorphic**: SNN (LIF neurons), ANN2SNN conversion
- **Building Blocks**: RMSNorm, SwiGLU, FFN, RoPE, ALiBi, PatchEmbed, SinusoidalPE, AdaptiveNorm, CrossAttention
- 12 conceptual guide documents covering theory, evolution, and decision tables for all families
- `CONTRIBUTING.md` with architecture addition guide, test patterns, and Nx/Axon gotchas
- ~1160 tests covering all architecture families
