# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Expanded from 44 to 91 registered architectures across 16 families
- **State Space Models**: S4, S4D, H3, Hyena, MambaCumsum, MambaHillisSteele, BiMamba, HybridBuilder
- **Attention**: GQA, Perceiver, FNet, LinearTransformer, Nystromformer, Performer
- **Recurrent**: MinGRU, MinLSTM, DeltaNet, TTT (Test-Time Training), Titans
- **Vision**: ViT, DeiT, Swin Transformer, U-Net, ConvNeXt, MLP-Mixer
- **Convolutional**: MobileNet, EfficientNet
- **Generative**: DDIM, DiT (Diffusion Transformer), Latent Diffusion, Consistency Model, Score SDE
- **Contrastive**: SimCLR, BYOL, Barlow Twins, MAE (Masked Autoencoder), VICReg
- **Graph**: GIN, GraphSAGE, Graph Transformer, PNA, SchNet
- **Energy**: Neural ODE
- **Probabilistic**: Evidential Neural Networks
- **Meta**: Switch MoE, Soft MoE, LoRA, Adapter
- **Neuromorphic**: ANN2SNN (ANN-to-SNN conversion)
- **Feedforward**: TabNet (attentive tabular learning)
- **Building Blocks**: RMSNorm, SwiGLU, FFN, RoPE, ALiBi, PatchEmbed, SinusoidalPE, AdaptiveNorm, CrossAttention
- 12 conceptual guide documents covering theory, evolution, and decision tables for all families
- `groups_for_extras` in ExDoc for organized guide sidebar navigation
- Benchee-based profiling scripts (`bench/architecture_profile.exs`, `bench/scaling_profile.exs`)
- `CONTRIBUTING.md` with architecture addition guide, test patterns, and Nx/Axon gotchas
- `.credo.exs` config with documented threshold adjustments for ML library patterns
- `mix_audit` for dependency vulnerability scanning
- Explicit `:files` key in `package/0` for clean Hex.pm tarballs
- ~1159 tests covering all architecture families

### Changed

- Migrated 8 modules from private FFN builders to shared `Edifice.Blocks.FFN` (S4, S4D, S5, H3, Hyena, BiMamba, Liquid, MoE)
- Precomputed chunk A cumulative products in Mamba-2 SSD inter-chunk propagation (perf optimization)

### Fixed

- Nx tensor arithmetic outside `defn` context (NTM, Capsule, NormalizingFlow, EBM)
- `Nx.dot` batching with explicit batch axes (GAT, MessagePassing)
- Activation naming (`:silu` instead of `:swish` for Axon 0.8)
- Removed broken `sliding_window` registry alias (called wrong function)
- Removed dead `_chunk_a_products` code in Mamba-2 SSD
- README architecture count accuracy (was "90+", now exact "91 across 16 families")
- All 72 Credo warnings resolved (alias ordering, `length/1`, nested module refs)

## [0.1.0] - 2026-02-10

### Added

- Initial release with 44 architectures across 14 families
- Unified interface: `Edifice.build(:name, opts)` and `Edifice.list_architectures()`
- **Feedforward**: MLP, KAN (Kolmogorov-Arnold Networks)
- **Convolutional**: Conv1D/2D, ResNet, DenseNet, TCN
- **Recurrent**: LSTM, GRU, xLSTM, Echo State Networks (Reservoir)
- **State Space Models**: Mamba (parallel scan), Mamba-2 (SSD), S5, GatedSSM, Jamba, Zamba
- **Attention**: Multi-Head (sliding window, hybrid), RetNet, RWKV-7, GLA, HGRN-2, Griffin/Hawk
- **Generative**: VAE, VQ-VAE, GAN (WGAN-GP), DDPM Diffusion, Flow Matching, Normalizing Flows
- **Graph**: GCN, GAT, generic Message Passing (MPNN)
- **Sets**: DeepSets, PointNet
- **Energy**: EBM (contrastive divergence), Modern Hopfield Networks
- **Probabilistic**: Bayesian Neural Networks, MC Dropout
- **Memory**: Neural Turing Machine, Memory Networks
- **Meta**: Mixture of Experts, Hypernetworks, Capsule Networks
- **Liquid**: Liquid Neural Networks (continuous-time ODE)
- **Neuromorphic**: Spiking Neural Networks (LIF neurons)
- 184 tests covering all architecture families
