# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
