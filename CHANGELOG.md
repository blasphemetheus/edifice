# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
