# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-25

### Added

186 registered architectures across 25 families (up from 92 across 16 in v0.1.0). 94 new architectures grouped by family:

- **Attention** (35 total, +20): Hawk, RetNet v2, Megalodon, Lightning Attention, GLA v2, HGRN v2, Flash Linear Attention, KDA (Kernelized Deformable Attention), Gated Attention, SSMax (Scalable-Softmax), Softpick (non-saturating sparse normalization), RNoPE-SWA (sliding window without positional encoding), YaRN (context window extension via frequency-scaled RoPE), NSA (Native Sparse Attention from DeepSeek-V3/V4), TMRoPE (Time-aligned Multimodal RoPE), Dual Chunk Attention, Based (Taylor expansion linear attention), InfiniAttention (compressive memory + local attention), Conformer (conv + transformer for audio), Mega (EMA + single-head gated attention), RingAttention (chunked ring-distributed), MLA (Multi-Head Latent Attention), DiffTransformer (dual softmax noise-cancelling)
- **Audio** (3, NEW family): SoundStorm (parallel audio generation via masked prediction), EnCodec (neural audio codec), VALL-E (zero-shot TTS via neural codec language modeling)
- **Contrastive** (8, +3): JEPA (Joint Embedding Predictive Architecture), Temporal JEPA, SigLIP (sigmoid contrastive loss for language-image pretraining)
- **Generative** (22, +11): MMDiT (multi-modal DiT), SoFlow, VAR (Visual Autoregressive Modeling, NeurIPS 2024 Best Paper), Linear DiT/SANA (DiT with linear attention), SiT (Scalable Interpolant Transformer), Transfusion (unified AR text + diffusion image), MAR (Masked Autoregressive Generation), CogVideoX (text-to-video diffusion with 3D causal VAE), TRELLIS (structured 3D latents with sparse transformer + rectified flow), DiT v2, Consistency Model
- **Graph** (9, +2): GIN v2 (GIN with edge features), EGNN (E(n)-equivariant graph neural network)
- **Inference** (1, NEW family): Medusa (multi-head speculative decoding for 2-3x speedup)
- **Interpretability** (2, NEW family): Sparse Autoencoder, Transcoder
- **Memory** (3, +1): Engram (O(1) hash-based associative memory via locality-sensitive hashing)
- **Meta** (22, +11): DPO (Direct Preference Optimization), KTO (Kahneman-Tversky Optimization), GRPO (Group Relative Policy Optimization), MoE v2 (aux-loss-free load balancing), DoRA, Speculative Decoding, Test-Time Compute, Mixture of Tokenizers, Speculative Head, Distillation Head, QAT (Quantization-Aware Training), Hybrid Builder (flexible hybrid architecture composition), MixtureOfDepths, MixtureOfAgents, RLHFHead
- **Multimodal** (1, NEW family): Multimodal Fusion
- **Recurrent** (15, +7): sLSTM, xLSTM v2, Gated DeltaNet, TTT-E2E (end-to-end test-time training), Native Recurrence, plus previously added recurrent variants
- **RL** (1, NEW family): PolicyValue
- **Robotics** (2, NEW family): ACT (Action Chunking Transformer for robot imitation learning), OpenVLA (Vision-Language-Action model)
- **Scientific** (1, NEW family): FNO (Fourier Neural Operator)
- **SSM** (19, +5): StripedHyena (gated conv + Hyena hybrid), Mamba-3 (complex state dynamics, trapezoidal discretization, MIMO rank-r), GSS (Gated State Spaces), Hyena v2, Hymba, SS Transformer
- **Transformer** (4, NEW family): Decoder-Only (GPT-style with GQA, RoPE, SwiGLU, RMSNorm), Multi-Token Prediction, Byte Latent Transformer, Nemotron-H (NVIDIA's hybrid Mamba-Transformer)
- **Vision** (15, +9): FocalNet (focal modulation), PoolFormer (pooling-based MetaFormer), NeRF (positional encoding + MLP for radiance fields), Gaussian Splatting (real-time differentiable radiance field rendering), MambaVision, DINOv2 (self-supervised vision backbone via self-distillation), MetaFormer + CAFormer (pluggable token mixer framework), EfficientViT (O(n) linear attention with cascaded group attention)
- **World Model** (1, NEW family): World Model
- **Feedforward**: KAT (Kolmogorov-Arnold Transformer), BitNet (ternary/binary weight quantization)
- **Blocks**: CausalMask (unified mask creation), DepthwiseConv (1D depthwise separable convolution)

Infrastructure and tooling:

- GGUF export for decoder-only models
- KV cache for inference
- Quantization toolkit (QAT module)
- `shell.nix` for reproducible Erlang 27 + Elixir 1.18 + CUDA dev environment
- `livebook.sh` script for attached-mode Livebook with EXLA/CUDA
- `ARCHITECTURE_ROADMAP.md` tracking remaining architectures by priority tier
- `.credo.exs` configuration and `CONTRIBUTING.md` with architecture addition guide

Notebooks (12 Livebook notebooks):

- Architecture zoo guided tour
- Architecture comparison (decision boundaries)
- Sequence modeling (RNN vs SSM vs Transformer)
- MLP training end-to-end walkthrough
- Graph classification (GCN vs GAT vs GIN)
- Generative models (VAE)
- Small language model (Transformer + Mamba char-level LM)
- Liquid neural networks
- LM architecture shootout
- Softmax shootout (Softmax vs SSMax vs Softpick)
- Guided tour demo with detailed ML explanations
- Notebook index with descriptions and categories

Documentation:

- 18 conceptual guide documents (up from 12) covering architecture taxonomy, ML foundations, learning path, meta-learning, and reading Edifice source
- Architecture landscape survey and research docs
- 100% moduledoc coverage across all 211 modules
- 100% `@spec` coverage on all public functions
- Typed `@type build_opt` for all `build/1` modules

Benchmarks:

- Full architecture sweep benchmark covering all families
- Training throughput and memory profile benchmarks
- GPU runtime warmup phases for accurate measurements

Testing:

- 2822+ tests (up from ~1160 in v0.1.0)
- Gradient smoke tests with JIT-wrapped `value_and_grad`
- Parameter sensitivity tests and EXLA.Backend variants for conv models
- Dialyzer added to CI, zero warnings enforced

### Enhanced

- Decoder-Only transformer: added `:attention_type` option, iRoPE (interleaved RoPE) support
- MultiHead and GQA attention: added `:rope` option for built-in RoPE integration
- TTT (Test-Time Training): added `:variant` option for `:linear` and `:mlp` inner models
- TransformerBlock: added `:custom_ffn` callback for non-standard feed-forward networks
- xLSTM: added `:mlstm` registry alias (`Edifice.build(:mlstm, opts)`)
- sLSTM: log-domain stabilization (m_t state), recurrent connections (R*h_{t-1}), proper normalization (max(|n_t|, 1))
- MoE v2: aux-loss-free load balancing via bias mode
- DiffTransformer: simplified V2 with scalar lambda and RMSNorm only
- Liquid Neural Networks: exact analytical ODE solver added, set as default
- API option names normalized across all modules for consistency

### Changed

- Removed unnecessary `require Axon` from 104 modules (Axon has zero macros)
- BitNet `bitlinear_impl` comment clarified: STE is implicit via Axon's param/callback architecture
- Removed broken `sliding_window` registry alias
- Dependency constraints tightened to match tested versions
- All 72 Credo warnings resolved across 41 files
- All Dialyzer errors resolved; strict formatting enforced
- Notebooks default to 10 epochs with EXLA optional and dual setup cells (standalone / attached mode)

### Fixed

- **EnCodec**: channels-first bug fixed across all conv/conv_transpose layers
- **Gaussian Splatting**: render pipeline rewritten for JIT/EXLA compatibility; `render_layer` arity mismatch resolved
- **Gradient smoke tests**: JIT-wrapped `value_and_grad` for conv model gradients; `put_nested` no longer destroys sibling params
- **MessagePassing**: aggregate batch axes added to `Nx.dot` for correct batched matrix multiplication; `global_pool` refactored for 100% coverage
- **RetNet**: corrected `recurrent_retention_step` batching
- **TTT**: paper-faithful initialization for numerical stability
- **FNet**: replaced `Nx.fft` with real DFT matrix multiply for compatibility; `Nx.real` taken after each FFT to avoid complex intermediates
- **RWKV**: fixed seq_len=1 compile failure; silenced Range warnings in parallel scans
- **sLSTM**: log-domain stabilization for numerical stability
- **MoE routing**: top-k uses `Nx.top_k` with one-hot mask; hash routing properly selects expert; Switch MoE uses straight-through top-1 selection
- Paper-faithfulness corrections across 8 architecture modules
- 5 GPU test failures resolved in capsule and conv gradient tests
- VAE training fixed (single Axon graph); graph viz range bug resolved
- FocalNet bench spec corrected to match flat-architecture API

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
