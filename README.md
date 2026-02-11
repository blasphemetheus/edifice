# Edifice

A comprehensive ML architecture library for Elixir, built on [Nx](https://github.com/elixir-nx/nx) and [Axon](https://github.com/elixir-nx/axon).

103 neural network architectures across 19 families — from MLPs to Mamba, transformers to graph networks, VAEs to spiking neurons.

## Why Edifice?

The Elixir ML ecosystem has excellent numerical computing (Nx) and model building (Axon) foundations, but no comprehensive collection of ready-to-use architectures. Edifice fills that gap:

- **One dependency** for all major architecture families
- **Consistent API** — every architecture follows `Module.build(opts)` returning an Axon model
- **Unified registry** — `Edifice.build(:mamba, opts)` discovers and builds any architecture by name
- **Pure Elixir** — no Python, no ONNX imports, just Nx/Axon all the way down
- **GPU-ready** — works with EXLA/CUDA out of the box

## Installation

Add `edifice` to your dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:edifice, "~> 0.1.0"}
  ]
end
```

Edifice requires Nx ~> 0.9 and Axon ~> 0.7. For GPU acceleration, add EXLA:

```elixir
{:exla, "~> 0.9"}
```

## Quick Start

```elixir
# Build any architecture by name
model = Edifice.build(:mamba, embed_size: 256, hidden_size: 512, num_layers: 4)

# Or use the module directly for more control
model = Edifice.SSM.Mamba.build(
  embed_size: 256,
  hidden_size: 512,
  state_size: 16,
  num_layers: 4,
  window_size: 60
)

# Build and run
{init_fn, predict_fn} = Axon.build(model)
params = init_fn.(Nx.template({1, 60, 256}, :f32), Axon.ModelState.empty())
output = predict_fn.(params, input)

# Explore what's available
Edifice.list_architectures()
# => [:attention, :bayesian, :capsule, :deep_sets, :densenet, :diffusion, ...]

Edifice.list_families()
# => %{ssm: [:mamba, :mamba_ssd, :s5, ...], attention: [:attention, :retnet, ...], ...}
```

## Architecture Families

### State Space Models

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **S4** | `Edifice.SSM.S4` | HiPPO DPLR initialization, long-range memory |
| **S4D** | `Edifice.SSM.S4D` | Diagonal state space, simplified S4 |
| **S5** | `Edifice.SSM.S5` | MIMO diagonal SSM with D skip connection |
| **H3** | `Edifice.SSM.H3` | Two SSMs with multiplicative gating + short convolution |
| **Hyena** | `Edifice.SSM.Hyena` | Long convolution hierarchy, implicit filters |
| **Mamba** | `Edifice.SSM.Mamba` | Selective SSM, parallel associative scan |
| **Mamba-2 (SSD)** | `Edifice.SSM.MambaSSD` | Structured state space duality, chunk-wise matmul |
| **Mamba (Cumsum)** | `Edifice.SSM.MambaCumsum` | Mamba with configurable scan algorithm |
| **Mamba (Hillis-Steele)** | `Edifice.SSM.MambaHillisSteele` | Mamba with max-parallelism scan |
| **BiMamba** | `Edifice.SSM.BiMamba` | Bidirectional Mamba for non-causal tasks |
| **GatedSSM** | `Edifice.SSM.GatedSSM` | Gated temporal with gradient checkpointing |
| **Jamba** | `Edifice.SSM.Hybrid` | Mamba + Attention hybrid (configurable ratio) |
| **Zamba** | `Edifice.SSM.Zamba` | Mamba + single shared attention layer |

### Attention & Linear Attention

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **Multi-Head Attention** | `Edifice.Attention.MultiHead` | Sliding window, QK LayerNorm |
| **GQA** | `Edifice.Attention.GQA` | Grouped Query Attention, fewer KV heads |
| **Perceiver** | `Edifice.Attention.Perceiver` | Cross-attention to learned latents, input-agnostic |
| **FNet** | `Edifice.Attention.FNet` | Fourier Transform replacing attention |
| **Linear Transformer** | `Edifice.Attention.LinearTransformer` | Kernel-based O(N) attention |
| **Nystromformer** | `Edifice.Attention.Nystromformer` | Nystrom approximation of attention matrix |
| **Performer** | `Edifice.Attention.Performer` | FAVOR+ random feature attention |
| **RetNet** | `Edifice.Attention.RetNet` | Multi-scale retention, O(1) recurrent inference |
| **RWKV-7** | `Edifice.Attention.RWKV` | Linear attention, O(1) space, "Goose" architecture |
| **GLA** | `Edifice.Attention.GLA` | Gated Linear Attention with data-dependent decay |
| **HGRN-2** | `Edifice.Attention.HGRN` | Hierarchically gated linear RNN, state expansion |
| **Griffin/Hawk** | `Edifice.Attention.Griffin` | RG-LRU + local attention (Griffin) or pure RG-LRU (Hawk) |

### Recurrent Networks

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **LSTM/GRU** | `Edifice.Recurrent` | Classic recurrent with multi-layer stacking |
| **xLSTM** | `Edifice.Recurrent.XLSTM` | Exponential gating, matrix memory (sLSTM/mLSTM) |
| **MinGRU** | `Edifice.Recurrent.MinGRU` | Minimal GRU, parallel-scannable |
| **MinLSTM** | `Edifice.Recurrent.MinLSTM` | Minimal LSTM, parallel-scannable |
| **DeltaNet** | `Edifice.Recurrent.DeltaNet` | Delta rule-based linear RNN |
| **TTT** | `Edifice.Recurrent.TTT` | Test-Time Training, self-supervised at inference |
| **Titans** | `Edifice.Recurrent.Titans` | Neural long-term memory, surprise-gated |
| **Reservoir** | `Edifice.Recurrent.Reservoir` | Echo State Networks with fixed random reservoir |

### Vision

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **ViT** | `Edifice.Vision.ViT` | Vision Transformer, patch embedding |
| **DeiT** | `Edifice.Vision.DeiT` | Data-efficient ViT with distillation token |
| **Swin** | `Edifice.Vision.SwinTransformer` | Shifted window attention, hierarchical features |
| **U-Net** | `Edifice.Vision.UNet` | Encoder-decoder with skip connections |
| **ConvNeXt** | `Edifice.Vision.ConvNeXt` | Modernized ConvNet with transformer-inspired design |
| **MLP-Mixer** | `Edifice.Vision.MLPMixer` | Pure MLP with token/channel mixing |

### Convolutional

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **Conv1D/2D** | `Edifice.Convolutional.Conv` | Configurable convolution blocks |
| **ResNet** | `Edifice.Convolutional.ResNet` | Residual/bottleneck blocks, configurable depth |
| **DenseNet** | `Edifice.Convolutional.DenseNet` | Dense connections, feature reuse |
| **TCN** | `Edifice.Convolutional.TCN` | Dilated causal convolutions for sequences |
| **MobileNet** | `Edifice.Convolutional.MobileNet` | Depthwise separable convolutions |
| **EfficientNet** | `Edifice.Convolutional.EfficientNet` | Compound scaling (depth, width, resolution) |

### Generative Models

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **VAE** | `Edifice.Generative.VAE` | Reparameterization trick, KL divergence, beta-VAE |
| **VQ-VAE** | `Edifice.Generative.VQVAE` | Discrete codebook, straight-through estimator |
| **GAN** | `Edifice.Generative.GAN` | Generator/discriminator, WGAN-GP support |
| **Diffusion (DDPM)** | `Edifice.Generative.Diffusion` | Denoising diffusion, sinusoidal time embedding |
| **DDIM** | `Edifice.Generative.DDIM` | Deterministic diffusion sampling, fast inference |
| **DiT** | `Edifice.Generative.DiT` | Diffusion Transformer, AdaLN-Zero conditioning |
| **Latent Diffusion** | `Edifice.Generative.LatentDiffusion` | Diffusion in compressed latent space |
| **Consistency Model** | `Edifice.Generative.ConsistencyModel` | Single-step generation via consistency training |
| **Score SDE** | `Edifice.Generative.ScoreSDE` | Continuous SDE framework (VP-SDE, VE-SDE) |
| **Flow Matching** | `Edifice.Generative.FlowMatching` | ODE-based generation, multiple loss variants |
| **Normalizing Flow** | `Edifice.Generative.NormalizingFlow` | Affine coupling layers (RealNVP-style) |

### Contrastive & Self-Supervised

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **SimCLR** | `Edifice.Contrastive.SimCLR` | NT-Xent contrastive loss, projection head |
| **BYOL** | `Edifice.Contrastive.BYOL` | No negatives, momentum encoder |
| **Barlow Twins** | `Edifice.Contrastive.BarlowTwins` | Cross-correlation redundancy reduction |
| **MAE** | `Edifice.Contrastive.MAE` | Masked Autoencoder, 75% patch masking |
| **VICReg** | `Edifice.Contrastive.VICReg` | Variance-Invariance-Covariance regularization |

### Graph & Set Networks

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **GCN** | `Edifice.Graph.GCN` | Spectral graph convolutions (Kipf & Welling) |
| **GAT** | `Edifice.Graph.GAT` | Graph attention with multi-head support |
| **GIN** | `Edifice.Graph.GIN` | Graph Isomorphism Network, maximally expressive |
| **GraphSAGE** | `Edifice.Graph.GraphSAGE` | Inductive learning, neighborhood sampling |
| **Graph Transformer** | `Edifice.Graph.GraphTransformer` | Full attention over nodes with edge features |
| **PNA** | `Edifice.Graph.PNA` | Principal Neighbourhood Aggregation |
| **SchNet** | `Edifice.Graph.SchNet` | Continuous-filter convolutions for molecules |
| **Message Passing** | `Edifice.Graph.MessagePassing` | Generic MPNN framework, global pooling |
| **DeepSets** | `Edifice.Sets.DeepSets` | Permutation-invariant set functions |
| **PointNet** | `Edifice.Sets.PointNet` | Point cloud processing with T-Net alignment |

### Energy, Probabilistic & Memory

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **EBM** | `Edifice.Energy.EBM` | Energy-based models, contrastive divergence |
| **Hopfield** | `Edifice.Energy.Hopfield` | Modern continuous Hopfield networks |
| **Neural ODE** | `Edifice.Energy.NeuralODE` | Continuous-depth networks via ODE solvers |
| **Bayesian NN** | `Edifice.Probabilistic.Bayesian` | Weight uncertainty, variational inference |
| **MC Dropout** | `Edifice.Probabilistic.MCDropout` | Uncertainty estimation via dropout at inference |
| **Evidential NN** | `Edifice.Probabilistic.EvidentialNN` | Dirichlet priors for uncertainty |
| **NTM** | `Edifice.Memory.NTM` | Neural Turing Machine, differentiable memory |
| **Memory Network** | `Edifice.Memory.MemoryNetwork` | End-to-end memory with multi-hop attention |

### Meta-Learning & Specialized

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **MoE** | `Edifice.Meta.MoE` | Mixture of Experts with top-k/hash routing |
| **Switch MoE** | `Edifice.Meta.SwitchMoE` | Top-1 routing with load balancing |
| **Soft MoE** | `Edifice.Meta.SoftMoE` | Fully differentiable soft token routing |
| **LoRA** | `Edifice.Meta.LoRA` | Low-Rank Adaptation for parameter-efficient fine-tuning |
| **Adapter** | `Edifice.Meta.Adapter` | Bottleneck adapter modules for transfer learning |
| **Hypernetwork** | `Edifice.Meta.Hypernetwork` | Networks that generate other networks' weights |
| **Capsule** | `Edifice.Meta.Capsule` | Dynamic routing between capsules |
| **Liquid NN** | `Edifice.Liquid` | Continuous-time ODE dynamics (LTC cells) |
| **SNN** | `Edifice.Neuromorphic.SNN` | Leaky integrate-and-fire, surrogate gradients |
| **ANN2SNN** | `Edifice.Neuromorphic.ANN2SNN` | Convert trained ANNs to spiking networks |

### Building Blocks

| Block | Module | Key Feature |
|-------|--------|-------------|
| **RMSNorm** | `Edifice.Blocks.RMSNorm` | Root Mean Square normalization |
| **SwiGLU** | `Edifice.Blocks.SwiGLU` | Gated FFN with SiLU activation |
| **RoPE** | `Edifice.Blocks.RoPE` | Rotary position embedding |
| **ALiBi** | `Edifice.Blocks.ALiBi` | Attention with linear biases |
| **Patch Embed** | `Edifice.Blocks.PatchEmbed` | Image-to-patch tokenization |
| **Sinusoidal PE** | `Edifice.Blocks.SinusoidalPE` | Fixed sinusoidal position encoding |
| **Adaptive Norm** | `Edifice.Blocks.AdaptiveNorm` | Condition-dependent normalization (AdaLN) |
| **Cross Attention** | `Edifice.Blocks.CrossAttention` | Cross-attention between two sequences |

## Guides

### New to ML?

Start here if you're new to machine learning. These guides build from zero to fluency with Edifice's API and architecture families.

1. **[ML Foundations](guides/ml_foundations.md)** — What neural networks are, how they learn, tensors and shapes
2. **[Core Vocabulary](guides/core_vocabulary.md)** — Essential terminology used across all guides
3. **[The Problem Landscape](guides/problem_landscape.md)** — Classification, generation, sequence modeling — which architectures solve which problems
4. **[Reading Edifice](guides/reading_edifice.md)** — The build/init/predict pattern, Axon graphs, shapes, and runnable examples
5. **[Learning Path](guides/learning_path.md)** — A guided tour through the 19 architecture families

### Architecture Guides

Conceptual guides covering theory, architecture evolution, and decision tables for each family.

#### Sequence Processing

- **[State Space Models](guides/state_space_models.md)** — S4 through Mamba to hybrid architectures
- **[Attention Mechanisms](guides/attention_mechanisms.md)** — Quadratic to linear to Fourier to retention
- **[Recurrent Networks](guides/recurrent_networks.md)** — LSTM through xLSTM, MinGRU, TTT, and Titans

#### Representation Learning

- **[Vision Architectures](guides/vision_architectures.md)** — ViT, Swin, UNet, ConvNeXt, MLP-Mixer
- **[Convolutional Networks](guides/convolutional_networks.md)** — ResNet, DenseNet, MobileNet, TCN
- **[Contrastive Learning](guides/contrastive_learning.md)** — SimCLR, BYOL, BarlowTwins, MAE, VICReg
- **[Graph & Set Networks](guides/graph_and_set_networks.md)** — Message passing, spectral, invariance

#### Generative & Dynamic

- **[Generative Models](guides/generative_models.md)** — VAEs, GANs, diffusion, flows
- **[Dynamic & Continuous](guides/dynamic_and_continuous.md)** — ODE dynamics, energy landscapes, spiking

#### Composition & Enhancement

- **[Building Blocks](guides/building_blocks.md)** — RoPE vs ALiBi, RMSNorm, SwiGLU, composition
- **[Meta-Learning](guides/meta_learning.md)** — MoE, PEFT (LoRA/Adapter), capsules
- **[Uncertainty & Memory](guides/uncertainty_and_memory.md)** — Bayesian, NTM, MLP/KAN/TabNet foundations

## Examples

### Mamba for Sequence Modeling

```elixir
model = Edifice.SSM.Mamba.build(
  embed_size: 128,
  hidden_size: 256,
  state_size: 16,
  num_layers: 4,
  window_size: 100
)

{init_fn, predict_fn} = Axon.build(model)
params = init_fn.(Nx.template({1, 100, 128}, :f32), Axon.ModelState.empty())
output = predict_fn.(params, Nx.broadcast(0.5, {1, 100, 128}))
# => {1, 256}
```

### Graph Classification with GCN

```elixir
model = Edifice.Graph.GCN.build_classifier(
  input_dim: 16,
  hidden_dims: [64, 64],
  num_classes: 2,
  pool: :mean
)

{init_fn, predict_fn} = Axon.build(model)

params = init_fn.(
  %{
    "nodes" => Nx.template({4, 10, 16}, :f32),
    "adjacency" => Nx.template({4, 10, 10}, :f32)
  },
  Axon.ModelState.empty()
)

output = predict_fn.(params, %{
  "nodes" => Nx.broadcast(0.5, {4, 10, 16}),
  "adjacency" => Nx.eye(10) |> Nx.broadcast({4, 10, 10})
})
# => {4, 2}
```

### VAE with Reparameterization

```elixir
{encoder, decoder} = Edifice.Generative.VAE.build(
  input_size: 784,
  latent_size: 32,
  encoder_sizes: [512, 256],
  decoder_sizes: [256, 512]
)

# Encoder outputs mu and log_var
{init_fn, predict_fn} = Axon.build(encoder)
params = init_fn.(Nx.template({1, 784}, :f32), Axon.ModelState.empty())
%{mu: mu, log_var: log_var} = predict_fn.(params, Nx.broadcast(0.5, {1, 784}))

# Sample latent vector
z = Edifice.Generative.VAE.reparameterize(mu, log_var)

# KL divergence for training
kl_loss = Edifice.Generative.VAE.kl_divergence(mu, log_var)
```

### Permutation-Invariant Set Processing

```elixir
model = Edifice.Sets.DeepSets.build(
  input_dim: 3,
  hidden_dim: 64,
  output_dim: 10,
  pool: :mean
)

{init_fn, predict_fn} = Axon.build(model)
params = init_fn.(Nx.template({4, 20, 3}, :f32), Axon.ModelState.empty())
# Process sets of 20 3D points
output = predict_fn.(params, Nx.broadcast(0.5, {4, 20, 3}))
# => {4, 10}
```

## API Design

Every architecture module follows the same pattern:

```elixir
# Module.build(opts) returns an Axon model
model = Edifice.SSM.Mamba.build(embed_size: 256, hidden_size: 512)

# Some modules expose layer-level builders for composition
layer = Edifice.Graph.GCN.gcn_layer(nodes, adjacency, output_dim)

# Generative models may return tuples
{encoder, decoder} = Edifice.Generative.VAE.build(input_size: 784)

# Utility functions for training
loss = Edifice.Generative.VAE.loss(reconstruction, target, mu, log_var)
energy = Edifice.Energy.Hopfield.energy(query, patterns, beta)
```

The unified registry lets you build any architecture by name:

```elixir
# Useful for hyperparameter search, config-driven experiments
for arch <- [:mamba, :retnet, :griffin, :gla] do
  model = Edifice.build(arch, embed_size: 256, hidden_size: 512, num_layers: 4)
  # ... train and evaluate
end
```

## Requirements

- Elixir >= 1.18
- Nx ~> 0.9
- Axon ~> 0.7
- Polaris ~> 0.1
- EXLA ~> 0.9 (optional, for GPU acceleration)

## License

MIT License. See [LICENSE](LICENSE) for details.
