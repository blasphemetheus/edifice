# Edifice

A comprehensive ML architecture library for Elixir, built on [Nx](https://github.com/elixir-nx/nx) and [Axon](https://github.com/elixir-nx/axon).

44 neural network architectures across 14 families — from MLPs to Mamba, transformers to graph networks, VAEs to spiking neurons.

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

### Sequence Models

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **Mamba** | `Edifice.SSM.Mamba` | Parallel associative scan, selective state spaces |
| **Mamba-2 (SSD)** | `Edifice.SSM.MambaSSD` | Structured state space duality, chunk-wise |
| **S5** | `Edifice.SSM.S5` | MIMO diagonal SSM, simplified |
| **GatedSSM** | `Edifice.SSM.GatedSSM` | Gated temporal with gradient checkpointing |
| **Jamba** | `Edifice.SSM.Hybrid` | Mamba + Attention hybrid (configurable ratio) |
| **Zamba** | `Edifice.SSM.Zamba` | Mamba + single shared attention layer |
| **LSTM/GRU** | `Edifice.Recurrent` | Classic recurrent with multi-layer stacking |
| **xLSTM** | `Edifice.Recurrent.XLSTM` | Exponential gating, matrix memory (sLSTM/mLSTM) |
| **Reservoir** | `Edifice.Recurrent.Reservoir` | Echo State Networks with fixed random reservoir |

### Attention & Linear Attention

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **Multi-Head Attention** | `Edifice.Attention.MultiHead` | Sliding window, hybrid (LSTM+attn) |
| **RetNet** | `Edifice.Attention.RetNet` | Multi-scale retention, O(1) recurrent inference |
| **RWKV-7** | `Edifice.Attention.RWKV` | Linear attention, O(1) space, "Goose" architecture |
| **GLA** | `Edifice.Attention.GLA` | Gated Linear Attention with data-dependent decay |
| **HGRN-2** | `Edifice.Attention.HGRN` | Hierarchically gated linear RNN, state expansion |
| **Griffin/Hawk** | `Edifice.Attention.Griffin` | RG-LRU + local attention (Griffin) or pure RG-LRU (Hawk) |

### Feedforward & Convolutional

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **MLP** | `Edifice.Feedforward.MLP` | Residual connections, layer norm, temporal mode |
| **KAN** | `Edifice.Feedforward.KAN` | Learnable activation functions (B-spline, sine, Chebyshev) |
| **ResNet** | `Edifice.Convolutional.ResNet` | Residual/bottleneck blocks, configurable depth |
| **DenseNet** | `Edifice.Convolutional.DenseNet` | Dense connections, feature reuse |
| **TCN** | `Edifice.Convolutional.TCN` | Dilated causal convolutions for sequences |

### Generative Models

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **VAE** | `Edifice.Generative.VAE` | Reparameterization trick, KL divergence, beta-VAE |
| **VQ-VAE** | `Edifice.Generative.VQVAE` | Discrete codebook, straight-through estimator |
| **GAN** | `Edifice.Generative.GAN` | Generator/discriminator, WGAN-GP support |
| **Diffusion (DDPM)** | `Edifice.Generative.Diffusion` | Denoising diffusion, sinusoidal time embedding |
| **Flow Matching** | `Edifice.Generative.FlowMatching` | ODE-based generation, multiple loss variants |
| **Normalizing Flow** | `Edifice.Generative.NormalizingFlow` | Affine coupling layers (RealNVP-style) |

### Graph & Set Networks

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **GCN** | `Edifice.Graph.GCN` | Spectral graph convolutions (Kipf & Welling) |
| **GAT** | `Edifice.Graph.GAT` | Graph attention with multi-head support |
| **Message Passing** | `Edifice.Graph.MessagePassing` | Generic MPNN framework, global pooling |
| **DeepSets** | `Edifice.Sets.DeepSets` | Permutation-invariant set functions |
| **PointNet** | `Edifice.Sets.PointNet` | Point cloud processing with T-Net alignment |

### Energy, Probabilistic & Memory

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **EBM** | `Edifice.Energy.EBM` | Energy-based models, contrastive divergence |
| **Hopfield** | `Edifice.Energy.Hopfield` | Modern continuous Hopfield networks |
| **Bayesian NN** | `Edifice.Probabilistic.Bayesian` | Weight uncertainty, variational inference |
| **MC Dropout** | `Edifice.Probabilistic.MCDropout` | Uncertainty estimation via dropout at inference |
| **NTM** | `Edifice.Memory.NTM` | Neural Turing Machine, differentiable memory |
| **Memory Network** | `Edifice.Memory.MemoryNetwork` | End-to-end memory with multi-hop attention |

### Meta-Learning & Specialized

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **MoE** | `Edifice.Meta.MoE` | Mixture of Experts with top-k/hash routing |
| **Hypernetwork** | `Edifice.Meta.Hypernetwork` | Networks that generate other networks' weights |
| **Capsule** | `Edifice.Meta.Capsule` | Dynamic routing between capsules |
| **Liquid NN** | `Edifice.Liquid` | Continuous-time ODE dynamics (LTC cells) |
| **SNN** | `Edifice.Neuromorphic.SNN` | Leaky integrate-and-fire, surrogate gradients |

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
