# Edifice

[![Hex.pm](https://img.shields.io/hexpm/v/edifice.svg)](https://hex.pm/packages/edifice)
[![Hex Docs](https://img.shields.io/badge/hex-docs-blue.svg)](https://hexdocs.pm/edifice)
[![License](https://img.shields.io/hexpm/l/edifice.svg)](https://github.com/blasphemetheus/edifice/blob/main/LICENSE)

A comprehensive ML architecture library for Elixir, built on [Nx](https://github.com/elixir-nx/nx) and [Axon](https://github.com/elixir-nx/axon).

196 neural network architectures across 26 families — from MLPs to Mamba, transformers to graph networks, VAEs to spiking neurons, audio codecs to robotics, scientific ML to 3D generation.

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
    {:edifice, "~> 0.2.0"}
  ]
end
```

Edifice requires Nx ~> 0.10 and Axon ~> 0.8. For GPU acceleration, add EXLA:

```elixir
{:exla, "~> 0.10"}
```

> **Tip:** On Elixir 1.19+, set `MIX_OS_DEPS_COMPILE_PARTITION_COUNT=4` to compile dependencies in parallel (up to 4x faster first build).

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

196 architectures across 26 families, plus 20 shared building blocks. See the **[full architecture index](guides/architecture_index.md)** for every module.

| Family | Count | Highlights |
|--------|------:|------------|
| **Attention** | 36 | Multi-Head, GQA, MLA, Perceiver, RetNet, RWKV-7, GLA, Griffin, NSA, Conformer |
| **Generative** | 24 | VAE, GAN, Diffusion, DiT, MMDiT, Flow Matching, Transfusion, CogVideoX, MDLM |
| **Meta** | 23 | MoE, LoRA, DoRA, DPO, GRPO, Capsules, Speculative Decoding, QAT |
| **SSM** | 19 | Mamba, Mamba-2, Mamba-3, S4, Hyena, Hymba, Jamba, StripedHyena |
| **Recurrent** | 16 | LSTM, GRU, xLSTM, MinGRU, DeltaNet, TTT, Titans |
| **Vision** | 15 | ViT, DeiT, Swin, U-Net, ConvNeXt, MLP-Mixer, DINOv2, MambaVision |
| **Graph** | 9 | GCN, GAT, GIN, GraphSAGE, SchNet, EGNN |
| **Contrastive** | 8 | SimCLR, BYOL, MAE, VICReg, JEPA, SigLIP |
| **Convolutional** | 6 | ResNet, DenseNet, TCN, MobileNet, EfficientNet |
| **Feedforward** | 5 | MLP, KAN, KAT, TabNet, BitNet |
| **Transformer** | 4 | Decoder-Only, Multi-Token Prediction, BLT, Nemotron-H |
| **Audio** | 4 | EnCodec, VALL-E, SoundStorm, Whisper |
| **Detection** | 3 | DETR, RT-DETR, SAM 2 |
| **Energy** | 3 | EBM, Hopfield, Neural ODE |
| **Memory** | 3 | NTM, Memory Networks, Engram |
| **Probabilistic** | 3 | Bayesian, MC Dropout, Evidential |
| **Sets** | 2 | DeepSets, PointNet |
| **Robotics** | 2 | ACT, OpenVLA |
| **RL** | 2 | PolicyValue, Decision Transformer |
| **Interpretability** | 2 | Sparse Autoencoder, Transcoder |
| **Neuromorphic** | 2 | SNN, ANN2SNN |
| **+ 6 more** | 6 | Liquid NN, FNO, World Model, Medusa, Multimodal Fusion, Hybrid Builder |

## Guides

### New to ML?

Start here if you're new to machine learning. These guides build from zero to fluency with Edifice's API and architecture families.

1. **[ML Foundations](guides/ml_foundations.md)** — What neural networks are, how they learn, tensors and shapes
2. **[Core Vocabulary](guides/core_vocabulary.md)** — Essential terminology used across all guides
3. **[The Problem Landscape](guides/problem_landscape.md)** — Classification, generation, sequence modeling — which architectures solve which problems
4. **[Reading Edifice](guides/reading_edifice.md)** — The build/init/predict pattern, Axon graphs, shapes, and runnable examples
5. **[Learning Path](guides/learning_path.md)** — A guided tour through the architecture families

### Reference

- **[Architecture Index](guides/architecture_index.md)** — Full listing of all 196 architectures with modules and descriptions
- **[Architecture Taxonomy](guides/architecture_taxonomy.md)** — Paper references, strengths/weaknesses, adoption context, and gap analysis

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

See [`examples/`](https://github.com/blasphemetheus/edifice/tree/main/examples) for runnable scripts including `mlp_basics.exs`, `sequence_comparison.exs`, `graph_classification.exs`, `vae_generation.exs`, and `architecture_tour.exs`.

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

# Sample latent vector (requires PRNG key for stochastic sampling)
key = Nx.Random.key(42)
{z, _new_key} = Edifice.Generative.VAE.reparameterize(mu, log_var, key)

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
- Nx ~> 0.10
- Axon ~> 0.8
- Polaris ~> 0.1
- EXLA ~> 0.10 (optional, for GPU acceleration)

## License

MIT License. See [LICENSE](LICENSE) for details.
