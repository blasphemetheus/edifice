# Reading Edifice
> How to understand and use the code patterns in this library -- Axon computation graphs, the build API, tensor shapes, and running inference.

## What This Guide Covers

Every architecture in Edifice follows the same patterns. Once you understand these patterns,
you can pick up any of the 90+ architectures without re-learning the API. This guide walks
through those patterns with runnable examples.

**Prerequisites:** You should be comfortable with the concepts in
[ML Foundations](ml_foundations.md) and [Core Vocabulary](core_vocabulary.md). Familiarity with
basic Elixir syntax is helpful but not strictly required -- the patterns are simple enough to
follow even if you're new to the language.

## The Stack: Nx, Axon, and Edifice

Edifice sits on top of two foundational Elixir libraries:

```
┌─────────────────────────────────────┐
│           Edifice                    │  90+ architectures, consistent API
│  "What architecture do I want?"      │
├─────────────────────────────────────┤
│           Axon                       │  Model building, computation graphs
│  "How do layers connect?"            │
├─────────────────────────────────────┤
│           Nx                         │  Numerical computing, tensors, autograd
│  "How do I do math on tensors?"      │
├─────────────────────────────────────┤
│     EXLA (optional)                  │  GPU acceleration via XLA compiler
│  "Make it fast on GPU"               │
└─────────────────────────────────────┘
```

**Nx** is Elixir's numerical computing library. It provides tensors (multi-dimensional arrays),
mathematical operations, and automatic differentiation. Think of it as Elixir's equivalent of
NumPy + autograd.

**Axon** builds on Nx to provide a model-building API. You define a neural network as a
**computation graph** -- a description of how data flows through layers. The graph is then
compiled into efficient functions for initialization and prediction.

**Edifice** uses Axon to implement 90+ architectures with a consistent API. Instead of manually
wiring up attention heads, SSM blocks, and normalization layers, you call `Edifice.build/2` and
get a ready-to-use Axon model.

## The Build Pattern

Every architecture module in Edifice has a `build/1` function that returns an Axon model:

```elixir
# The universal pattern
model = SomeModule.build(option1: value1, option2: value2)
```

The model isn't a trained network -- it's a **computation graph** that describes the network's
structure. No weights exist yet. No computation has happened. It's a blueprint.

### Building by Module

You can use any architecture module directly:

```elixir
# Simple feedforward network
model = Edifice.Feedforward.MLP.build(input_size: 256, hidden_sizes: [512, 256])

# Mamba state space model
model = Edifice.SSM.Mamba.build(
  embed_size: 128,
  hidden_size: 256,
  state_size: 16,
  num_layers: 4,
  window_size: 60
)

# Graph convolutional network for classification
model = Edifice.Graph.GCN.build_classifier(
  input_dim: 16,
  hidden_dims: [64, 64],
  num_classes: 2,
  pool: :mean
)
```

### Building by Name (Registry)

The unified registry lets you build any architecture with an atom name:

```elixir
# Same Mamba model, built through the registry
model = Edifice.build(:mamba,
  embed_size: 128,
  hidden_size: 256,
  state_size: 16,
  num_layers: 4,
  window_size: 60
)

# Useful for config-driven experiments
arch_name = :retnet  # could come from a config file
model = Edifice.build(arch_name, embed_size: 256, hidden_size: 512, num_layers: 4)
```

You can explore what's available:

```elixir
# List all 90+ architecture names
Edifice.list_architectures()
# => [:adapter, :ann2snn, :attention, :barlow_twins, :bayesian, :bimamba, ...]

# See architectures grouped by family
Edifice.list_families()
# => %{
#   ssm: [:mamba, :mamba_ssd, :s4, :s4d, :s5, :h3, :hyena, ...],
#   attention: [:attention, :retnet, :rwkv, :gla, :hgrn, ...],
#   feedforward: [:mlp, :kan, :tabnet],
#   ...
# }

# Get the module behind a name
Edifice.module_for(:mamba)
# => Edifice.SSM.Mamba
```

## From Graph to Functions: Axon.build

An Axon model is just a graph. To actually run it, you compile it with `Axon.build/1`:

```elixir
model = Edifice.Feedforward.MLP.build(input_size: 10, hidden_sizes: [64, 32])

# Compile the graph into two functions
{init_fn, predict_fn} = Axon.build(model)
```

This gives you two functions:

- **`init_fn`**: creates the initial (random) parameters
- **`predict_fn`**: runs the forward pass

### Initializing Parameters

`init_fn` takes a **template** (a tensor describing the expected input shape) and an empty
model state:

```elixir
# Template: 1 sample, 10 features -- matches input_size: 10
template = Nx.template({1, 10}, :f32)

# Create random initial parameters
params = init_fn.(template, Axon.ModelState.empty())
```

The template doesn't contain real data -- it just tells Axon the shape and type of inputs to
expect so it can create parameters of the right sizes. `Nx.template/2` creates a placeholder
that takes no memory.

`params` is now an `Axon.ModelState` containing all the network's weights and biases, randomly
initialized. For a 2-layer MLP with sizes [64, 32], this includes:
- Layer 0: a {10, 64} weight matrix + a {64} bias vector
- Layer 1: a {64, 32} weight matrix + a {32} bias vector

### Running Inference

`predict_fn` takes parameters and input data, and runs the forward pass:

```elixir
# Create some input data: 4 samples, 10 features each
input = Nx.broadcast(0.5, {4, 10})

# Run the forward pass
output = predict_fn.(params, input)
# => a tensor of shape {4, 32} (4 samples, 32 features from the last hidden layer)
```

That's it. Three steps: **build** the graph, **init** the parameters, **predict** with data.

## Understanding Tensor Shapes

Shapes are how you reason about what's happening inside a network. Every Edifice architecture
documents its expected input and output shapes.

### Common Shape Patterns

```
{batch_size, features}
  Used by: MLP, classification heads, pooled outputs
  Example: {32, 256} = 32 samples, 256 features each

{batch_size, seq_len, features}
  Used by: Sequence models (Mamba, attention, recurrent, TCN)
  Example: {1, 60, 128} = 1 sample, 60 timesteps, 128 features per step

{batch_size, height, width, channels}
  Used by: Vision models (ViT, ResNet, UNet)
  Example: {16, 224, 224, 3} = 16 RGB images at 224x224

Map with named inputs
  Used by: Graph models (GCN, GAT)
  Example: %{"nodes" => {4, 10, 16}, "adjacency" => {4, 10, 10}}
```

### The Batch Dimension

The first dimension is **always** the batch size. When you see `{nil, 60, 128}` in an Axon
input specification, `nil` means "any batch size." The network doesn't care how many samples
you feed it at once.

```elixir
# These all work with the same model:
predict_fn.(params, Nx.broadcast(0.5, {1, 60, 128}))    # 1 sample
predict_fn.(params, Nx.broadcast(0.5, {32, 60, 128}))   # 32 samples
predict_fn.(params, Nx.broadcast(0.5, {256, 60, 128}))  # 256 samples
```

### Shape Transformations

Most Edifice sequence models output `{batch, hidden_size}` -- they reduce the sequence dimension
by taking the last timestep or pooling. This is because the common use case is classification or
regression from sequences, where you need a fixed-size output regardless of sequence length.

```elixir
# Mamba: sequence in, fixed vector out
model = Edifice.build(:mamba, embed_size: 128, hidden_size: 256, num_layers: 2, window_size: 60)
{init_fn, predict_fn} = Axon.build(model)
params = init_fn.(Nx.template({1, 60, 128}, :f32), Axon.ModelState.empty())
output = predict_fn.(params, Nx.broadcast(0.5, {1, 60, 128}))
# output shape: {1, 256}  -- the 60 timesteps have been reduced to a single vector
```

## Generative Models: The Tuple Pattern

Most architectures return a single Axon model. Generative architectures return **tuples** of
models because they have multiple components that are trained differently:

```elixir
# VAE returns an encoder and a decoder
{encoder, decoder} = Edifice.Generative.VAE.build(
  input_size: 784,
  latent_size: 32,
  encoder_sizes: [512, 256],
  decoder_sizes: [256, 512]
)

# Each is a separate Axon model
{enc_init, enc_predict} = Axon.build(encoder)
{dec_init, dec_predict} = Axon.build(decoder)

# GAN returns a generator and a discriminator
{generator, discriminator} = Edifice.Generative.GAN.build(
  latent_size: 128,
  output_size: 784,
  gen_sizes: [256, 512],
  disc_sizes: [512, 256]
)
```

Generative modules also provide associated utility functions for training:

```elixir
# VAE: reparameterization trick and KL divergence
z = Edifice.Generative.VAE.reparameterize(mu, log_var)
kl_loss = Edifice.Generative.VAE.kl_divergence(mu, log_var)
```

## Graph Models: Map Inputs

Graph models expect **maps** as input because graphs have multiple components (nodes, edges,
adjacency matrices):

```elixir
model = Edifice.Graph.GCN.build_classifier(
  input_dim: 16,
  hidden_dims: [64, 64],
  num_classes: 2,
  pool: :mean
)

{init_fn, predict_fn} = Axon.build(model)

# Graph input is a map with named tensors
input = %{
  "nodes" => Nx.broadcast(0.5, {4, 10, 16}),       # 4 graphs, 10 nodes, 16 features
  "adjacency" => Nx.eye(10) |> Nx.broadcast({4, 10, 10})  # adjacency matrices
}

params = init_fn.(
  %{
    "nodes" => Nx.template({4, 10, 16}, :f32),
    "adjacency" => Nx.template({4, 10, 10}, :f32)
  },
  Axon.ModelState.empty()
)

output = predict_fn.(params, input)
# output shape: {4, 2} -- 4 graphs, 2 class probabilities each
```

## Common Options Across Architectures

While each architecture has unique options, several appear across many modules:

| Option | Meaning | Typical Values |
|--------|---------|----------------|
| `embed_size` | Input feature dimension per token | 64, 128, 256, 512 |
| `hidden_size` | Internal representation width | 128, 256, 512, 1024 |
| `num_layers` | Depth of the network (stacked blocks) | 2, 4, 6, 8, 12 |
| `num_heads` | Number of attention heads | 4, 8, 16 |
| `window_size` | Expected sequence length | 60, 128, 512, 1024 |
| `dropout` | Dropout rate for regularization | 0.0, 0.1, 0.2 |
| `activation` | Activation function | `:relu`, `:silu`, `:gelu` |

**Larger values = more capacity** (can learn more complex patterns) but also **more compute
and more data needed** to train effectively.

## Putting It All Together: A Complete Example

Here's a full example showing the lifecycle from architecture selection to inference:

```elixir
# 1. Choose an architecture for sequence classification
#    We have 60-frame game state sequences with 128 features per frame
#    and want to classify into 5 actions
model = Edifice.build(:mamba,
  embed_size: 128,
  hidden_size: 256,
  state_size: 16,
  num_layers: 4,
  window_size: 60
)

# 2. Add a classification head on top
#    Edifice models output a feature vector; we need class probabilities
classifier =
  model
  |> Axon.dense(5, name: "action_head")
  |> Axon.activation(:softmax)

# 3. Compile the full model
{init_fn, predict_fn} = Axon.build(classifier)

# 4. Initialize parameters
template = Nx.template({1, 60, 128}, :f32)
params = init_fn.(template, Axon.ModelState.empty())

# 5. Run inference on a batch of game states
game_states = Nx.broadcast(0.5, {8, 60, 128})  # 8 sequences of 60 frames
predictions = predict_fn.(params, game_states)
# predictions shape: {8, 5} -- probability distribution over 5 actions for each sequence
```

Notice step 2: Edifice models are composable Axon graphs. You can pipe them into additional
layers, combine multiple Edifice models, or use Edifice layers as components in a larger
architecture. This composability is fundamental to the design.

## Comparing Architectures

Because every architecture follows the same API, swapping one for another is trivial:

```elixir
# Try several sequence models with the same input/output contract
architectures = [
  {:mamba, [embed_size: 128, hidden_size: 256, num_layers: 4, window_size: 60]},
  {:retnet, [embed_size: 128, hidden_size: 256, num_layers: 4, num_heads: 4, window_size: 60]},
  {:lstm, [embed_size: 128, hidden_size: 256, num_layers: 4, window_size: 60]},
  {:griffin, [embed_size: 128, hidden_size: 256, num_layers: 4, window_size: 60]}
]

for {name, opts} <- architectures do
  model = Edifice.build(name, opts)
  {init_fn, predict_fn} = Axon.build(model)
  params = init_fn.(Nx.template({1, 60, 128}, :f32), Axon.ModelState.empty())
  output = predict_fn.(params, Nx.broadcast(0.5, {1, 60, 128}))
  IO.puts("#{name}: output shape #{inspect(Nx.shape(output))}")
end
```

This is one of Edifice's core value propositions: the cost of trying a different architecture
is a one-line change.

## Reading Architecture Moduledocs

Every module in Edifice includes documentation you can access in IEx:

```elixir
# In IEx
h Edifice.SSM.Mamba           # Module overview
h Edifice.SSM.Mamba.build     # Build function options and return type
```

The moduledocs follow a consistent pattern:
1. One-line description of the architecture
2. ASCII diagram of the computation flow
3. Options with types and defaults
4. Usage examples with shapes annotated

## What's Next

With the API patterns understood, you're ready to explore architectures:

1. **[Learning Path](learning_path.md)** -- a guided tour through the 19 families in a logical order
2. Any architecture-specific guide (e.g., [State Space Models](state_space_models.md),
   [Attention Mechanisms](attention_mechanisms.md)) -- you now have the vocabulary and API
   knowledge to follow them
