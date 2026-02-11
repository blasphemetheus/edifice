# Learning Path
> A guided tour through Edifice's 19 architecture families -- what to learn first, what builds on what, and where to go deep.

## How to Use This Guide

Edifice has 90+ architectures across 19 families. That's overwhelming if you try to learn them
all at once. This guide gives you a structured path through the families, organized so that
each step builds naturally on the previous ones. At each step, you get:

- **What it is** and why it matters
- **Prerequisites** -- what you should understand first
- **Key ideas** to focus on
- **Try it** -- a runnable Edifice example
- **Go deeper** -- links to the detailed architecture guide

You don't need to follow this linearly. If you know what problem you're solving, jump to the
relevant section using the [Problem Landscape](problem_landscape.md) as your map. But if
you're here to learn ML architectures from the ground up, start at Phase 1 and work through.

## Phase 1: The Fundamentals

These are the building blocks that everything else is built on. Learn these first.

### Step 1: Feedforward Networks (MLP)

**What:** The simplest neural network -- stacked dense layers with activations. Input goes in,
passes through layers, output comes out. No recurrence, no attention, no fancy routing.

**Why it matters:** MLPs appear *inside* almost every other architecture. The feed-forward
block in a transformer? An MLP. The classification head at the end of a vision model? An MLP.
The expert networks in Mixture of Experts? MLPs. Understanding MLPs means understanding the
fundamental building block.

**Key ideas:**
- Dense layers (matrix multiply + bias + activation)
- How depth (more layers) and width (more neurons per layer) affect capacity
- Dropout for regularization
- Residual connections for training deep networks

**Try it:**

```elixir
# A simple 3-layer MLP for tabular data
model = Edifice.Feedforward.MLP.build(
  input_size: 32,
  hidden_sizes: [128, 64, 16],
  activation: :relu,
  dropout: 0.1
)

{init_fn, predict_fn} = Axon.build(model)
params = init_fn.(Nx.template({1, 32}, :f32), Axon.ModelState.empty())
output = predict_fn.(params, Nx.broadcast(0.5, {4, 32}))
# => {4, 16}  -- 4 samples, 16 features from the last hidden layer
```

**Also explore:** KAN (Kolmogorov-Arnold Networks -- learnable activation functions instead of
fixed ones) and TabNet (attention-based feature selection for tabular data).

**Go deeper:** The MLP module docs cover residual connections and layer normalization options.

---

### Step 2: Convolutional Networks

**Prerequisites:** Step 1 (understand dense layers and activations)

**What:** Networks that use shared filters sliding across the input. Instead of every neuron
connecting to every input (dense), a convolutional filter looks at a small local region and
reuses the same weights across all positions.

**Why it matters:** Convolutions encode the insight that local patterns matter and can appear
anywhere in the input. This is why they dominate image processing and are widely used for
sequence modeling (TCN). The concept of "receptive field" -- how much input context a layer
can see -- carries over to understanding attention and SSMs.

**Key ideas:**
- Filters/kernels: small weight matrices that slide across input
- Feature maps: the output of applying a filter
- Stride and padding: how the filter moves and handles edges
- Residual blocks (ResNet): the skip connection pattern that enables very deep networks
- Depthwise separable convolutions (MobileNet): factoring convolutions for efficiency

**Try it:**

```elixir
# ResNet for image-like data
model = Edifice.Convolutional.ResNet.build(
  input_channels: 3,
  num_classes: 10,
  depth: 18
)

{init_fn, predict_fn} = Axon.build(model)
params = init_fn.(Nx.template({1, 32, 32, 3}, :f32), Axon.ModelState.empty())
output = predict_fn.(params, Nx.broadcast(0.5, {2, 32, 32, 3}))
# => {2, 10}  -- 2 images, 10 class probabilities

# TCN for temporal sequences (causal convolutions)
model = Edifice.Convolutional.TCN.build(
  embed_size: 64,
  hidden_size: 128,
  num_layers: 4,
  kernel_size: 3,
  window_size: 100
)

{init_fn, predict_fn} = Axon.build(model)
params = init_fn.(Nx.template({1, 100, 64}, :f32), Axon.ModelState.empty())
output = predict_fn.(params, Nx.broadcast(0.5, {1, 100, 64}))
# => {1, 128}
```

**Go deeper:** [Convolutional Networks guide](convolutional_networks.md)

---

### Step 3: Building Blocks

**Prerequisites:** Steps 1-2

**What:** The composable primitives -- normalization (RMSNorm), position encoding (RoPE, ALiBi),
gating (SwiGLU), and patching (PatchEmbed) -- that appear inside transformers, SSMs, and
vision models.

**Why it matters:** When you read about a transformer using "pre-RMSNorm with RoPE and SwiGLU
FFN," you need to know what each of those pieces does. These blocks are the vocabulary of
modern architecture design.

**Key ideas:**
- Why normalization is essential (training stability)
- RMSNorm vs LayerNorm (speed vs mean centering)
- Position encoding: how networks know where tokens are in a sequence
- RoPE (rotary): relative position via rotation, good extrapolation
- ALiBi: no learned parameters, linear bias
- SwiGLU: gated feed-forward with multiplicative interactions

**Go deeper:** [Building Blocks guide](building_blocks.md)

---

## Phase 2: Sequence Processing

The three major approaches to processing ordered data. Understanding the tradeoffs between
these families is one of the most important skills in modern ML.

### Step 4: Recurrent Networks

**Prerequisites:** Phase 1

**What:** Networks that maintain a hidden state which is updated at each timestep. They process
sequences one token at a time, carrying forward a compressed summary of everything seen so far.

**Why it matters:** Recurrence is the most intuitive approach to sequences -- it mirrors how
you might mentally process a sentence word by word. Modern recurrent architectures (xLSTM,
MinGRU, Titans) have closed much of the gap with transformers while retaining constant-memory
inference.

**Key ideas:**
- Hidden state: the network's running memory
- Gates: mechanisms that control what to remember and what to forget
- LSTM/GRU: the classic gated architectures
- The vanishing gradient problem and how gates solve it
- Parallel scan: how MinGRU/MinLSTM make recurrence parallelizable for training

**Try it:**

```elixir
# Classic LSTM
model = Edifice.build(:lstm,
  embed_size: 64,
  hidden_size: 128,
  num_layers: 2,
  window_size: 60
)

{init_fn, predict_fn} = Axon.build(model)
params = init_fn.(Nx.template({1, 60, 64}, :f32), Axon.ModelState.empty())
output = predict_fn.(params, Nx.broadcast(0.5, {1, 60, 64}))
# => {1, 128}

# Modern minimal GRU (parallel-scannable)
model = Edifice.build(:min_gru,
  embed_size: 64,
  hidden_size: 128,
  num_layers: 4,
  window_size: 60
)
```

**Go deeper:** [Recurrent Networks guide](recurrent_networks.md)

---

### Step 5: Attention Mechanisms

**Prerequisites:** Phase 1, especially Building Blocks (Step 3)

**What:** A mechanism where each position in a sequence computes relevance scores against all
other positions, then aggregates information based on those scores. This is the core of the
transformer architecture.

**Why it matters:** Transformers (built on attention) are the dominant architecture for language
models and increasingly for other domains. Understanding attention -- and its quadratic cost --
is essential context for understanding why SSMs, linear attention, and hybrid architectures
exist.

**Key ideas:**
- Queries, keys, and values: the three projections
- Scaled dot-product attention: the core computation
- Multi-head attention: parallel attention with different learned perspectives
- The quadratic bottleneck: O(L^2) in sequence length
- Linear attention variants: approximating attention in O(L)
- Retention and RWKV: recurrence-based alternatives

**Try it:**

```elixir
# Standard multi-head attention transformer
model = Edifice.build(:attention,
  embed_size: 128,
  hidden_size: 256,
  num_heads: 8,
  num_layers: 4,
  window_size: 60
)

{init_fn, predict_fn} = Axon.build(model)
params = init_fn.(Nx.template({1, 60, 128}, :f32), Axon.ModelState.empty())
output = predict_fn.(params, Nx.broadcast(0.5, {1, 60, 128}))
# => {1, 256}

# RetNet: attention-like quality with recurrent inference
model = Edifice.build(:retnet,
  embed_size: 128,
  hidden_size: 256,
  num_heads: 4,
  num_layers: 4,
  window_size: 60
)
```

**Go deeper:** [Attention Mechanisms guide](attention_mechanisms.md)

---

### Step 6: State Space Models

**Prerequisites:** Steps 4 and 5 (understand both recurrence and attention tradeoffs)

**What:** Models that treat sequences as discretized continuous-time dynamical systems. A
hidden state evolves according to learned dynamics, combining the parallel training of
convolutions with the constant-memory inference of recurrence.

**Why it matters:** SSMs (especially Mamba) are the strongest alternative to transformers for
sequence modeling. They scale linearly with sequence length while matching or exceeding
transformer quality on many tasks. Understanding the SSM-attention tradeoff is crucial for
architecture selection.

**Key ideas:**
- State space equations: the continuous-time formulation
- Discretization: converting continuous to discrete for digital computation
- Selective SSMs (Mamba): input-dependent parameters
- Parallel scan: how linear recurrences train in parallel
- Hybrid models (Jamba, Zamba): SSM + attention for the best of both

**Try it:**

```elixir
# Mamba: the flagship SSM
model = Edifice.build(:mamba,
  embed_size: 128,
  hidden_size: 256,
  state_size: 16,
  num_layers: 4,
  window_size: 60
)

{init_fn, predict_fn} = Axon.build(model)
params = init_fn.(Nx.template({1, 60, 128}, :f32), Axon.ModelState.empty())
output = predict_fn.(params, Nx.broadcast(0.5, {1, 60, 128}))
# => {1, 256}

# Hybrid: Mamba + Attention
model = Edifice.build(:jamba,
  embed_size: 128,
  hidden_size: 256,
  num_layers: 6,
  attention_ratio: 0.33,  # 1/3 of layers use attention
  num_heads: 4,
  window_size: 60
)
```

**Go deeper:** [State Space Models guide](state_space_models.md)

---

## Phase 3: Specialized Domains

With the sequence processing fundamentals down, branch out into domain-specific families.

### Step 7: Vision Architectures

**Prerequisites:** Steps 2 (convolutions), 3 (building blocks), 5 (attention)

**What:** Architectures designed for image understanding, from Vision Transformers (ViT) that
treat images as sequences of patches, to U-Net for pixel-level segmentation.

**Key ideas:**
- Patch embedding: converting images into token sequences
- ViT: applying the transformer to vision
- Swin: hierarchical vision transformer with shifted windows
- U-Net: encoder-decoder with skip connections for dense prediction

**Try it:**

```elixir
# Vision Transformer
model = Edifice.Vision.ViT.build(
  image_size: 32,
  patch_size: 8,
  num_channels: 3,
  embed_size: 256,
  num_heads: 8,
  num_layers: 6,
  num_classes: 10
)

{init_fn, predict_fn} = Axon.build(model)
params = init_fn.(Nx.template({1, 32, 32, 3}, :f32), Axon.ModelState.empty())
output = predict_fn.(params, Nx.broadcast(0.5, {2, 32, 32, 3}))
# => {2, 10}
```

**Go deeper:** [Vision Architectures guide](vision_architectures.md)

---

### Step 8: Graph & Set Networks

**Prerequisites:** Step 1 (MLPs), basic understanding of attention helps

**What:** Architectures for data with relational structure (graphs) or no ordering (sets).
Graph networks propagate information along edges between nodes. Set networks process
unordered collections with permutation-invariant operations.

**Key ideas:**
- Message passing: nodes aggregate information from their neighbors
- Adjacency matrices: how graph structure is represented as tensors
- Permutation invariance: the output shouldn't change if you reorder the nodes/elements
- Pooling: going from node-level to graph-level predictions

**Try it:**

```elixir
# Graph attention network
model = Edifice.Graph.GAT.build_classifier(
  input_dim: 16,
  hidden_dims: [64, 64],
  num_classes: 3,
  num_heads: 4,
  pool: :mean
)

# DeepSets: process unordered collections
model = Edifice.Sets.DeepSets.build(
  input_dim: 3,
  hidden_dim: 64,
  output_dim: 10,
  pool: :mean
)

{init_fn, predict_fn} = Axon.build(model)
params = init_fn.(Nx.template({4, 20, 3}, :f32), Axon.ModelState.empty())
output = predict_fn.(params, Nx.broadcast(0.5, {4, 20, 3}))
# => {4, 10}  -- set-level predictions for 4 sets of 20 points
```

**Go deeper:** [Graph & Set Networks guide](graph_and_set_networks.md)

---

### Step 9: Generative Models

**Prerequisites:** Steps 1 and 5 (MLPs and attention); probability concepts from
[Core Vocabulary](core_vocabulary.md)

**What:** Architectures that learn to create new data. Instead of predicting a label, they
learn the distribution of the training data and can generate new samples from it.

**Key ideas:**
- Latent space and the encoder-decoder pattern (VAE)
- The reparameterization trick: making sampling differentiable
- Adversarial training (GAN): generator vs. discriminator
- Diffusion: adding noise then learning to reverse it
- Flow matching: learning ODE trajectories from noise to data

**Try it:**

```elixir
# Variational Autoencoder
{encoder, decoder} = Edifice.Generative.VAE.build(
  input_size: 784,
  latent_size: 32,
  encoder_sizes: [512, 256],
  decoder_sizes: [256, 512]
)

# Build encoder
{enc_init, enc_predict} = Axon.build(encoder)
enc_params = enc_init.(Nx.template({1, 784}, :f32), Axon.ModelState.empty())
%{mu: mu, log_var: log_var} = enc_predict.(enc_params, Nx.broadcast(0.5, {4, 784}))

# Sample from latent space
z = Edifice.Generative.VAE.reparameterize(mu, log_var)
# z shape: {4, 32}  -- 4 latent vectors
```

**Go deeper:** [Generative Models guide](generative_models.md)

---

### Step 10: Contrastive & Self-Supervised Learning

**Prerequisites:** Steps 1-2 (feedforward + convolutions), basic understanding of encoders

**What:** Methods that learn useful representations without labeled data. They create their
own training signal by comparing different views of the same data (contrastive) or by
reconstructing masked portions (self-supervised).

**Key ideas:**
- Positive and negative pairs: what to pull together, what to push apart
- Projection heads: small networks that transform representations for the contrastive objective
- Momentum encoders (BYOL): a slowly-updating copy of the network
- Masked autoencoders (MAE): reconstruct what you can't see

**Go deeper:** [Contrastive Learning guide](contrastive_learning.md)

---

## Phase 4: Advanced and Specialized

These families build on the foundations and address specific needs.

### Step 11: Meta-Learning (MoE, LoRA, Adapter, Capsules)

**Prerequisites:** Phase 2

**What:** Techniques that modify or compose other architectures: Mixture of Experts routes
different inputs to different sub-networks; LoRA and Adapters add small trainable modules to
frozen pretrained models; Capsules encode part-whole relationships.

**Why it matters:** MoE is how modern large language models scale to hundreds of billions of
parameters while keeping inference cost manageable. LoRA is how you fine-tune those models on
your specific task with limited compute.

**Go deeper:** [Meta-Learning guide](meta_learning.md)

---

### Step 12: Dynamic & Continuous Systems (NeuralODE, Liquid, Energy)

**Prerequisites:** Phase 2, comfort with the idea of differential equations

**What:** Architectures that model continuous dynamics: Neural ODEs define depth as a
continuous variable, Liquid Networks use ODE-based cells that adapt over time, Energy-Based
Models learn energy landscapes, and Hopfield networks provide associative memory.

**Go deeper:** [Dynamic & Continuous guide](dynamic_and_continuous.md)

---

### Step 13: Uncertainty & Memory (Bayesian, NTM, Evidential)

**Prerequisites:** Phase 1, basic probability

**What:** Networks that know what they don't know (Bayesian, MC Dropout, Evidential) and
networks with external memory banks (NTM, Memory Networks) for tasks requiring storage and
retrieval.

**Go deeper:** [Uncertainty & Memory guide](uncertainty_and_memory.md)

---

### Step 14: Neuromorphic (SNN, ANN2SNN)

**Prerequisites:** Phase 2

**What:** Spiking neural networks that communicate through discrete spikes rather than
continuous activations, inspired by biological neurons. ANN2SNN converts conventional networks
to spiking equivalents for deployment on neuromorphic hardware.

---

## The Dependency Graph

A visual summary of what builds on what:

```
Phase 1: Fundamentals
┌──────────────┐   ┌──────────────────┐   ┌─────────────────┐
│  1. MLP      │──→│ 2. Convolutional │──→│ 3. Blocks       │
│  (foundation)│   │   (local pattern)│   │   (components)  │
└──────┬───────┘   └────────┬─────────┘   └────────┬────────┘
       │                    │                       │
       ▼                    ▼                       ▼
Phase 2: Sequence Processing
┌──────────────┐   ┌──────────────────┐   ┌─────────────────┐
│ 4. Recurrent │   │ 5. Attention     │   │ 6. SSM          │
│  (sequential)│──→│  (parallel)      │──→│  (best of both) │
└──────────────┘   └──────────────────┘   └─────────────────┘
       │                    │                       │
       └────────────┬───────┴───────────────────────┘
                    ▼
Phase 3: Specialized Domains
┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌─────────────┐
│ 7. Vision│ │ 8. Graph │ │ 9. Generative│ │10. Contrast.│
└──────────┘ └──────────┘ └──────────────┘ └─────────────┘
                    │
                    ▼
Phase 4: Advanced
┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌─────────────┐
│11. Meta  │ │12. ODE/  │ │13. Uncertain.│ │14. Neuromor.│
│  (MoE)   │ │  Energy  │ │  & Memory    │ │  (Spiking)  │
└──────────┘ └──────────┘ └──────────────┘ └─────────────┘
```

## Suggested Exercises

For each phase, a concrete exercise to test your understanding:

**Phase 1:** Build an MLP classifier for a simple dataset (e.g., Iris or synthetic data).
Train it using Axon's training API. Add dropout and observe the effect on overfitting.

**Phase 2:** Take the same dataset but structure it as sequences. Compare LSTM, attention, and
Mamba on the same task. Measure parameter counts and observe output shapes.

**Phase 3:** Pick a domain that interests you:
- Vision: classify MNIST digits with both ResNet and ViT
- Graphs: classify synthetic graph structures with GCN
- Generation: train a VAE on a simple distribution and sample from it

**Phase 4:** Take your Phase 2 model and apply LoRA fine-tuning. Compare training a full model
from scratch vs. fine-tuning with LoRA -- observe the difference in trainable parameter count.

## Quick Reference: All 19 Families

| # | Family | Guide | Key Architecture | Core Idea |
|---|--------|-------|-----------------|-----------|
| 1 | Feedforward | - | MLP | Stacked dense layers |
| 2 | Convolutional | [Guide](convolutional_networks.md) | ResNet | Shared local filters + skip connections |
| 3 | Building Blocks | [Guide](building_blocks.md) | RMSNorm, RoPE, SwiGLU | Composable primitives |
| 4 | Recurrent | [Guide](recurrent_networks.md) | LSTM, xLSTM | Sequential hidden state |
| 5 | Attention | [Guide](attention_mechanisms.md) | Multi-Head, GQA | Pairwise relevance scoring |
| 6 | State Space | [Guide](state_space_models.md) | Mamba | Discretized dynamical system |
| 7 | Vision | [Guide](vision_architectures.md) | ViT, Swin | Images as patch sequences |
| 8 | Graph | [Guide](graph_and_set_networks.md) | GCN, GAT | Message passing on edges |
| 9 | Sets | [Guide](graph_and_set_networks.md) | DeepSets | Permutation-invariant aggregation |
| 10 | Generative | [Guide](generative_models.md) | VAE, Diffusion | Learn and sample from p(data) |
| 11 | Contrastive | [Guide](contrastive_learning.md) | SimCLR, BYOL | Learn representations without labels |
| 12 | Energy | [Guide](dynamic_and_continuous.md) | EBM, Hopfield | Energy landscape minimization |
| 13 | Liquid | [Guide](dynamic_and_continuous.md) | LiquidNN | Continuous-time ODE cells |
| 14 | Probabilistic | [Guide](uncertainty_and_memory.md) | Bayesian, Evidential | Calibrated uncertainty |
| 15 | Memory | [Guide](uncertainty_and_memory.md) | NTM | External differentiable memory |
| 16 | Meta | [Guide](meta_learning.md) | MoE, LoRA | Compose and adapt architectures |
| 17 | Neuromorphic | - | SNN | Spike-based communication |
| 18 | Capsule | [Guide](meta_learning.md) | Capsule | Part-whole relationships |
| 19 | Hypernetwork | [Guide](meta_learning.md) | Hypernetwork | Networks generating networks |
