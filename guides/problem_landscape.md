# The Problem Landscape
> Different problems have different shapes -- classification, regression, sequence modeling, generation, and structured prediction each demand different architectural choices.

## Why Problems Dictate Architecture

It's tempting to think there's one best neural network architecture. There isn't. The reason
Edifice has 19 families is that different problems have fundamentally different structures, and
the right architecture encodes the right structural assumptions. Choosing an architecture is
really about answering: "what does my data look like, and what do I need to predict?"

This guide maps the landscape of ML problems and connects each one to the Edifice families
designed for it. After reading this, you should be able to look at a new problem and have a
reasonable sense of which part of Edifice to reach for.

## The Five Problem Types

### 1. Classification

**Question:** "Which category does this belong to?"

You have an input and a fixed set of possible labels. The network outputs a probability
distribution over those labels.

```
Input:  an image, a game frame, a sentence, a molecule graph
Output: probabilities over classes

Examples:
  - "Is this image a cat or a dog?"                    (binary)
  - "Which of 10 digits is this?"                      (multiclass)
  - "What action should the player take?"              (multiclass)
  - "Which tags apply to this article?"                (multi-label)
```

**How it works:** The final layer has one neuron per class and applies softmax to produce
probabilities. Training uses cross-entropy loss, which heavily penalizes confident wrong
predictions.

**Edifice families for classification:**
- **Feedforward (MLP)**: simplest classifier for tabular data
- **Convolutional (ResNet, EfficientNet)**: image classification
- **Vision (ViT, Swin)**: image classification with transformers
- **Attention/SSM/Recurrent**: sequence classification (text sentiment, game state evaluation)
- **Graph (GCN, GAT)**: classify molecules, social network nodes

### 2. Regression

**Question:** "What number should this produce?"

Like classification, but the output is a continuous value (or vector of values) instead of a
category. Training uses mean squared error or similar distance-based losses.

```
Input:  features describing a state or situation
Output: one or more continuous numbers

Examples:
  - "What will the temperature be tomorrow?"           (scalar)
  - "What (x, y) position will this object move to?"   (vector)
  - "What is this house worth?"                        (scalar)
  - "What energy does this molecule have?"             (scalar, SchNet)
```

**How it works:** The final layer outputs raw numbers (no softmax). Loss measures the gap
between predicted and actual values.

**Edifice families for regression:**
- **Feedforward (MLP, KAN, TabNet)**: tabular regression
- **Graph (SchNet, PNA)**: molecular property prediction
- **Recurrent/SSM**: time-series forecasting

### 3. Sequence Modeling

**Question:** "Given this sequence so far, what comes next?"

This is the problem that drives language models, music generators, and game AI. The network
processes a sequence of tokens and predicts the next one (autoregressive) or fills in missing
ones (masked). This is technically classification at each timestep (choosing the next token from
a vocabulary), but the sequential structure demands specialized architectures.

```
Input:  a sequence of tokens (words, game frames, notes)
Output: prediction for the next token (or all tokens)

Examples:
  - "Given these words, what's the next word?"         (language modeling)
  - "Given these game frames, what controller input?"  (game AI)
  - "Given this melody so far, what's the next note?"  (music)
  - "Translate this sentence to another language"      (seq-to-seq)
```

**The core challenge:** sequences have temporal dependencies. Word 50 might depend on word 3.
Frame 120 might depend on what happened at frame 10. The architecture needs some mechanism to
carry or access information across time.

**Edifice families for sequence modeling:**

```
                     Sequence Modeling Approaches
                     ============================

Recurrent (LSTM, GRU, xLSTM, MinGRU, Titans)
  Process one token at a time, maintaining a hidden state.
  Constant memory per step, but sequential (can't parallelize training).
  Best for: streaming/online inference, moderate sequence lengths.

Attention (Multi-Head, GQA, Perceiver, RetNet)
  Every token attends to every other token simultaneously.
  Parallel training, but O(L²) cost in sequence length.
  Best for: tasks requiring precise long-range recall.

State Space Models (Mamba, S4, H3, Hyena)
  Model sequences as discretized dynamical systems.
  Parallel training AND constant-memory inference.
  Best for: long sequences where O(L²) attention is too expensive.

Linear Attention (LinearTransformer, Performer, GLA, RWKV, Griffin)
  Approximate attention with O(L) cost.
  Bridge between attention quality and SSM efficiency.
  Best for: when you need attention-like behavior at SSM-like cost.
```

The choice between these families is one of the most important architectural decisions, and it's
driven by your sequence length, whether you need autoregressive inference, and how much compute
you have. The architecture guides cover this tradeoff in depth.

### 4. Generation

**Question:** "Create new data that looks like the training data."

Generative models learn the underlying distribution of the data and can sample new examples
from it. This is fundamentally different from classification or regression -- there's no single
right answer. The network needs to capture the full range of variation in the data.

```
Input:  random noise, a text prompt, a conditioning signal
Output: a new image, audio clip, molecule, game trajectory

Examples:
  - "Generate a new face image"                        (unconditional)
  - "Create an image matching this description"        (conditional)
  - "Design a molecule with these properties"          (conditional)
  - "Generate realistic game trajectories"             (simulation)
```

**Edifice families for generation:**

```
                       Generative Paradigms
                       ====================

Latent Variable (VAE, VQ-VAE)
  Encode data to a compressed latent space, decode back.
  Smooth latent space enables interpolation and manipulation.
  Trade-off: smooth generation but sometimes blurry outputs.

Adversarial (GAN)
  Generator creates, discriminator judges. Minimax game.
  Sharp outputs but unstable training, mode collapse risk.

Diffusion (DDPM, DDIM, DiT, LatentDiffusion, ConsistencyModel)
  Gradually add noise to data, then learn to reverse the process.
  Current state of the art for image generation.
  Trade-off: high quality but slow sampling (many denoising steps).

Flow-Based (NormalizingFlow, FlowMatching, ScoreSDE)
  Learn invertible transformations between noise and data.
  Exact likelihood computation, principled training.
```

### 5. Structured Prediction

**Question:** "The input has non-trivial structure -- how do I respect it?"

Some data doesn't fit neatly into sequences or grids. Graphs have nodes and edges. Sets have
no ordering. Point clouds are unordered 3D coordinates. These require architectures that
respect the data's inherent symmetries.

```
Input:  a graph, a set of items, a point cloud
Output: node labels, graph labels, set-level prediction

Examples:
  - "Classify each atom in this molecule"              (node classification)
  - "Is this molecule toxic?"                          (graph classification)
  - "What's the aggregate property of this item set?"  (set regression)
  - "Segment this 3D scene"                            (point cloud)
```

**Edifice families for structured prediction:**
- **Graph (GCN, GAT, GIN, GraphSAGE, PNA, SchNet)**: data with relational structure
- **Sets (DeepSets, PointNet)**: unordered collections

The key property these architectures encode is **equivariance** or **invariance**:
- A graph network produces the same result regardless of how you number the nodes
- DeepSets produces the same result regardless of how you order the set elements
- PointNet handles arbitrary point orderings in 3D space

## Combining Problems

Real applications often combine multiple problem types:

```
Game AI (ExPhil):
  Sequence modeling  → process history of game frames
  Classification     → choose which button to press
  Regression         → choose stick position (continuous)

Image Captioning:
  Structured pred.   → encode image patches (ViT)
  Generation         → decode caption text autoregressively

Drug Discovery:
  Structured pred.   → encode molecular graph (SchNet)
  Regression         → predict binding energy
  Generation         → generate candidate molecules (VAE/Flow)
```

When you see a compound problem like this, you'll typically compose modules from different
Edifice families. Edifice's consistent API (everything returns an Axon model) makes this
composition natural.

## The Decision Map

When approaching a new problem, walk through this:

```
What is your data?
│
├─ Tabular (rows and columns, no structure)
│   └─ Feedforward: MLP, KAN, TabNet
│
├─ Sequential (ordered in time or position)
│   ├─ Short sequences (< 1K tokens)
│   │   └─ Attention or Recurrent
│   ├─ Long sequences (1K-100K+ tokens)
│   │   └─ SSM (Mamba) or Linear Attention
│   └─ Need both recall AND efficiency?
│       └─ Hybrid (Jamba, Zamba)
│
├─ Images
│   ├─ Classification / understanding
│   │   └─ Convolutional (ResNet, EfficientNet) or Vision (ViT, Swin)
│   ├─ Segmentation / dense prediction
│   │   └─ Vision: UNet, Swin
│   └─ Generation
│       └─ Generative: Diffusion, VAE, GAN
│
├─ Graphs / molecules
│   └─ Graph: GCN, GAT, SchNet, GraphTransformer
│
├─ Unordered sets / point clouds
│   └─ Sets: DeepSets, PointNet
│
└─ Want to generate new data?
    ├─ High quality images    → Diffusion (DiT, LatentDiffusion)
    ├─ Fast sampling needed   → ConsistencyModel, FlowMatching
    ├─ Smooth latent space    → VAE, VQ-VAE
    └─ Exact likelihood       → NormalizingFlow
```

## Supervised, Unsupervised, and Self-Supervised

One more axis to consider -- not what you're predicting, but what kind of training signal
you have:

**Supervised learning**: you have labeled data (input-output pairs). Classification and
regression are typically supervised. Most Edifice architectures can be used in supervised
settings.

**Unsupervised learning**: no labels. The network finds structure in the data on its own.
Generative models (VAE, GAN, Diffusion) are unsupervised -- they learn from the data
distribution itself. Clustering and dimensionality reduction also fall here.

**Self-supervised learning**: a clever middle ground. You create labels from the data itself
by hiding part of it and asking the network to predict the hidden part. Examples:
- Masked language modeling: hide words, predict them
- Contrastive learning: different views of the same image should have similar representations
- MAE: mask 75% of image patches, reconstruct them

The Contrastive family in Edifice (SimCLR, BYOL, BarlowTwins, MAE, VICReg) is entirely
self-supervised. These methods learn powerful representations without any human-provided labels.

**Reinforcement learning**: the network learns by interacting with an environment and receiving
rewards. This is how game AI (like ExPhil) trains after the initial behavioral cloning phase.
Edifice provides the architecture (the policy network), and a separate RL framework handles the
training loop.

## What's Next

Now that you can identify your problem type and narrow down which Edifice families to consider:

1. **[Reading Edifice](reading_edifice.md)** -- understand the code patterns so you can actually use the architectures
2. **[Learning Path](learning_path.md)** -- a guided tour through the families in a logical learning order
3. Jump to any specific architecture guide for the family you need
