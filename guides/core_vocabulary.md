# Core Vocabulary
> The essential terminology of machine learning, defined precisely and connected to how Edifice uses each concept.

## Why a Vocabulary Guide?

Every Edifice guide uses these terms. The existing architecture guides (attention mechanisms,
state space models, generative models, etc.) assume you already know what "embedding" and
"softmax" mean. This guide is your reference -- skim it now to build familiarity, then come
back whenever you hit a term you don't recognize.

Terms are grouped by how you encounter them: first the data, then the network, then training,
then evaluation.

## Data

### Features
The individual measurements or attributes that describe each data point. In an image, each pixel
value is a feature. In a game state, the player's x-position, y-position, damage percentage,
and action state are all features. Features are the raw inputs your network sees.

### Labels
The correct answers for supervised learning. If you're training a network to classify images of
cats and dogs, the label for each image is "cat" or "dog." If you're predicting a player's next
action, the label is the action they actually took. Not all ML tasks have labels -- generative
models and self-supervised methods learn from the data itself.

### Samples (Examples)
A single data point: one image, one game frame, one sentence. Your dataset is a collection of
samples, and each sample consists of features (input) and optionally a label (target output).

### Batch
A group of samples processed together in one forward pass. Instead of feeding the network one
sample at a time, you feed it a batch of 32, 64, or 256 samples simultaneously. This is faster
because GPUs are designed for parallel computation. In Edifice, the batch dimension is always the
first dimension of a tensor: `{batch_size, ...}`.

### Epoch
One complete pass through the entire training dataset. If your dataset has 10,000 samples and
your batch size is 100, one epoch = 100 batch updates. Training typically runs for many epochs.

### Dataset Split
Training data is divided into three parts:

```
Training set (80%)     What the network learns from
Validation set (10%)   Checked during training to detect overfitting
Test set (10%)         Evaluated once at the end to measure true performance
```

The network never trains on validation or test data. The validation set is your early warning
system -- if training loss keeps dropping but validation loss starts rising, the network is
overfitting.

## Network Architecture

### Parameters (Weights and Biases)
The learnable numbers in a network. **Weights** scale inputs (how much does this input matter?).
**Biases** shift the result (what's the baseline output regardless of input?). When we say a
network has "7 billion parameters," we mean 7 billion individually adjustable numbers. During
training, every parameter gets nudged a little bit each update step.

### Layer
A distinct processing step in the network. Each layer takes a tensor in and produces a tensor
out. Layers are the building blocks of all architectures. Common types:

- **Dense (fully connected)**: every input connects to every output
- **Convolutional**: a shared filter slides across the input
- **Attention**: each position weighs contributions from all other positions
- **Normalization**: rescales activations to stabilize training
- **Recurrent**: maintains a hidden state that carries information across timesteps

### Activation Function
A non-linear function applied after a layer's linear transformation. Without activations, a
stack of layers would collapse into a single linear function. Common activations:

```
ReLU:     max(0, x)              Most common. Simple, fast. Zero for negatives.
SiLU:     x * sigmoid(x)        Smoother than ReLU. Used in modern transformers.
GELU:     x * Φ(x)              Gaussian-weighted. Popular in language models.
Sigmoid:  1 / (1 + e^-x)        Squashes to (0, 1). Used for probabilities.
Tanh:     (e^x - e^-x)/(e^x + e^-x)  Squashes to (-1, 1). Used in gates.
Softmax:  e^xi / sum(e^xj)      Squashes a vector to a probability distribution.
```

You'll see "SiLU" and "GELU" frequently in Edifice's modern architectures. Older architectures
tend to use ReLU.

### Embedding
A learned mapping from discrete items (words, tokens, categories) to continuous vectors. Instead
of representing the word "cat" as an arbitrary integer ID like 4271, you represent it as a
256-dimensional vector of learned floating-point numbers. This lets the network discover that
similar things have similar vectors -- "cat" and "kitten" end up nearby in embedding space.

In Edifice, `embed_size` is one of the most common parameters. It determines the dimensionality
of these vector representations. Larger embeddings can capture more nuance but require more
computation.

### Hidden State
Internal information a network carries forward, either across layers or across timesteps.
In recurrent networks, the hidden state is explicitly maintained -- it's the network's "memory"
of what it has seen so far. In transformers, the concept is more implicit: the evolving
representations at each layer serve as the hidden state.

When Edifice architectures have a `hidden_size` parameter, it controls the width of these
internal representations. Larger hidden sizes give the network more capacity to represent
complex patterns.

### Residual Connection (Skip Connection)
A shortcut that adds a layer's input directly to its output:

```
output = layer(input) + input
```

This seemingly trivial change was revolutionary (ResNet, 2015). It solves the vanishing gradient
problem in deep networks: gradients can flow directly through the addition, bypassing the layer
entirely if needed. Nearly every modern architecture uses residual connections. When Edifice
guides mention "residual stream" or "skip connections," this is what they mean.

### Normalization
Rescaling activations to have consistent statistics (roughly zero mean, unit variance). Without
normalization, activations can grow or shrink exponentially through many layers, making training
unstable. You'll encounter several types in Edifice:

- **Layer Normalization**: normalizes across features within each sample
- **RMSNorm**: a faster variant that skips mean centering (used in most modern architectures)
- **Batch Normalization**: normalizes across the batch (common in CNNs, less so in transformers)

### Attention
A mechanism where each element in a sequence computes how relevant every other element is to it,
then aggregates information based on those relevance scores. This is the core operation in
transformers and the subject of an entire [Edifice guide](attention_mechanisms.md). The key
intuition: attention lets any token directly access any other token, regardless of distance.

### Encoder and Decoder
Two complementary roles in many architectures:

```
Encoder:  raw data → compressed representation (understanding)
Decoder:  compressed representation → output (generation)
```

An encoder takes high-dimensional input and distills it into a lower-dimensional representation
that captures the essential information. A decoder takes a representation and expands it back
into the output space. Some architectures use only an encoder (classification), some only a
decoder (generation), and some use both (translation, VAEs).

## Training

### Loss Function
A function that measures the distance between the network's prediction and the correct answer.
The choice of loss function defines what "correct" means. Common losses:

- **Mean Squared Error (MSE)**: `mean((predicted - actual)²)` -- for regression
- **Cross-Entropy**: `-sum(target * log(predicted))` -- for classification
- **KL Divergence**: measures how one probability distribution differs from another -- used in VAEs
- **Contrastive losses**: push similar things together and different things apart in embedding space

### Optimizer
The algorithm that updates parameters based on gradients. Gradient descent is the simplest
version, but in practice everyone uses more sophisticated optimizers:

- **SGD**: basic gradient descent with optional momentum
- **Adam**: adapts the learning rate per parameter based on gradient history (the default choice)
- **AdamW**: Adam with decoupled weight decay (current standard for transformers)

The optimizer is how the network actually learns. Each optimizer makes different tradeoffs
between speed, stability, and memory usage.

### Learning Rate
The step size for parameter updates. The single most important hyperparameter:

```
Too high:    loss oscillates or diverges (overshooting the valley)
Too low:     training is painfully slow (tiny steps)
Just right:  loss decreases steadily toward a good solution
```

Modern practice often uses a **learning rate schedule** that starts high and decreases over
training, or warms up from a low value and then decays.

### Gradient
The derivative of the loss with respect to a parameter. It tells you: "if I increase this
parameter slightly, how much does the loss change, and in which direction?" Gradients point
toward increasing loss, so you step in the opposite direction.

### Backpropagation
The algorithm for computing gradients efficiently by working backward from the loss through
the network. See the [ML Foundations](ml_foundations.md) guide for the intuition. In Edifice,
you never implement this -- Nx's automatic differentiation handles it.

### Hyperparameters
Settings that you choose *before* training, as opposed to parameters that are *learned during*
training. Examples: learning rate, batch size, number of layers, hidden size, dropout rate.
The options you pass to `Edifice.build(:mamba, embed_size: 256, num_layers: 4)` are
hyperparameters.

### Regularization
Techniques that prevent overfitting by constraining the network:

- **Dropout**: randomly set a fraction of neurons to zero during training
- **Weight decay**: add a penalty proportional to the magnitude of weights
- **Early stopping**: stop training when validation performance degrades

### Fine-Tuning
Taking a network that was trained on one task and continuing training on a different (usually
smaller) dataset. The pretrained weights provide a strong starting point. Edifice's LoRA and
Adapter modules (in the Meta family) are specifically designed for parameter-efficient fine-tuning,
where you freeze most of the pretrained weights and only train a small number of new parameters.

## Architecture-Specific Terms

These appear frequently across the Edifice guides:

### Sequence Length
The number of timesteps or tokens in a sequential input. A sentence of 20 words has sequence
length 20. A recording of 60 game frames has sequence length 60. Many architecture choices
(attention vs. SSMs vs. recurrence) are driven by how sequence length affects computation cost.

### Context Window
The maximum sequence length a model can process. Attention-based models have quadratic cost in
sequence length (doubling the context costs 4x), which is why efficient alternatives like SSMs
and linear attention exist. In Edifice, `window_size` or `max_seq_len` control this.

### Latent Space
A compressed, learned representation space. When an encoder maps a 784-dimensional image to a
32-dimensional vector, that 32-dimensional space is the latent space. Points that are close in
latent space should correspond to similar data. Generative models like VAEs sample from this
space to create new data.

### Token
The fundamental unit a sequence model operates on. In text, a token might be a word or a subword
piece. In vision, a token is an image patch. In game AI, a token is a frame's worth of state.
The idea is always the same: break continuous input into discrete chunks that the network
processes one at a time.

### Pooling
Aggregating a variable-length representation into a fixed-size one. After processing a
sequence of 60 tokens, you might need a single vector for classification. Common strategies:

- **Mean pooling**: average all token representations
- **Max pooling**: take the maximum across tokens per feature
- **Last token**: use only the final token's representation (common in causal models)
- **CLS token**: use a special learned token prepended to the sequence

### Projection
A linear transformation (matrix multiply + optional bias) that changes a tensor's feature
dimension. Used constantly in neural networks to map between different sizes:

```
Input:  {batch, seq, 256}
Dense:  W is {256, 512}
Output: {batch, seq, 512}
```

When Edifice guides say "project to hidden size," they mean a dense layer that maps from one
feature dimension to another.

## What's Next

With this vocabulary in hand, you can:

1. **[Problem Landscape](problem_landscape.md)** -- understand what types of problems exist and which architectures solve them
2. **[Reading Edifice](reading_edifice.md)** -- learn the code patterns that every architecture in this library follows
3. Start reading any architecture guide and look up terms here as needed
