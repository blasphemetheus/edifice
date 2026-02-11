# ML Foundations
> What neural networks are, how they learn, and why any of this works at all.

## Why This Guide Exists

Edifice gives you 90+ neural network architectures. Before you explore them, you need a mental
model of what a neural network actually *is* and what it means for one to "learn." This guide
builds that foundation from scratch. No prior ML knowledge is assumed -- just basic comfort with
the idea that numbers go in and numbers come out.

If you already know what a loss function is and can explain backpropagation at a high level,
skip to [Reading Edifice](reading_edifice.md) or the [Learning Path](learning_path.md).

## What Is a Neural Network?

A neural network is a function made of simple, stacked building blocks. Each block takes numbers
in, transforms them, and passes them forward. Stack enough of these blocks together with the
right transformations, and the network can approximate remarkably complex patterns.

The fundamental unit is the **neuron** (also called a node or unit). A neuron does three things:

```
1. Multiply each input by a weight        (how important is this input?)
2. Add all the weighted inputs together    (combine the evidence)
3. Apply an activation function            (introduce non-linearity)

Concretely:
  output = activation( w1*x1 + w2*x2 + ... + wN*xN + bias )
```

The weights and bias are the neuron's **parameters** -- the knobs the network adjusts during
learning. The activation function is what makes neural networks more powerful than simple linear
regression: without it, stacking layers would just produce another linear function, no matter
how many layers you add.

### Layers

Neurons are organized into **layers**. A layer is just a group of neurons that all process the
same inputs and produce their outputs together:

```
Input         Layer 1         Layer 2         Output
[x1] ──────→ [n1] ──────→ [n5] ──────→ [prediction]
[x2] ──╲ ╱→ [n2] ──╲ ╱→ [n6] ──────→
       ╳       ╳
[x3] ──╱ ╲→ [n3] ──╱ ╲→ [n7]
         ╲→ [n4] ──╱
```

The key insight: **each neuron in a layer connects to every neuron in the next layer** (in a
standard "dense" or "fully connected" layer). This means a layer with 4 neurons connecting to a
layer with 3 neurons has 4 × 3 = 12 weight parameters, plus 3 biases.

Three terms you'll see everywhere:

- **Input layer**: the raw data entering the network (not really a "layer" of neurons -- just the data)
- **Hidden layers**: the intermediate layers where the network builds up internal representations
- **Output layer**: the final layer that produces the prediction

A network with many hidden layers is called a **deep** neural network -- that's where "deep learning"
comes from. The depth is what gives these networks their power: early layers learn simple features,
and later layers combine those into increasingly abstract representations.

### The Forward Pass

When data flows from input to output through the network, that's called the **forward pass**.
Nothing mysterious -- it's just function composition. The output of layer 1 becomes the input
to layer 2, and so on:

```
input → layer_1(input) → layer_2(...) → layer_3(...) → prediction
```

Every architecture in Edifice -- whether it's a simple MLP, a transformer, a Mamba SSM, or a
graph network -- ultimately performs a forward pass. What differs is the *structure* of those
intermediate transformations. Some architectures look at sequences one token at a time (recurrent
networks). Some let every token attend to every other token (transformers). Some model the data
as continuous dynamical systems (state space models). But the forward pass concept is universal.

## What Does "Learning" Mean?

A neural network starts with random parameters. Its predictions are garbage. Learning is the
process of adjusting those parameters so the predictions get better.

This requires three ingredients:

### 1. A Loss Function

The **loss function** (also called cost function or objective) measures how wrong the network's
predictions are. It takes the network's output and the correct answer, and produces a single
number: the loss. Lower is better.

```
                    ┌────────────────┐
  network output ──→│  Loss Function  │──→ single number (the loss)
  correct answer ──→│                 │
                    └────────────────┘

Examples:
  - Predicting a number?  Loss = (predicted - actual)²
  - Classifying images?   Loss = -log(probability of correct class)
```

The choice of loss function tells the network what "better" means. Different problems use
different loss functions, and this choice shapes how the network learns.

### 2. Gradient Descent

Once we have a loss, we need a way to reduce it. **Gradient descent** is the core algorithm.
The idea is intuitive: the gradient tells you which direction increases the loss fastest, so
you step in the opposite direction.

```
Think of it like descending a mountain in fog:
  - You can't see the valley, but you can feel the slope under your feet
  - At each step, you move in the steepest downhill direction
  - Eventually you reach a low point

The "slope" is the gradient -- the derivative of the loss with respect to each parameter.
The "step size" is the learning rate -- how far you move each update.

  new_weight = old_weight - learning_rate × gradient
```

A small learning rate means slow, careful progress. A large learning rate means faster movement
but with the risk of overshooting the valley entirely. Choosing the right learning rate is one
of the most impactful decisions in training.

### 3. Backpropagation

**Backpropagation** is how the network figures out each parameter's gradient. It's just the
chain rule from calculus, applied systematically backward through the network:

```
Forward:   input → layer_1 → layer_2 → layer_3 → prediction → loss
Backward:  input ← layer_1 ← layer_2 ← layer_3 ← prediction ← loss
                                                                 ↑
                                                         "how does each
                                                          weight affect
                                                          this loss?"
```

For each weight in the network, backpropagation computes: "if I increase this weight by a
tiny amount, how much does the loss change?" Weights that contribute a lot to the error get
large gradients (big updates). Weights that barely affect the loss get small gradients (small
updates). This is what makes learning efficient -- the network focuses its adjustments where
they matter most.

You don't need to implement backpropagation yourself. Nx (the numerical computing library
under Edifice) handles this automatically through **automatic differentiation**. You define
the forward pass, and Nx computes all the gradients for you.

## The Training Loop

Training a neural network is a repetitive cycle:

```
repeat until good enough:
  1. Forward pass:    feed a batch of data through the network
  2. Compute loss:    measure how wrong the predictions are
  3. Backward pass:   compute gradients via backpropagation
  4. Update weights:  adjust parameters in the direction that reduces loss
```

One pass through the entire training dataset is called an **epoch**. In practice, you don't
feed the whole dataset at once -- you split it into **batches** (typically 32-512 samples) and
update weights after each batch. This is called **mini-batch gradient descent**, and it's what
virtually everyone uses because:

- Full-dataset gradient computation is too expensive for large datasets
- The noise from random batches actually helps escape shallow local minima
- It enables training on data that doesn't fit in memory

A typical training run might be 10-100 epochs, with hundreds or thousands of batch updates per
epoch.

## Tensors and Shapes

Neural networks operate on **tensors** -- multi-dimensional arrays of numbers. If you know what
a matrix is, a tensor is just the generalization to any number of dimensions:

```
Scalar:     42                          shape: ()        0 dimensions
Vector:     [1, 2, 3]                   shape: {3}       1 dimension
Matrix:     [[1, 2], [3, 4], [5, 6]]    shape: {3, 2}    2 dimensions
3D Tensor:  a stack of matrices          shape: {4, 3, 2} 3 dimensions
```

In Edifice and Nx, shapes are written as tuples. The most common shapes you'll encounter:

```
{batch_size, features}                     Tabular data or network output
{batch_size, sequence_length, features}    Sequences (text, time series, game frames)
{batch_size, height, width, channels}      Images
```

The **batch dimension** (always first) is how many samples the network processes simultaneously.
Processing samples in batches is more efficient than one at a time because modern hardware
(GPUs especially) is optimized for parallel operations on large blocks of numbers.

Understanding shapes is critical for working with Edifice. When you see something like
`{1, 60, 256}`, that means: 1 sample, 60 timesteps, 256 features per timestep. A Mamba model
with `embed_size: 256` and `window_size: 60` expects exactly that input shape.

## Generalization: The Actual Goal

The point of training isn't to memorize the training data -- it's to learn patterns that apply
to **new, unseen data**. This is called **generalization**, and it's the central challenge of
machine learning.

Two failure modes:

```
Underfitting                          Overfitting
───────────                          ──────────
Network is too simple or             Network memorizes the training data
undertrained to capture               but fails on new data.
the underlying pattern.

Training loss: high                   Training loss: very low
Test loss: high                       Test loss: high
"Can't learn the pattern"            "Learned the noise, not the signal"
```

Think of it like studying for a test. Underfitting is not studying enough -- you don't know
the material. Overfitting is memorizing specific practice problems without understanding the
concepts -- you ace the practice test but fail the real one.

Techniques for fighting overfitting (called **regularization**) include:

- **Dropout**: randomly zeroing out neurons during training, forcing the network to not rely on
  any single neuron
- **Weight decay**: penalizing large weights, encouraging simpler solutions
- **Early stopping**: stop training when performance on a held-out validation set starts to degrade
- **Data augmentation**: artificially expanding training data through transformations

## Why Architecture Matters

If all neural networks do the same basic thing (forward pass, loss, gradient descent), why do
we need so many different architectures?

Because **structure encodes assumptions about the data**. The right architecture builds in the
right biases for your problem:

```
Data Type          Key Property              Architecture Bias
─────────          ────────────              ─────────────────
Images             Spatial locality          Convolutions: share filters
                                             across positions

Sequences          Temporal ordering         Recurrence or attention:
                                             model dependencies over time

Graphs             Relational structure      Message passing: aggregate
                                             information from neighbors

Sets               Permutation invariance    Symmetric aggregation:
                                             order doesn't matter
```

A convolutional network "knows" that a cat's ear looks the same regardless of where it appears
in an image. A recurrent network "knows" that word order matters. A graph network "knows" that
nodes interact through edges. These structural biases mean the network needs less data and
less training to learn the pattern, because the architecture already encodes part of the answer.

This is why Edifice has 19 families -- each family encodes a different set of assumptions about
what the data looks like and how it should be processed.

## What's Next

With these foundations in place, you're ready for:

1. **[Core Vocabulary](core_vocabulary.md)** -- the precise terminology used across all Edifice guides
2. **[Problem Landscape](problem_landscape.md)** -- how different ML problems map to different architecture families
3. **[Reading Edifice](reading_edifice.md)** -- understanding the code patterns in this library
4. **[Learning Path](learning_path.md)** -- a guided tour through the 19 architecture families
