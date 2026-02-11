# Convolutional Networks
> The foundational architecture family built on translation equivariance, local receptive fields, and parameter sharing -- from simple convolutions to compound-scaled efficient networks.

## Overview

Convolutional networks form the bedrock of deep learning for structured data. The core insight is deceptively simple: if a feature is useful at one spatial position, it is likely useful at other positions too. By sharing filter weights across all positions, convolutions achieve translation equivariance with far fewer parameters than fully-connected layers. A 3x3 convolutional filter uses 9 parameters regardless of whether the input is 32x32 or 1024x1024.

The Convolutional family in Edifice spans the full history of CNN design, from basic Conv blocks through the architecture innovations that defined a decade of deep learning. ResNet's skip connections (2015) solved vanishing gradients and enabled training of networks hundreds of layers deep. DenseNet (2016) pushed feature reuse further by connecting every layer to every subsequent layer. TCN (2018) adapted causal convolutions for sequence modeling, providing a parallelizable alternative to RNNs. MobileNet (2017) introduced depthwise separable convolutions for mobile deployment, and EfficientNet (2019) showed that systematically scaling depth, width, and resolution together yields better results than scaling any single dimension.

These six modules give you the vocabulary to build convolutional architectures for any task, from a simple three-layer classifier to a production-optimized network for edge devices. While transformer-based vision models (see vision_architectures.md) have matched CNNs on many benchmarks, convolutions remain the preferred choice when computational efficiency, hardware optimization, or small-data regimes are priorities.

## Conceptual Foundation

A discrete convolution applies a learned filter (kernel) by sliding it across the input and computing a dot product at each position. For a 1D input x with filter w of size K:

    y[t] = sum_{k=0}^{K-1} w[k] * x[t + k]

The receptive field -- the region of input that influences a given output position -- grows with network depth. For a stack of L layers each with kernel size K:

    receptive_field = L * (K - 1) + 1    (without dilation)

With dilated convolutions (dilation rate d), the receptive field grows exponentially:

    receptive_field = 1 + 2 * (K - 1) * (2^L - 1)    (TCN with doubling dilation)

This exponential growth is what makes TCNs competitive with RNNs for sequence modeling: four layers of dilated convolutions with kernel size 3 cover a receptive field of 31 timesteps, while eight layers cover 511.

## Architecture Evolution

```
  LeNet (1998)
      |
  AlexNet (2012) --- Deep convolutions + ReLU + dropout
      |
  VGGNet (2014) --- Deeper, smaller kernels (3x3 everywhere)
      |
      +---------- "How do we go deeper?"
      |
  ResNet (2015) ---------- Skip connections solve vanishing gradients
      |                        |
      |                   Bottleneck blocks
      |                   (1x1 -> 3x3 -> 1x1)
      |
  DenseNet (2016) -------- Feature reuse via concatenation
      |
      +---------- "How do we go efficient?"
      |
  MobileNet (2017) ------- Depthwise separable convolutions
      |                    (~8-9x fewer operations)
      |
  EfficientNet (2019) ---- Compound scaling (depth + width + resolution)
      |                    MBConv blocks with squeeze-excitation
      |
      +---------- "Can convolutions replace RNNs?"
      |
  TCN (2018) ------------- Dilated causal convolutions for sequences
                           Parallelizable, exponential receptive field
```

## When to Use What

| Scenario | Module | Why |
|----------|--------|-----|
| Simple feature extraction baseline | `Conv` | Configurable 1D/2D conv blocks with BN and activation |
| Deep image classification | `ResNet` | Skip connections enable 50-152+ layers; well-understood |
| Feature-rich tasks, small datasets | `DenseNet` | Dense connections maximize feature reuse; fewer parameters than ResNet at same accuracy |
| Temporal/causal sequence modeling | `TCN` | Parallelizable unlike RNNs; exponential receptive field; maintains causal ordering |
| Mobile or edge deployment | `MobileNet` | Depthwise separable convolutions minimize compute and model size |
| Optimal accuracy-efficiency trade-off | `EfficientNet` | Compound scaling finds the best balance of depth, width, and resolution |

### Choosing a Convolutional Architecture

```
What is your deployment target?
  |
  +-- Server/GPU (compute is cheap)
  |     |
  |     +-- Need maximum depth? --> ResNet-101/152 with bottleneck blocks
  |     +-- Need feature reuse? --> DenseNet-201
  |     +-- Want modern best practice? --> EfficientNet-B4+
  |
  +-- Mobile/Edge (compute is limited)
  |     |
  |     +-- MobileNet with width_multiplier < 1.0
  |     +-- EfficientNet-B0 (smallest compound-scaled model)
  |
  +-- Sequence data (temporal)
        |
        +-- Need causal processing? --> TCN
        +-- Need very long context? --> TCN with many layers
        +-- Compare with RNN? --> TCN (same task, parallelizable)
```

## Key Concepts

### Skip Connections and the Residual Learning Principle

ResNet's skip connections are arguably the most influential architectural idea in deep learning. Instead of learning a direct mapping H(x), each block learns a residual F(x) = H(x) - x, so the output is:

    y = F(x) + x

If the optimal mapping is close to identity (which is common in deep networks), learning a near-zero residual is far easier than learning an identity mapping through multiple nonlinear layers. This also provides a direct gradient path from loss to early layers, solving the vanishing gradient problem that limited pre-ResNet networks to roughly 20 layers.

```
    Input x ----+
        |       |
        v       |  (identity shortcut)
    [Conv-BN]   |
        |       |
      [ReLU]    |
        |       |
    [Conv-BN]   |
        |       |
        v       |
      F(x) + ---+  <-- addition
        |
      [ReLU]
        |
      Output
```

The bottleneck variant uses a 1x1 convolution to reduce channels, a 3x3 convolution in the reduced space, and another 1x1 to expand back. This is more parameter-efficient for deep networks: ResNet-50 uses bottleneck blocks (3 layers per block) and has fewer parameters than a hypothetical non-bottleneck ResNet-50 would.

### Dense Connectivity and Feature Reuse

DenseNet takes a different approach to deep network training. Instead of additive skip connections, each layer receives the concatenated feature maps of all preceding layers within its dense block:

```
    Layer 0: x_0
    Layer 1: x_0, f_1(x_0)
    Layer 2: x_0, f_1(x_0), f_2(x_0, f_1(x_0))
    Layer 3: x_0, f_1(x_0), f_2(...), f_3(...)
```

Each layer produces only a small number of new feature maps (the growth rate, typically 32). The total feature count at layer L is initial_channels + L * growth_rate. Transition layers between dense blocks use 1x1 convolutions to compress the accumulated features (by a compression factor, typically 0.5) and 2x2 average pooling to downsample spatially.

This design has two advantages: it encourages feature reuse (early features are directly available to all later layers) and it requires fewer parameters per layer since each layer only needs to produce growth_rate new features rather than the full channel count.

### Causal Convolutions for Sequences (TCN)

TCNs adapt convolutions for sequence modeling with a strict constraint: the output at time t must depend only on inputs at times t and earlier. This causal property is achieved through asymmetric (left-only) padding:

```
Dilation = 1:     Dilation = 2:     Dilation = 4:
  o   o   o         o   o   o         o   o   o
 /|\ /|\ /|\       / \ / \ / \       /   |   \
o o o o o o o     o   o   o   o     o    o    o    o    o
                                    ^-receptive field grows
                                     exponentially-^
```

Each temporal block doubles the dilation rate (1, 2, 4, 8, ...), so the receptive field grows exponentially with depth. TCN provides a `receptive_field/1` function and a `layers_for_receptive_field/2` function to help configure the network for your sequence length. Each temporal block contains two dilated causal convolutions with batch normalization, activation, and a residual skip connection.

### Efficiency-Oriented Designs

MobileNet's core contribution is the depthwise separable convolution, which factorizes a standard convolution into two steps: a depthwise convolution (one filter per input channel) and a pointwise 1x1 convolution (mixing channels). For a K x K convolution with C_in input and C_out output channels, this reduces operations from K^2 * C_in * C_out to K^2 * C_in + C_in * C_out -- roughly a K^2 factor reduction. The width multiplier parameter further scales all channel dimensions to trade accuracy for speed.

EfficientNet builds on MobileNet's MBConv blocks (inverted residual with squeeze-excitation) and adds compound scaling. Rather than independently scaling depth, width, or resolution, EfficientNet uses fixed scaling coefficients that scale all three together. The base model (B0) is found via neural architecture search, and larger variants (B1-B7) are derived by applying increasing compound scaling factors. In Edifice, the `depth_multiplier` and `width_multiplier` parameters control this scaling.

## Complexity Comparison

| Module | Param Efficiency | Compute Efficiency | Receptive Field | Typical Depth | Hardware Optimization |
|--------|-----------------|-------------------|-----------------|---------------|---------------------|
| Conv | Baseline | Baseline | Linear in depth | 3-10 layers | Excellent (cuDNN) |
| ResNet | Moderate (bottleneck helps) | Moderate | Linear in depth | 18-152 layers | Excellent |
| DenseNet | High (feature reuse) | Moderate (concat cost) | Linear in depth | 121-264 layers | Good |
| TCN | Moderate | Good (parallelizable) | Exponential (dilation) | 4-10 blocks | Good |
| MobileNet | Very high | Very high (depthwise sep.) | Linear in depth | 13-17 layers | Optimized for mobile |
| EfficientNet | Highest (compound scaled) | Highest (NAS-optimized) | Compound-scaled | Scaled by depth_mult | Good (SE overhead) |

## Module Reference

- `Edifice.Convolutional.Conv` -- Generic Conv1D/Conv2D building blocks with configurable kernels, batch normalization, activation, dropout, and optional max/avg pooling.
- `Edifice.Convolutional.ResNet` -- Residual networks with both standard residual blocks (2 convolutions) and bottleneck blocks (1x1, 3x3, 1x1); supports ResNet-18 through ResNet-152.
- `Edifice.Convolutional.DenseNet` -- Dense blocks where each layer receives all prior feature maps via concatenation; transition layers with compression and spatial downsampling.
- `Edifice.Convolutional.TCN` -- Temporal Convolutional Network with dilated causal convolutions, exponentially growing receptive fields, and residual blocks; includes receptive field calculation utilities.
- `Edifice.Convolutional.MobileNet` -- Depthwise separable convolution blocks with width multiplier for scaling; uses ReLU6 activation by default for mobile quantization.
- `Edifice.Convolutional.EfficientNet` -- Compound-scaled architecture with MBConv inverted residual blocks, squeeze-excitation attention, and configurable depth/width multipliers.

## Cross-References

- **vision_architectures.md** -- ConvNeXt modernizes this family's design patterns with transformer-era techniques (LayerNorm, GELU, inverted bottleneck); it lives in the Vision family but is spiritually a successor to ResNet.
- **state_space_models.md** -- H3 and similar state-space models use short depthwise convolutions as a component, connecting to the efficiency techniques pioneered by MobileNet.
- **recurrent_networks.md** -- TCN provides a direct alternative to RNNs for sequence modeling; benchmark comparisons typically show TCNs matching or exceeding LSTMs on many tasks while being fully parallelizable.

## Further Reading

1. He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016) -- introduces skip connections and the residual learning framework.
2. Huang et al., "Densely Connected Convolutional Networks" (CVPR 2017) -- dense connectivity for feature reuse and parameter efficiency.
3. Bai et al., "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (2018) -- demonstrates TCN competitiveness with RNNs.
4. Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" (2017) -- depthwise separable convolutions for efficient inference.
5. Tan and Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (ICML 2019) -- compound scaling method for balanced network scaling.
