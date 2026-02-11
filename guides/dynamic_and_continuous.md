# Dynamic and Continuous Architectures
> Models defined by energy functions, differential equations, and event-driven computation rather than discrete layer stacks.

## Overview

Most neural networks compute in discrete steps: input flows through layer 1, then layer 2, then layer 3, each applying a fixed transformation. The architectures in this guide break that pattern. They define computation through continuous dynamics -- energy landscapes to minimize, differential equations to integrate, or spike trains to accumulate. The output emerges from a physical process rather than a sequence of matrix multiplications.

This guide covers three families that share this continuous-process perspective. The Energy family (EBM, Hopfield, NeuralODE) defines models through scalar energy functions or continuous-time differential equations. The Liquid family provides biologically-inspired ODEs with adaptive time constants. The Neuromorphic family (SNN, ANN2SNN) uses discrete spikes as the unit of communication, mimicking biological neurons and enabling deployment on specialized hardware.

The unifying theme is that time and dynamics are first-class concepts. Where a standard MLP asks "what is the output given this input?", these architectures ask "where does this dynamical system converge?" or "what energy minimum does this configuration settle into?" This perspective opens up capabilities that feedforward networks lack: adaptive computation depth, continuous-time sequence processing, associative memory, and event-driven inference with extreme energy efficiency.

## Conceptual Foundation

The three families converge on a single mathematical idea: the state of the network evolves according to a rule that depends on the current state.

For energy-based models, the rule is gradient descent on an energy function:

```
dx/dt = -dE(x)/dx
```

The network learns E(x) such that low-energy configurations correspond to valid data. Hopfield networks are a special case where the energy has the form E(x) = -log(SUM exp(beta * x^T * y_i)), and the minimum-energy retrieval rule turns out to be exactly the softmax attention mechanism.

For ODE-based models (NeuralODE, Liquid), the dynamics are parameterized directly:

```
dx/dt = f(x, t; theta)        (NeuralODE)
dx/dt = (-x + f(x, I)) / tau  (Liquid Time-Constant)
```

NeuralODE learns the velocity field f, while Liquid networks add a learnable time constant tau that controls how quickly the system adapts to new inputs. Larger tau means slower decay and longer memory; smaller tau means faster response.

For spiking networks, the dynamics are integrate-and-fire:

```
dV/dt = -V/tau + I(t)         (leak + input current)
spike when V > threshold       (fire)
V <- V - threshold             (reset)
```

The spike is a discrete event in an otherwise continuous process. Information is encoded in spike timing and rates rather than continuous activations.

## Architecture Evolution

```
1982                  2016              2018               2020               2021
 |                     |                 |                  |                  |
 v                     v                 v                  v                  v

 Hopfield             SNN/LIF        NeuralODE           Modern           Liquid
 (Classical)          (Surrogate     (Chen et al.)       Hopfield         Time-Constant
  |                    Gradients)      |                 (Ramsauer)        Networks
  |                     |              |                  |                  |
  |                     |              |                  |                  |
  |                     |              |                  +--- Exponential    |
  |                     |              |                  |    storage       |
  |                     |              |                  |    capacity      |
  |                     |              |                  |                  |
  |                     |              +--- Adjoint       +--- Equivalent    +--- Adaptive
  |                     |              |    method for     |    to softmax    |    time
  |                     |              |    O(1) memory    |    attention     |    constants
  |                     |              |                   |                  |
  |                     +--- ANN2SNN   |                   |                  |
  |                     |    Conversion|                   |                  |
  v                     v              v                   v                  v

ASSOCIATIVE         EVENT-DRIVEN     CONTINUOUS-         ENERGY-BASED     BIOLOGICAL
MEMORY              COMPUTATION      DEPTH NETWORKS      ATTENTION        ODE DYNAMICS


                        EBM (Energy-Based Models)
                        |
                        +--- Contrastive divergence training
                        +--- Langevin dynamics sampling
                        +--- Scalar energy landscape
```

## When to Use What

| Criterion             | EBM              | Hopfield          | NeuralODE         | Liquid            | SNN               | ANN2SNN           |
|-----------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| **Core idea**         | Energy landscape  | Associative mem   | Continuous depth  | Adaptive ODE      | Spike trains      | ANN -> spike conv |
| **Dynamics**          | Gradient descent  | Softmax retrieval | dx/dt = f(x,t)   | dx/dt = -x/tau+f  | Integrate & fire  | Rate coding       |
| **Time concept**      | Implicit          | Single-step       | Continuous        | Continuous        | Discrete spikes   | Discrete spikes   |
| **Adaptive compute**  | Langevin steps    | No                | Solver-controlled | Solver-controlled | Timesteps         | Timesteps         |
| **Memory cost**       | Standard          | O(N*M) patterns   | O(1) via adjoint  | O(hidden * layers)| O(layers * T)     | Same as ANN       |
| **Hardware target**   | GPU               | GPU               | GPU               | GPU               | Neuromorphic      | Neuromorphic      |
| **Returns**           | Single model      | Single model      | Single model      | Single model      | Single model      | ANN or SNN model  |
| **Best for**          | Generative model, anomaly detection | Pattern retrieval, key-value lookup | Irregular time series, physics | Real-time adaptive sequences | Ultra-low power inference | Deploy existing models on neuromorphic chips |

**Quick selection guide:**
- Learning a probability distribution without explicit likelihood: **EBM** -- defines p(x) proportional to exp(-E(x)).
- Key-value associative memory with exponential capacity: **Hopfield** -- mathematically equivalent to attention.
- Irregularly sampled time series or variable-depth computation: **NeuralODE** -- the solver adapts step count to accuracy needs.
- Real-time sequence processing that must adapt to distributional shift: **Liquid** -- time constants enable automatic adaptation.
- Deploying on Intel Loihi or IBM TrueNorth: **SNN** or **ANN2SNN** -- event-driven computation at milliwatt power.
- Already have a trained ReLU network, need neuromorphic deployment: **ANN2SNN** -- direct weight transfer, no retraining.

## Key Concepts

### Energy Landscapes and Contrastive Divergence

An energy-based model learns a scalar function E(x; theta) over inputs. Training pushes energy down on observed data and up elsewhere. The contrastive divergence procedure alternates between two steps: (1) evaluate E on real data, (2) generate negative samples via Langevin dynamics (gradient descent on E with injected noise) and evaluate E on those. The loss is simply E(real) - E(negative) plus a regularizer to prevent energy magnitudes from diverging.

```
Energy
  ^
  |      *          *
  |     / \   *    / \
  |    /   \ / \  /   \
  |   /     V    V     \       * = high energy (unlikely)
  |  /   (data)  (data) \      V = low energy (likely data)
  | /                     \
  +-----------------------------> x
        Training pushes V down, * up
```

Langevin dynamics sampling follows the negative gradient of E with Gaussian noise injection. In the limit of many steps and small step sizes, it samples from the Boltzmann distribution p(x) proportional to exp(-E(x)). The Edifice EBM module provides `langevin_sample/4` with configurable step count, step size, and noise scale.

### Modern Hopfield Networks and the Attention Connection

Classical Hopfield networks (1982) store binary patterns and retrieve them by energy minimization, but can only store O(N) patterns in N neurons. Modern continuous Hopfield networks (Ramsauer et al., 2020) replace the quadratic energy with an exponential interaction, achieving exponential storage capacity.

The key insight is that the retrieval update rule of the modern Hopfield network is mathematically identical to the attention mechanism:

```
retrieval(X) = softmax(beta * X * Y^T) * Y
```

This is exactly attention with queries X, keys Y, values Y, and inverse temperature beta replacing the 1/sqrt(d_k) scaling. Higher beta produces sharper retrieval (nearest-neighbor), lower beta produces softer retrieval (averaging over patterns). The Edifice Hopfield module provides both single-layer retrieval and multi-head stacked architectures with residual connections.

### Continuous-Time Dynamics: NeuralODE and Liquid Networks

NeuralODE replaces the discrete residual block h_{t+1} = h_t + f(h_t) with its continuous-time limit dh/dt = f(h, t). The network learns the velocity field f, and integration is performed by a numerical ODE solver (Euler, RK4, or adaptive Dormand-Prince 4/5 in the Edifice implementation). This provides three advantages: constant memory via the adjoint method, adaptive computation depth controlled by solver tolerance, and a natural framework for irregularly-sampled time series.

Liquid Time-Constant networks extend this by adding a learnable time constant tau to each hidden unit:

```
dx/dt = (-x + activation) / tau
```

The time constant controls the decay rate. Units with large tau respond slowly and integrate over longer time windows; units with small tau respond quickly to recent inputs. This creates a heterogeneous temporal bank that automatically adapts to the input's temporal structure. The Edifice Liquid module supports four ODE solvers (Euler, midpoint, RK4, Dormand-Prince 4/5) selectable via the `:solver` option.

```
Input signal:  ___/\___/\/\/\____

Large tau:     ___/--\__/---\____    (slow response, smoothed)
Small tau:     ___/\___/\/\/\____    (fast response, tracks input)
```

### Spiking Networks and the ANN-to-SNN Bridge

Spiking neural networks communicate through discrete events (spikes) rather than continuous activations. A Leaky Integrate-and-Fire (LIF) neuron accumulates input current, leaks charge over time (controlled by membrane time constant tau), and emits a spike when the membrane potential exceeds a threshold. After spiking, the potential resets.

The primary challenge for training SNNs is that the spike function (Heaviside step) has zero gradient almost everywhere. The surrogate gradient approach replaces the true gradient with the derivative of a smooth approximation (sigmoid) during backpropagation, while keeping the hard spike during the forward pass. The Edifice SNN module uses this approach with a configurable surrogate slope parameter.

ANN2SNN provides a complementary path: train a standard ReLU network with backpropagation, then convert it to a spiking network by replacing ReLU activations with integrate-and-fire neurons. The time-averaged firing rate of an IF neuron converges to the ReLU activation value as the number of timesteps increases. Both the ANN (for training) and SNN (for inference) share identical weight names in Edifice, enabling direct parameter transfer.

```
ANN:  ReLU(W*x) = max(0, W*x) = 3.7

SNN (10 steps):  spike at t=2,5,8  -> rate = 3/10 = 0.3  (rough approx)
SNN (100 steps): spike at t=2,5,8,10,...  -> rate -> 3.7   (converges)
```

## Complexity Comparison

| Module       | Time Complexity        | Memory Complexity      | Params (typical)       | Training Method           |
|--------------|------------------------|------------------------|------------------------|---------------------------|
| EBM          | O(S * forward_pass)    | O(model + S * input)   | MLP params             | Contrastive divergence    |
| Hopfield     | O(B * N * M)           | O(N * M) patterns      | N*M + projections      | Standard backprop         |
| NeuralODE    | O(T * forward_pass)    | O(1) with adjoint      | Single dynamics net     | Backprop through solver   |
| Liquid       | O(L * T * S * forward) | O(L * hidden)          | 3 * hidden^2 per layer | Standard backprop         |
| SNN          | O(L * T * forward)     | O(L * hidden * T)      | Dense layer weights     | Surrogate gradient        |
| ANN2SNN      | O(L * forward) or O(L * T * forward) | Same as source ANN | Same as source ANN | Standard backprop (ANN phase) |

S = sampling/integration steps, T = timesteps, L = layers, N = num_patterns, M = pattern_dim, B = batch size.

## Module Reference

- `Edifice.Energy.EBM` -- Energy function network with contrastive divergence training and Langevin dynamics sampling. Outputs scalar energy per input. Includes `langevin_sample/4` and `contrastive_divergence_loss/3`.
- `Edifice.Energy.Hopfield` -- Modern continuous Hopfield network with exponential storage capacity. Provides both single-layer `hopfield_layer/2` and multi-head `build_associative_memory/1`. Includes `energy/3` for analysis.
- `Edifice.Energy.NeuralODE` -- Continuous-depth networks via Euler integration. Provides `build/1` (per-step dynamics) and `build_shared/1` (shared dynamics weights across steps).
- `Edifice.Liquid` -- Liquid Time-Constant networks with selectable ODE solvers (`:euler`, `:midpoint`, `:rk4`, `:dopri5`). Provides `build/1` and `build_with_ffn/1` for interleaved FFN variants. Includes `init_cache/1` for incremental inference.
- `Edifice.Neuromorphic.SNN` -- Spiking Neural Networks with LIF neurons and sigmoid surrogate gradients. Provides `lif_neuron/4` for single-step dynamics and `rate_decode/1` for spike-to-rate conversion.
- `Edifice.Neuromorphic.ANN2SNN` -- ANN-to-SNN conversion with shared weight names. Provides `build/1` for the trainable ANN and `build_snn/1` for the spiking inference model. Includes `if_neuron/4` for the integrate-and-fire simulation.

See also: [Generative Models](generative_models.md) for flow matching (ODE dynamics) and diffusion (SDE dynamics), [State Space Models](state_space_models.md) for SSMs as discretized continuous systems, [Recurrent Networks](recurrent_networks.md) for Liquid NNs compared to discrete-time RNNs.

## Further Reading

- Chen et al., "Neural Ordinary Differential Equations" (NeurIPS 2018) -- https://arxiv.org/abs/1806.07366
- Ramsauer et al., "Hopfield Networks is All You Need" (ICLR 2021) -- https://arxiv.org/abs/2008.02217
- Hasani et al., "Liquid Time-constant Networks" (AAAI 2021) -- https://arxiv.org/abs/2006.04439
- Neftci et al., "Surrogate Gradient Learning in Spiking Neural Networks" (IEEE Signal Processing 2019) -- https://arxiv.org/abs/1901.09948
- Du & Mordatch, "Implicit Generation and Modeling with Energy-Based Models" (NeurIPS 2019) -- https://arxiv.org/abs/1903.08689
