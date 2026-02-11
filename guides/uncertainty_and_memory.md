# Uncertainty, Memory, and Feedforward Foundations
> Three complementary threads: knowing what you don't know, reasoning over stored knowledge, and the building blocks everything else is built on.

## Overview

This guide covers three conceptual threads that cut across the rest of the Edifice library. The Probabilistic family (Bayesian, MCDropout, EvidentialNN) addresses a critical gap in standard neural networks: they produce confident predictions even on inputs far from their training distribution. These modules provide calibrated uncertainty estimates, essential for safety-critical deployments in medical diagnosis, autonomous systems, and financial modeling.

The Memory family (NTM, MemoryNetwork) augments neural networks with explicit external storage. Standard networks store knowledge implicitly in their weights, which limits their ability to perform algorithmic tasks or reason over large knowledge bases. Neural Turing Machines provide differentiable read/write operations on a memory matrix, while Memory Networks enable multi-hop attention over a set of memory slots for question answering and reasoning.

The Feedforward family (MLP, KAN, TabNet) provides the foundational architectures that other families build upon. The MLP is the universal approximation baseline. KAN replaces fixed activations with learnable functions grounded in the Kolmogorov-Arnold representation theorem. TabNet brings attention-based feature selection to tabular data, the most common data format in industry that tree-based methods have long dominated.

## Conceptual Foundation

The three threads share a common concern: making neural networks more trustworthy and capable as standalone reasoning systems rather than pattern matchers.

For uncertainty, the key equation is Bayes' rule applied to network weights:

```
p(W | D) = p(D | W) * p(W) / p(D)
```

The posterior p(W | D) over weights given data D is intractable for neural networks. Each method approximates it differently: Bayesian NNs learn a Gaussian q(W) = N(mu, sigma^2) via the reparameterization trick, MC Dropout uses dropout masks as an implicit variational family, and Evidential NNs place a Dirichlet prior over class probabilities to derive uncertainty in a single forward pass.

For memory, the foundation is differentiable addressing. The NTM's content-based addressing computes:

```
w_i = softmax(beta * cosine_similarity(key, memory[i]))
```

This allows gradient-based learning of when and what to read/write, making the memory fully differentiable and trainable end-to-end.

For feedforward networks, the Kolmogorov-Arnold representation theorem states that any multivariate continuous function can be written as:

```
f(x1, ..., xn) = SUM_q  Phi_q( SUM_p  phi_{q,p}(x_p) )
```

KAN operationalizes this by placing learnable univariate functions (parameterized as spline or sine bases) on the edges of the network graph, replacing the fixed activations of MLPs.

## Architecture Evolution

```
1943                    1986             2014              2015              2024
 |                       |                |                 |                 |
 v                       v                v                 v                 v

 Perceptron             Backprop         NTM              MemoryNetwork     KAN
 (McCulloch-Pitts)      + MLP            (Graves)         (Sukhbaatar)      (Liu et al.)
  |                      |                |                 |                 |
  |                      |                |                 |                 |
  |  +--- Universal Approximation        |                 |                 |
  |  |    Theorem (Hornik 1989)          |                 |                 |
  |  |                                   |                 |                 |
  |  +-- MLP baseline                   +-- Differentiable +-- Multi-hop    +-- Learnable
  |      (fixed activations)            |   read/write     |   attention     |   activations
  |                                     |   on external    |   over memory   |   on edges
  |                                     |   memory         |                 |
  |                                     |                  |                 |
  v                                     v                  v                 v

       +--- Bayesian NN (2015, Blundell)
       |    Weight distributions
       |
       +--- MC Dropout (2016, Gal)             +--- TabNet (2019, Arik)
       |    Cheap uncertainty via               |    Attention-based
       |    training-mode inference             |    feature selection
       |                                        |    for tabular data
       +--- Evidential NN (2018, Sensoy)        |
            Dirichlet priors for                |
            single-pass uncertainty             |
                                                |
UNCERTAINTY                MEMORY           FEEDFORWARD
QUANTIFICATION             AUGMENTATION     FOUNDATIONS
```

## When to Use What

### Uncertainty Methods

| Criterion              | Bayesian              | MCDropout             | EvidentialNN          |
|------------------------|-----------------------|-----------------------|-----------------------|
| **Core idea**          | Weight distributions  | Dropout as inference  | Dirichlet over classes|
| **Forward passes**     | 1 (stochastic)        | N (typically 20-30)   | 1 (deterministic)     |
| **Training change**    | ELBO loss + KL term   | None (standard train) | Evidential loss       |
| **Inference cost**     | ~1x (sample weights)  | ~Nx (multiple passes) | 1x (single pass)     |
| **Uncertainty types**  | Epistemic             | Epistemic + aleatoric | Epistemic + aleatoric |
| **OOD detection**      | Good                  | Moderate              | Strong                |
| **Calibration**        | Good with tuned prior | Good with tuned rate  | Good with KL annealing|
| **Best for**           | Full posterior approx | Retrofitting existing models | Fast OOD detection |

### Memory Methods

| Criterion              | NTM                   | MemoryNetwork         |
|------------------------|-----------------------|-----------------------|
| **Core idea**          | Differentiable R/W    | Multi-hop attention   |
| **Memory access**      | Content + location    | Content-based only    |
| **Write mechanism**    | Erase + add vectors   | Fixed (read-only)     |
| **Controller**         | LSTM                  | Dense projections     |
| **Turing-complete**    | In principle, yes     | No                    |
| **Best for**           | Algorithmic tasks (copy, sort) | QA, reasoning over facts |

### Feedforward Methods

| Criterion              | MLP                   | KAN                   | TabNet                |
|------------------------|-----------------------|-----------------------|-----------------------|
| **Core idea**          | Fixed activations     | Learnable activations | Feature selection     |
| **Activation**         | On nodes (ReLU, etc.) | On edges (sine/spline)| On features (masks)   |
| **Interpretability**   | Low                   | High (visualizable)   | High (feature masks)  |
| **Params scaling**     | O(n^2)                | O(n^2 * grid_size)    | O(n * hidden * steps) |
| **Data type**          | Any                   | Smooth functions      | Tabular               |
| **Residual support**   | Optional              | Built-in              | Via step aggregation   |
| **Best for**           | General baseline      | Scientific/symbolic   | Tabular classification|

**Quick selection guide -- Uncertainty:**
- Need uncertainty from an existing trained model without retraining: **MCDropout** -- just keep dropout on at inference and run N passes.
- Building a new model where uncertainty is critical: **Bayesian** -- provides the most principled posterior approximation.
- Need fast OOD detection in a single forward pass: **EvidentialNN** -- Dirichlet strength directly measures evidence.

**Quick selection guide -- Memory:**
- Learning algorithms (copy, sort, associative recall): **NTM** -- the write mechanism enables procedural memory.
- Question answering over a knowledge base: **MemoryNetwork** -- multi-hop attention refines the answer iteratively.

**Quick selection guide -- Feedforward:**
- Baseline or building block for larger architectures: **MLP** -- simple, fast, well-understood.
- Fitting smooth mathematical functions with interpretability: **KAN** -- learnable edge activations are visualizable and can recover symbolic expressions.
- Structured tabular data (the kind you would use XGBoost for): **TabNet** -- instance-wise feature selection competes with tree methods.

## Key Concepts

### Uncertainty Quantification: Why It Matters

A standard neural network outputs a softmax probability vector, but these probabilities are notoriously miscalibrated. A model can assign 99% confidence to an input it has never seen anything like. The three uncertainty methods address this by modeling the distribution over possible predictions rather than a single point estimate.

```
Standard NN:                     Uncertainty-aware NN:

  Input -> [0.95, 0.04, 0.01]     Input -> mean: [0.95, 0.04, 0.01]
  "95% confident, class A"                 var:  [0.001, 0.001, 0.001]
                                           "95% confident, LOW uncertainty"

  OOD Input -> [0.80, 0.15, 0.05]  OOD Input -> mean: [0.40, 0.35, 0.25]
  "80% confident, class A"                    var:  [0.15, 0.12, 0.10]
  (WRONG - overconfident)                     "Uncertain, HIGH variance"
                                              (CORRECT - flags for review)
```

**Bayesian NNs** model each weight as a Gaussian distribution N(mu, softplus(rho)^2). During each forward pass, weights are sampled via the reparameterization trick: W = mu + softplus(rho) * epsilon, where epsilon is drawn from N(0,1). Training maximizes the Evidence Lower Bound (ELBO), which balances data fit against KL divergence from a standard normal prior. The beta parameter (typically 1/num_batches) controls how strongly the prior regularizes.

**MC Dropout** requires no changes to training at all. The insight from Gal & Ghahramani is that a network trained with dropout is implicitly performing approximate Bayesian inference. By keeping dropout active at inference time and running N forward passes, each pass uses a different random subnetwork. The variance across predictions estimates epistemic uncertainty. The Edifice module provides `predict_with_uncertainty/4` that returns both mean and variance.

**Evidential NNs** take a different approach entirely. Instead of placing a distribution over weights, they place a Dirichlet distribution over class probabilities. The network outputs concentration parameters alpha_k (one per class), from which epistemic uncertainty is derived as u = K / SUM(alpha) -- when evidence is low, alpha values are close to 1, and uncertainty is high. This requires only a single forward pass.

### External Memory: Beyond Weight Storage

Standard neural networks store all learned knowledge in their weight matrices. This works well for pattern recognition but poorly for tasks requiring variable-length storage, random access, or sequential algorithmic steps. The Memory family provides explicit external storage accessible through differentiable operations.

The **Neural Turing Machine** maintains a memory matrix M of shape [N, M] (N memory slots, each of dimension M). An LSTM controller generates parameters for read and write heads that address memory through two complementary mechanisms:

```
Content Addressing:
  "Find the memory row most similar to this key"
  w_i = softmax(beta * cosine(key, M[i]))

Location Addressing (NTM paper):
  "Shift attention left/right from current position"
  w_shifted = circular_convolution(w, shift_kernel)

Combined:
  Interpolate content and location weights,
  then sharpen to prevent blur accumulation
```

The read operation computes a weighted sum over memory rows. The write operation uses an erase vector (what to forget) and an add vector (what to remember), both gated by the write attention weights. This enables the NTM to learn copy, sort, and recall algorithms from input-output examples alone.

**Memory Networks** use a simpler read-only protocol with multi-hop attention. Given a query q and a set of memory slots, each hop computes attention over the memories, reads a weighted sum, and adds it to the query. After K hops (typically 3), the refined query is projected to produce an answer. The power comes from the iterative refinement: hop 1 might identify relevant facts, hop 2 might chain them together, and hop 3 might extract the final answer.

```
Hop 1: "Where was the ball?"  -> attends to "John picked up the ball"
Hop 2: "Where is John?"       -> attends to "John went to the kitchen"
Hop 3: "Answer: kitchen"      -> combines evidence from both facts
```

### Feedforward Foundations: MLP, KAN, and TabNet

The **MLP** remains the most important architecture in deep learning -- not because it is the most powerful, but because it is the universal building block. Every transformer FFN layer, every projection head, every classification head is an MLP. The Edifice MLP module supports optional residual connections and layer normalization, which transform it from a basic network into a competitive architecture for many tasks. When residual connections are enabled with matching dimensions, each layer computes h + f(h), which is a single Euler step of an ODE (connecting back to the NeuralODE perspective from the [Dynamic and Continuous guide](dynamic_and_continuous.md)).

**KAN** challenges the MLP's fundamental design by moving learnable functions from nodes to edges. Where an MLP computes y = W2 * sigma(W1 * x) with a fixed sigma (ReLU, GELU, etc.), KAN computes y = SUM Phi_j(x_j) where each Phi_j is a learnable function parameterized as:

```
Phi(x) = w_base * SiLU(x) + w_spline * SUM sin(omega * x + phi)
```

The SiLU base activation ensures good gradient flow, while the sine terms (or Chebyshev polynomials, or RBF kernels) provide the learnable component. The grid_size parameter controls how many basis functions each edge uses -- higher values increase expressiveness at the cost of more parameters. KAN excels at learning smooth mathematical relationships and can sometimes recover the symbolic form of a function.

**TabNet** addresses a domain where neural networks have historically underperformed gradient-boosted trees: tabular data. The key innovation is sequential attention over input features. At each of several decision steps, TabNet learns a sparse mask that selects which features to process. A relaxation factor gamma controls how much previously attended features can be reused. The masks provide instance-wise feature importance, making the model inherently interpretable -- you can see exactly which features drove each prediction.

```
Step 1 mask: [1, 0, 1, 0, 0, 1, 0, 0]  -> Uses features 0, 2, 5
Step 2 mask: [0, 1, 0, 0, 1, 0, 0, 0]  -> Uses features 1, 4
Step 3 mask: [0, 0, 0, 1, 0, 0, 1, 0]  -> Uses features 3, 6
```

Each step produces a decision output; the final prediction aggregates all step decisions. This multi-step reasoning over different feature subsets is what enables TabNet to match or exceed tree-based methods on many tabular benchmarks.

## Complexity Comparison

| Module         | Forward Cost           | Params (typical)       | Inference Passes | Training Method           |
|----------------|------------------------|------------------------|------------------|---------------------------|
| Bayesian       | 2x MLP (mu + rho)     | 2x MLP (mu + rho)     | 1 (stochastic)   | ELBO maximization         |
| MCDropout      | 1x MLP                | 1x MLP                | N (default 20)   | Standard + dropout        |
| EvidentialNN   | ~1x MLP               | ~1x MLP + evidence    | 1                | Evidential loss           |
| NTM            | O(controller + N*M)   | LSTM + heads + N*M    | 1                | Backprop through addressing |
| MemoryNetwork  | O(K * N * D)          | K * 2 * projections   | 1                | Standard backprop         |
| MLP            | O(L * D^2)            | SUM(D_i * D_{i+1})    | 1                | Standard backprop         |
| KAN            | O(L * D^2 * G)        | O(L * D^2 * G)        | 1                | Standard backprop         |
| TabNet         | O(S * D * F)          | O(S * (F + D) * D)    | 1                | Standard backprop         |

L = layers, D = hidden dim, N = memory slots, M = memory dim, K = hops, G = grid size, S = steps, F = input features.

## Module Reference

- `Edifice.Probabilistic.Bayesian` -- Bayesian NN with reparameterized weight sampling. Provides `bayesian_dense/3` for building Bayesian layers, `kl_cost/3` for KL divergence, and `elbo_loss/4` for ELBO training.
- `Edifice.Probabilistic.MCDropout` -- Monte Carlo Dropout for uncertainty estimation. Provides `build_mc_layer/3` for always-on dropout layers and `predict_with_uncertainty/4` for N-pass inference. Includes `predictive_entropy/1`.
- `Edifice.Probabilistic.EvidentialNN` -- Evidential deep learning with Dirichlet priors. Outputs alpha parameters. Provides `uncertainty/1` (returns epistemic + aleatoric tuple), `expected_probability/1`, and `evidential_loss/3`.
- `Edifice.Memory.NTM` -- Neural Turing Machine with LSTM controller and content-based addressing. Provides `read_head/3`, `write_head/3`, and `content_addressing/3`. Takes both input and memory tensors.
- `Edifice.Memory.MemoryNetwork` -- End-to-end memory with multi-hop attention. Provides `memory_hop/3` for single hops and `build_multi_hop/3` for stacking. Takes query and memories inputs.
- `Edifice.Feedforward.MLP` -- Multi-layer perceptron with optional residual connections and layer normalization. Provides `build/1` (standalone), `build_temporal/1` (sequence last-frame), and `build_backbone/5` (composable).
- `Edifice.Feedforward.KAN` -- Kolmogorov-Arnold Networks with sine basis learnable activations. Provides `build_kan_layer/3` and `build_kan_block/2`. Includes `sine_basis/3`, `chebyshev_basis/2`, and `rbf_basis/3` for reference.
- `Edifice.Feedforward.TabNet` -- Attention-based tabular learning with instance-wise feature selection. Configurable step count and relaxation factor. Optional classification head via `:num_classes`.

See also: [Meta-Learning](meta_learning.md) for MoE, LoRA, and Adapter which build on feedforward blocks, [Attention Mechanisms](attention_mechanisms.md) for how MemoryNetwork's attention relates to transformer attention, [Dynamic and Continuous](dynamic_and_continuous.md) for Bayesian uncertainty in ODE-based models.

## Further Reading

- Blundell et al., "Weight Uncertainty in Neural Networks" (ICML 2015) -- https://arxiv.org/abs/1505.05424
- Gal & Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning" (ICML 2016) -- https://arxiv.org/abs/1506.02142
- Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty" (NeurIPS 2018) -- https://arxiv.org/abs/1806.01768
- Graves et al., "Neural Turing Machines" (2014) -- https://arxiv.org/abs/1410.5401
- Liu et al., "KAN: Kolmogorov-Arnold Networks" (2024) -- https://arxiv.org/abs/2404.19756
