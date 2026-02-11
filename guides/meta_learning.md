# Meta-Learning and Conditional Computation
> Architectures that adapt their structure, routing, or parameters based on the input or task at hand.

## Overview

The Meta family in Edifice addresses a fundamental tension in neural network design: how to increase model capacity without proportionally increasing the computation required for every input. A model with billions of parameters can represent far richer functions than a small one, but running every parameter on every input is wasteful when different inputs need different kinds of processing.

This family contains three distinct groups of modules that approach this problem from different angles. The Mixture of Experts modules (MoE, SwitchMoE, SoftMoE) use conditional computation -- routing each input to a subset of specialist subnetworks. The parameter-efficient adaptation modules (LoRA, Adapter) solve the related problem of adapting large pretrained models without retraining all their weights. Finally, the meta-architecture modules (Hypernetwork, Capsule) take more radical approaches: Hypernetwork generates the weights of another network entirely, and Capsule replaces scalar activations with vector representations that encode part-whole relationships through iterative routing.

What unites these modules is that they all involve some form of dynamic, input-dependent computation -- whether that means selecting experts, generating weights, or iteratively refining routing coefficients between capsules.

## Conceptual Foundation

The core idea behind Mixture of Experts is that a gating function G(x) decides which of N expert networks E_1...E_N should process each input x:

    y = sum_i( G(x)_i * E_i(x) )    where G(x) is sparse (most entries zero)

When G selects only K << N experts per input, the model has N experts worth of capacity but only K experts worth of compute. The challenge is training G to route effectively without collapsing to always using the same few experts (the "rich get richer" problem), which is why load-balancing auxiliary losses are essential.

For parameter-efficient fine-tuning, LoRA decomposes the weight update into a low-rank product:

    W' = W_0 + (alpha/r) * B * A

where W_0 is frozen, A has shape [d_in, r], B has shape [r, d_out], and r << min(d_in, d_out). At rank r=8 on a 4096x4096 weight matrix, this reduces trainable parameters from 16M to 65K per layer -- a 250x reduction.

## Architecture Evolution

```
                        Conditional Computation
                               |
               +---------------+---------------+
               |                               |
         Hard Routing                    Soft Routing
               |                               |
      +--------+--------+                      |
      |                 |                      |
  Top-K MoE (2017)  Hash Routing         Soft MoE (2024)
      |                                  (fully differentiable,
      |                                   no token dropping)
      v
  Switch Transformer (2022)
  (top-1 routing, simplified,
   load balancing loss)


         Parameter Efficiency              Meta-Architecture
               |                               |
      +--------+--------+              +-------+-------+
      |                 |              |               |
   LoRA (2022)    Adapter (2019)   Hypernetwork    Capsule Net
  (low-rank        (bottleneck      (2016)          (2017)
   weight delta)    modules)        (generate       (dynamic routing
                                     weights)        by agreement)
```

## When to Use What

| Scenario | Module | Why |
|----------|--------|-----|
| Scale model capacity without proportional compute | `MoE` | Top-K routing activates only K of N experts per input |
| Maximize sparsity, simplest routing | `SwitchMoE` | Top-1 routing with load balancing is simplest to train |
| Avoid token dropping and routing instability | `SoftMoE` | Fully differentiable soft dispatch; every token reaches every expert |
| Fine-tune a large pretrained model cheaply | `LoRA` | Adds <1% trainable parameters; merges back at inference |
| Modular transfer across many tasks | `Adapter` | Bottleneck modules can be swapped per-task |
| Generate task-specific or input-conditioned networks | `Hypernetwork` | One network produces weights for another |
| Preserve spatial part-whole hierarchies | `Capsule` | Dynamic routing captures pose relationships lost by pooling |

### Choosing a Routing Strategy

```
Do you need differentiable routing?
  |
  +-- Yes --> SoftMoE (no load balancing needed, no token dropping)
  |
  +-- No
       |
       +-- How many active experts per token?
            |
            +-- Exactly 1 --> SwitchMoE (sparse, good load balance)
            |
            +-- 2 or more --> MoE with top_k routing
            |
            +-- Deterministic/reproducible --> MoE with :hash routing
```

## Key Concepts

### Expert Specialization and Load Balancing

In hard-routing MoE systems, a learned router assigns each token to its top-K experts. Without intervention, this creates a positive feedback loop: popular experts get more training signal, become even better, and attract even more tokens. The load-balancing auxiliary loss counteracts this by penalizing uneven expert utilization:

    aux_loss = alpha * N * sum(f_i * P_i)

where f_i is the fraction of tokens routed to expert i and P_i is the average router probability for expert i. A perfectly balanced router produces aux_loss close to 1.0. In Edifice, `MoE.compute_aux_loss/3` implements this directly.

SoftMoE sidesteps load balancing entirely. Instead of hard token-to-expert assignment, it computes dispatch weights D = softmax(X * Phi) and combine weights, creating soft token mixtures as expert inputs. Every expert processes a weighted combination of all tokens, and every token receives a weighted combination of all expert outputs. This eliminates token dropping but removes sparsity -- all experts run on all inputs.

### Low-Rank Adaptation (LoRA)

LoRA exploits the observation that weight updates during fine-tuning have low intrinsic rank. Instead of updating a full d_in x d_out matrix, LoRA learns two small matrices: A (d_in x r) initialized with small random values, and B (r x d_out) initialized to zero. At initialization, the LoRA delta is zero, so the model starts from the pretrained weights exactly. The scaling factor alpha/r controls the magnitude of the adaptation relative to the pretrained weights.

At inference time, the LoRA delta can be merged back into the original weights (W' = W_0 + BA), adding zero latency. Multiple LoRA adapters for different tasks can share the same base model and be hot-swapped.

### Adapter Modules

Adapters take a complementary approach to LoRA: instead of modifying existing weight matrices, they insert new bottleneck modules between frozen pretrained layers. Each adapter is a down-projection to a small bottleneck dimension, a nonlinearity, and an up-projection back to the original dimension, with a residual connection. The up-projection is initialized to zero so the adapter starts as an identity function.

The key difference from LoRA: adapters add sequential computation (extra layers in the forward pass), while LoRA adds parallel computation (a parallel path that merges with the original). Adapters also cannot be merged away at inference time, so they add a small latency cost.

### Hypernetworks and Capsule Routing

Hypernetworks are the most general form of conditional computation: rather than routing inputs to different experts, a hypernetwork generates the weights of a target network conditioned on some input. This enables a single hypernetwork to produce entirely different target networks for different tasks, inputs, or conditions. The trade-off is that generating weights for large target networks requires substantial hypernetwork capacity.

Capsule networks take a different approach to representation. Instead of scalar activations, capsules use vector outputs where the vector length represents entity probability and the direction represents entity properties (pose, deformation, texture). Dynamic routing iteratively refines the connections between capsule layers based on agreement: lower capsules send their predictions to higher capsules whose outputs most closely match those predictions. This routing-by-agreement mechanism captures part-whole relationships that standard architectures discard through pooling.

## Complexity Comparison

| Module | Trainable Params (relative) | Inference Cost | Training Stability | Merges at Inference |
|--------|---------------------------|----------------|-------------------|-------------------|
| MoE | N x expert_params (but K active) | Base + K/N overhead | Requires aux loss | N/A |
| SwitchMoE | N x expert_params (1 active) | Base + 1/N overhead | Good with load balance | N/A |
| SoftMoE | N x expert_params (all active) | N x expert_cost | Most stable | N/A |
| LoRA | r * (d_in + d_out) per layer | Zero (after merge) | Very stable | Yes |
| Adapter | bottleneck * 2 * d per layer | Small overhead | Very stable | No |
| Hypernetwork | Generator network params | Generator + target | Can be unstable | N/A |
| Capsule | Transform matrices + routing | Iterative routing overhead | Sensitive to routing iters | N/A |

## Module Reference

- `Edifice.Meta.MoE` -- Mixture of Experts with top-k, switch, soft, and hash routing strategies; supports FFN, GLU, and Mamba expert types.
- `Edifice.Meta.SwitchMoE` -- Switch Transformer with top-1 routing and softmax-weighted expert combination for sparse, efficient scaling.
- `Edifice.Meta.SoftMoE` -- Soft Mixture of Experts using differentiable dispatch-combine weights; no token dropping or load balancing needed.
- `Edifice.Meta.LoRA` -- Low-Rank Adaptation that decomposes weight updates into A and B matrices; supports standalone layers and wrapping existing dense layers.
- `Edifice.Meta.Adapter` -- Bottleneck adapter modules with down-project, activate, up-project, and residual; can wrap any existing layer.
- `Edifice.Meta.Hypernetwork` -- Weight-generating networks that produce target network parameters conditioned on an input; supports multi-layer targets.
- `Edifice.Meta.Capsule` -- Capsule Network with primary capsule layers, squash activation, and iterative dynamic routing by agreement.

## Cross-References

- **building_blocks.md** -- MoE experts commonly use SwiGLU or standard FFN blocks as their internal architecture; see the Blocks family for FFN design patterns.
- **attention_mechanisms.md** -- MoE layers typically replace the FFN sublayer in transformer blocks while keeping the attention sublayer unchanged; SwitchMoE and SoftMoE follow this pattern.
- **graph_and_set_networks.md** -- Expert routing can be viewed as a bipartite assignment problem between tokens and experts, analogous to message passing in graph networks.

## Further Reading

1. Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (ICLR 2017) -- foundational MoE with top-K gating.
2. Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (JMLR 2022) -- simplifies MoE to top-1 routing.
3. Puigcerver et al., "From Sparse to Soft Mixtures of Experts" (ICLR 2024) -- eliminates hard routing with fully differentiable dispatch-combine.
4. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022) -- parameter-efficient fine-tuning via low-rank weight decomposition.
5. Houlsby et al., "Parameter-Efficient Transfer Learning for NLP" (ICML 2019) -- introduces bottleneck adapter modules for transfer learning.
