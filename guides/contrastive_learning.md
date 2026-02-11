# Contrastive and Self-Supervised Learning
> Learn visual representations from unlabeled data by teaching networks what should look similar.

## Overview

Self-supervised learning extracts meaningful representations from data without human-annotated labels. The core insight is that data augmentations provide a free supervisory signal: two augmented views of the same image should produce similar representations, while views of different images should not. This family of methods has closed the gap with supervised pretraining on ImageNet and often surpasses it on transfer learning benchmarks.

Edifice implements five self-supervised approaches spanning three paradigms: contrastive methods that use negative pairs (SimCLR), non-contrastive methods that avoid collapse through architectural or loss design (BYOL, BarlowTwins, VICReg), and masked reconstruction methods (MAE). Each method makes different tradeoffs between computational cost, batch size requirements, and the complexity of the collapse-prevention mechanism.

All five modules follow the same high-level pattern: produce two views of each input, pass them through a shared encoder and projection head, then compute a loss that encourages agreement between views. The differences lie entirely in how the loss is computed and how representational collapse is prevented.

## Conceptual Foundation

The fundamental problem these methods solve is learning an encoder f such that f(aug1(x)) is similar to f(aug2(x)) for any input x and any augmentation pair. The challenge is avoiding collapse -- the trivial solution where f maps everything to a constant vector. The loss landscape has a deep, wide basin at collapse, and the key innovation in each method is the mechanism that prevents the optimizer from finding it.

The NT-Xent loss used by SimCLR formalizes this as:

```
L = -log( exp(sim(z_i, z_j) / tau) / SUM_k exp(sim(z_i, z_k) / tau) )
```

where sim is cosine similarity, tau is a temperature parameter, (z_i, z_j) is a positive pair, and the sum runs over all other examples in the batch as negatives. The denominator explicitly pushes apart representations of different images, preventing collapse. Other methods achieve the same effect through different mechanisms.

## Architecture Evolution

```
2020                                                       2022
  |                                                          |
  v                                                          v

  SimCLR (Chen et al.)             BYOL (Grill et al.)       MAE (He et al.)
  "Negatives prevent collapse"     "Momentum prevents        "Reconstruction prevents
   |                                collapse, no negatives"   collapse, no pairs at all"
   |                                 |                         |
   |    +------ BarlowTwins ---------+                         |
   |    |   (Zbontar et al., 2021)   |                         |
   |    |   "Decorrelation prevents  |                         |
   |    |    collapse"               |                         |
   |    |                            |                         |
   |    +------ VICReg -------------+                          |
   |    |   (Bardes et al., 2022)                              |
   |    |   "Explicit V+I+C terms                              |
   |    |    prevent collapse"                                 |
   |    |                                                      |
   v    v                                                      v

CONTRASTIVE        NON-CONTRASTIVE                    MASKED RECONSTRUCTION
(needs negatives)  (no negatives needed)              (no pairs needed)
```

## When to Use What

| Criterion              | SimCLR           | BYOL             | BarlowTwins      | MAE              | VICReg           |
|------------------------|------------------|------------------|------------------|------------------|------------------|
| **Conceptual model**   | Contrastive      | Momentum teacher | Decorrelation    | Reconstruction   | Regularization   |
| **Collapse mechanism** | Negative pairs   | EMA target net   | Cross-corr -> I  | Pixel prediction | V + I + C terms  |
| **Batch size need**    | Large (4096+)    | Moderate (256+)  | Moderate (256+)  | Any              | Moderate (256+)  |
| **Compute overhead**   | Low              | 2x forward pass  | Low              | Encoder-only     | Low              |
| **Negative pairs?**    | Yes              | No               | No               | N/A              | No               |
| **Momentum encoder?**  | No               | Yes              | No               | No               | No               |
| **Returns**            | Single model     | Tuple (online, target) | Single model | Tuple (enc, dec) | Single model     |
| **Best for**           | Learning basics  | Small batches    | Interpretable loss| Vision pretraining | Clean theory  |

**Quick selection guide:**
- Starting out or teaching SSL concepts: **SimCLR** -- simplest to understand and debug.
- Limited GPU memory or small batch sizes: **BYOL** -- works without large negative batches.
- Want to understand what each loss term does: **BarlowTwins** or **VICReg** -- each term has a clear geometric meaning.
- Pretraining a ViT backbone at scale: **MAE** -- processes only 25% of patches (most efficient).
- Need theoretical guarantees on the representation: **VICReg** -- explicit variance, invariance, and covariance control.

## Key Concepts

### The Collapse Problem

Every self-supervised method must prevent the encoder from mapping all inputs to the same point. There are two modes of collapse:

1. **Complete collapse**: All representations converge to a single constant vector. The loss reaches a trivial minimum because all pairs become perfectly similar.

2. **Dimensional collapse**: Representations use only a low-dimensional subspace of the embedding space. The model appears to work but discards information.

```
Complete Collapse              Dimensional Collapse           Good Representation
  All points at one spot        Points on a line               Points fill the sphere

       *                              /                          *    *    *
       * (all here)                  / (all on this line)       *   *   *   *
       *                            /                            *    *    *
                                   /
```

Each method addresses these differently: SimCLR uses repulsion from negatives, BYOL uses the asymmetric predictor + momentum to break symmetry, BarlowTwins pushes the cross-correlation matrix toward identity (diagonal = 1, off-diagonal = 0), VICReg has explicit variance and covariance regularizers, and MAE sidesteps the issue entirely by using pixel-level reconstruction.

### Augmentation as Inductive Bias

The choice of augmentations defines what information the representation should be invariant to versus what it should preserve. This is the primary inductive bias in self-supervised learning -- more important than the specific loss function used.

Standard augmentation pipelines for vision include random crop, color jitter, Gaussian blur, and horizontal flip. Aggressive augmentations force the network to learn higher-level semantic features rather than low-level texture or color statistics.

For non-vision domains (tabular, time series, graph), designing appropriate augmentations is an open research problem. The Edifice contrastive modules accept any encoder and operate on feature vectors, so augmentation is handled externally.

### Projection Head and Representation Quality

All contrastive/non-contrastive methods use a projection head (typically a 2-3 layer MLP) on top of the encoder. A critical finding from the SimCLR paper is that the representation before the projection head (the encoder output) transfers better than the representation after it. The projection head absorbs information about the specific augmentations used during pretraining, allowing the encoder to preserve more general features.

```
Encoder Output (use this for downstream tasks)
     |
     v
Projection Head (discarded after pretraining)
     |
     v
Contrastive Loss
```

All Edifice contrastive modules include both the encoder and projection head in the built model. For downstream evaluation, extract the encoder output by inspecting the Axon graph or building a truncated model.

### Downstream Evaluation Protocols

After pretraining, representations are evaluated through:

- **Linear probing**: Freeze the encoder, train a linear classifier on top. Measures representation quality directly.
- **Fine-tuning**: Unfreeze the encoder, train end-to-end on the downstream task with a lower learning rate. Typically gives the best downstream performance.
- **k-NN evaluation**: Use k-nearest neighbors in the representation space. No training required -- useful for quick evaluation during pretraining.

## Complexity Comparison

| Module       | Forward Passes | Memory (relative) | Params (beyond encoder) | Loss Computation    |
|--------------|----------------|--------------------|-------------------------|---------------------|
| SimCLR       | 2              | O(B^2) for sim     | 1 projection head       | O(B^2) pairwise sim |
| BYOL         | 2 + EMA update | 2x model params    | 2 projection + predictor| O(B) MSE            |
| BarlowTwins  | 2              | O(D^2) cross-corr  | 1 projection head       | O(D^2) correlation  |
| MAE          | 1 (25% tokens) | Encoder + decoder   | Decoder params          | O(P) reconstruction |
| VICReg       | 2              | O(D^2) covariance  | 1 projection head       | O(B*D + D^2) V+I+C  |

B = batch size, D = projection dimension, P = number of masked patches.

## Module Reference

- `Edifice.Contrastive.SimCLR` -- Contrastive learning with NT-Xent loss, shared encoder and 2-layer projection head. Includes `nt_xent_loss/3` for loss computation.
- `Edifice.Contrastive.BYOL` -- Bootstrap Your Own Latent with separate online/target networks. Returns `{online, target}` tuple. Includes `ema_update/3` and `loss/2`.
- `Edifice.Contrastive.BarlowTwins` -- Redundancy reduction via cross-correlation identity constraint. Includes `barlow_twins_loss/3` with configurable lambda.
- `Edifice.Contrastive.MAE` -- Masked Autoencoder with configurable mask ratio (default 75%). Returns `{encoder, decoder}` tuple. Includes `generate_mask/2` and `reconstruction_loss/3`.
- `Edifice.Contrastive.VICReg` -- Variance-Invariance-Covariance regularization with explicit per-term control. Includes `vicreg_loss/3`, `variance_loss/2`, `invariance_loss/2`, and `covariance_loss/1`.

See also: [Vision Architectures](vision_architectures.md) for ViT backbones used with MAE, [Building Blocks](building_blocks.md) for the FFN projection heads these modules use internally.

## Further Reading

- Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (ICML 2020) -- https://arxiv.org/abs/2002.05709
- Grill et al., "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning" (NeurIPS 2020) -- https://arxiv.org/abs/2006.07733
- Zbontar et al., "Barlow Twins: Self-Supervised Learning via Redundancy Reduction" (ICML 2021) -- https://arxiv.org/abs/2103.03230
- He et al., "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022) -- https://arxiv.org/abs/2111.06377
- Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning" (ICLR 2022) -- https://arxiv.org/abs/2105.04906
