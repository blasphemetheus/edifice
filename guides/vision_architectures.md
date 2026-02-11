# Vision Architectures
> Treating images as structured token sequences to enable transformers, hierarchical attention, and pure MLPs to match or surpass convolutional networks on visual tasks.

## Overview

The Vision family in Edifice captures a pivotal shift in computer vision: the move from hand-designed convolutional hierarchies to general-purpose architectures that process images as sequences of patches. This shift was catalyzed by the Vision Transformer (ViT), which showed that a standard transformer encoder -- with no convolution-specific inductive bias -- could match state-of-the-art CNNs when given enough data.

The six modules in this family represent the major branches that grew from this insight. ViT established the patch-embedding paradigm. DeiT showed that ViT could be trained data-efficiently with distillation. Swin Transformer reintroduced locality through windowed attention, producing hierarchical feature maps suitable for detection and segmentation. UNet provides the encoder-decoder architecture essential for dense prediction tasks like segmentation and diffusion-based generation. ConvNeXt turned the question around, showing that a pure CNN designed with transformer-era training techniques could match ViT. And MLPMixer demonstrated that neither attention nor convolution is strictly necessary -- pure MLPs with the right mixing structure can compete.

Together, these modules cover the spectrum from pure attention (ViT, DeiT, Swin) to pure convolution (ConvNeXt, which bridges to the Convolutional family) to pure MLP (MLPMixer), providing a complete toolkit for vision tasks of any scale.

## Conceptual Foundation

All transformer-based vision models begin with the same core operation: converting a 2D image into a 1D sequence of patch embeddings. Given an image of size H x W with C channels and patch size P:

    num_patches = (H/P) * (W/P)
    patch_embed: [B, C, H, W] -> [B, num_patches, embed_dim]

Each P x P patch is linearly projected to an embedding vector. This is mathematically equivalent to a convolution with kernel size P and stride P, but conceptually it converts a spatial grid into a sequence that any sequence model can process.

The key trade-off across these architectures is between global and local information flow. ViT computes full attention over all patches (O(N^2) in patch count), giving every patch a global view but at high cost. Swin restricts attention to local windows of M x M patches (O(N * M^2)) and uses shifted windows for cross-region communication. MLPMixer avoids attention entirely, using fixed-size MLPs that implicitly learn spatial mixing patterns.

## Architecture Evolution

```
    Traditional CNNs (AlexNet, VGG, ResNet)
              |
              | "Can we replace convolutions with attention?"
              |
         ViT (2021) -------- Patch embedding paradigm
          /   |    \
         /    |     \
        /     |      \
  DeiT (2021) |   Swin (2021) ---------- Hierarchical attention
  (training   |   (windowed,              for detection/segmentation
   recipe,    |    multi-scale)
   distill.)  |
              |
      "Do we even need attention?"
         /              \
        /                \
  MLPMixer (2021)    ConvNeXt (2022)
  (pure MLP,          (pure CNN,
   token/channel       transformer
   mixing)             training tricks)

  Meanwhile, for dense prediction:
  UNet (2015) -------> UNet + ViT backbone (DiT, 2023)
  (encoder-decoder     for diffusion generation
   with skip connects)
```

## When to Use What

| Scenario | Module | Why |
|----------|--------|-----|
| Large-scale classification with abundant data | `ViT` | Global attention captures long-range dependencies; scales well |
| Classification with limited data | `DeiT` | Distillation token enables learning from a teacher; strong regularization recipe |
| Object detection or segmentation | `SwinTransformer` | Hierarchical multi-scale features like a CNN; efficient windowed attention |
| Dense prediction (segmentation, generation) | `UNet` | Encoder-decoder with skip connections preserves fine spatial detail |
| Pure-CNN simplicity with modern performance | `ConvNeXt` | No attention overhead; convolutions are hardware-optimized; matches ViT |
| Attention-free research baseline | `MLPMixer` | Tests whether attention is necessary; simple to implement and analyze |

### Resolution and Scale Decision Tree

```
What is your input resolution?
  |
  +-- Small (32x32, e.g., CIFAR)
  |     |
  |     +-- ViT or DeiT with small patch size (4)
  |     +-- ConvNeXt with shallow depths
  |
  +-- Medium (224x224, e.g., ImageNet)
  |     |
  |     +-- Classification? --> ViT, DeiT, ConvNeXt, or Swin
  |     +-- Detection/Segmentation? --> Swin (hierarchical features)
  |
  +-- Large (512+ or variable)
        |
        +-- Dense prediction? --> UNet or Swin backbone
        +-- Classification? --> Swin (linear attention cost)

Do you need multi-scale feature maps?
  |
  +-- Yes --> SwinTransformer (native hierarchy)
  |           or ConvNeXt (natural CNN pyramid)
  |
  +-- No  --> ViT (single-scale, add FPN if needed)
```

## Key Concepts

### The Patch Embedding Paradigm

Every module in this family (except UNet in its classical form) begins by converting images to patches. In Edifice, `Edifice.Blocks.PatchEmbed` handles this conversion. A patch size of 16 on a 224x224 image produces 196 patches (14x14 grid), each a 768-dimensional vector in ViT-Base. Smaller patch sizes give more tokens and finer spatial resolution but quadratically increase attention cost.

ViT prepends a learnable CLS token to the sequence, whose output after the transformer serves as the image representation. DeiT adds a second distillation token at position 1 -- during training, it learns to match the teacher model's predictions, providing an additional training signal that does not interfere with the CLS token's classification loss.

### Windowed and Shifted Attention (Swin)

Swin Transformer solves ViT's quadratic attention cost by restricting attention to non-overlapping M x M windows (typically M=7). Within each window, standard multi-head self-attention is computed. To enable cross-window information flow, alternating layers shift the window grid by M/2 positions in both dimensions using a cyclic shift:

```
Regular Windows          Shifted Windows (cyclic shift by M/2)

+---+---+---+---+       --+--+---+---+-
|   |   |   |   |         |  |   |   |
+---+---+---+---+       --+--+---+---+-
|   |   |   |   |  -->    |  |   |   |
+---+---+---+---+       --+--+---+---+-
|   |   |   |   |         |  |   |   |
+---+---+---+---+       --+--+---+---+-
```

After the cyclic shift, some windows contain tokens from non-adjacent spatial regions. Swin applies an attention mask to prevent these tokens from attending to each other, maintaining correct spatial locality. The Edifice implementation in `SwinTransformer` precomputes both the relative position bias and shift mask as constants to avoid runtime overhead.

Between stages, patch merging groups 2x2 neighboring patches and projects their concatenated features to double the channel dimension. This produces hierarchical feature maps at 1/4, 1/8, 1/16, and 1/32 of the original resolution -- the same multi-scale structure that makes CNNs effective for detection and segmentation.

### Encoder-Decoder with Skip Connections (UNet)

UNet's architecture is symmetric: an encoder progressively downsamples while increasing feature channels, and a decoder upsamples back to the original resolution. The defining feature is skip connections that concatenate encoder features at each resolution level with the corresponding decoder features. This allows the decoder to combine high-level semantic information (from deep layers) with fine-grained spatial detail (from shallow layers).

In Edifice, UNet operates on flattened spatial representations using dense layers rather than spatial convolutions, making it compatible with the Axon pipeline. The architecture supports an optional attention mechanism at the bottleneck for enhanced feature interaction. UNet is the standard backbone for diffusion model denoising networks, though recent work (DiT) replaces it with transformer-based architectures.

### ConvNeXt: Modernizing Convolutions

ConvNeXt systematically applies transformer-era design choices to a standard ResNet and demonstrates that the resulting pure-CNN architecture matches Swin Transformer. The key modifications include: replacing BatchNorm with LayerNorm, using GELU instead of ReLU, adopting an inverted bottleneck (expand then project, not project then expand), using fewer activations (only one per block), and increasing kernel sizes. The ConvNeXt block pattern in Edifice is LayerNorm, expand 4x, GELU, project back, scale, residual add -- a simpler structure than a traditional residual block.

## Complexity Comparison

| Module | Attention Cost | Hierarchical | Position Encoding | Params (Base config) | Best For |
|--------|---------------|-------------|-------------------|---------------------|----------|
| ViT | O(N^2) global | No (single scale) | Learned absolute | ~86M (ViT-B/16) | Large-scale classification |
| DeiT | O(N^2) global | No (single scale) | Learned absolute | ~86M + dist. token | Data-efficient training |
| SwinTransformer | O(N * M^2) windowed | Yes (4 stages) | Relative position bias | ~28M (Swin-T) | Detection, segmentation |
| UNet | Optional at bottleneck | Yes (encoder-decoder) | None | Varies by depth | Dense prediction, diffusion |
| ConvNeXt | None (pure conv) | Yes (4 stages) | Implicit in convolution | ~28M (ConvNeXt-T) | Hardware-efficient vision |
| MLPMixer | None (pure MLP) | No (single scale) | Implicit in token dim | ~59M (Mixer-B/16) | Research baseline |

## Module Reference

- `Edifice.Vision.ViT` -- Vision Transformer with patch embedding, learnable CLS token, absolute position embeddings, and standard transformer encoder blocks.
- `Edifice.Vision.DeiT` -- Data-efficient Image Transformer extending ViT with a distillation token; supports dual-head output for teacher-student training.
- `Edifice.Vision.SwinTransformer` -- Hierarchical vision transformer with windowed self-attention, cyclic shift, relative position bias, and patch merging between stages.
- `Edifice.Vision.UNet` -- Encoder-decoder architecture with skip connections for dense prediction; supports optional bottleneck attention.
- `Edifice.Vision.ConvNeXt` -- Modernized ConvNet with inverted bottleneck blocks, LayerNorm, GELU, and hierarchical downsampling via patch merging.
- `Edifice.Vision.MLPMixer` -- All-MLP architecture with alternating token-mixing (spatial) and channel-mixing (feature) MLP sublayers.

## Cross-References

- **building_blocks.md** -- `PatchEmbed` is the shared foundation for ViT, DeiT, Swin, ConvNeXt, and MLPMixer; all five modules use it from the Blocks family.
- **attention_mechanisms.md** -- Swin's windowed attention with relative position bias is a specialized form of local attention; ViT and DeiT use standard multi-head self-attention.
- **convolutional_networks.md** -- ConvNeXt bridges this family with the Convolutional family, showing that modern CNN design can match transformer-based vision models.
- **generative_models.md** -- UNet is the standard backbone for diffusion models; DiT replaces UNet with a ViT-based architecture for generation tasks.

## Further Reading

1. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ICLR 2021) -- introduces ViT and the patch embedding paradigm.
2. Touvron et al., "Training Data-Efficient Image Transformers & Distillation Through Attention" (ICML 2021) -- DeiT's training recipe and distillation token.
3. Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (ICCV 2021) -- windowed attention with shifted windows for efficient hierarchical features.
4. Liu et al., "A ConvNet for the 2020s" (CVPR 2022) -- ConvNeXt modernizes ResNet with transformer design principles.
5. Tolstikhin et al., "MLP-Mixer: An all-MLP Architecture for Vision" (NeurIPS 2021) -- demonstrates competitive vision performance without attention or convolution.
