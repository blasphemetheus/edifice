# Architecture Index

Catalog of Edifice architectures and shared blocks.

**AI-generated validation labels.** The evidence column distinguishes reference
comparisons, real-checkpoint comparisons, and local behavior checks. Read the
[validation guide](validation.md) for scope and backend limits. **Unreviewed**
means the evidence has not been assessed here; it does not mean there are no tests.
An architecture name or a passing shape test is not a claim of paper fidelity.

For descriptions, paper references, and decision guidance, see [Architecture Taxonomy](architecture_taxonomy.md).

---

## Verified in use

Most of the 267 architectures are backed by the library's own tests: shapes,
gradients, batch sweeps, and the registry checks. The ones below have been
exercised beyond that. The list is `Edifice.verified_architectures/0`; this
section is checked against it by `test/edifice/doc_counts_test.exs`, and
`Edifice.verification/1` answers for a single name.

| Level | Meaning | Architectures |
|---|---|---|
| **Deployed** | Trained and run as an ExPhil policy backbone, with the stateful step and replay path checked against the batch forward on an actual checkpoint | `:gru`, `:lstm`, `:min_gru`, `:mamba`, `:mamba_ssd` |
| **Trained** | ExPhil carries tuned training defaults for it, so it has been trained end to end at least in bake-offs | `:attention`, `:delta_net`, `:gated_delta_net`, `:gla`, `:griffin`, `:gru`, `:h3`, `:hgrn`, `:jamba`, `:lstm`, `:mamba`, `:mamba3`, `:mamba_ssd`, `:min_gru`, `:min_lstm`, `:performer`, `:retnet`, `:rwkv`, `:ttt`, `:ttt_e2e`, `:xlstm`, `:zamba` |
| **Pretrained parity** | A published checkpoint loads through `Edifice.Pretrained` and its outputs match the PyTorch reference (`test/edifice/pretrained/numerical_validation_test.exs`) | `:convnext`, `:detr`, `:resnet`, `:vit`, `:whisper` |
| **Reference parity** | Forward (and for DeltaNet, gradients) matches a PyTorch fixture at random weights (`test/edifice/pretrained/architecture_numerical_test.exs`) | `:deep_res_lstm`, `:delta_net`, `:dit`, `:gat`, `:gqa`, `:mamba`, `:min_gru` |

Anything not listed has not been trained to convergence or compared with a
reference implementation by us. Treat it as a faithful reading of the paper
that compiles and runs, not as a certified port.

## Feedforward

| Architecture | Module | Key Feature | Validation evidence |
|-------------|--------|-------------|---------------------|
| **MLP** | `Edifice.Feedforward.MLP` | Multi-layer perceptron with configurable hidden sizes | [Unreviewed](validation.md) |
| **KAN** | `Edifice.Feedforward.KAN` | Kolmogorov-Arnold Networks, learnable activation functions | [Unreviewed](validation.md) |
| **KAT** | `Edifice.Feedforward.KAT` | Kolmogorov-Arnold Transformer (KAN + attention) | [Unreviewed](validation.md) |
| **TabNet** | `Edifice.Feedforward.TabNet` | Attentive feature selection for tabular data | [Unreviewed](validation.md) |
| **BitNet** | `Edifice.Feedforward.BitNet` | Ternary/binary weight quantization (1.58-bit) | [Unreviewed](validation.md) |

## Transformer

| Architecture | Module | Key Feature | Validation evidence |
|-------------|--------|-------------|---------------------|
| **Decoder-Only** | `Edifice.Transformer.DecoderOnly` | GPT-style with GQA, RoPE/iRoPE, SwiGLU, RMSNorm | [Behavior](validation.md) |
| **Multi-Token Prediction** | `Edifice.Transformer.MultiTokenPrediction` | Predict next N tokens simultaneously | [Unreviewed](validation.md) |
| **Byte Latent Transformer** | `Edifice.Transformer.ByteLatentTransformer` | Byte-level processing via encoder-latent-decoder | [Unreviewed](validation.md) |
| **Nemotron-H** | `Edifice.Transformer.NemotronH` | NVIDIA's hybrid Mamba-Transformer | [Unreviewed](validation.md) |
| **Free Transformer** | `Edifice.Transformer.FreeTransformer` | Decoder with discrete latent variable injected at midpoint | [Unreviewed](validation.md) |

## State Space Models

| Architecture | Module | Key Feature | Validation evidence |
|-------------|--------|-------------|---------------------|
| **S4** | `Edifice.SSM.S4` | HiPPO DPLR initialization, long-range memory | [Unreviewed](validation.md) |
| **S4D** | `Edifice.SSM.S4D` | Diagonal state space, simplified S4 | [Unreviewed](validation.md) |
| **S5** | `Edifice.SSM.S5` | MIMO diagonal SSM with D skip connection | [Unreviewed](validation.md) |
| **H3** | `Edifice.SSM.H3` | Two SSMs with multiplicative gating + short convolution | [Unreviewed](validation.md) |
| **Hyena** | `Edifice.SSM.Hyena` | Long convolution hierarchy, implicit filters | [Unreviewed](validation.md) |
| **Mamba** | `Edifice.SSM.Mamba` | Selective SSM, parallel associative scan | [Reference (limited scope)](validation.md) |
| **Mamba-2 (SSD)** | `Edifice.SSM.MambaSSD` | Structured state space duality, chunk-wise matmul | [Reference (limited scope)](validation.md) |
| **Mamba-2 Full** | `Edifice.SSM.Mamba2` | Canonical full-SSM mixer, learned decay and skip, grouped chunked SSD (`:mamba2_full`) | [Reference (limited scope)](validation.md) |
| **Mamba (Cumsum)** | `Edifice.SSM.MambaCumsum` | Mamba with configurable scan algorithm | [Unreviewed](validation.md) |
| **Mamba (Hillis-Steele)** | `Edifice.SSM.MambaHillisSteele` | Mamba with max-parallelism scan | [Unreviewed](validation.md) |
| **BiMamba** | `Edifice.SSM.BiMamba` | Bidirectional Mamba for non-causal tasks | [Unreviewed](validation.md) |
| **GatedSSM** | `Edifice.SSM.GatedSSM` | Gated temporal model; causal mode supports carried inference | [Behavior](validation.md) |
| **Jamba** | `Edifice.SSM.Hybrid` | Mamba + Attention hybrid (configurable ratio) | [Unreviewed](validation.md) |
| **Zamba** | `Edifice.SSM.Zamba` | Mamba + single shared attention layer | [Unreviewed](validation.md) |
| **StripedHyena** | `Edifice.SSM.StripedHyena` | Interleaved Hyena long conv + gated conv | [Unreviewed](validation.md) |
| **Mamba-3** | `Edifice.SSM.Mamba3` | Complex states, trapezoidal discretization, MIMO | [Unreviewed](validation.md) |
| **GSS** | `Edifice.SSM.GSS` | Gated State Space (simplified S4 with gating) | [Unreviewed](validation.md) |
| **Hymba** | `Edifice.SSM.Hymba` | Hybrid Mamba + attention with learnable meta tokens | [Unreviewed](validation.md) |
| **SS Transformer** | `Edifice.SSM.SSTransformer` | State Space Transformer | [Unreviewed](validation.md) |
| **Hyena v2** | `Edifice.SSM.HyenaV2` | Improved implicit long convolution with short conv and better decay | [Unreviewed](validation.md) |
| **Longhorn** | `Edifice.SSM.Longhorn` | SSM derived from online associative recall | [Unreviewed](validation.md) |
| **Mixture-of-Mamba** | `Edifice.SSM.MixtureOfMamba` | MoM: modality-aware sparse Mamba blocks | [Unreviewed](validation.md) |
| **Samba** | `Edifice.SSM.Samba` | Hybrid Mamba + sliding window attention + SwiGLU MLP | [Unreviewed](validation.md) |

## Attention & Linear Attention

| Architecture | Module | Key Feature | Validation evidence |
|-------------|--------|-------------|---------------------|
| **Multi-Head Attention** | `Edifice.Attention.MultiHead` | Sliding window, QK LayerNorm | [Unreviewed](validation.md) |
| **GQA** | `Edifice.Attention.GQA` | Grouped Query Attention, fewer KV heads | [Reference (limited scope)](validation.md) |
| **Perceiver** | `Edifice.Attention.Perceiver` | Cross-attention to learned latents, input-agnostic | [Unreviewed](validation.md) |
| **FNet** | `Edifice.Attention.FNet` | Fourier Transform replacing attention | [Unreviewed](validation.md) |
| **Linear Transformer** | `Edifice.Attention.LinearTransformer` | Kernel-based O(N) attention | [Unreviewed](validation.md) |
| **Nystromformer** | `Edifice.Attention.Nystromformer` | Nystrom approximation of attention matrix | [Unreviewed](validation.md) |
| **Performer** | `Edifice.Attention.Performer` | FAVOR+ random feature attention | [Unreviewed](validation.md) |
| **RetNet** | `Edifice.Attention.RetNet` | Multi-scale retention, O(1) recurrent inference | [Behavior](validation.md) |
| **RWKV-7** | `Edifice.Attention.RWKV` | Linear attention, O(1) space, "Goose" architecture | [Unreviewed](validation.md) |
| **GLA** | `Edifice.Attention.GLA` | Gated Linear Attention with data-dependent decay | [Unreviewed](validation.md) |
| **HGRN-2** | `Edifice.Attention.HGRN` | Hierarchically gated linear RNN, state expansion | [Unreviewed](validation.md) |
| **Griffin/Hawk** | `Edifice.Attention.Griffin` | RG-LRU + local attention (Griffin) or pure RG-LRU (Hawk) | [Unreviewed](validation.md) |
| **Diff Transformer** | `Edifice.Attention.DiffTransformer` | Noise-cancelling dual softmax subtraction | [Unreviewed](validation.md) |
| **MLA** | `Edifice.Attention.MLA` | Multi-Head Latent Attention (DeepSeek KV compression) | [Unreviewed](validation.md) |
| **Based** | `Edifice.Attention.Based` | Taylor expansion linear attention | [Unreviewed](validation.md) |
| **Mega** | `Edifice.Attention.Mega` | Moving average + gated attention | [Unreviewed](validation.md) |
| **InfiniAttention** | `Edifice.Attention.InfiniAttention` | Compressive memory for unbounded context | [Unreviewed](validation.md) |
| **Conformer** | `Edifice.Attention.Conformer` | Conv-augmented transformer for audio/speech | [Unreviewed](validation.md) |
| **Ring Attention** | `Edifice.Attention.RingAttention` | Distributed chunked attention for long sequences | [Unreviewed](validation.md) |
| **Lightning Attention** | `Edifice.Attention.LightningAttention` | Hybrid linear/softmax with I/O-aware tiling | [Unreviewed](validation.md) |
| **Gated Attention** | `Edifice.Attention.GatedAttention` | Sigmoid post-attention gate (NeurIPS 2025) | [Unreviewed](validation.md) |
| **NSA** | `Edifice.Attention.NSA` | Native Sparse Attention (DeepSeek three-path) | [Behavior](validation.md) |
| **KDA** | `Edifice.Attention.KDA` | Kimi Delta Attention, channel-wise decay | [Unreviewed](validation.md) |
| **Flash Linear Attention** | `Edifice.Attention.FlashLinearAttention` | Optimized linear attention | [Unreviewed](validation.md) |
| **YaRN** | `Edifice.Attention.YARN` | RoPE context extension via frequency scaling | [Unreviewed](validation.md) |
| **Dual Chunk** | `Edifice.Attention.DualChunk` | Dual Chunk Attention for long-context | [Unreviewed](validation.md) |
| **TMRoPE** | `Edifice.Attention.TMRoPE` | Time-aligned Multimodal RoPE | [Unreviewed](validation.md) |
| **RNoPE-SWA** | `Edifice.Attention.RNoPESWA` | No positional encoding + sliding window | [Unreviewed](validation.md) |
| **FoX** | `Edifice.Attention.FoX` | Forgetting Transformer with learnable per-head forget gates | [Unreviewed](validation.md) |
| **GLA v2** | `Edifice.Attention.GLAv2` | Improved Gated Linear Attention | [Unreviewed](validation.md) |
| **GSA** | `Edifice.Attention.GSA` | Gated Slot Attention with low-rank memory bottleneck | [Unreviewed](validation.md) |
| **Hawk** | `Edifice.Attention.Hawk` | Pure RG-LRU recurrent model (RecurrentGemma) | [Unreviewed](validation.md) |
| **HGRN v2** | `Edifice.Attention.HGRNv2` | Multi-resolution hierarchical gating with outer product state | [Unreviewed](validation.md) |
| **InfLLM-V2** | `Edifice.Attention.InfLLMV2` | Dense-sparse switchable attention | [Unreviewed](validation.md) |
| **LASER** | `Edifice.Attention.LASER` | Attention with exponential transformation | [Unreviewed](validation.md) |
| **Log-Linear Attention** | `Edifice.Attention.LogLinear` | O(log T) memory bridging linear and softmax attention | [Unreviewed](validation.md) |
| **MEGALODON** | `Edifice.Attention.Megalodon` | Complex EMA and timestep normalization at mega scale | [Unreviewed](validation.md) |
| **MoBA** | `Edifice.Attention.MoBA` | Mixture of Block Attention | [Behavior](validation.md) |
| **MTA** | `Edifice.Attention.MTA` | Multi-Token Attention, 2D convolution on attention logits | [Unreviewed](validation.md) |
| **NHA** | `Edifice.Attention.NHA` | Native Hybrid Attention: learned per-layer linear vs softmax selection | [Unreviewed](validation.md) |
| **RLA / RDN** | `Edifice.Attention.RLA` | Residual Linear Attention with error-correcting dual-state recurrence | [Unreviewed](validation.md) |
| **RetNet v2** | `Edifice.Attention.RetNetV2` | Improved Retentive Network with enhanced chunkwise retention | [Unreviewed](validation.md) |
| **Sigmoid Attention** | `Edifice.Attention.SigmoidAttention` | Drop-in softmax replacement using normalized sigmoid | [Unreviewed](validation.md) |
| **SPLA** | `Edifice.Attention.SPLA` | Block Sparse Plus Linear Attention | [Unreviewed](validation.md) |
| **TNN** | `Edifice.Attention.TNN` | Toeplitz Neural Network, learned Toeplitz convolutions | [Unreviewed](validation.md) |

## Recurrent Networks

| Architecture | Module | Key Feature | Validation evidence |
|-------------|--------|-------------|---------------------|
| **LSTM/GRU** | `Edifice.Recurrent` | Classic recurrent with multi-layer stacking | [GRU: reference; LSTM: behavior](validation.md) |
| **xLSTM** | `Edifice.Recurrent.XLSTM` | Exponential gating, matrix memory (sLSTM/mLSTM) | [Unreviewed](validation.md) |
| **MinGRU** | `Edifice.Recurrent.MinGRU` | Minimal GRU, parallel-scannable | [Reference (limited scope)](validation.md) |
| **MinLSTM** | `Edifice.Recurrent.MinLSTM` | Minimal LSTM, parallel-scannable | [Unreviewed](validation.md) |
| **DeltaNet** | `Edifice.Recurrent.DeltaNet` | Delta rule-based linear RNN | [Reference (limited scope)](validation.md) |
| **TTT** | `Edifice.Recurrent.TTT` | Test-Time Training, self-supervised at inference | [Unreviewed](validation.md) |
| **Titans** | `Edifice.Recurrent.Titans` | Neural long-term memory, surprise-gated | [Unreviewed](validation.md) |
| **Reservoir** | `Edifice.Recurrent.Reservoir` | Echo State Networks with fixed random reservoir | [Behavior](validation.md) |
| **sLSTM** | `Edifice.Recurrent.SLSTM` | Scalar LSTM with exponential gating | [Unreviewed](validation.md) |
| **xLSTM v2** | `Edifice.Recurrent.XLSTMv2` | Updated mLSTM with matrix memory | [Unreviewed](validation.md) |
| **Gated DeltaNet** | `Edifice.Recurrent.GatedDeltaNet` | Linear attention with data-dependent gating | [Behavior](validation.md) |
| **TTT-E2E** | `Edifice.Recurrent.TTTE2E` | End-to-end test-time training | [Unreviewed](validation.md) |
| **Native Recurrence** | `Edifice.Recurrent.NativeRecurrence` | Native recurrence block | [Unreviewed](validation.md) |
| **DeepResLSTM** | `Edifice.Recurrent.DeepResLSTM` | Stacked residual LSTM blocks (slippi-ai) | [Reference (limited scope)](validation.md) |
| **DeltaProduct** | `Edifice.Recurrent.DeltaProduct` | Multi-step DeltaNet via Householder products | [Unreviewed](validation.md) |
| **Huginn** | `Edifice.Recurrent.Huginn` | Depth-recurrent transformer with adaptive iteration | [Unreviewed](validation.md) |
| **MIRAS** | `Edifice.Recurrent.MIRAS` | Memory as Iterative Reasoning over Associative Structures | [Unreviewed](validation.md) |
| **Multi-Timescale Recurrence** | `Edifice.Recurrent.MultiTimescaleRecurrence` | Parallel recurrent cores at different temporal strides | [Unreviewed](validation.md) |
| **TransformerLike** | `Edifice.Recurrent.TransformerLike` | Hybrid recurrent + FFN architecture (slippi-ai) | [Unreviewed](validation.md) |

## Vision

| Architecture | Module | Key Feature | Validation evidence |
|-------------|--------|-------------|---------------------|
| **ViT** | `Edifice.Vision.ViT` | Vision Transformer, patch embedding | [Checkpoint (limited scope)](validation.md) |
| **DeiT** | `Edifice.Vision.DeiT` | Data-efficient ViT with distillation token | [Unreviewed](validation.md) |
| **Swin** | `Edifice.Vision.SwinTransformer` | Shifted window attention, hierarchical features | [Unreviewed](validation.md) |
| **U-Net** | `Edifice.Vision.UNet` | Encoder-decoder with skip connections | [Unreviewed](validation.md) |
| **ConvNeXt** | `Edifice.Vision.ConvNeXt` | Modernized ConvNet with transformer-inspired design | [Checkpoint (limited scope)](validation.md) |
| **MLP-Mixer** | `Edifice.Vision.MLPMixer` | Pure MLP with token/channel mixing | [Unreviewed](validation.md) |
| **FocalNet** | `Edifice.Vision.FocalNet` | Focal modulation, hierarchical context | [Unreviewed](validation.md) |
| **PoolFormer** | `Edifice.Vision.PoolFormer` | Average pooling token mixer (MetaFormer) | [Unreviewed](validation.md) |
| **NeRF** | `Edifice.Vision.NeRF` | Neural radiance field, coordinate-to-color mapping | [Unreviewed](validation.md) |
| **Gaussian Splat** | `Edifice.Vision.GaussianSplat` | 3D Gaussian Splatting (NeRF successor) | [Unreviewed](validation.md) |
| **MambaVision** | `Edifice.Vision.MambaVision` | 4-stage hierarchical CNN+Mamba+Attention | [Unreviewed](validation.md) |
| **DINOv2** | `Edifice.Vision.DINOv2` | Self-distillation vision backbone | [Unreviewed](validation.md) |
| **MetaFormer** | `Edifice.Vision.MetaFormer` | Architecture-first framework (+ CAFormer variant) | [Unreviewed](validation.md) |
| **EfficientViT** | `Edifice.Vision.EfficientViT` | Linear attention ViT | [Unreviewed](validation.md) |
| **DINOv3** | `Edifice.Vision.DINOv3` | Self-supervised backbone with axial RoPE and text alignment | [Unreviewed](validation.md) |
| **Janus** | `Edifice.Vision.Janus` | Unified multimodal understanding and generation | [Unreviewed](validation.md) |
| **Vision KAN** | `Edifice.Vision.VisionKAN` | Attention-free hierarchical backbone using RBF KAN layers | [Unreviewed](validation.md) |

## Convolutional

| Architecture | Module | Key Feature | Validation evidence |
|-------------|--------|-------------|---------------------|
| **Conv1D/2D** | `Edifice.Convolutional.Conv` | Configurable convolution blocks with BN, activation, dropout | [Unreviewed](validation.md) |
| **ResNet** | `Edifice.Convolutional.ResNet` | Residual/bottleneck blocks, configurable depth | [Checkpoint (limited scope)](validation.md) |
| **DenseNet** | `Edifice.Convolutional.DenseNet` | Dense connections, feature reuse | [Unreviewed](validation.md) |
| **TCN** | `Edifice.Convolutional.TCN` | Dilated causal convolutions for sequences | [Unreviewed](validation.md) |
| **MobileNet** | `Edifice.Convolutional.MobileNet` | Depthwise separable convolutions | [Unreviewed](validation.md) |
| **EfficientNet** | `Edifice.Convolutional.EfficientNet` | Compound scaling (depth, width, resolution) | [Unreviewed](validation.md) |

## Generative

| Architecture | Module | Key Feature | Validation evidence |
|-------------|--------|-------------|---------------------|
| **VAE** | `Edifice.Generative.VAE` | Reparameterization trick, KL divergence, beta-VAE | [Unreviewed](validation.md) |
| **VQ-VAE** | `Edifice.Generative.VQVAE` | Discrete codebook, straight-through estimator | [Unreviewed](validation.md) |
| **GAN** | `Edifice.Generative.GAN` | Generator/discriminator, WGAN-GP support | [Unreviewed](validation.md) |
| **Diffusion (DDPM)** | `Edifice.Generative.Diffusion` | Denoising diffusion, sinusoidal time embedding | [Unreviewed](validation.md) |
| **DDIM** | `Edifice.Generative.DDIM` | Deterministic diffusion sampling, fast inference | [Unreviewed](validation.md) |
| **DiT** | `Edifice.Generative.DiT` | Diffusion Transformer, AdaLN-Zero conditioning | [Reference (limited scope)](validation.md) |
| **DiT v2** | `Edifice.Generative.DiTv2` | Improved adaptive norm conditioning | [Unreviewed](validation.md) |
| **Latent Diffusion** | `Edifice.Generative.LatentDiffusion` | Diffusion in compressed latent space | [Unreviewed](validation.md) |
| **Consistency Model** | `Edifice.Generative.ConsistencyModel` | Single-step generation via consistency training | [Unreviewed](validation.md) |
| **Score SDE** | `Edifice.Generative.ScoreSDE` | Continuous SDE framework (VP-SDE, VE-SDE) | [Unreviewed](validation.md) |
| **Flow Matching** | `Edifice.Generative.FlowMatching` | ODE-based generation, multiple loss variants | [Unreviewed](validation.md) |
| **Rectified Flow** | `Edifice.Generative.RectifiedFlow` | Straight-trajectory flow matching, fewer steps | [Unreviewed](validation.md) |
| **Normalizing Flow** | `Edifice.Generative.NormalizingFlow` | Affine coupling layers (RealNVP-style) | [Unreviewed](validation.md) |
| **MMDiT** | `Edifice.Generative.MMDiT` | Multimodal Diffusion Transformer (FLUX.1, SD3) | [Unreviewed](validation.md) |
| **SoFlow** | `Edifice.Generative.SoFlow` | Flow matching + consistency loss | [Unreviewed](validation.md) |
| **VAR** | `Edifice.Generative.VAR` | Visual Autoregressive (next-scale prediction) | [Unreviewed](validation.md) |
| **Linear DiT (SANA)** | `Edifice.Generative.LinearDiT` | Linear attention for diffusion, 100x speedup | [Unreviewed](validation.md) |
| **SiT** | `Edifice.Generative.SiT` | Scalable Interpolant Transformer | [Unreviewed](validation.md) |
| **Transfusion** | `Edifice.Generative.Transfusion` | Unified AR text + diffusion images | [Unreviewed](validation.md) |
| **MAR** | `Edifice.Generative.MAR` | Masked Autoregressive generation | [Unreviewed](validation.md) |
| **CogVideoX** | `Edifice.Generative.CogVideoX` | 3D causal VAE + expert transformer for video | [Unreviewed](validation.md) |
| **TRELLIS** | `Edifice.Generative.TRELLIS` | Sparse 3D lattice + rectified flow | [Unreviewed](validation.md) |
| **MDLM** | `Edifice.Generative.MDLM` | Discrete diffusion language model | [Unreviewed](validation.md) |
| **CaDDi** | `Edifice.Generative.CaDDi` | Causal Discrete Diffusion | [Unreviewed](validation.md) |
| **DeepFlow** | `Edifice.Generative.DeepFlow` | Deeply supervised flow-based generative model | [Unreviewed](validation.md) |
| **JanusFlow** | `Edifice.Generative.JanusFlow` | Unified AR text + rectified flow image generation | [Unreviewed](validation.md) |
| **LLaDA** | `Edifice.Generative.LLaDA` | Large Language Diffusion with Masking | [Unreviewed](validation.md) |
| **MAGVIT-v2** | `Edifice.Generative.MAGVIT2` | Lookup-Free Quantization for visual tokens | [Unreviewed](validation.md) |
| **Meissonic** | `Edifice.Generative.Meissonic` | Masked generative transformer for images | [Unreviewed](validation.md) |
| **Show-o** | `Edifice.Generative.ShowO` | Unified autoregressive + discrete diffusion multimodal model | [Unreviewed](validation.md) |
| **STARFlow** | `Edifice.Generative.STARFlow` | Scaled transformer autoregressive flow | [Unreviewed](validation.md) |
| **TarFlow** | `Edifice.Generative.TarFlow` | Transformer-based autoregressive normalizing flow | [Unreviewed](validation.md) |

## Contrastive & Self-Supervised

| Architecture | Module | Key Feature | Validation evidence |
|-------------|--------|-------------|---------------------|
| **SimCLR** | `Edifice.Contrastive.SimCLR` | NT-Xent contrastive loss, projection head | [Unreviewed](validation.md) |
| **BYOL** | `Edifice.Contrastive.BYOL` | No negatives, momentum encoder | [Unreviewed](validation.md) |
| **Barlow Twins** | `Edifice.Contrastive.BarlowTwins` | Cross-correlation redundancy reduction | [Unreviewed](validation.md) |
| **MAE** | `Edifice.Contrastive.MAE` | Masked Autoencoder, 75% patch masking | [Unreviewed](validation.md) |
| **VICReg** | `Edifice.Contrastive.VICReg` | Variance-Invariance-Covariance regularization | [Unreviewed](validation.md) |
| **JEPA** | `Edifice.Contrastive.JEPA` | Joint Embedding Predictive Architecture | [Unreviewed](validation.md) |
| **Temporal JEPA** | `Edifice.Contrastive.TemporalJEPA` | V-JEPA for video/temporal sequences | [Unreviewed](validation.md) |
| **SigLIP** | `Edifice.Contrastive.SigLIP` | Sigmoid contrastive learning (CLIP improvement) | [Unreviewed](validation.md) |
| **V-JEPA 2** | `Edifice.Contrastive.VJEPA2` | Video Joint Embedding Predictive Architecture v2 | [Unreviewed](validation.md) |

## Graph & Set Networks

| Architecture | Module | Key Feature | Validation evidence |
|-------------|--------|-------------|---------------------|
| **GCN** | `Edifice.Graph.GCN` | Spectral graph convolutions (Kipf & Welling) | [Unreviewed](validation.md) |
| **GAT** | `Edifice.Graph.GAT` | Graph attention with multi-head support | [Reference (limited scope)](validation.md) |
| **GIN** | `Edifice.Graph.GIN` | Graph Isomorphism Network, maximally expressive | [Unreviewed](validation.md) |
| **GraphSAGE** | `Edifice.Graph.GraphSAGE` | Inductive learning, neighborhood sampling | [Unreviewed](validation.md) |
| **Graph Transformer** | `Edifice.Graph.GraphTransformer` | Full attention over nodes with edge features | [Unreviewed](validation.md) |
| **PNA** | `Edifice.Graph.PNA` | Principal Neighbourhood Aggregation | [Unreviewed](validation.md) |
| **GINv2** | `Edifice.Graph.GINv2` | GIN with edge features | [Unreviewed](validation.md) |
| **SchNet** | `Edifice.Graph.SchNet` | Continuous-filter convolutions for molecules | [Unreviewed](validation.md) |
| **EGNN** | `Edifice.Graph.EGNN` | E(n)-equivariant GNN for molecular simulation | [Unreviewed](validation.md) |
| **DeepSets** | `Edifice.Sets.DeepSets` | Permutation-invariant set functions | [Unreviewed](validation.md) |
| **PointNet** | `Edifice.Sets.PointNet` | Point cloud processing with T-Net alignment | [Unreviewed](validation.md) |
| **DimeNet++** | `Edifice.Graph.DimeNet` | Directional message passing for molecules | [Unreviewed](validation.md) |
| **GPS** | `Edifice.Graph.GPS` | General, Powerful, Scalable graph transformer | [Unreviewed](validation.md) |
| **KA-GNN** | `Edifice.Graph.KAGNN` | Kolmogorov-Arnold Graph Neural Network | [Unreviewed](validation.md) |
| **SE(3)-Transformer** | `Edifice.Graph.SE3Transformer` | 3D roto-translation equivariant attention | [Unreviewed](validation.md) |
| **Entity Encoder** | `Edifice.Sets.EntityEncoder` | Type-conditioned encoder for heterogeneous game entities | [Unreviewed](validation.md) |
| **PointNet++** | `Edifice.Sets.PointNetPP` | Hierarchical feature learning on point sets | [Unreviewed](validation.md) |

## Energy, Probabilistic & Memory

| Architecture | Module | Key Feature | Validation evidence |
|-------------|--------|-------------|---------------------|
| **EBM** | `Edifice.Energy.EBM` | Energy-based models, contrastive divergence | [Unreviewed](validation.md) |
| **Hopfield** | `Edifice.Energy.Hopfield` | Modern continuous Hopfield networks | [Unreviewed](validation.md) |
| **Neural ODE** | `Edifice.Energy.NeuralODE` | Continuous-depth networks via ODE solvers | [Unreviewed](validation.md) |
| **Bayesian NN** | `Edifice.Probabilistic.Bayesian` | Weight uncertainty, variational inference | [Unreviewed](validation.md) |
| **MC Dropout** | `Edifice.Probabilistic.MCDropout` | Uncertainty estimation via dropout at inference | [Unreviewed](validation.md) |
| **Evidential NN** | `Edifice.Probabilistic.EvidentialNN` | Dirichlet priors for uncertainty | [Unreviewed](validation.md) |
| **NTM** | `Edifice.Memory.NTM` | Neural Turing Machine, differentiable memory | [Unreviewed](validation.md) |
| **Memory Network** | `Edifice.Memory.MemoryNetwork` | End-to-end memory with multi-hop attention | [Unreviewed](validation.md) |
| **Engram** | `Edifice.Memory.Engram` | O(1) hash-based associative memory | [Unreviewed](validation.md) |
| **Memory Layer** | `Edifice.Memory.MemoryLayer` | Sparse key-value lookup replacing dense FFN | [Unreviewed](validation.md) |

## Meta-Learning & Specialized

| Architecture | Module | Key Feature | Validation evidence |
|-------------|--------|-------------|---------------------|
| **MoE** | `Edifice.Meta.MoE` | Mixture of Experts with top-k/hash routing | [Unreviewed](validation.md) |
| **MoE v2** | `Edifice.Meta.MoEv2` | Expert-choice routing + shared experts + bias balancing | [Unreviewed](validation.md) |
| **Switch MoE** | `Edifice.Meta.SwitchMoE` | Top-1 routing with load balancing | [Unreviewed](validation.md) |
| **Soft MoE** | `Edifice.Meta.SoftMoE` | Fully differentiable soft token routing | [Unreviewed](validation.md) |
| **ReMoE** | `Edifice.Meta.ReMoE` | ReLU-based differentiable sparse routing | [Unreviewed](validation.md) |
| **LoRA** | `Edifice.Meta.LoRA` | Low-Rank Adaptation for parameter-efficient fine-tuning | [Unreviewed](validation.md) |
| **DoRA** | `Edifice.Meta.DoRA` | Weight-decomposed LoRA | [Unreviewed](validation.md) |
| **Adapter** | `Edifice.Meta.Adapter` | Bottleneck adapter modules for transfer learning | [Unreviewed](validation.md) |
| **Hypernetwork** | `Edifice.Meta.Hypernetwork` | Networks that generate other networks' weights | [Unreviewed](validation.md) |
| **Capsule** | `Edifice.Meta.Capsule` | Dynamic routing between capsules | [Unreviewed](validation.md) |
| **MixtureOfDepths** | `Edifice.Meta.MixtureOfDepths` | Dynamic per-token compute allocation | [Unreviewed](validation.md) |
| **MixtureOfAgents** | `Edifice.Meta.MixtureOfAgents` | Multi-model proposer + aggregator | [Unreviewed](validation.md) |
| **RLHF Head** | `Edifice.Meta.RLHFHead` | Reward model and preference heads | [Unreviewed](validation.md) |
| **DPO** | `Edifice.Meta.DPO` | Direct Preference Optimization | [Unreviewed](validation.md) |
| **GRPO** | `Edifice.Meta.GRPO` | Group Relative Policy Optimization (DeepSeek-R1) | [Unreviewed](validation.md) |
| **KTO** | `Edifice.Meta.KTO` | Kahneman-Tversky Optimization (binary feedback) | [Unreviewed](validation.md) |
| **Speculative Decoding** | `Edifice.Meta.SpeculativeDecoding` | Draft + verify inference acceleration | [Unreviewed](validation.md) |
| **Test-Time Compute** | `Edifice.Meta.TestTimeCompute` | Adaptive test-time compute | [Unreviewed](validation.md) |
| **Mixture of Tokenizers** | `Edifice.Meta.MixtureOfTokenizers` | Multi-tokenization expert routing | [Unreviewed](validation.md) |
| **QAT** | `Edifice.Meta.QAT` | Quantization-Aware Training | [Unreviewed](validation.md) |
| **Hybrid Builder** | `Edifice.Meta.HybridBuilder` | Configurable SSM/Attention ratio | [Unreviewed](validation.md) |
| **Agent Swarm** | `Edifice.Meta.AgentSwarm` | Communication-augmented multi-agent ensemble | [Unreviewed](validation.md) |
| **Coconut** | `Edifice.Meta.Coconut` | Continuous Chain of Thought via latent reasoning | [Unreviewed](validation.md) |
| **Distillation Head** | `Edifice.Meta.DistillationHead` | Projects student hidden states to match teacher representations | [Unreviewed](validation.md) |
| **EAGLE-3** | `Edifice.Meta.Eagle3` | Multi-level feature fusion draft head for speculative decoding | [Unreviewed](validation.md) |
| **Kron-LoRA** | `Edifice.Meta.KronLoRA` | Hybrid Kronecker-LoRA adapters | [Unreviewed](validation.md) |
| **mHC** | `Edifice.Meta.ManifoldHC` | Manifold-Constrained Hyper-Connections | [Unreviewed](validation.md) |
| **Message-Passing Agents** | `Edifice.Meta.MessagePassingAgents` | GNN-inspired agent communication graph | [Unreviewed](validation.md) |
| **Mixture of Expert Depths** | `Edifice.Meta.MixtureOfExpertDepths` | MoED: joint expert selection + depth routing | [Unreviewed](validation.md) |
| **Mixture of Recursions** | `Edifice.Meta.MixtureOfRecursions` | MoR: dynamic recursive depth with per-token routing | [Unreviewed](validation.md) |
| **Mixture of Transformers** | `Edifice.Meta.MixtureOfTransformers` | MoT: modality-sparse multi-modal architecture | [Unreviewed](validation.md) |
| **PiSSA** | `Edifice.Meta.PiSSA` | Principal Singular Values and Singular Vectors Adaptation | [Unreviewed](validation.md) |
| **Router Network** | `Edifice.Meta.RouterNetwork` | Learned input-level dispatch to specialist models | [Unreviewed](validation.md) |
| **Speculative Head** | `Edifice.Meta.SpeculativeHead` | Multi-head parallel draft with per-head MLPs (Medusa/EAGLE) | [Unreviewed](validation.md) |
| **Stateful Agent** | `Edifice.Meta.StatefulAgent` | Multi-turn architecture wrapper with persistent memory | [Unreviewed](validation.md) |
| **VeRA** | `Edifice.Meta.VeRA` | Vector-based Random Matrix Adaptation | [Unreviewed](validation.md) |

## Detection & Segmentation

| Architecture | Module | Key Feature | Validation evidence |
|-------------|--------|-------------|---------------------|
| **DETR** | `Edifice.Detection.DETR` | Set-based detection with bipartite matching | [Checkpoint (limited scope)](validation.md) |
| **RT-DETR** | `Edifice.Detection.RTDETR` | Real-time DETR, hybrid CNN+transformer encoder | [Unreviewed](validation.md) |
| **SAM 2** | `Edifice.Detection.SAM2` | Promptable segmentation for images + video | [Unreviewed](validation.md) |

## Audio

| Architecture | Module | Key Feature | Validation evidence |
|-------------|--------|-------------|---------------------|
| **EnCodec** | `Edifice.Audio.EnCodec` | Neural audio codec (encoder + RVQ + decoder) | [Unreviewed](validation.md) |
| **VALL-E** | `Edifice.Audio.VALLE` | Codec language model for zero-shot TTS | [Unreviewed](validation.md) |
| **SoundStorm** | `Edifice.Audio.SoundStorm` | Parallel audio token generation | [Unreviewed](validation.md) |
| **Whisper** | `Edifice.Audio.Whisper` | Encoder-decoder ASR (OpenAI) | [Checkpoint; backend limitation](validation.md) |
| **F5-TTS** | `Edifice.Audio.F5TTS` | Flow-matching text-to-speech | [Unreviewed](validation.md) |
| **Wav2Vec 2.0** | `Edifice.Audio.Wav2Vec2` | Self-supervised speech representation learning | [Unreviewed](validation.md) |

## Robotics

| Architecture | Module | Key Feature | Validation evidence |
|-------------|--------|-------------|---------------------|
| **ACT** | `Edifice.Robotics.ACT` | Action Chunking Transformer for imitation learning | [Unreviewed](validation.md) |
| **OpenVLA** | `Edifice.Robotics.OpenVLA` | Vision-Language-Action model for robot control | [Unreviewed](validation.md) |
| **Diffusion Policy** | `Edifice.Robotics.DiffusionPolicy` | Visuomotor policy learning via action diffusion | [Unreviewed](validation.md) |

## RL & World Models

| Architecture | Module | Key Feature | Validation evidence |
|-------------|--------|-------------|---------------------|
| **PolicyValue** | `Edifice.RL.PolicyValue` | Actor-critic policy-value network | [Unreviewed](validation.md) |
| **Decision Transformer** | `Edifice.RL.DecisionTransformer` | Offline RL as conditional sequence generation | [Unreviewed](validation.md) |
| **World Model** | `Edifice.WorldModel.WorldModel` | Encoder + dynamics + reward head | [Unreviewed](validation.md) |
| **Medusa** | `Edifice.Inference.Medusa` | Multi-head speculative decoding | [Unreviewed](validation.md) |
| **Autoregressive Head** | `Edifice.RL.AutoregressiveHead` | Autoregressive action head with cross-component conditioning | [Unreviewed](validation.md) |
| **Pointer Network** | `Edifice.RL.PointerNetwork` | Attention-based pointer network for entity selection | [Unreviewed](validation.md) |

## Multimodal

| Architecture | Module | Key Feature | Validation evidence |
|-------------|--------|-------------|---------------------|
| **Multimodal Fusion** | `Edifice.Multimodal.Fusion` | MLP projection, cross-attention, Perceiver resampler | [Unreviewed](validation.md) |

## Interpretability

| Architecture | Module | Key Feature | Validation evidence |
|-------------|--------|-------------|---------------------|
| **BatchTopK SAE** | `Edifice.Interpretability.BatchTopKSAE` | BatchTopK sparse autoencoder | [Unreviewed](validation.md) |
| **Concept Bottleneck** | `Edifice.Interpretability.ConceptBottleneck` | CBM for inherently interpretable prediction | [Unreviewed](validation.md) |
| **Cross-Layer Transcoder** | `Edifice.Interpretability.CrossLayerTranscoder` | Cross-layer transcoder (status: broken, needs redesign) | [Unreviewed](validation.md) |
| **Crosscoder** | `Edifice.Interpretability.Crosscoder` | Joint sparse dictionary learning across multiple sources | [Unreviewed](validation.md) |
| **DAS Probe** | `Edifice.Interpretability.DASProbe` | Rank-limited linear probe (status: mislabeled, not true DAS) | [Unreviewed](validation.md) |
| **Gated SAE** | `Edifice.Interpretability.GatedSAE` | Gated sparse autoencoder (status: broken, gate does not train) | [Unreviewed](validation.md) |
| **JumpReLU SAE** | `Edifice.Interpretability.JumpReluSAE` | JumpReLU sparse autoencoder | [Unreviewed](validation.md) |
| **LEACE** | `Edifice.Interpretability.LEACE` | LEAst-squares Concept Erasure (Belrose et al., 2023) | [Unreviewed](validation.md) |
| **Linear Probe** | `Edifice.Interpretability.LinearProbe` | Linear probe for interpretability analysis | [Unreviewed](validation.md) |
| **Matryoshka SAE** | `Edifice.Interpretability.MatryoshkaSAE` | Matryoshka SAE (status: broken, nested-prefix loss missing) | [Unreviewed](validation.md) |

## Other

| Architecture | Module | Key Feature | Validation evidence |
|-------------|--------|-------------|---------------------|
| **Liquid NN** | `Edifice.Liquid` | Continuous-time ODE dynamics (LTC cells) | [Unreviewed](validation.md) |
| **SNN** | `Edifice.Neuromorphic.SNN` | Leaky integrate-and-fire, surrogate gradients | [Unreviewed](validation.md) |
| **ANN2SNN** | `Edifice.Neuromorphic.ANN2SNN` | Convert trained ANNs to spiking networks | [Unreviewed](validation.md) |
| **Sparse Autoencoder** | `Edifice.Interpretability.SparseAutoencoder` | Feature extraction from model activations | [Unreviewed](validation.md) |
| **Transcoder** | `Edifice.Interpretability.Transcoder` | Cross-layer mechanistic interpretability | [Unreviewed](validation.md) |
| **FNO** | `Edifice.Scientific.FNO` | Fourier Neural Operator for solving PDEs | [Unreviewed](validation.md) |
| **DeepONet** | `Edifice.Scientific.DeepONet` | Deep Operator Network | [Unreviewed](validation.md) |
| **TNO** | `Edifice.Scientific.TNO` | Temporal Neural Operator for time-dependent PDEs | [Unreviewed](validation.md) |

## Shared Building Blocks

| Block | Module | Purpose |
|-------|--------|---------|
| **RMSNorm** | `Edifice.Blocks.RMSNorm` | Root Mean Square normalization |
| **SwiGLU** | `Edifice.Blocks.SwiGLU` | Gated FFN with SiLU activation |
| **RoPE** | `Edifice.Blocks.RoPE` | Rotary position embedding (3D + 4D) |
| **ALiBi** | `Edifice.Blocks.ALiBi` | Attention with linear biases |
| **Patch Embed** | `Edifice.Blocks.PatchEmbed` | Image-to-patch tokenization |
| **Sinusoidal PE** | `Edifice.Blocks.SinusoidalPE` | Fixed sinusoidal position encoding + timestep embedding |
| **Sinusoidal PE 2D** | `Edifice.Blocks.SinusoidalPE2D` | 2D spatial positional encoding |
| **Adaptive Norm** | `Edifice.Blocks.AdaptiveNorm` | Condition-dependent normalization (AdaLN) |
| **Cross Attention** | `Edifice.Blocks.CrossAttention` | Cross-attention between two sequences |
| **SDPA** | `Edifice.Blocks.SDPA` | Scaled Dot-Product Attention |
| **FFN** | `Edifice.Blocks.FFN` | Standard and gated feed-forward networks |
| **Transformer Block** | `Edifice.Blocks.TransformerBlock` | Pre-norm block with pluggable attention |
| **Causal Mask** | `Edifice.Blocks.CausalMask` | Unified causal mask creation |
| **Depthwise Conv** | `Edifice.Blocks.DepthwiseConv` | 1D depthwise separable convolution |
| **BBox Head** | `Edifice.Blocks.BBoxHead` | Bounding box regression MLP |
| **Upsample 2x** | `Edifice.Blocks.Upsample2x` | Nearest-neighbor 2x spatial upsample |
| **Model Builder** | `Edifice.Blocks.ModelBuilder` | Sequence/vision model skeletons |
| **Message Passing** | `Edifice.Graph.MessagePassing` | Generic MPNN framework, global pooling |
| **Scalable-Softmax** | `Edifice.Blocks.SSMax` | Drop-in softmax replacement for long sequences |
| **Softpick** | `Edifice.Blocks.Softpick` | Non-saturating sparse attention function |
| **KV Cache** | `Edifice.Blocks.KVCache` | Inference-time KV caching |
