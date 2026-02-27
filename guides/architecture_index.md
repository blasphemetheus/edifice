# Architecture Index

Complete listing of all 196 architectures and 20 shared blocks in Edifice.

For descriptions, paper references, and decision guidance, see [Architecture Taxonomy](architecture_taxonomy.md).

---

## Feedforward

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **MLP** | `Edifice.Feedforward.MLP` | Multi-layer perceptron with configurable hidden sizes |
| **KAN** | `Edifice.Feedforward.KAN` | Kolmogorov-Arnold Networks, learnable activation functions |
| **KAT** | `Edifice.Feedforward.KAT` | Kolmogorov-Arnold Transformer (KAN + attention) |
| **TabNet** | `Edifice.Feedforward.TabNet` | Attentive feature selection for tabular data |
| **BitNet** | `Edifice.Feedforward.BitNet` | Ternary/binary weight quantization (1.58-bit) |

## Transformer

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **Decoder-Only** | `Edifice.Transformer.DecoderOnly` | GPT-style with GQA, RoPE/iRoPE, SwiGLU, RMSNorm |
| **Multi-Token Prediction** | `Edifice.Transformer.MultiTokenPrediction` | Predict next N tokens simultaneously |
| **Byte Latent Transformer** | `Edifice.Transformer.ByteLatentTransformer` | Byte-level processing via encoder-latent-decoder |
| **Nemotron-H** | `Edifice.Transformer.NemotronH` | NVIDIA's hybrid Mamba-Transformer |

## State Space Models

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **S4** | `Edifice.SSM.S4` | HiPPO DPLR initialization, long-range memory |
| **S4D** | `Edifice.SSM.S4D` | Diagonal state space, simplified S4 |
| **S5** | `Edifice.SSM.S5` | MIMO diagonal SSM with D skip connection |
| **H3** | `Edifice.SSM.H3` | Two SSMs with multiplicative gating + short convolution |
| **Hyena** | `Edifice.SSM.Hyena` | Long convolution hierarchy, implicit filters |
| **Mamba** | `Edifice.SSM.Mamba` | Selective SSM, parallel associative scan |
| **Mamba-2 (SSD)** | `Edifice.SSM.MambaSSD` | Structured state space duality, chunk-wise matmul |
| **Mamba (Cumsum)** | `Edifice.SSM.MambaCumsum` | Mamba with configurable scan algorithm |
| **Mamba (Hillis-Steele)** | `Edifice.SSM.MambaHillisSteele` | Mamba with max-parallelism scan |
| **BiMamba** | `Edifice.SSM.BiMamba` | Bidirectional Mamba for non-causal tasks |
| **GatedSSM** | `Edifice.SSM.GatedSSM` | Gated temporal with gradient checkpointing |
| **Jamba** | `Edifice.SSM.Hybrid` | Mamba + Attention hybrid (configurable ratio) |
| **Zamba** | `Edifice.SSM.Zamba` | Mamba + single shared attention layer |
| **StripedHyena** | `Edifice.SSM.StripedHyena` | Interleaved Hyena long conv + gated conv |
| **Mamba-3** | `Edifice.SSM.Mamba3` | Complex states, trapezoidal discretization, MIMO |
| **GSS** | `Edifice.SSM.GSS` | Gated State Space (simplified S4 with gating) |
| **Hymba** | `Edifice.SSM.Hymba` | Hybrid Mamba + attention with learnable meta tokens |
| **SS Transformer** | `Edifice.SSM.SSTransformer` | State Space Transformer |

## Attention & Linear Attention

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **Multi-Head Attention** | `Edifice.Attention.MultiHead` | Sliding window, QK LayerNorm |
| **GQA** | `Edifice.Attention.GQA` | Grouped Query Attention, fewer KV heads |
| **Perceiver** | `Edifice.Attention.Perceiver` | Cross-attention to learned latents, input-agnostic |
| **FNet** | `Edifice.Attention.FNet` | Fourier Transform replacing attention |
| **Linear Transformer** | `Edifice.Attention.LinearTransformer` | Kernel-based O(N) attention |
| **Nystromformer** | `Edifice.Attention.Nystromformer` | Nystrom approximation of attention matrix |
| **Performer** | `Edifice.Attention.Performer` | FAVOR+ random feature attention |
| **RetNet** | `Edifice.Attention.RetNet` | Multi-scale retention, O(1) recurrent inference |
| **RWKV-7** | `Edifice.Attention.RWKV` | Linear attention, O(1) space, "Goose" architecture |
| **GLA** | `Edifice.Attention.GLA` | Gated Linear Attention with data-dependent decay |
| **HGRN-2** | `Edifice.Attention.HGRN` | Hierarchically gated linear RNN, state expansion |
| **Griffin/Hawk** | `Edifice.Attention.Griffin` | RG-LRU + local attention (Griffin) or pure RG-LRU (Hawk) |
| **Diff Transformer** | `Edifice.Attention.DiffTransformer` | Noise-cancelling dual softmax subtraction |
| **MLA** | `Edifice.Attention.MLA` | Multi-Head Latent Attention (DeepSeek KV compression) |
| **Based** | `Edifice.Attention.Based` | Taylor expansion linear attention |
| **Mega** | `Edifice.Attention.Mega` | Moving average + gated attention |
| **InfiniAttention** | `Edifice.Attention.InfiniAttention` | Compressive memory for unbounded context |
| **Conformer** | `Edifice.Attention.Conformer` | Conv-augmented transformer for audio/speech |
| **Ring Attention** | `Edifice.Attention.RingAttention` | Distributed chunked attention for long sequences |
| **Lightning Attention** | `Edifice.Attention.LightningAttention` | Hybrid linear/softmax with I/O-aware tiling |
| **Gated Attention** | `Edifice.Attention.GatedAttention` | Sigmoid post-attention gate (NeurIPS 2025) |
| **NSA** | `Edifice.Attention.NSA` | Native Sparse Attention (DeepSeek three-path) |
| **KDA** | `Edifice.Attention.KDA` | Kimi Delta Attention, channel-wise decay |
| **Flash Linear Attention** | `Edifice.Attention.FlashLinearAttention` | Optimized linear attention |
| **YaRN** | `Edifice.Attention.YARN` | RoPE context extension via frequency scaling |
| **Dual Chunk** | `Edifice.Attention.DualChunk` | Dual Chunk Attention for long-context |
| **TMRoPE** | `Edifice.Attention.TMRoPE` | Time-aligned Multimodal RoPE |
| **RNoPE-SWA** | `Edifice.Attention.RNoPESWA` | No positional encoding + sliding window |

## Recurrent Networks

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **LSTM/GRU** | `Edifice.Recurrent` | Classic recurrent with multi-layer stacking |
| **xLSTM** | `Edifice.Recurrent.XLSTM` | Exponential gating, matrix memory (sLSTM/mLSTM) |
| **MinGRU** | `Edifice.Recurrent.MinGRU` | Minimal GRU, parallel-scannable |
| **MinLSTM** | `Edifice.Recurrent.MinLSTM` | Minimal LSTM, parallel-scannable |
| **DeltaNet** | `Edifice.Recurrent.DeltaNet` | Delta rule-based linear RNN |
| **TTT** | `Edifice.Recurrent.TTT` | Test-Time Training, self-supervised at inference |
| **Titans** | `Edifice.Recurrent.Titans` | Neural long-term memory, surprise-gated |
| **Reservoir** | `Edifice.Recurrent.Reservoir` | Echo State Networks with fixed random reservoir |
| **sLSTM** | `Edifice.Recurrent.SLSTM` | Scalar LSTM with exponential gating |
| **xLSTM v2** | `Edifice.Recurrent.XLSTMv2` | Updated mLSTM with matrix memory |
| **Gated DeltaNet** | `Edifice.Recurrent.GatedDeltaNet` | Linear attention with data-dependent gating |
| **TTT-E2E** | `Edifice.Recurrent.TTTE2E` | End-to-end test-time training |
| **Native Recurrence** | `Edifice.Recurrent.NativeRecurrence` | Native recurrence block |

## Vision

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **ViT** | `Edifice.Vision.ViT` | Vision Transformer, patch embedding |
| **DeiT** | `Edifice.Vision.DeiT` | Data-efficient ViT with distillation token |
| **Swin** | `Edifice.Vision.SwinTransformer` | Shifted window attention, hierarchical features |
| **U-Net** | `Edifice.Vision.UNet` | Encoder-decoder with skip connections |
| **ConvNeXt** | `Edifice.Vision.ConvNeXt` | Modernized ConvNet with transformer-inspired design |
| **MLP-Mixer** | `Edifice.Vision.MLPMixer` | Pure MLP with token/channel mixing |
| **FocalNet** | `Edifice.Vision.FocalNet` | Focal modulation, hierarchical context |
| **PoolFormer** | `Edifice.Vision.PoolFormer` | Average pooling token mixer (MetaFormer) |
| **NeRF** | `Edifice.Vision.NeRF` | Neural radiance field, coordinate-to-color mapping |
| **Gaussian Splat** | `Edifice.Vision.GaussianSplat` | 3D Gaussian Splatting (NeRF successor) |
| **MambaVision** | `Edifice.Vision.MambaVision` | 4-stage hierarchical CNN+Mamba+Attention |
| **DINOv2** | `Edifice.Vision.DINOv2` | Self-distillation vision backbone |
| **MetaFormer** | `Edifice.Vision.MetaFormer` | Architecture-first framework (+ CAFormer variant) |
| **EfficientViT** | `Edifice.Vision.EfficientViT` | Linear attention ViT |

## Convolutional

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **Conv1D/2D** | `Edifice.Convolutional.Conv` | Configurable convolution blocks with BN, activation, dropout |
| **ResNet** | `Edifice.Convolutional.ResNet` | Residual/bottleneck blocks, configurable depth |
| **DenseNet** | `Edifice.Convolutional.DenseNet` | Dense connections, feature reuse |
| **TCN** | `Edifice.Convolutional.TCN` | Dilated causal convolutions for sequences |
| **MobileNet** | `Edifice.Convolutional.MobileNet` | Depthwise separable convolutions |
| **EfficientNet** | `Edifice.Convolutional.EfficientNet` | Compound scaling (depth, width, resolution) |

## Generative

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **VAE** | `Edifice.Generative.VAE` | Reparameterization trick, KL divergence, beta-VAE |
| **VQ-VAE** | `Edifice.Generative.VQVAE` | Discrete codebook, straight-through estimator |
| **GAN** | `Edifice.Generative.GAN` | Generator/discriminator, WGAN-GP support |
| **Diffusion (DDPM)** | `Edifice.Generative.Diffusion` | Denoising diffusion, sinusoidal time embedding |
| **DDIM** | `Edifice.Generative.DDIM` | Deterministic diffusion sampling, fast inference |
| **DiT** | `Edifice.Generative.DiT` | Diffusion Transformer, AdaLN-Zero conditioning |
| **DiT v2** | `Edifice.Generative.DiTv2` | Improved adaptive norm conditioning |
| **Latent Diffusion** | `Edifice.Generative.LatentDiffusion` | Diffusion in compressed latent space |
| **Consistency Model** | `Edifice.Generative.ConsistencyModel` | Single-step generation via consistency training |
| **Score SDE** | `Edifice.Generative.ScoreSDE` | Continuous SDE framework (VP-SDE, VE-SDE) |
| **Flow Matching** | `Edifice.Generative.FlowMatching` | ODE-based generation, multiple loss variants |
| **Rectified Flow** | `Edifice.Generative.RectifiedFlow` | Straight-trajectory flow matching, fewer steps |
| **Normalizing Flow** | `Edifice.Generative.NormalizingFlow` | Affine coupling layers (RealNVP-style) |
| **MMDiT** | `Edifice.Generative.MMDiT` | Multimodal Diffusion Transformer (FLUX.1, SD3) |
| **SoFlow** | `Edifice.Generative.SoFlow` | Flow matching + consistency loss |
| **VAR** | `Edifice.Generative.VAR` | Visual Autoregressive (next-scale prediction) |
| **Linear DiT (SANA)** | `Edifice.Generative.LinearDiT` | Linear attention for diffusion, 100x speedup |
| **SiT** | `Edifice.Generative.SiT` | Scalable Interpolant Transformer |
| **Transfusion** | `Edifice.Generative.Transfusion` | Unified AR text + diffusion images |
| **MAR** | `Edifice.Generative.MAR` | Masked Autoregressive generation |
| **CogVideoX** | `Edifice.Generative.CogVideoX` | 3D causal VAE + expert transformer for video |
| **TRELLIS** | `Edifice.Generative.TRELLIS` | Sparse 3D lattice + rectified flow |
| **MDLM** | `Edifice.Generative.MDLM` | Discrete diffusion language model |

## Contrastive & Self-Supervised

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **SimCLR** | `Edifice.Contrastive.SimCLR` | NT-Xent contrastive loss, projection head |
| **BYOL** | `Edifice.Contrastive.BYOL` | No negatives, momentum encoder |
| **Barlow Twins** | `Edifice.Contrastive.BarlowTwins` | Cross-correlation redundancy reduction |
| **MAE** | `Edifice.Contrastive.MAE` | Masked Autoencoder, 75% patch masking |
| **VICReg** | `Edifice.Contrastive.VICReg` | Variance-Invariance-Covariance regularization |
| **JEPA** | `Edifice.Contrastive.JEPA` | Joint Embedding Predictive Architecture |
| **Temporal JEPA** | `Edifice.Contrastive.TemporalJEPA` | V-JEPA for video/temporal sequences |
| **SigLIP** | `Edifice.Contrastive.SigLIP` | Sigmoid contrastive learning (CLIP improvement) |

## Graph & Set Networks

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **GCN** | `Edifice.Graph.GCN` | Spectral graph convolutions (Kipf & Welling) |
| **GAT** | `Edifice.Graph.GAT` | Graph attention with multi-head support |
| **GIN** | `Edifice.Graph.GIN` | Graph Isomorphism Network, maximally expressive |
| **GraphSAGE** | `Edifice.Graph.GraphSAGE` | Inductive learning, neighborhood sampling |
| **Graph Transformer** | `Edifice.Graph.GraphTransformer` | Full attention over nodes with edge features |
| **PNA** | `Edifice.Graph.PNA` | Principal Neighbourhood Aggregation |
| **GINv2** | `Edifice.Graph.GINv2` | GIN with edge features |
| **SchNet** | `Edifice.Graph.SchNet` | Continuous-filter convolutions for molecules |
| **EGNN** | `Edifice.Graph.EGNN` | E(n)-equivariant GNN for molecular simulation |
| **DeepSets** | `Edifice.Sets.DeepSets` | Permutation-invariant set functions |
| **PointNet** | `Edifice.Sets.PointNet` | Point cloud processing with T-Net alignment |

## Energy, Probabilistic & Memory

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **EBM** | `Edifice.Energy.EBM` | Energy-based models, contrastive divergence |
| **Hopfield** | `Edifice.Energy.Hopfield` | Modern continuous Hopfield networks |
| **Neural ODE** | `Edifice.Energy.NeuralODE` | Continuous-depth networks via ODE solvers |
| **Bayesian NN** | `Edifice.Probabilistic.Bayesian` | Weight uncertainty, variational inference |
| **MC Dropout** | `Edifice.Probabilistic.MCDropout` | Uncertainty estimation via dropout at inference |
| **Evidential NN** | `Edifice.Probabilistic.EvidentialNN` | Dirichlet priors for uncertainty |
| **NTM** | `Edifice.Memory.NTM` | Neural Turing Machine, differentiable memory |
| **Memory Network** | `Edifice.Memory.MemoryNetwork` | End-to-end memory with multi-hop attention |
| **Engram** | `Edifice.Memory.Engram` | O(1) hash-based associative memory |

## Meta-Learning & Specialized

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **MoE** | `Edifice.Meta.MoE` | Mixture of Experts with top-k/hash routing |
| **MoE v2** | `Edifice.Meta.MoEv2` | Expert-choice routing + shared experts + bias balancing |
| **Switch MoE** | `Edifice.Meta.SwitchMoE` | Top-1 routing with load balancing |
| **Soft MoE** | `Edifice.Meta.SoftMoE` | Fully differentiable soft token routing |
| **ReMoE** | `Edifice.Meta.ReMoE` | ReLU-based differentiable sparse routing |
| **LoRA** | `Edifice.Meta.LoRA` | Low-Rank Adaptation for parameter-efficient fine-tuning |
| **DoRA** | `Edifice.Meta.DoRA` | Weight-decomposed LoRA |
| **Adapter** | `Edifice.Meta.Adapter` | Bottleneck adapter modules for transfer learning |
| **Hypernetwork** | `Edifice.Meta.Hypernetwork` | Networks that generate other networks' weights |
| **Capsule** | `Edifice.Meta.Capsule` | Dynamic routing between capsules |
| **MixtureOfDepths** | `Edifice.Meta.MixtureOfDepths` | Dynamic per-token compute allocation |
| **MixtureOfAgents** | `Edifice.Meta.MixtureOfAgents` | Multi-model proposer + aggregator |
| **RLHF Head** | `Edifice.Meta.RLHFHead` | Reward model and preference heads |
| **DPO** | `Edifice.Meta.DPO` | Direct Preference Optimization |
| **GRPO** | `Edifice.Meta.GRPO` | Group Relative Policy Optimization (DeepSeek-R1) |
| **KTO** | `Edifice.Meta.KTO` | Kahneman-Tversky Optimization (binary feedback) |
| **Speculative Decoding** | `Edifice.Meta.SpeculativeDecoding` | Draft + verify inference acceleration |
| **Test-Time Compute** | `Edifice.Meta.TestTimeCompute` | Adaptive test-time compute |
| **Mixture of Tokenizers** | `Edifice.Meta.MixtureOfTokenizers` | Multi-tokenization expert routing |
| **QAT** | `Edifice.Meta.QAT` | Quantization-Aware Training |
| **Hybrid Builder** | `Edifice.Meta.HybridBuilder` | Configurable SSM/Attention ratio |

## Detection & Segmentation

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **DETR** | `Edifice.Detection.DETR` | Set-based detection with bipartite matching |
| **RT-DETR** | `Edifice.Detection.RTDETR` | Real-time DETR, hybrid CNN+transformer encoder |
| **SAM 2** | `Edifice.Detection.SAM2` | Promptable segmentation for images + video |

## Audio

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **EnCodec** | `Edifice.Audio.EnCodec` | Neural audio codec (encoder + RVQ + decoder) |
| **VALL-E** | `Edifice.Audio.VALLE` | Codec language model for zero-shot TTS |
| **SoundStorm** | `Edifice.Audio.SoundStorm` | Parallel audio token generation |
| **Whisper** | `Edifice.Audio.Whisper` | Encoder-decoder ASR (OpenAI) |

## Robotics

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **ACT** | `Edifice.Robotics.ACT` | Action Chunking Transformer for imitation learning |
| **OpenVLA** | `Edifice.Robotics.OpenVLA` | Vision-Language-Action model for robot control |

## RL & World Models

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **PolicyValue** | `Edifice.RL.PolicyValue` | Actor-critic policy-value network |
| **Decision Transformer** | `Edifice.RL.DecisionTransformer` | Offline RL as conditional sequence generation |
| **World Model** | `Edifice.WorldModel.WorldModel` | Encoder + dynamics + reward head |
| **Medusa** | `Edifice.Inference.Medusa` | Multi-head speculative decoding |

## Multimodal

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **Multimodal Fusion** | `Edifice.Multimodal.Fusion` | MLP projection, cross-attention, Perceiver resampler |

## Other

| Architecture | Module | Key Feature |
|-------------|--------|-------------|
| **Liquid NN** | `Edifice.Liquid` | Continuous-time ODE dynamics (LTC cells) |
| **SNN** | `Edifice.Neuromorphic.SNN` | Leaky integrate-and-fire, surrogate gradients |
| **ANN2SNN** | `Edifice.Neuromorphic.ANN2SNN` | Convert trained ANNs to spiking networks |
| **Sparse Autoencoder** | `Edifice.Interpretability.SparseAutoencoder` | Feature extraction from model activations |
| **Transcoder** | `Edifice.Interpretability.Transcoder` | Cross-layer mechanistic interpretability |
| **FNO** | `Edifice.Scientific.FNO` | Fourier Neural Operator for solving PDEs |

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
