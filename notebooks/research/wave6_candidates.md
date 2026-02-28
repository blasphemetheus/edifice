# Wave 6 Architecture Candidates

> Research compiled 2026-02-28. Filtered against 234 existing Edifice architectures.

---

## Status Overview

| Category | Candidates | Top Picks |
|----------|-----------|-----------|
| Attention / Sequence | 10 | AdaSplash, EFLA, GDA, M-DGSA, MoSA, CLA |
| SSM / Recurrent | 8 | RWKV-7, S7, LinOSS, Falcon-H1, GAM |
| Generative | 12 | Block Diffusion, FlowAR, HART, Diffusion Forcing, NOVA, Latte |
| Vision / Multimodal | 8 | Hiera, NaViT, Vision-LSTM, Vim, Chameleon |
| Meta / PEFT / MoE | 12 | PiSSA, MoH, DeepSeekMoE, MatFormer, BOFT, SwitchHead |
| Audio / Speech | 4 | MusicGen, MaskGCT, Stable Audio, Spark-TTS |
| RL / Robotics | 6 | DreamerV3, TD-MPC2, DIAMOND, pi0, Octo |
| Graph / Scientific | 10 | PaiNN, LNO, MeshGraphNet, DiGress, Neural SDE, GFlowNet |

---

## 1. Attention / Sequence

### 1.1 AdaSplash (Adaptive Sparse Flash Attention)
- **Paper**: arXiv:2502.12082 (2025). Deep-Spin group (IST Lisboa).
- **Core**: Replaces softmax with alpha-entmax (learnable sparse activation producing exact zeros). Hybrid Halley-bisection algorithm makes entmax GPU-efficient. Data-dependent exact sparsity, not fixed patterns.
- **Value**: The only attention variant with differentiable exact sparsity. Approaches FlashAttention-2 speed.
- **Complexity**: Medium. Iterative root-finding for entmax, but standard attention structure otherwise.

### 1.2 Error-Free Linear Attention (EFLA)
- **Paper**: arXiv:2512.12602 (Dec 2025). DECLARE Lab (SUTD).
- **Core**: Reformulates delta rule update as continuous-time ODE with exact closed-form solution, eliminating discretization error in DeltaNet. Computable in linear time with full parallelism via rank-1 matrix exponential trick.
- **Value**: Direct improvement over DeltaNet (already implemented). Drop-in replacement with lower perplexity.
- **Complexity**: Medium. ODE-based exact solution with rank-1 trick. Similar scan structure to DeltaNet.

### 1.3 Grouped Differential Attention (GDA)
- **Paper**: arXiv:2510.06949 (Oct 2025). Motif Technologies.
- **Core**: Extends DiffTransformer with unbalanced head allocation between signal-preserving and noise-control groups. More heads for signal, fewer for noise. Noise-control heads repeated via controlled repetition (analogous to GQA sharing).
- **Value**: Natural evolution of DiffTransformer (already implemented). Combines GQA sharing with differential attention.
- **Complexity**: Low-Medium. Asymmetric head group management on top of existing DiffTransformer.

### 1.4 Differential Gated Self-Attention (M-DGSA)
- **Paper**: arXiv:2505.24054 (NeurIPS 2025). TU Wien.
- **Core**: Biologically inspired lateral inhibition. Excitatory and inhibitory attention branches fused by input-dependent sigmoid gate. Unlike DiffTransformer (fixed lambda subtraction), uses dynamic gating.
- **Value**: NeurIPS 2025. Distinct from both DiffTransformer and FoX. Minimal overhead, cross-domain.
- **Complexity**: Low-Medium. Sigmoid gate modulating subtraction of two softmax maps.

### 1.5 Threshold Differential Attention (TDA)
- **Paper**: arXiv:2601.12145 (Jan 2026).
- **Core**: Row-wise extreme-value thresholding with length-dependent gate on attention scores. Retains only exceedances above threshold. Ultra-sparse, sink-free, non-dispersive patterns.
- **Value**: Novel sparsity via extreme value theory. Different angle from entmax or top-k.
- **Complexity**: Medium. Extreme-value thresholding with length-dependent gate.

### 1.6 Mixture of Sparse Attention (MoSA)
- **Paper**: arXiv:2505.00315 (May 2025).
- **Core**: Expert-choice routing selects k tokens per attention head from sequence of length T. Reduces per-head complexity from O(T^2) to O(k^2 + T). Content-based, not position-based. Only sparse attention that outperforms dense baselines (up to 27%).
- **Value**: Unique MoE+attention intersection. Outperforms dense attention.
- **Complexity**: Medium-High. Expert-choice routing for token selection within attention.

### 1.7 Cross-Layer Attention (CLA)
- **Paper**: arXiv:2405.12981 (NAACL 2025). MIT.
- **Core**: Shares KV activations across adjacent transformer layers. CLA2 shares pairs, CLA3 shares triples. 2x KV cache reduction on top of GQA/MQA. Adjacent layers compute highly similar KV representations.
- **Value**: Orthogonal to existing attention mechanisms. Purely a model-builder pattern.
- **Complexity**: Low-Medium. KV routing across layers.

### 1.8 GatedFWA (Gated Flash Windowed Attention)
- **Paper**: arXiv:2512.07782 (Dec 2025).
- **Core**: Learnable per-token/head gate added to SWA, accumulated as decay bias on attention logits. Acts as learnable contraction in associative memory recurrence.
- **Value**: Integrates cleanly with existing SWA infrastructure. Negligible overhead.
- **Complexity**: Low-Medium. Gate on top of sliding window attention.

### 1.9 Superlinear Multi-Step Attention
- **Paper**: arXiv:2601.18401 (Jan 2026). Concavity AI.
- **Core**: Reformulates causal attention as multi-step search. O(L^(1+1/N)) complexity. No eligible position structurally excluded.
- **Value**: Novel complexity class between linear and quadratic.
- **Complexity**: High. Multi-step search with accumulation, span-search, span-attention.

### 1.10 LoLA (Low-Rank Linear Attention + Sparse Cache)
- **Paper**: arXiv:2505.23666 (2025).
- **Core**: Three-tier memory: sliding window cache (recent), sparse global cache (difficult pairs), recurrent hidden state (generic). 97.4% passkey retrieval vs 0.6% base.
- **Value**: Novel three-tier memory for linear attention inference.
- **Complexity**: Medium. Primarily an inference strategy rather than training architecture.

---

## 2. SSM / Recurrent

### 2.1 RWKV-7 "Goose"
- **Paper**: arXiv:2503.14456 (Mar 2025). RWKV Foundation.
- **Core**: Generalized delta rule with vector-valued gating and in-context learning rates. Diagonal-plus-rank-one state update. Separates removal and replacement keys. Can recognize all regular languages.
- **Value**: Major evolution beyond RWKV-6 (implemented as :rwkv). New 3B SoTA. Production-deployed.
- **Complexity**: Medium-High. Complex discretization with separate removal/replacement keys.

### 2.2 S7 (Selective and Simplified State Space)
- **Paper**: arXiv:2410.03464 (ICLR 2025).
- **Core**: Simplified SSM with input-dependent transition matrices and stable reparameterization preventing vanishing/exploding gradients. Provable stability guarantees unlike Mamba.
- **Value**: Clean, simple design complementing S4/S4D/S5. Stability guarantees unique among SSMs.
- **Complexity**: Low-Medium. Fewer moving parts than Mamba.

### 2.3 LinOSS (Linear Oscillatory State-Space)
- **Paper**: arXiv:2410.03943 (ICLR 2025).
- **Core**: SSM based on forced harmonic oscillators with nonneg diagonal state matrix. Implicit/IMEX discretization via parallel scans. Provably universal. IMEX variant conserves time-reversibility.
- **Value**: 2x better than Mamba on 50k-length sequences. Biologically inspired. Different SSM paradigm.
- **Complexity**: Medium. Nontrivial discretization but clean recurrence.

### 2.4 D-LinOSS (Damped LinOSS)
- **Paper**: arXiv:2505.12171 (May 2025).
- **Core**: Extends LinOSS with learnable energy dissipation. Stable dynamics under simple parameterization. Halves hyperparameter search space.
- **Value**: Natural extension of LinOSS. Faster convergence.
- **Complexity**: Low (incremental over LinOSS).

### 2.5 Falcon-H1 (Hybrid-Head)
- **Paper**: arXiv:2507.22448 (Jul 2025). TII.
- **Core**: Parallel hybrid-head: attention heads and Mamba-2 SSM heads operate side by side within same layer (not interleaved across layers). Large RoPE base. 256K context.
- **Value**: Production-validated 0.5B-34B. 4x input throughput. Distinct hybrid pattern.
- **Complexity**: Medium. Parallel attention+SSM heads within a layer.

### 2.6 GAM (Gated Associative Memory)
- **Paper**: arXiv:2509.00605 (Sep 2025).
- **Core**: Fully parallel O(N) architecture. Two pathways: causal conv for local context + parallel associative memory for global patterns, fused via gating.
- **Value**: Attention-free, distinct from SSM and linear-attention families.
- **Complexity**: Medium. Two parallel pathways plus gating.

### 2.7 mGRADE (Minimal Gated Recurrence + Delay Conv)
- **Paper**: arXiv:2507.01829 (Jul 2025).
- **Core**: Temporal 1D-conv with learnable spacings + minGRU. 20% less memory than pure alternatives.
- **Value**: Designed for edge deployment. Builds on existing MinGRU. Relevant to ExPhil's 16ms budget.
- **Complexity**: Low.

### 2.8 Zamba2 (Shared-Attention SSM Hybrid)
- **Paper**: arXiv:2405.16712 (2024). Zyphra.
- **Core**: Mamba-2 backbone with two shared attention blocks (ABAB pattern). Shared weights + LoRA projections for per-position specialization.
- **Value**: Very parameter-efficient hybrid. Shared-attention-with-LoRA pattern is unique.
- **Complexity**: Medium.

---

## 3. Generative

### 3.1 Block Diffusion (BD3-LM)
- **Paper**: arXiv:2503.09573 (ICLR 2025 Oral).
- **Core**: Decomposes sequence into fixed-size blocks. Within-block: discrete diffusion. Between-blocks: autoregressive conditioning. Interpolates AR (block=1) and diffusion (block=seq_len). KV caching between blocks.
- **Value**: Elegant AR+diffusion unification for language. Directly complements existing MDLM.
- **Complexity**: Medium.

### 3.2 FlowAR (Scale-wise AR + Flow Matching)
- **Paper**: arXiv:2412.15205 (ICML 2025).
- **Core**: Next-scale AR prediction (like VAR) but with per-scale flow matching instead of discrete VQ. Any off-the-shelf continuous VAE works as tokenizer. FID 1.65 with 1.9B params.
- **Value**: Elegant hybrid of AR + flow matching. Modular -- any VAE works.
- **Complexity**: Medium.

### 3.3 HART (Hybrid AR Transformer)
- **Paper**: arXiv:2410.10812 (ICLR 2025).
- **Core**: Decomposes continuous latents into discrete tokens (coarse) + continuous residuals (detail). AR transformer for discrete + lightweight 37M residual diffusion head (8 steps). 4.5-7.7x faster than diffusion.
- **Value**: Principled decomposition of what AR captures vs what needs diffusion. Direct 1024x1024 generation.
- **Complexity**: Medium.

### 3.4 Diffusion Forcing
- **Paper**: arXiv:2407.01392 (NeurIPS 2024).
- **Core**: Causal model denoising tokens with independent per-token noise levels. Each token at different diffusion timestep. Enables rolling generation past training horizon. Variational bound on all subsequence likelihoods.
- **Value**: Unique bridge between next-token prediction and full-sequence diffusion. General framework.
- **Complexity**: Medium.

### 3.5 NOVA (Non-Quantized Video AR)
- **Paper**: arXiv:2412.14169 (ICLR 2025).
- **Core**: Temporal frame-by-frame AR + spatial bidirectional diffusion within each frame. No VQ needed. 0.6B params outperforms larger models.
- **Value**: Clean decomposition of temporal causality and spatial bidirectionality.
- **Complexity**: Medium.

### 3.6 Latte (Video DiT with Factored Attention)
- **Paper**: arXiv:2401.03048 (TMLR 2025).
- **Core**: Interleaved spatial + temporal transformer blocks for video DiT. Four factorization variants. Systematic study of PE, timestep injection, training.
- **Value**: Clean video DiT reference. Simpler than CogVideoX.
- **Complexity**: Low-Medium.

### 3.7 MaskBit (Embedding-Free Bit Token Generation)
- **Paper**: arXiv:2409.16211 (TMLR 2024).
- **Core**: LFQ binary bit tokens + masked prediction. No embedding lookup. FID 1.52 on ImageNet 256 with only 305M params.
- **Value**: Extremely parameter-efficient. Embedding-free is architecturally novel.
- **Complexity**: Low-Medium.

### 3.8 Dream 7B (Diffusion LLM)
- **Paper**: arXiv:2508.15487 (Aug 2025).
- **Core**: Mask diffusion with shifted prediction. AR weight initialization. Adaptive per-token noise. Matches AR LLMs on general/math/code.
- **Value**: Most powerful open diffusion LM. Shifted prediction is novel.
- **Complexity**: Medium-High.

### 3.9 FAR (Frequency AR)
- **Paper**: arXiv:2503.05305 (Mar 2025).
- **Core**: Frequency-progressive AR -- generates low-to-high frequency components. Orthogonal to spatial and scale-wise approaches.
- **Value**: Unique generation axis (frequency domain).
- **Complexity**: Medium.

### 3.10 D2iT (Dynamic Diffusion Transformer)
- **Paper**: arXiv:2504.09454 (CVPR 2025).
- **Core**: Dynamic VAE encoding different regions at different rates. Two-stage: rough global + region-specific refinement.
- **Value**: Adaptive-resolution latent space. Multi-grained noise prediction.
- **Complexity**: Medium-High.

### 3.11 LiT (Linear Diffusion Transformer)
- **Paper**: arXiv:2501.12976 (ICCV 2025).
- **Core**: Converts pretrained DiT to linear-attention DiT via depthwise conv + weight inheritance. 20-33% of original training cost.
- **Value**: Practical DiT efficiency recipe. Edge/laptop hardware.
- **Complexity**: Low-Medium.

### 3.12 Pyramidal Flow Matching
- **Paper**: arXiv:2410.05954 (ICLR 2025).
- **Core**: Multi-resolution pyramid stages for video flow matching. Temporal pyramid compresses full-res history.
- **Value**: Efficient video generation at 768p/24fps. Competitive with commercial models.
- **Complexity**: Medium-High.

---

## 4. Vision / Multimodal

### 4.1 Hiera
- **Paper**: arXiv:2306.00989 (ICML 2023, widely adopted 2025).
- **Core**: Stripped-down hierarchical ViT with no relative position biases, no shifted windows. Relies on MAE pretraining for inductive biases. 30-40% faster than comparable models.
- **Value**: SAM 2 backbone (already in Edifice). Simple hierarchical ViT distinct from Swin.
- **Complexity**: Low-Medium.

### 4.2 NaViT (Native Resolution ViT)
- **Paper**: arXiv:2307.06304 (NeurIPS 2023, Google DeepMind).
- **Core**: Sequence packing for arbitrary resolution/aspect ratio in single batch. Masked attention, masked pooling, factorized fractional positional embeddings.
- **Value**: Solves fundamental ViT resolution limitation. Used in PaLI-3, Gemini.
- **Complexity**: Low-Medium.

### 4.3 Vision-LSTM (ViL)
- **Paper**: arXiv:2406.04303 (ICLR 2025).
- **Core**: xLSTM (mLSTM blocks) for vision. Alternating top-to-bottom / bottom-to-top scanning.
- **Value**: Composes directly with existing xLSTM/mLSTM. Lower FLOPs than transformers.
- **Complexity**: Medium.

### 4.4 Vision Mamba (Vim)
- **Paper**: arXiv:2401.09417 (ICML 2024).
- **Core**: Bidirectional Mamba for vision patches. 2.8x faster than DeiT, 86.8% less GPU memory.
- **Value**: Pure Mamba for vision. Complements MambaVision (hybrid).
- **Complexity**: Medium.

### 4.5 Chameleon (Early-Fusion Multimodal)
- **Paper**: arXiv:2405.09818 (Meta 2024).
- **Core**: VQ-VAE image tokenizer + text tokens interleaved in single decoder-only transformer. No modality-specific components after tokenization.
- **Value**: Simplest possible multimodal. Composes existing VQ-VAE + DecoderOnly.
- **Complexity**: Low-Medium.

### 4.6 Florence-2 (Unified Vision via Seq2Seq)
- **Paper**: arXiv:2311.06242 (CVPR 2024).
- **Core**: DaViT encoder + encoder-decoder transformer. All vision tasks as text generation.
- **Value**: Single architecture for captioning, detection, segmentation, grounding. 0.2-0.7B.
- **Complexity**: Medium.

### 4.7 Grounding DINO
- **Paper**: arXiv:2303.05499 (ECCV 2024).
- **Core**: Dual-encoder (image + text) with cross-modal fusion at three levels. Open-vocabulary detection.
- **Value**: Standard open-set detection. Complements existing DETR/RT-DETR.
- **Complexity**: Medium-High.

### 4.8 VGGT (Visual Geometry Grounded Transformer)
- **Paper**: arXiv:2503.11651 (CVPR 2025 Best Paper).
- **Core**: Feed-forward transformer inferring all 3D scene attributes from arbitrary views.
- **Value**: CVPR 2025 Best Paper. Clean 3D reconstruction pipeline.
- **Complexity**: High.

---

## 5. Meta / PEFT / MoE

### 5.1 PiSSA (Principal SVD Adaptation)
- **Paper**: arXiv:2404.02948 (NeurIPS 2024 Spotlight).
- **Core**: Same architecture as LoRA but initializes A, B with principal SVD components of W, freezing residual. Fine-tunes most important weight directions.
- **Value**: Drop-in LoRA upgrade. NeurIPS spotlight. Same inference cost.
- **Complexity**: Low. Identical to LoRA structurally, different initialization via SVD.

### 5.2 MoH (Mixture-of-Head Attention)
- **Paper**: arXiv:2410.11842 (2024).
- **Core**: Router selects top-K attention heads per token. Weighted summation replaces standard sum. MoH-LLaMA3-8B uses 75% of heads, outperforms by 2.4%.
- **Value**: Simple, proven on LLaMA3. 10-50% attention compute reduction.
- **Complexity**: Low. Router + top-K head masking.

### 5.3 DeepSeekMoE (Fine-Grained + Shared Experts)
- **Paper**: arXiv:2401.06066 (2024).
- **Core**: Fine-grained expert segmentation (split each FFN into m smaller experts) + shared expert isolation (always-on shared experts capturing common knowledge).
- **Value**: Dominant MoE design in DeepSeek V2/V3/R1. Fundamentally different topology.
- **Complexity**: Medium.

### 5.4 MatFormer (Nested Transformer for Elastic Inference)
- **Paper**: arXiv:2310.07707 (NeurIPS 2024).
- **Core**: Nested sub-FFN blocks of increasing width (matryoshka). Train once, extract hundreds of models. Self-contained speculative decoding.
- **Value**: Single-training multi-deployment. Highly practical.
- **Complexity**: Low-Medium.

### 5.5 BOFT (Butterfly Orthogonal Fine-Tuning)
- **Paper**: arXiv:2311.06243 (2023).
- **Core**: Orthogonal transformations via butterfly factorization (FFT-inspired). O(d log d) params. Preserves hyperspherical energy of weight space.
- **Value**: Fundamentally different from low-rank (LoRA family). Preserves angular structure.
- **Complexity**: Medium. Butterfly matrix construction.

### 5.6 SwitchHead (MoE Attention Heads)
- **Paper**: arXiv:2312.07987 (2023).
- **Core**: MoE for K, Q, V, O projections in attention. Up to 8x fewer attention matrices. 262M matches standard with 44% compute.
- **Value**: Only MoE that sparsifies attention computation directly.
- **Complexity**: Medium.

### 5.7 UMoE (Unified Attention+FFN MoE)
- **Paper**: arXiv:2505.07260 (NeurIPS 2025 Spotlight).
- **Core**: Decomposes attention into token mixing + token-wise expert processing. Unified MoE sharing experts between attention and FFN.
- **Value**: NeurIPS 2025 spotlight. Novel unification.
- **Complexity**: Medium-High.

### 5.8 LoRA-XS (Extremely Small LoRA)
- **Paper**: arXiv:2405.17604 (2024).
- **Core**: Small trainable R matrix between two frozen SVD-derived matrices. 100x fewer params than LoRA.
- **Value**: Extreme parameter efficiency for edge/multi-task.
- **Complexity**: Low.

### 5.9 Hydra (Sequentially-Dependent Draft Heads)
- **Paper**: arXiv:2402.05109 (2024).
- **Core**: Draft heads sequentially dependent on earlier heads' predictions (not just base hidden state). 1.31x speedup over Medusa.
- **Value**: Direct upgrade to existing Medusa. Same deployment model.
- **Complexity**: Low.

### 5.10 Auxiliary-Loss-Free Load Balancing
- **Paper**: arXiv:2408.15664 (2024). Used in DeepSeek-V3.
- **Core**: Learnable bias per expert adjusted based on load. No auxiliary loss interference.
- **Value**: Composable with any top-K MoE. Eliminates performance-balance tradeoff.
- **Complexity**: Low.

### 5.11 Expert Choice Routing
- **Paper**: arXiv:2202.09368 (2022, widely adopted 2025).
- **Core**: Inverts routing: each expert selects top-K tokens. Perfect load balance by construction.
- **Value**: Foundational routing alternative. Composable with existing MoE modules.
- **Complexity**: Low.

### 5.12 DynMoLE (Dynamic Mixture of LoRA Experts)
- **Paper**: arXiv:2504.00661 (2025).
- **Core**: Multiple LoRA adapters as experts with Tsallis entropy-based dynamic routing. Variable number of active experts per token.
- **Value**: Bridges MoE + PEFT. Novel dynamic routing.
- **Complexity**: Medium.

---

## 6. Audio / Speech

### 6.1 MusicGen (Codebook Delay Pattern)
- **Paper**: arXiv:2306.05284 (Meta 2023, dominant through 2025).
- **Core**: Single-stage AR transformer with codebook delay pattern for parallel multi-codebook generation. 1-step delay between RVQ codebooks. Cross-attention on T5/CLAP text embeddings.
- **Value**: No music generation in Edifice. Delay pattern is architecturally novel. Dominant open music architecture.
- **Complexity**: Medium.

### 6.2 MaskGCT (Masked Generative Codec Transformer)
- **Paper**: arXiv:2409.00750 (Sep 2024).
- **Core**: Fully non-autoregressive zero-shot TTS via two-stage masked prediction: text->semantic tokens, semantic->acoustic tokens. No duration prediction needed.
- **Value**: Novel paradigm vs AR (VALL-E) and diffusion (F5-TTS). Masked generative for audio.
- **Complexity**: Medium-High.

### 6.3 Stable Audio (Latent DiT for Audio)
- **Paper**: arXiv:2402.04825, arXiv:2407.14358 (2024).
- **Core**: Audio VAE + DiT backbone with timing conditioning (start time + duration). Variable-length stereo 44.1kHz up to 47s.
- **Value**: Builds on existing DiT + VAE. Audio-specific latent diffusion.
- **Complexity**: Medium.

### 6.4 Spark-TTS (BiCodec)
- **Paper**: arXiv:2503.01710 (Mar 2025).
- **Core**: Dual-path speech codec: low-bitrate semantic tokens (wav2vec features) + fixed-length global tokens (speaker attributes). Explicit content/timbre disentanglement.
- **Value**: Disentangled speech codec distinct from EnCodec's multi-codebook approach.
- **Complexity**: Medium.

---

## 7. RL / Robotics

### 7.1 DreamerV3
- **Paper**: arXiv:2301.04104 (Nature 2025).
- **Core**: RSSM (Recurrent State Space Model) with categorical discrete latents. Symlog normalization, percentile returns, symexp twohot loss. Actor-critic trains in imagination.
- **Value**: THE reference model-based RL architecture. Masters 150+ tasks. Enhances existing WorldModel.
- **Complexity**: Medium.

### 7.2 TD-MPC2 (Implicit World Model)
- **Paper**: arXiv:2310.16828 (ICLR 2024).
- **Core**: Decoder-free world model -- all MLPs with LayerNorm + Mish + SimNorm. Plans via MPC in latent space. 317M agent handles 80 tasks.
- **Value**: Simplest world model. All MLPs. SimNorm is novel normalization.
- **Complexity**: Low.

### 7.3 DIAMOND (Diffusion World Model)
- **Paper**: arXiv:2405.12399 (NeurIPS 2024 Spotlight).
- **Core**: Diffusion model predicting next frame from previous frames + actions. Preserves visual details that tokenization discards. Best Atari 100k world-model result.
- **Value**: Diffusion for environment simulation. Complements existing diffusion modules.
- **Complexity**: Medium.

### 7.4 IRIS (Transformer World Model)
- **Paper**: arXiv:2209.00588 (ICLR 2023).
- **Core**: VQ-VAE tokenizer + autoregressive transformer for dynamics. Policy trains on imagined trajectories. "GPT for environment simulation."
- **Value**: Clean composition of existing VQ-VAE + decoder-only transformer.
- **Complexity**: Low-Medium.

### 7.5 pi0 (VLM + Flow Matching Action)
- **Paper**: arXiv:2410.24164 (Physical Intelligence, Oct 2024).
- **Core**: VLM backbone + flow-matching action expert. FAST tokenizer (DCT + BPE for actions). Leading open-source generalist robot policy.
- **Value**: Novel VLM + flow matching combination for robotics.
- **Complexity**: Medium-High.

### 7.6 Octo (Generalist Robot Policy)
- **Paper**: arXiv:2405.12213 (RSS 2024).
- **Core**: Transformer backbone + CNN patch encoders + diffusion action MLP. Pretrained on 800K trajectories. Supports flexible observation/action spaces.
- **Value**: Clean, well-documented. Good middle-ground between Diffusion Policy and pi0.
- **Complexity**: Medium.

---

## 8. Graph / Scientific

### 8.1 PaiNN (Polarizable Atom Interaction NN)
- **Paper**: arXiv:2102.03150 (ICML 2021, widely used 2025).
- **Core**: Scalar + vectorial features throughout message passing. Scalarization approach (scalar coefficients on 3D vectors). Predicts tensorial properties.
- **Value**: Simpler equivariant GNN than NequIP/MACE. Widely used baseline.
- **Complexity**: Medium.

### 8.2 LNO (Latent Neural Operator)
- **Paper**: arXiv:2406.03923 (NeurIPS 2024).
- **Core**: Physics-Cross-Attention transforms data to latent space, solves PDE operator via transformer, decodes back. Only transforms at first/last layer. 50% less GPU memory.
- **Value**: Successor to FNO/DeepONet. SOTA on 4/6 PDE benchmarks.
- **Complexity**: Medium.

### 8.3 MeshGraphNet
- **Paper**: arXiv:2010.03409 (ICLR 2021, NVIDIA 2024 extensions).
- **Core**: Simulation meshes as graphs. Encode-process-decode with 10-15 GNN layers. Multi-type edges (mesh/world/collision). Resolution-independent.
- **Value**: Standard for learned physics simulation. Natural for scientific family.
- **Complexity**: Medium.

### 8.4 DiGress (Discrete Graph Diffusion)
- **Paper**: arXiv:2209.14734 (ICLR 2023).
- **Core**: Discrete diffusion progressively editing graphs (add/remove edges, change categories). Graph transformer denoiser. 3x validity improvement for molecular generation.
- **Value**: Extends diffusion family into graph domain.
- **Complexity**: Medium.

### 8.5 Neural SDE
- **Paper**: Multiple works, including arXiv:2402.14989 (ICLR 2024).
- **Core**: Neural ODE + learnable diffusion (noise) term. Three stable classes: Langevin, Linear Noise, Geometric. Uncertainty quantification.
- **Value**: Natural extension of existing Neural ODE. Fills stochastic gap.
- **Complexity**: Medium.

### 8.6 GFlowNet
- **Paper**: arXiv:2106.04399 (NeurIPS 2021), arXiv:2111.09266 (JMLR 2023).
- **Core**: Generative flow through DAG where trajectories build structures (e.g., molecules). Flow proportional to reward. Diverse high-reward samples.
- **Value**: Unique architecture for molecular design/combinatorial optimization. Not diffusion or VAE.
- **Complexity**: Medium-High.

### 8.7 GNOT (General Neural Operator Transformer)
- **Paper**: arXiv:2302.14376 (ICML 2023).
- **Core**: Heterogeneous normalized attention for multiple input functions + irregular meshes. Geometric gating for soft domain decomposition.
- **Value**: More general than FNO. Handles irregular meshes.
- **Complexity**: Medium.

### 8.8 Pairformer (AlphaFold 3 Style)
- **Paper**: Nature 2024 (AlphaFold 3).
- **Core**: Triangle attention on pair representations. 48 blocks of triangular attention + updates. Diffusion module for atom coordinates.
- **Value**: AlphaFold 3 core. Triangle attention is unique primitive.
- **Complexity**: High.

### 8.9 PIKAN (Physics-Informed KAN)
- **Paper**: Multiple 2025 variants.
- **Core**: KAN replaces MLPs in PINNs. Learnable spline activations. Addresses spectral bias.
- **Value**: Bridges existing KAN + scientific families.
- **Complexity**: Medium.

### 8.10 CatFlow (Variational Flow Matching for Graphs)
- **Paper**: arXiv:2406.04843 (NeurIPS 2024).
- **Core**: Flow matching for categorical data (nodes/edges). Reduces to classifier training per component.
- **Value**: Extends flow matching to graph domain.
- **Complexity**: Medium.

---

## Recommended Wave 6 Selection

### Tier 1 -- High impact, low-medium complexity (implement first)

| # | Architecture | Family | Complexity | Rationale |
|---|-------------|--------|-----------|-----------|
| 1 | PiSSA | Meta/PEFT | Low | Drop-in LoRA upgrade, NeurIPS spotlight |
| 2 | MoH | Meta/Attention | Low | Simple head routing, proven on LLaMA3 |
| 3 | GDA | Attention | Low-Med | Natural DiffTransformer evolution |
| 4 | M-DGSA | Attention | Low-Med | NeurIPS 2025, lateral inhibition |
| 5 | Hydra | Inference | Low | Direct Medusa upgrade |
| 6 | S7 | SSM | Low-Med | Simple stable SSM, ICLR 2025 |
| 7 | Hiera | Vision | Low-Med | SAM 2 backbone, widely used |
| 8 | Latte | Generative | Low-Med | Clean video DiT reference |
| 9 | MaskBit | Generative | Low-Med | Embedding-free, 305M params |
| 10 | TD-MPC2 | RL | Low | All-MLP world model, SimNorm |

### Tier 2 -- Medium complexity, high value

| # | Architecture | Family | Complexity | Rationale |
|---|-------------|--------|-----------|-----------|
| 11 | EFLA | Attention | Medium | DeltaNet improvement, exact ODE solution |
| 12 | LinOSS | SSM | Medium | Oscillatory SSM, ICLR 2025, universal |
| 13 | Block Diffusion | Generative | Medium | AR+diffusion unification, ICLR 2025 Oral |
| 14 | FlowAR | Generative | Medium | AR + flow matching, ICML 2025 |
| 15 | Diffusion Forcing | Generative | Medium | Per-token noise, NeurIPS 2024 |
| 16 | MusicGen | Audio | Medium | First music gen, delay pattern |
| 17 | DreamerV3 | RL | Medium | Reference model-based RL, Nature 2025 |
| 18 | PaiNN | Graph | Medium | Equivariant GNN baseline |
| 19 | LNO | Scientific | Medium | FNO/DeepONet successor |
| 20 | Neural SDE | Energy | Medium | Neural ODE extension |

### Tier 3 -- Higher complexity, ambitious

| # | Architecture | Family | Complexity | Rationale |
|---|-------------|--------|-----------|-----------|
| 21 | RWKV-7 | Recurrent | Med-High | Production, major RWKV evolution |
| 22 | MoSA | Attention | Med-High | Outperforms dense, MoE+attention |
| 23 | DeepSeekMoE | Meta | Medium | Dominant MoE design |
| 24 | Falcon-H1 | SSM/Hybrid | Medium | Parallel hybrid-head |
| 25 | HART | Generative | Medium | Hybrid discrete+continuous AR |
| 26 | NOVA | Generative | Medium | Video AR without VQ |
| 27 | DiGress | Graph | Medium | Graph diffusion |
| 28 | GFlowNet | Graph | Med-High | Unique flow-network sampler |
| 29 | pi0 | Robotics | Med-High | VLM + flow matching for robots |
| 30 | Pairformer | Scientific | High | AlphaFold 3 triangle attention |
