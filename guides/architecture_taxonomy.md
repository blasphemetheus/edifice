# Architecture Taxonomy

Edifice provides **184 registered architectures** across **25 families**, spanning from foundational MLPs to cutting-edge state space models, audio codecs, robotics, and 3D generation. This document serves as a comprehensive reference catalog — organized as a taxonomy with provenance, strengths/weaknesses, and adoption context for every architecture.

For guided learning, see [Learning Path](learning_path.md). For choosing architectures by problem type, see [Problem Landscape](problem_landscape.md).

---

## Family Overview

| Family | Count | Primary Input | Core Use Case | Era |
|--------|-------|---------------|---------------|-----|
| **Transformer** | 4 | Sequences | Autoregressive generation backbone | 2017–2025 |
| **SSM** (State Space Models) | 19 | Sequences | Efficient long-range sequence modeling | 2021–2025 |
| **Attention & Linear Attention** | 34 | Sequences | Token mixing, global dependencies | 2017–2025 |
| **Recurrent** | 15 | Sequences | Sequential state tracking, temporal memory | 1997–2025 |
| **Vision** | 15 | Images/Patches | Image classification, segmentation, synthesis | 2020–2025 |
| **Convolutional** | 6 | Images/Sequences | Local pattern extraction | 2015–2019 |
| **Generative** | 22 | Noise/Latents | Data generation, density estimation | 2014–2025 |
| **Graph** | 9 | Graphs | Node/graph classification, molecular modeling | 2017–2025 |
| **Contrastive & Self-Supervised** | 8 | Paired views | Representation learning without labels | 2020–2025 |
| **Meta-Learning & Composition** | 22 | Varied | Routing, adaptation, preference optimization | 2016–2025 |
| **Feedforward** | 5 | Tabular/Vectors | Feature transformation, tabular data | 1986–2024 |
| **Memory** | 3 | Sequences | External differentiable memory | 2014–2025 |
| **Energy** | 3 | Varied | Energy landscapes, continuous dynamics | 2006–2020 |
| **Probabilistic** | 3 | Varied | Uncertainty quantification | 2015–2018 |
| **Audio** | 3 | Audio/Speech | Neural audio codecs, TTS | 2023–2025 |
| **Robotics** | 2 | Vision+Actions | Imitation learning, VLA | 2024–2025 |
| **Interpretability** | 2 | Activations | Feature extraction, mechanistic analysis | 2024–2025 |
| **Sets** | 2 | Point sets | Permutation-invariant set functions | 2017 |
| **Neuromorphic** | 2 | Spike trains | Biologically-plausible, ultra-low-power | 2015–2019 |
| **Scientific** | 1 | Functions/PDEs | Operator learning, PDE solving | 2021–2025 |
| **Multimodal** | 1 | Multi-modal | Cross-modal fusion | 2024–2025 |
| **World Model** | 1 | Observations | Latent dynamics, planning | 2024–2025 |
| **RL** | 1 | States/Actions | Actor-critic policy networks | 2024–2025 |
| **Liquid** | 1 | Sequences | Continuous-time adaptive dynamics | 2021 |
| **Inference** | 1 | Sequences | Speculative decoding acceleration | 2024–2025 |

---

## Detailed Taxonomy

### Sequence Processing

#### State Space Models (19 architectures)

State Space Models (SSMs) model sequences through continuous-time linear dynamical systems discretized for neural networks. The key equation is:

```
h[t] = A * h[t-1] + B * x[t]    (state update)
y[t] = C * h[t]                  (output)
```

SSMs achieve O(N) complexity for sequence length N (vs O(N²) for attention), making them the leading architecture family for long-range sequence modeling. The evolution from S4 to Mamba to Mamba-3 represents one of the most active research fronts in deep learning.

##### S4

| | |
|---|---|
| **Module** | `Edifice.SSM.S4` |
| **Paper** | Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces" (ICLR 2022) |
| **Reference** | [arXiv:2111.00396](https://arxiv.org/abs/2111.00396) |
| **Origin** | Stanford / Albert Gu |

The foundational SSM architecture. S4 introduced HiPPO initialization — a mathematically principled way to initialize the state matrix A so it optimally compresses continuous signals into finite-dimensional state. Uses DPLR (Diagonal Plus Low-Rank) decomposition for efficient computation.

**Strengths:** Exceptional long-range modeling (Path-X, LRA benchmarks), mathematically principled initialization, parallelizable via convolution mode.
**Weaknesses:** Complex implementation (DPLR decomposition), fixed (non-selective) dynamics, superseded by simpler variants.
**Who uses it:** Research benchmarking, long-range dependency tasks, historical baseline for SSM comparisons.

##### S4D

| | |
|---|---|
| **Module** | `Edifice.SSM.S4D` |
| **Paper** | Gu et al., "On the Parameterization and Initialization of Diagonal State Space Models" |
| **Reference** | [arXiv:2206.11893](https://arxiv.org/abs/2206.11893) |
| **Origin** | Stanford / Albert Gu |

Simplified S4 with purely diagonal state matrix, eliminating the DPLR decomposition. Serves as the bridge between original S4 and modern SSMs like Mamba.

**Strengths:** Dramatically simpler than S4, nearly identical performance, fewer parameters.
**Weaknesses:** Still uses fixed (non-selective) dynamics.
**Who uses it:** Ablation studies, educational purposes, lightweight SSM needs.

##### S5

| | |
|---|---|
| **Module** | `Edifice.SSM.S5` |
| **Paper** | Smith et al., "Simplified State Space Layers for Sequence Modeling" (ICLR 2023) |
| **Reference** | [arXiv:2208.04933](https://arxiv.org/abs/2208.04933) |
| **Origin** | University of Toronto |

Uses a single MIMO (Multi-Input Multi-Output) SSM instead of many parallel SISO systems. Simpler architecture with fewer parameters than Mamba while maintaining strong performance.

**Strengths:** Conceptually simple, efficient, good ablation baseline.
**Weaknesses:** Fixed dynamics (not input-dependent like Mamba), less expressive.
**Who uses it:** Research ablations to isolate what Mamba's selectivity contributes.

##### H3

| | |
|---|---|
| **Module** | `Edifice.SSM.H3` |
| **Paper** | Fu et al., "Hungry Hungry Hippos: Towards Language Modeling with State Space Models" (ICLR 2023) |
| **Reference** | [arXiv:2212.14052](https://arxiv.org/abs/2212.14052) |
| **Origin** | Stanford / Christopher Ré |

Combines two SSM layers with a short convolution and multiplicative gating. The "two-SSM + short conv" pattern bridges the gap between SSMs and Transformers on language modeling. One SSM captures local shifts, the other broader patterns, and a short convolution handles very local (1-4 token) patterns.

**Strengths:** Closes SSM-Transformer gap on language modeling, elegant multiplicative design.
**Weaknesses:** More complex than single-SSM approaches, superseded by Mamba.
**Who uses it:** Language modeling research, hybrid architecture exploration.

##### Hyena

| | |
|---|---|
| **Module** | `Edifice.SSM.Hyena` |
| **Paper** | Poli et al., "Hyena Hierarchy: Towards Larger Convolutional Language Models" (ICML 2023) |
| **Reference** | [arXiv:2302.10866](https://arxiv.org/abs/2302.10866) |
| **Origin** | Together AI / Michael Poli |

Replaces attention with a hierarchy of long convolutions and element-wise gating. Uses implicit filters (small MLPs that generate convolution kernels) for O(L log L) training via FFT and O(L) inference via recurrence.

**Strengths:** Sub-quadratic complexity, no attention needed, strong on language tasks.
**Weaknesses:** FFT-based training can be tricky to optimize, long-conv requires careful implementation.
**Who uses it:** Together AI (StripedHyena models), long-sequence processing.

##### Mamba

| | |
|---|---|
| **Module** | `Edifice.SSM.Mamba` |
| **Paper** | Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023) |
| **Reference** | [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) |
| **Origin** | Carnegie Mellon / Albert Gu & Tri Dao |

The breakthrough SSM architecture. Mamba makes the SSM parameters (B, C, Δ) **input-dependent** (selective), enabling content-based reasoning that fixed SSMs cannot do. Uses a parallel associative scan for O(N) training.

**Strengths:** O(N) complexity, selective dynamics, excellent language/sequence modeling, simple gated block design, widely adopted.
**Weaknesses:** Sequential scan limits GPU utilization vs attention's matmul-heavy approach, inference not yet as optimized as FlashAttention.
**Who uses it:** Widely adopted — state-of-the-art for efficient sequence models, AI21 (Jamba), Zyphra (Zamba), many research groups.

##### Mamba-2 (SSD)

| | |
|---|---|
| **Module** | `Edifice.SSM.MambaSSD` |
| **Paper** | Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms" (2024) |
| **Origin** | Princeton / Tri Dao & Albert Gu |

Reformulates Mamba as structured matrix multiplication (State Space Duality), enabling tensor core utilization. Splits sequences into chunks and uses dense matmul within chunks, tiny scan between chunks.

**Strengths:** 2-8x faster training than Mamba-1 via tensor cores, mathematically equivalent output.
**Weaknesses:** Chunk boundary effects, XLA implementation less optimized than fused CUDA kernels.
**Who uses it:** Production training of large SSM models.

##### Mamba-3

| | |
|---|---|
| **Module** | `Edifice.SSM.Mamba3` |
| **Paper** | Extension of Gu & Dao (2023, 2024) |
| **Origin** | Carnegie Mellon / Gu & Dao |

Extends Mamba with three innovations: (1) complex-valued state dynamics via 2×2 rotation matrices, (2) generalized trapezoidal discretization reducing approximation error, (3) MIMO rank-r updates for better hardware utilization.

**Strengths:** More expressive state dynamics, reduced discretization error, better GPU/TPU utilization.
**Weaknesses:** More parameters and complexity than Mamba-1.
**Who uses it:** Cutting-edge SSM research, high-performance sequence modeling.

##### Mamba (Cumsum)

| | |
|---|---|
| **Module** | `Edifice.SSM.MambaCumsum` |
| **Origin** | Edifice experimental variant |

Experimental Mamba variant for testing alternative scan algorithms (cumsum-based log-space reformulation). Currently uses Blelloch scan (same as regular Mamba) as the cumsum approach proved slower in XLA.

**Strengths:** Useful for scan algorithm experimentation.
**Weaknesses:** No performance advantage over standard Mamba.
**Who uses it:** Internal benchmarking and scan algorithm research.

##### Mamba (Hillis-Steele)

| | |
|---|---|
| **Module** | `Edifice.SSM.MambaHillisSteele` |
| **Origin** | Edifice experimental variant |

Mamba variant using Hillis-Steele parallel scan instead of Blelloch. O(L log L) work but ALL elements active at every level, maximizing GPU occupancy at the cost of more total operations.

**Strengths:** Maximum parallelism per scan level, potentially better GPU occupancy.
**Weaknesses:** O(L log L) total work vs O(L) for Blelloch.
**Who uses it:** GPU utilization experiments, comparison of scan algorithms.

##### BiMamba

| | |
|---|---|
| **Module** | `Edifice.SSM.BiMamba` |
| **Origin** | Multiple concurrent works extending Mamba bidirectionally |

Bidirectional Mamba that runs two parallel SSMs (forward and backward) for non-causal tasks. Combines outputs via concatenation and projection.

**Strengths:** Full sequence context for classification and offline analysis.
**Weaknesses:** Not usable for causal/real-time tasks, 2x compute of standard Mamba.
**Who uses it:** Sequence classification, offline analysis, fill-in-the-blank tasks.

##### GatedSSM

| | |
|---|---|
| **Module** | `Edifice.SSM.GatedSSM` |
| **Origin** | Edifice simplified variant |

Simplified gated temporal network inspired by SSMs but using sigmoid gating instead of true parallel scan. NOT a true Mamba implementation — uses mean pooling + projection instead of depthwise separable convolution.

**Strengths:** Numerically stable, lightweight, simple implementation, no NaN issues.
**Weaknesses:** Less expressive than true Mamba, no parallel scan.
**Who uses it:** Lightweight temporal processing, prototyping, when stability is paramount.

##### Jamba

| | |
|---|---|
| **Module** | `Edifice.SSM.Hybrid` |
| **Paper** | AI21 Labs, "Jamba: A Hybrid Transformer-Mamba Language Model" (2024) |
| **Origin** | AI21 Labs |

Hybrid architecture alternating Mamba blocks with periodic attention blocks. Most layers are O(L) Mamba blocks with attention at configurable intervals (default: every 4 layers) for long-range dependencies.

**Strengths:** Best of both worlds — Mamba efficiency with attention's long-range capability, configurable ratio.
**Weaknesses:** More complex than pure Mamba, still has some O(L²) attention layers.
**Who uses it:** AI21 Labs (Jamba-1.5 production model), hybrid architecture research.

##### Zamba

| | |
|---|---|
| **Module** | `Edifice.SSM.Zamba` |
| **Paper** | Zyphra, "Zamba: A Compact 7B SSM Hybrid Model" (2024) |
| **Reference** | [arXiv:2405.16712](https://arxiv.org/abs/2405.16712) |
| **Origin** | Zyphra |

Like Jamba but uses a **single shared attention layer** with reused weights instead of multiple independent attention layers. This achieves 10x KV cache reduction versus Jamba.

**Strengths:** 10x KV cache reduction, fewer parameters, insight that attention mainly provides global information flow.
**Weaknesses:** Less attention diversity than Jamba.
**Who uses it:** Zyphra (Zamba production models), memory-constrained deployments.

##### StripedHyena

| | |
|---|---|
| **Module** | `Edifice.SSM.StripedHyena` |
| **Paper** | Together AI, "StripedHyena: Moving Beyond Transformers with Hybrid Signal Processing Models" (2023) |
| **Origin** | Together AI |

Interleaves Hyena long convolution blocks (even layers, global context) with gated depthwise convolution blocks (odd layers, local patterns). The striped pattern balances efficiency and expressivity.

**Strengths:** Better efficiency than pure Hyena, retains global context capability, attention-free.
**Weaknesses:** Complex implementation, less widely adopted than Mamba-based hybrids.
**Who uses it:** Together AI (StripedHyena-7B), attention-free model research.

---

#### Attention & Linear Attention (19 architectures)

The attention family spans the full evolution from O(N²) quadratic softmax attention to O(N) linear approximations. This is the largest family in Edifice, reflecting the central role of attention mechanisms in modern deep learning.

##### Multi-Head Attention

| | |
|---|---|
| **Module** | `Edifice.Attention.MultiHead` |
| **Paper** | Vaswani et al., "Attention Is All You Need" (NeurIPS 2017) |
| **Origin** | Google Brain |

The foundational attention mechanism. Implements sliding window attention (O(K²) for window size K) and hybrid LSTM + attention. QK LayerNorm option for stable training.

**Strengths:** Universally understood, highly expressive, hardware-optimized (FlashAttention).
**Weaknesses:** O(N²) complexity for full attention, quadratic memory.
**Who uses it:** Universal — the backbone of GPT, BERT, and virtually all modern LLMs.

##### Grouped Query Attention (GQA)

| | |
|---|---|
| **Module** | `Edifice.Attention.GQA` |
| **Paper** | Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023) |
| **Origin** | Google Research |

Interpolation between Multi-Head Attention (MHA) and Multi-Query Attention (MQA). Groups of query heads share key/value heads, reducing KV cache size while maintaining near-MHA quality.

**Strengths:** Significant KV cache reduction, near-MHA quality, now standard in production LLMs.
**Weaknesses:** Slightly lower quality than full MHA.
**Who uses it:** LLaMA 2 70B, Mistral 7B, Gemma — now the default for production LLMs.

##### Multi-Head Latent Attention (MLA)

| | |
|---|---|
| **Module** | `Edifice.Attention.MLA` |
| **Paper** | DeepSeek-AI, "DeepSeek-V2" (2024) |
| **Reference** | [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) |
| **Origin** | DeepSeek |

Compresses key-value representations into low-rank latent vectors, reconstructing K,V on-the-fly during attention. Also uses decoupled RoPE to keep position information separate from compressed content.

**Strengths:** Dramatic KV cache reduction (beyond GQA), maintains attention quality via low-rank reconstruction.
**Weaknesses:** Additional computation for KV reconstruction, complex implementation.
**Who uses it:** DeepSeek-V2/V3 production models.

##### Differential Transformer

| | |
|---|---|
| **Module** | `Edifice.Attention.DiffTransformer` |
| **Paper** | Ye et al., "Differential Transformer" (Microsoft Research, 2024) |
| **Origin** | Microsoft Research |

Computes two independent attention maps per head and subtracts them. Shared noise patterns (tokens that universally attract attention) cancel out, amplifying signal-to-noise ratio — analogous to differential amplifiers in electronics.

**Strengths:** Better signal-to-noise ratio, reduces attention noise, same complexity as standard attention.
**Weaknesses:** 2x attention score computation per head (minor overhead), relatively new.
**Who uses it:** Microsoft Research, noise-sensitive attention applications.

##### RetNet

| | |
|---|---|
| **Module** | `Edifice.Attention.RetNet` |
| **Paper** | Sun et al., "Retentive Network: A Successor to Transformer for Large Language Models" (Microsoft, 2023) |
| **Reference** | [arXiv:2307.08621](https://arxiv.org/abs/2307.08621) |
| **Origin** | Microsoft Research |

Replaces attention with "retention" — a decay-based mechanism supporting three computation modes: parallel (training, O(L²)), recurrent (inference, O(1) per token), and chunkwise (long sequences, O(L)).

**Strengths:** Triple paradigm (parallel/recurrent/chunkwise), O(1) inference, multi-scale decay rates.
**Weaknesses:** Decay-based mechanism may lose fine-grained long-range patterns, less widely adopted than attention.
**Who uses it:** Microsoft (RetNet research models), efficient inference research.

##### RWKV-7

| | |
|---|---|
| **Module** | `Edifice.Attention.RWKV` |
| **Paper** | Peng et al., "RWKV: Reinventing RNNs for the Transformer Era" |
| **Reference** | [arXiv:2305.13048](https://arxiv.org/abs/2305.13048) |
| **Origin** | Bo Peng / RWKV community |

Linear attention architecture with O(1) space complexity per inference step. RWKV-7 ("Goose") uses a generalized delta rule surpassing the TC0 constraint. Combines WKV (weighted key-value) attention with channel-mixing FFN using R-gate and K-gate.

**Strengths:** O(1) inference, parallelizable training, shipped to 1.5B Windows devices, large community.
**Weaknesses:** O(1) state may lose long-range fine detail, community-driven (less corporate backing).
**Who uses it:** RWKV community (14B+ parameter models), Microsoft (on-device Copilot).

##### GLA

| | |
|---|---|
| **Module** | `Edifice.Attention.GLA` |
| **Paper** | "Gated Linear Attention Transformers with Hardware-Efficient Training" |
| **Origin** | Flash-linear-attention project |

Linear attention with data-dependent gating for improved expressiveness. Particularly effective on short sequences (<2K tokens) where it can outperform FlashAttention-2.

**Strengths:** O(L) complexity, data-dependent gating, competitive with attention on short sequences, native tensor ops.
**Weaknesses:** Less effective on very long sequences, less widely adopted.
**Who uses it:** Efficient attention research, short-sequence modeling.

##### HGRN-2

| | |
|---|---|
| **Module** | `Edifice.Attention.HGRN` |
| **Paper** | "HGRN2: Gated Linear RNNs with State Expansion" |
| **Reference** | [arXiv:2404.07904](https://arxiv.org/abs/2404.07904) |
| **Origin** | Linear attention research community |

Linear RNN with hierarchical gating and state expansion. Expands hidden state dimension during recurrence for richer internal representation, then contracts back.

**Strengths:** O(1) inference, O(L) training, state expansion for richer representations.
**Weaknesses:** Less expressive than attention, state expansion increases compute.
**Who uses it:** Efficient sequence modeling research.

##### Griffin/Hawk

| | |
|---|---|
| **Module** | `Edifice.Attention.Griffin` |
| **Paper** | De et al., "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models" (2024) |
| **Reference** | [arXiv:2402.19427](https://arxiv.org/abs/2402.19427) |
| **Origin** | Google DeepMind |

Hybrid architecture using Real-Gated Linear Recurrent Unit (RG-LRU) with periodic local attention. Pattern: 2 RG-LRU blocks → 1 Local Attention block → repeat. The `sqrt(1-a²)` term preserves hidden state norm for stable long-sequence training.

**Strengths:** Simpler than Mamba (just gates, no SSM projections), stable training, Google DeepMind backing.
**Weaknesses:** Local attention windows limit truly global reasoning, less widely adopted than Mamba.
**Who uses it:** Google DeepMind (RecurrentGemma), efficient language model research.

##### Perceiver

| | |
|---|---|
| **Module** | `Edifice.Attention.Perceiver` |
| **Paper** | Jaegle et al., "Perceiver IO: A General Architecture for Structured Inputs & Outputs" (DeepMind, 2021) |
| **Origin** | Google DeepMind |

Uses cross-attention to map arbitrary-size inputs to a fixed-size learned latent array, then self-attends over latents. Total complexity: O(N·M + M²) where M = num_latents << N.

**Strengths:** Input-agnostic (works with any modality), decouples compute from input size, elegant design.
**Weaknesses:** Latent bottleneck may lose fine-grained detail, slower than specialized architectures.
**Who uses it:** Multi-modal learning, variable-length input processing, DeepMind research.

##### FNet

| | |
|---|---|
| **Module** | `Edifice.Attention.FNet` |
| **Paper** | Lee-Thorp et al., "FNet: Mixing Tokens with Fourier Transforms" (Google, 2021) |
| **Origin** | Google Research |

Replaces self-attention with an unparameterized Fourier Transform for token mixing. No learnable attention parameters — FFT provides global token mixing at O(N log N) with zero attention parameters.

**Strengths:** ~7x faster training than attention, zero attention parameters, 92-97% of BERT quality.
**Weaknesses:** Slightly lower quality than attention, no learnable selectivity.
**Who uses it:** Efficiency research, speed-critical applications where near-attention quality suffices.

##### Linear Transformer

| | |
|---|---|
| **Module** | `Edifice.Attention.LinearTransformer` |
| **Paper** | Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (2020) |
| **Origin** | EPFL |

Replaces softmax attention with kernel-based linear attention using the ELU+1 feature map. Avoids the N×N attention matrix entirely by computing phi(K)ᵀ·V first (a d×d matrix).

**Strengths:** O(N) complexity, causal formulation equivalent to an RNN, simple implementation.
**Weaknesses:** Lower quality than softmax attention (feature map approximation), best when N > d.
**Who uses it:** Long-sequence processing, real-time applications, efficiency research.

##### Nystromformer

| | |
|---|---|
| **Module** | `Edifice.Attention.Nystromformer` |
| **Paper** | Xiong et al., "Nystromformer: A Nystrom-Based Algorithm for Approximating Self-Attention" (AAAI 2021) |
| **Origin** | University of Wisconsin |

Approximates the full attention matrix using Nyström method with landmark points. Samples M landmarks and reconstructs attention through them: A ≈ F1 · pinv(F2) · F3.

**Strengths:** O(N·M) complexity, good attention quality with few landmarks (M=32-64 typical).
**Weaknesses:** Kernel inverse O(M³) overhead, landmark selection heuristic.
**Who uses it:** Efficient attention research, long-document processing.

##### Performer

| | |
|---|---|
| **Module** | `Edifice.Attention.Performer` |
| **Paper** | Choromanski et al., "Rethinking Attention with Performers" (ICLR 2021) |
| **Origin** | Google Research |

Approximates softmax attention using FAVOR+ (Fast Attention Via positive Orthogonal Random features). Uses orthogonal random features to approximate the exponential kernel.

**Strengths:** O(N) time and space, mathematically principled approximation, unbiased estimator.
**Weaknesses:** Approximation quality degrades for very sharp attention patterns, random features add variance.
**Who uses it:** Long-sequence tasks, Google Research, efficient attention research.

##### Mega

| | |
|---|---|
| **Module** | `Edifice.Attention.Mega` |
| **Paper** | Ma et al., "Mega: Moving Average Equipped Gated Attention" (ICLR 2023) |
| **Reference** | [arXiv:2209.10655](https://arxiv.org/abs/2209.10655) |
| **Origin** | Meta AI |

Combines exponential moving averages (EMA) for local context with single-head gated attention for global context. Sub-quadratic complexity: EMA is O(L·D_ema) and only a single attention head.

**Strengths:** Sub-quadratic complexity, elegant EMA + attention combination, strong benchmark results.
**Weaknesses:** Single-head attention may limit global context diversity.
**Who uses it:** Meta AI research, efficient sequence modeling.

##### Based

| | |
|---|---|
| **Module** | `Edifice.Attention.Based` |
| **Paper** | Arora et al., "Simple linear attention language models balance the recall-throughput tradeoff" (2024) |
| **Origin** | Stanford / Hazy Research |

Linear attention with Taylor expansion feature map. Replaces softmax(QK^T) with polynomial feature map phi(x) = [1, x, x²/√2!, ...] for linear-time computation.

**Strengths:** O(N·d²·p) complexity (p=Taylor order, typically 2-3), simple implementation, good recall.
**Weaknesses:** Quality depends on Taylor order, higher orders increase compute.
**Who uses it:** Efficient language model research, recall-throughput optimization.

##### InfiniAttention

| | |
|---|---|
| **Module** | `Edifice.Attention.InfiniAttention` |
| **Paper** | Munkhdalai et al., "Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention" (2024) |
| **Origin** | Google |

Extends standard attention with a compressive memory system for unbounded context length. Each layer maintains a learnable memory matrix that accumulates information from past segments, blended with local attention via a learnable gate.

**Strengths:** Effectively unbounded context, combines fine-grained local and compressed global information.
**Weaknesses:** Memory compression loses detail, more complex than standard attention.
**Who uses it:** Long-context modeling, infinite-context research.

##### Conformer

| | |
|---|---|
| **Module** | `Edifice.Attention.Conformer` |
| **Paper** | Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition" (2020) |
| **Origin** | Google |

Macaron-style architecture combining self-attention with convolution: Half-FFN → MHSA → Conv Module → Half-FFN. Captures both global (attention) and local (convolution) patterns.

**Strengths:** State-of-the-art for speech recognition, captures both local and global patterns elegantly.
**Weaknesses:** More complex block design, designed for audio/speech (less general).
**Who uses it:** Speech recognition (Google, many ASR systems), audio processing.

##### Ring Attention

| | |
|---|---|
| **Module** | `Edifice.Attention.RingAttention` |
| **Paper** | Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context" (2023) |
| **Reference** | [arXiv:2310.01889](https://arxiv.org/abs/2310.01889) |
| **Origin** | UC Berkeley |

Splits sequences into chunks and processes attention in a rotating ring pattern. Each query chunk attends to all key/value chunks in ring communication order. Enables distributed attention across devices.

**Strengths:** Near-infinite context on distributed systems, memory-efficient chunked attention.
**Weaknesses:** Communication overhead in distributed setting, single-device version is equivalent to chunked attention.
**Who uses it:** Distributed training, very long context processing.

---

#### Recurrent Networks (10 architectures)

The "recurrence renaissance" — a revival of RNN architectures with modern innovations like parallel scans, exponential gating, and test-time learning. These architectures combine the O(1) inference cost of RNNs with training parallelism.

##### LSTM

| | |
|---|---|
| **Module** | `Edifice.Recurrent` (cell_type: :lstm) |
| **Paper** | Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997) |
| **Origin** | TU Munich |

The classic gated recurrent network with forget, input, and output gates plus a cell state. Still one of the most widely deployed sequence models in production.

**Strengths:** Well-understood, robust, extensive tooling, good for moderate sequences.
**Weaknesses:** Purely sequential (not parallelizable), vanishing gradients on very long sequences, O(N) training.
**Who uses it:** Production systems everywhere — NLP, speech, time series, games.

##### GRU

| | |
|---|---|
| **Module** | `Edifice.Recurrent` (cell_type: :gru) |
| **Paper** | Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder" (2014) |
| **Origin** | Yoshua Bengio's lab |

Simplified LSTM with two gates (reset, update) instead of three. Merges cell and hidden state. Fewer parameters with comparable performance on many tasks.

**Strengths:** Simpler than LSTM, fewer parameters, comparable performance.
**Weaknesses:** Same parallelism limitations as LSTM.
**Who uses it:** Edge deployment, resource-constrained environments, sequence modeling.

##### xLSTM

| | |
|---|---|
| **Module** | `Edifice.Recurrent.XLSTM` |
| **Paper** | Beck et al., "xLSTM: Extended Long Short-Term Memory" (NeurIPS 2024) |
| **Reference** | [arXiv:2405.04517](https://arxiv.org/abs/2405.04517) |
| **Origin** | NXAI / Sepp Hochreiter |

Addresses three LSTM limitations with: (1) exponential gating for storage revision, (2) matrix memory (mLSTM) for increased capacity, (3) parallelizable mLSTM via covariance updates. Two variants: sLSTM (sequential, state-tracking) and mLSTM (parallelizable, memorization).

**Strengths:** Bridges LSTM and modern architectures, exponential gating, matrix memory, parallelizable mLSTM.
**Weaknesses:** More complex than original LSTM, sLSTM variant still sequential.
**Who uses it:** NXAI research, LSTM modernization research.

##### mLSTM

| | |
|---|---|
| **Module** | `Edifice.Recurrent.XLSTM` (variant: :mlstm) |
| **Paper** | Beck et al., "xLSTM" (NeurIPS 2024) |
| **Origin** | NXAI / Sepp Hochreiter |

The parallelizable variant of xLSTM with matrix memory cell: C_t = f_t · C_{t-1} + i_t · (v_t · k_t^T). Key-value storage mechanism similar to linear attention, fully parallelizable during training.

**Strengths:** Parallelizable, matrix memory for higher capacity, key-value storage.
**Weaknesses:** Higher memory than scalar LSTM, primarily a memorization engine.
**Who uses it:** Parallel-trainable RNN research, memorization-heavy tasks.

##### MinGRU

| | |
|---|---|
| **Module** | `Edifice.Recurrent.MinGRU` |
| **Paper** | Feng et al., "Were RNNs All We Needed?" (2024) |
| **Reference** | [arXiv:2410.01201](https://arxiv.org/abs/2410.01201) |
| **Origin** | Research community |

Strips GRU down to a single gate: z_t = σ(W_z · x_t), h_t = (1-z) · h_{t-1} + z · W_h · x_t. Gate depends only on input (not hidden state), enabling parallel prefix scan training. ~30 lines of core logic.

**Strengths:** Extremely simple (~30 lines), parallel-scannable, retains core gating mechanism.
**Weaknesses:** Less expressive than full GRU (no hidden-to-hidden gate dependency).
**Who uses it:** Minimalist RNN research, parallel-scannable RNN applications.

##### MinLSTM

| | |
|---|---|
| **Module** | `Edifice.Recurrent.MinLSTM` |
| **Paper** | Feng et al., "Were RNNs All We Needed?" (2024) |
| **Reference** | [arXiv:2410.01201](https://arxiv.org/abs/2410.01201) |
| **Origin** | Research community |

Simplified LSTM with normalized gates (f + i = 1), no output gate, and no hidden-to-hidden gate dependency. Cell state IS the hidden state. Parallel-scannable during training.

**Strengths:** Parallel-scannable, normalized gates ensure stability, simpler than LSTM.
**Weaknesses:** Less expressive than full LSTM (no output gate, no hidden-dependent gates).
**Who uses it:** Parallel RNN research, stable training applications.

##### DeltaNet

| | |
|---|---|
| **Module** | `Edifice.Recurrent.DeltaNet` |
| **Paper** | Schlag et al., "Linear Transformers with Learnable Kernel Functions are Better In-Context Models" (2021) |
| **Reference** | [arXiv:2102.11174](https://arxiv.org/abs/2102.11174) |
| **Origin** | IDSIA |

Linear attention with the delta rule update. Maintains an associative memory matrix S updated by error-correction: S_t = S_{t-1} + β · (v_t - S_{t-1}·k_t) · k_t^T. Subtracts current retrieval before adding, giving superior retrieval accuracy.

**Strengths:** Error-correcting memory updates, better retrieval than standard linear attention, O(d²) memory.
**Weaknesses:** Sequential memory update, learnable beta adds complexity.
**Who uses it:** In-context learning research, associative memory applications.

##### TTT (Test-Time Training)

| | |
|---|---|
| **Module** | `Edifice.Recurrent.TTT` |
| **Paper** | Sun et al., "Learning to (Learn at Test Time): RNNs with Expressive Hidden States" (2024) |
| **Reference** | [arXiv:2407.04620](https://arxiv.org/abs/2407.04620) |
| **Origin** | Stanford / Yu Sun |

The hidden state IS a model (a weight matrix) that is updated via self-supervised gradient steps at each token. TTT-Linear is mathematically equivalent to linear attention with the delta rule.

**Strengths:** Infinite expressiveness (hidden state is a full model), self-supervised adaptation at inference.
**Weaknesses:** Sequential per-token gradient steps, computationally expensive, requires careful initialization.
**Who uses it:** Test-time adaptation research, self-supervised sequence modeling.

##### Titans

| | |
|---|---|
| **Module** | `Edifice.Recurrent.Titans` |
| **Paper** | Behrouz et al., "Titans: Learning to Memorize at Test Time" (2025) |
| **Reference** | [arXiv:2501.00663](https://arxiv.org/abs/2501.00663) |
| **Origin** | Google Research |

Extends TTT with surprise-gated memory: the memory update magnitude scales with prediction error. Higher surprise → larger updates. Uses gradient momentum for smoother evolution.

**Strengths:** Surprise-gated adaptive memory, momentum-based updates, builds on TTT insights.
**Weaknesses:** Sequential, expensive per-token computation, very recent.
**Who uses it:** Cutting-edge memory-augmented sequence research.

##### Reservoir

| | |
|---|---|
| **Module** | `Edifice.Recurrent.Reservoir` |
| **Paper** | Jaeger, "Echo State Networks" (2001); Maass et al., "Real-Time Computing Without Stable States" (2002) |
| **Origin** | Herbert Jaeger / Wolfgang Maass |

Echo State Networks with fixed random reservoir weights. Only the readout (output) layer is trained, making training extremely fast. Spectral radius < 1 ensures the echo state property.

**Strengths:** Extremely fast training (only linear layer), simple, good for time series.
**Weaknesses:** Random reservoir limits expressiveness, can't learn complex representations.
**Who uses it:** Time series forecasting, chaotic system modeling, resource-constrained environments.

---

#### Transformer (1 architecture)

##### DecoderOnly

| | |
|---|---|
| **Module** | `Edifice.Transformer.DecoderOnly` |
| **Paper** | Radford et al. (GPT-2, 2019); Brown et al. (GPT-3, 2020); Touvron et al. (LLaMA, 2023) |
| **Origin** | OpenAI / Meta |

GPT-style decoder-only transformer combining modern LLM techniques: GQA for efficient KV cache, RoPE for position encoding, SwiGLU gated FFN, and RMSNorm. This is the backbone architecture of all major production LLMs.

**Strengths:** The de facto standard for autoregressive generation, extensively optimized, hardware-friendly.
**Weaknesses:** O(N²) attention, requires large parameter counts for best results.
**Who uses it:** GPT-4, Claude, LLaMA, Gemini — virtually all production LLMs.

---

### Perception

#### Vision (9 architectures)

Vision architectures process images (and related spatial data) through various token-mixing strategies — from attention (ViT) to convolution (ConvNeXt) to pure pooling (PoolFormer) to coordinate mapping (NeRF).

##### ViT (Vision Transformer)

| | |
|---|---|
| **Module** | `Edifice.Vision.ViT` |
| **Paper** | Dosovitskiy et al., "An Image is Worth 16x16 Words" (ICLR 2021) |
| **Origin** | Google Brain |

Treats an image as a sequence of fixed-size patches, linearly embeds each, prepends a CLS token, adds position embeddings, and processes through standard transformer blocks.

**Strengths:** Simple, scalable, strong with large datasets, established paradigm.
**Weaknesses:** Requires large datasets (less data-efficient than CNNs), O(N²) in patch count.
**Who uses it:** Google, Meta, virtually all vision research since 2021.

##### DeiT

| | |
|---|---|
| **Module** | `Edifice.Vision.DeiT` |
| **Paper** | Touvron et al., "Training data-efficient image transformers & distillation through attention" (ICML 2021) |
| **Origin** | Meta AI / Facebook |

Data-efficient ViT with a distillation token that learns from a teacher model. CLS token for classification, distillation token for teacher alignment; outputs averaged at inference.

**Strengths:** Data-efficient (works with ImageNet-1K, no JFT), knowledge distillation built in.
**Weaknesses:** Requires a teacher model for distillation, extra token adds minor overhead.
**Who uses it:** Vision tasks with limited data, knowledge distillation research.

##### Swin Transformer

| | |
|---|---|
| **Module** | `Edifice.Vision.SwinTransformer` |
| **Paper** | Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (ICCV 2021) |
| **Origin** | Microsoft Research Asia |

Hierarchical ViT computing attention within local windows, with shifted windows between layers for cross-window connections. Produces multi-scale feature maps like a CNN.

**Strengths:** Linear complexity in image size, hierarchical features for dense prediction, versatile backbone.
**Weaknesses:** Window-based attention limits global receptive field, complex implementation.
**Who uses it:** Object detection (COCO), semantic segmentation, Microsoft Research.

##### U-Net

| | |
|---|---|
| **Module** | `Edifice.Vision.UNet` |
| **Paper** | Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015) |
| **Origin** | University of Freiburg |

Symmetric encoder-decoder with skip connections concatenating encoder features at each level with decoder features. Uses real 2D convolutions, max-pooling, and transposed convolutions.

**Strengths:** Excellent for segmentation, preserves fine spatial detail via skip connections, well-proven.
**Weaknesses:** Memory-intensive (skip connections double feature maps), primarily for dense prediction.
**Who uses it:** Medical imaging, diffusion model backbones (Stable Diffusion), segmentation tasks.

##### ConvNeXt

| | |
|---|---|
| **Module** | `Edifice.Vision.ConvNeXt` |
| **Paper** | Liu et al., "A ConvNet for the 2020s" (CVPR 2022) |
| **Origin** | Meta AI / Facebook |

Modernized ResNet applying transformer-era techniques: depthwise-separable convolutions, inverted bottleneck, GELU, LayerNorm, fewer activations, LayerScale. Proves CNNs can match ViT quality.

**Strengths:** CNN simplicity with ViT-level accuracy, no attention needed, efficient.
**Weaknesses:** Limited global receptive field (vs attention), less flexible than ViT.
**Who uses it:** Vision backbone when attention isn't needed, efficiency-focused vision.

##### MLP-Mixer

| | |
|---|---|
| **Module** | `Edifice.Vision.MLPMixer` |
| **Paper** | Tolstikhin et al., "MLP-Mixer: An all-MLP Architecture for Vision" (NeurIPS 2021) |
| **Origin** | Google Brain |

Pure MLP architecture: token-mixing MLPs (across patches) and channel-mixing MLPs (within patches), alternated. No attention, no convolution.

**Strengths:** Extremely simple, demonstrates that MLPs alone can be competitive, fast.
**Weaknesses:** Requires large datasets, slightly below ViT/CNN on standard benchmarks.
**Who uses it:** Architecture research, simplicity-focused applications.

##### FocalNet

| | |
|---|---|
| **Module** | `Edifice.Vision.FocalNet` |
| **Paper** | Yang et al., "Focal Modulation Networks" (NeurIPS 2022) |
| **Reference** | [arXiv:2203.11926](https://arxiv.org/abs/2203.11926) |
| **Origin** | Microsoft Research |

Replaces self-attention with focal modulation — hierarchical depthwise convolutions aggregating context at multiple granularity levels with gated aggregation.

**Strengths:** Captures both local and global context without attention, simple yet effective.
**Weaknesses:** Less widely adopted, hierarchical convolutions add parameters.
**Who uses it:** Vision tasks where attention-free alternatives are preferred.

##### PoolFormer

| | |
|---|---|
| **Module** | `Edifice.Vision.PoolFormer` |
| **Paper** | Yu et al., "MetaFormer is Actually What You Need for Vision" (CVPR 2022) |
| **Reference** | [arXiv:2111.11418](https://arxiv.org/abs/2111.11418) |
| **Origin** | Sea AI Lab |

Replaces self-attention with simple average pooling. Demonstrates that the MetaFormer architecture (norm → mixer → residual → norm → FFN → residual) matters more than the specific attention mechanism.

**Strengths:** Extremely simple token mixer, competitive accuracy, very fast, proves MetaFormer thesis.
**Weaknesses:** No learnable token mixing, limited on tasks requiring precise attention.
**Who uses it:** Efficiency research, MetaFormer architecture studies.

##### NeRF

| | |
|---|---|
| **Module** | `Edifice.Vision.NeRF` |
| **Paper** | Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (ECCV 2020) |
| **Reference** | [arXiv:2003.08934](https://arxiv.org/abs/2003.08934) |
| **Origin** | UC Berkeley / Google |

Maps 3D coordinates (and optional viewing directions) to color and density values. Uses Fourier positional encoding and an MLP with skip connections. Unlike other vision models, takes raw coordinates rather than image inputs.

**Strengths:** Novel view synthesis from sparse images, implicit 3D representation, elegant design.
**Weaknesses:** Slow rendering (per-ray MLP evaluation), requires known camera poses.
**Who uses it:** 3D reconstruction, virtual reality, autonomous driving, visual effects.

---

#### Convolutional (6 architectures)

Classic convolutional networks for local pattern extraction. Despite the rise of transformers, CNNs remain relevant for efficiency, mobile deployment, and well-understood training dynamics.

##### Conv1D/2D

| | |
|---|---|
| **Module** | `Edifice.Convolutional.Conv` |
| **Origin** | Foundational CNN building blocks |

Configurable convolution blocks following the pattern: convolution → batch normalization → activation → dropout. Both 1D (sequence) and 2D (image) variants with optional pooling.

**Strengths:** Flexible building blocks, composable, well-understood.
**Weaknesses:** Building blocks only — need composition for full architectures.
**Who uses it:** Custom architecture building, feature extraction layers.

##### ResNet

| | |
|---|---|
| **Module** | `Edifice.Convolutional.ResNet` |
| **Paper** | He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016) |
| **Origin** | Microsoft Research |

Deep residual networks using skip connections. Residual and bottleneck block variants. Configurations from ResNet-18 (~11M params) to ResNet-152 (~60M params).

**Strengths:** Enables very deep networks (100+ layers), well-proven, extensive pretrained models available.
**Weaknesses:** Large models, older architecture compared to ConvNeXt/ViT.
**Who uses it:** Still widely used as backbone, benchmark baseline, transfer learning source.

##### DenseNet

| | |
|---|---|
| **Module** | `Edifice.Convolutional.DenseNet` |
| **Paper** | Huang et al., "Densely Connected Convolutional Networks" (CVPR 2017) |
| **Origin** | Cornell / Tsinghua |

Each layer receives feature maps of ALL preceding layers via concatenation. Encourages feature reuse and reduces parameter count. Transition layers compress features between dense blocks.

**Strengths:** Feature reuse, parameter efficient, strong gradient flow.
**Weaknesses:** Memory-intensive (concatenated features), slower than ResNet.
**Who uses it:** Medical imaging, small-dataset applications, feature-reuse research.

##### TCN

| | |
|---|---|
| **Module** | `Edifice.Convolutional.TCN` |
| **Paper** | Bai et al., "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (2018) |
| **Origin** | CMU |

Dilated causal convolutions with exponentially growing dilation rates (1, 2, 4, 8...). Receptive field grows exponentially with depth while parameters grow linearly. Causal: output at time t depends only on inputs ≤ t.

**Strengths:** Parallelizable (unlike RNNs), causal, flexible receptive field, residual connections.
**Weaknesses:** Fixed receptive field (must be designed for task), no content-based selection.
**Who uses it:** Audio processing, time series, real-time sequence classification.

##### MobileNet

| | |
|---|---|
| **Module** | `Edifice.Convolutional.MobileNet` |
| **Paper** | Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" (2017) |
| **Reference** | [arXiv:1704.04861](https://arxiv.org/abs/1704.04861) |
| **Origin** | Google |

Depthwise separable convolutions factoring standard convolution into per-channel and pointwise operations. Reduces computation by ~1/output_channels + 1/kernel_size².

**Strengths:** Extremely efficient, designed for mobile/edge deployment, width multiplier for scaling.
**Weaknesses:** Lower accuracy ceiling than full convolutions, approximated as dense layers in Edifice.
**Who uses it:** Mobile applications, edge deployment, real-time inference.

##### EfficientNet

| | |
|---|---|
| **Module** | `Edifice.Convolutional.EfficientNet` |
| **Paper** | Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (ICML 2019) |
| **Reference** | [arXiv:1905.11946](https://arxiv.org/abs/1905.11946) |
| **Origin** | Google Brain |

Compound scaling of depth, width, and resolution with fixed scaling coefficients. Core building block is MBConv (Mobile Inverted Bottleneck) with squeeze-and-excitation attention.

**Strengths:** Principled scaling strategy, excellent accuracy/efficiency tradeoff, SE attention.
**Weaknesses:** Complex scaling rules, approximated as dense layers in Edifice (designed for images).
**Who uses it:** Efficient image classification, mobile deployment, scaling research.

---

### Structured Data

#### Graph Networks (8 architectures)

Graph neural networks for processing graph-structured data — molecular graphs, social networks, knowledge graphs, and spatial interactions.

##### GCN

| | |
|---|---|
| **Module** | `Edifice.Graph.GCN` |
| **Paper** | Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks" (ICLR 2017) |
| **Origin** | University of Amsterdam |

Spectral graph convolutions approximated by first-order Chebyshev polynomials. Each layer: H' = σ(D^{-½}AD^{-½}HW). Symmetric normalization prevents feature magnitude scaling with node degree.

**Strengths:** Simple, foundational, well-understood, efficient.
**Weaknesses:** Fixed aggregation (no learned weighting), transductive (can't generalize to new graphs).
**Who uses it:** Baseline for all GNN research, node classification, social networks.

##### GAT

| | |
|---|---|
| **Module** | `Edifice.Graph.GAT` |
| **Paper** | Veličković et al., "Graph Attention Networks" (ICLR 2018) |
| **Origin** | University of Cambridge / DeepMind |

Attention-based message passing where each node attends to neighbors with learned attention weights. Multi-head attention for attending to different subspaces.

**Strengths:** Learned neighbor weighting (adaptive), multi-head attention, inductive.
**Weaknesses:** O(E) attention computation (E = edges), more parameters than GCN.
**Who uses it:** Citation networks, social networks, any graph task needing adaptive aggregation.

##### GIN

| | |
|---|---|
| **Module** | `Edifice.Graph.GIN` |
| **Paper** | Xu et al., "How Powerful are Graph Neural Networks?" (ICLR 2019) |
| **Reference** | [arXiv:1810.00826](https://arxiv.org/abs/1810.00826) |
| **Origin** | MIT |

Provably most expressive GNN under message passing, achieving Weisfeiler-Lehman graph isomorphism test power. Each layer: h_v' = MLP((1+ε)·h_v + Σ h_u).

**Strengths:** Maximally expressive under message passing, learnable ε for self-vs-neighbor weighting.
**Weaknesses:** Sum aggregation can be unstable, limited by WL expressiveness ceiling.
**Who uses it:** Graph classification, molecular property prediction, GNN expressiveness research.

##### GINv2

| | |
|---|---|
| **Module** | `Edifice.Graph.GINv2` |
| **Paper** | Hu et al., "Strategies for Pre-training Graph Neural Networks" (ICLR 2020) |
| **Reference** | [arXiv:1905.12265](https://arxiv.org/abs/1905.12265) |
| **Origin** | Stanford |

Extends GIN with edge feature incorporation. Edge features are projected and combined with neighbor features before aggregation, enabling learning from bond types, distances, and relationship properties.

**Strengths:** Edge feature awareness, more expressive for graphs with rich edge information.
**Weaknesses:** Requires edge features (3D input tensor), more complex than GIN.
**Who uses it:** Molecular graphs (bond types), knowledge graphs (relation types).

##### GraphSAGE

| | |
|---|---|
| **Module** | `Edifice.Graph.GraphSAGE` |
| **Paper** | Hamilton et al., "Inductive Representation Learning on Large Graphs" (NeurIPS 2017) |
| **Reference** | [arXiv:1706.02216](https://arxiv.org/abs/1706.02216) |
| **Origin** | Stanford |

Inductive graph learning via neighborhood sampling and aggregation. Concatenates self-features with aggregated neighbor features. Supports mean, max, and pool aggregation.

**Strengths:** Inductive (generalizes to unseen nodes), scalable via sampling, multiple aggregators.
**Weaknesses:** Sampling introduces variance, concatenation doubles feature dimension.
**Who uses it:** Large-scale graphs, evolving graphs, Pinterest recommendation system.

##### PNA

| | |
|---|---|
| **Module** | `Edifice.Graph.PNA` |
| **Paper** | Corso et al., "Principal Neighbourhood Aggregation for Graph Nets" (NeurIPS 2020) |
| **Reference** | [arXiv:2004.05718](https://arxiv.org/abs/2004.05718) |
| **Origin** | University of Cambridge |

Combines multiple aggregators (mean, max, sum, std) with degree-based scalers (identity, amplification) for maximally expressive message passing. Concatenates all aggregator×scaler combinations.

**Strengths:** Most expressive single-layer message passing, diverse aggregation captures richer structure.
**Weaknesses:** High feature dimension (num_aggregators × num_scalers × hidden), more compute.
**Who uses it:** Molecular property prediction, graph benchmarks requiring maximum expressiveness.

##### Graph Transformer

| | |
|---|---|
| **Module** | `Edifice.Graph.GraphTransformer` |
| **Paper** | Dwivedi & Bresson (AAAI 2021); Ying et al. (NeurIPS 2021) |
| **Origin** | Multiple research groups |

Full multi-head attention over graph nodes with adjacency as attention bias/mask. Includes structural positional encoding via random walk or adjacency matrix powers.

**Strengths:** Full global attention (any node can attend to any other), structural encoding.
**Weaknesses:** O(N²) in number of nodes, may ignore graph sparsity.
**Who uses it:** Molecular property prediction (Graphormer/OGB-LSC winner), knowledge graphs.

##### SchNet

| | |
|---|---|
| **Module** | `Edifice.Graph.SchNet` |
| **Paper** | Schütt et al., "SchNet: A continuous-filter convolutional neural network for modeling quantum interactions" (NeurIPS 2017) |
| **Reference** | [arXiv:1706.08566](https://arxiv.org/abs/1706.08566) |
| **Origin** | TU Berlin |

Continuous-filter convolutions for molecular graphs. Edge weights derived from interatomic distances via radial basis functions. Operates on continuous geometry, not discrete adjacency.

**Strengths:** Physics-aware (continuous distances, not binary edges), excellent for molecular properties.
**Weaknesses:** Requires distance information, specialized for molecular/atomic data.
**Who uses it:** Molecular property prediction, quantum chemistry, drug discovery.

---

#### Set Networks (2 architectures)

Architectures for permutation-invariant processing of unordered sets.

##### DeepSets

| | |
|---|---|
| **Module** | `Edifice.Sets.DeepSets` |
| **Paper** | Zaheer et al., "Deep Sets" (NeurIPS 2017) |
| **Origin** | CMU |

Processes each element independently through phi, aggregates with a permutation-invariant operation (sum), then post-processes with rho. Provably universal for permutation-invariant set functions.

**Strengths:** Theoretically principled, simple, universal approximator for set functions.
**Weaknesses:** Sum aggregation loses ordering information (by design), limited inter-element interaction.
**Who uses it:** Point cloud classification, set-based inference, multi-instance learning.

##### PointNet

| | |
|---|---|
| **Module** | `Edifice.Sets.PointNet` |
| **Paper** | Qi et al., "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation" (CVPR 2017) |
| **Origin** | Stanford |

Point cloud processing with per-point MLPs and max pooling for permutation invariance. Optional T-Net learns spatial transformation matrices for input canonicalization.

**Strengths:** Direct point cloud processing (no voxelization), T-Net for geometric invariance.
**Weaknesses:** No local structure modeling (each point processed independently before pooling).
**Who uses it:** 3D object classification, autonomous driving, robotics point cloud processing.

---

### Generation

#### Generative Models (11 architectures)

Architectures for generating data — from noise to images, actions, and latent representations.

##### VAE

| | |
|---|---|
| **Module** | `Edifice.Generative.VAE` |
| **Paper** | Kingma & Welling, "Auto-Encoding Variational Bayes" (ICLR 2014) |
| **Origin** | University of Amsterdam |

Encodes inputs into distributions (mu, log_var) via reparameterization trick. KL divergence regularizer pushes learned posterior toward standard normal prior. Beta-VAE variant for disentangled representations.

**Strengths:** Smooth latent space, principled probabilistic framework, generation + inference.
**Weaknesses:** Blurry outputs (KL regularization trades off with reconstruction), posterior collapse risk.
**Who uses it:** Latent space learning, disentangled representations, anomaly detection.

##### VQ-VAE

| | |
|---|---|
| **Module** | `Edifice.Generative.VQVAE` |
| **Paper** | van den Oord et al., "Neural Discrete Representation Learning" (NeurIPS 2017) |
| **Origin** | Google DeepMind |

Discrete codebook replaces continuous latent space. Encoder output quantized to nearest codebook vector via straight-through estimator. Avoids posterior collapse and produces sharper reconstructions.

**Strengths:** Discrete latents (composable, interpretable), sharp reconstructions, no posterior collapse.
**Weaknesses:** Codebook utilization can collapse, straight-through gradient is approximate.
**Who uses it:** Audio generation (Jukebox), image tokenization, discrete representation learning.

##### GAN

| | |
|---|---|
| **Module** | `Edifice.Generative.GAN` |
| **Paper** | Goodfellow et al., "Generative Adversarial Networks" (NeurIPS 2014) |
| **Origin** | University of Montreal |

Generator/discriminator adversarial training. Supports WGAN-GP variant for stable training. Generator transforms noise to data; discriminator classifies real vs fake.

**Strengths:** Sharp, high-quality generations, powerful implicit density estimation.
**Weaknesses:** Training instability (mode collapse, vanishing gradients), no latent inference.
**Who uses it:** Image generation, style transfer, data augmentation, super-resolution.

##### Diffusion (DDPM)

| | |
|---|---|
| **Module** | `Edifice.Generative.Diffusion` |
| **Paper** | Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (RSS 2023) |
| **Reference** | [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) |
| **Origin** | Columbia University |

Denoising diffusion probabilistic model adapted for action generation. Iteratively denoises random noise into actions conditioned on observations. Multi-modal output distribution (can represent multiple valid actions).

**Strengths:** Multi-modal output, stable MSE training, scales to high-dimensional action sequences.
**Weaknesses:** Slow inference (~100-1000 denoising steps), high compute.
**Who uses it:** Robot manipulation (Diffusion Policy), imitation learning, action generation.

##### DDIM

| | |
|---|---|
| **Module** | `Edifice.Generative.DDIM` |
| **Paper** | Song et al., "Denoising Diffusion Implicit Models" (ICLR 2021) |
| **Reference** | [arXiv:2010.02502](https://arxiv.org/abs/2010.02502) |
| **Origin** | Stanford |

Deterministic sampling variant of DDPM. Same training objective but reformulates reverse process as deterministic, enabling ~50 steps instead of ~1000. Eta parameter interpolates between deterministic and stochastic.

**Strengths:** 20x fewer sampling steps, deterministic generation, same training as DDPM.
**Weaknesses:** Slightly lower quality than full DDPM at equal training.
**Who uses it:** Fast diffusion sampling, real-time generation applications.

##### DiT

| | |
|---|---|
| **Module** | `Edifice.Generative.DiT` |
| **Paper** | Peebles & Xie, "Scalable Diffusion Models with Transformers" (ICCV 2023) |
| **Reference** | [arXiv:2212.09748](https://arxiv.org/abs/2212.09748) |
| **Origin** | UC Berkeley / Meta |

Replaces U-Net backbone in diffusion with a Transformer. Uses AdaLN-Zero conditioning: LayerNorm parameters modulated by conditioning signal, with alpha initialized to zero for stable deep training.

**Strengths:** Scalable (transformer backbone), elegant conditioning mechanism, state-of-the-art image generation.
**Weaknesses:** Higher compute than U-Net for small models, requires large-scale training.
**Who uses it:** Sora (OpenAI), state-of-the-art image generation, scalable diffusion research.

##### Latent Diffusion

| | |
|---|---|
| **Module** | `Edifice.Generative.LatentDiffusion` |
| **Paper** | Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022) |
| **Reference** | [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) |
| **Origin** | LMU Munich / Stability AI |

Runs diffusion in VAE latent space instead of pixel space. Train VAE first (perceptual compression), freeze it, then train diffusion in the compact latent space. Returns {encoder, decoder, denoiser}.

**Strengths:** Dramatically faster than pixel-space diffusion, lower memory, perceptual quality preserved.
**Weaknesses:** Two-stage training (VAE then diffusion), latent space may lose fine detail.
**Who uses it:** Stable Diffusion, Midjourney, virtually all production image generation.

##### Consistency Model

| | |
|---|---|
| **Module** | `Edifice.Generative.ConsistencyModel` |
| **Paper** | Song et al., "Consistency Models" (ICML 2023) |
| **Reference** | [arXiv:2303.01469](https://arxiv.org/abs/2303.01469) |
| **Origin** | OpenAI |

Learns a function f(x_t, t) that maps any point on a diffusion trajectory directly to its origin. Enables single-step generation (f(x_T, T) → x_0) or few-step refinement.

**Strengths:** Single-step generation (1000x faster than DDPM), can refine with more steps for quality.
**Weaknesses:** Lower quality than full diffusion with equivalent training, consistency constraint is hard to enforce.
**Who uses it:** Real-time generation, OpenAI research, fast diffusion inference.

##### Score SDE

| | |
|---|---|
| **Module** | `Edifice.Generative.ScoreSDE` |
| **Paper** | Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations" (ICLR 2021) |
| **Reference** | [arXiv:2011.13456](https://arxiv.org/abs/2011.13456) |
| **Origin** | Stanford |

Unifies DDPM, SMLD, and other score-based methods under a single SDE framework. VP-SDE (variance preserving, generalizes DDPM) and VE-SDE (variance exploding, generalizes SMLD) variants. Learns the score function ∇_x log p_t(x).

**Strengths:** Unified theoretical framework, exact log-likelihood via probability flow ODE, flexible.
**Weaknesses:** SDE/ODE solving is computationally expensive, complex mathematical framework.
**Who uses it:** Generative modeling theory, diffusion research, advanced sampling methods.

##### Flow Matching

| | |
|---|---|
| **Module** | `Edifice.Generative.FlowMatching` |
| **Paper** | Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023) |
| **Reference** | [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) |
| **Origin** | Meta AI |

Learns a velocity field transporting noise to data via ODE. Uses simple linear interpolation (optimal transport paths) — no noise schedule needed. Training: MSE on velocity prediction.

**Strengths:** No noise schedule, simpler than diffusion, fewer inference steps (10-20), deterministic ODE.
**Weaknesses:** ODE integration at inference, less established than diffusion.
**Who uses it:** Meta AI (Llama 3 tokenizer training), modern generative modeling.

##### Normalizing Flow

| | |
|---|---|
| **Module** | `Edifice.Generative.NormalizingFlow` |
| **Paper** | Dinh et al., "Density estimation using Real-NVP" (ICLR 2017) |
| **Origin** | University of Montreal |

Invertible transformations between simple base distribution and complex target. RealNVP-style affine coupling layers with tractable Jacobian determinant for exact log-likelihood computation.

**Strengths:** Exact log-likelihood (not a bound like VAE), invertible (both generation and inference).
**Weaknesses:** Limited expressiveness per layer (coupling splits), many layers needed, restricted architectures.
**Who uses it:** Density estimation, exact likelihood applications, physics simulations.

---

#### Contrastive & Self-Supervised (6 architectures)

Architectures for learning representations without labels, using contrastive losses or prediction-based objectives.

##### SimCLR

| | |
|---|---|
| **Module** | `Edifice.Contrastive.SimCLR` |
| **Paper** | Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (ICML 2020) |
| **Reference** | [arXiv:2002.05709](https://arxiv.org/abs/2002.05709) |
| **Origin** | Google Brain |

Contrastive learning with NT-Xent loss. Two augmented views of each example processed by shared encoder + projection head. Negative pairs from other examples in the batch.

**Strengths:** Simple framework, strong representations, well-understood.
**Weaknesses:** Requires large batch sizes for enough negatives, sensitive to augmentations.
**Who uses it:** Self-supervised pretraining, visual representation learning.

##### BYOL

| | |
|---|---|
| **Module** | `Edifice.Contrastive.BYOL` |
| **Paper** | Grill et al., "Bootstrap Your Own Latent" (NeurIPS 2020) |
| **Reference** | [arXiv:2006.07733](https://arxiv.org/abs/2006.07733) |
| **Origin** | Google DeepMind |

No negative pairs needed. Online network (with predictor head) trained to match EMA target network. Asymmetric design prevents collapse without negatives.

**Strengths:** No negatives needed (any batch size works), simple EMA update, strong representations.
**Weaknesses:** EMA schedule is sensitive, predictor head adds complexity.
**Who uses it:** Self-supervised learning when batch size is limited, general representation learning.

##### Barlow Twins

| | |
|---|---|
| **Module** | `Edifice.Contrastive.BarlowTwins` |
| **Paper** | Zbontar et al., "Barlow Twins: Self-Supervised Learning via Redundancy Reduction" (ICML 2021) |
| **Reference** | [arXiv:2103.03230](https://arxiv.org/abs/2103.03230) |
| **Origin** | Meta AI / FAIR |

Pushes cross-correlation matrix of two augmented views toward identity. Invariance term (diagonal → 1) and redundancy reduction term (off-diagonal → 0).

**Strengths:** Elegant loss function, no negatives/momentum/asymmetry, batch-size robust.
**Weaknesses:** Requires large projection dimension for decorrelation to work well.
**Who uses it:** Self-supervised learning research, decorrelated representation learning.

##### MAE

| | |
|---|---|
| **Module** | `Edifice.Contrastive.MAE` |
| **Paper** | He et al., "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022) |
| **Reference** | [arXiv:2111.06377](https://arxiv.org/abs/2111.06377) |
| **Origin** | Meta AI / FAIR |

Masks 75% of input patches and trains autoencoder to reconstruct them. Asymmetric encoder-decoder: encoder processes only unmasked patches (efficient), lightweight decoder processes all.

**Strengths:** Simple, scalable, efficient (encoder sees only 25% of patches), strong pretraining.
**Weaknesses:** Requires patch-based input, reconstruction quality depends on masking ratio.
**Who uses it:** Vision pretraining (BERT-style for images), Meta AI.

##### VICReg

| | |
|---|---|
| **Module** | `Edifice.Contrastive.VICReg` |
| **Paper** | Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning" (ICLR 2022) |
| **Reference** | [arXiv:2105.04906](https://arxiv.org/abs/2105.04906) |
| **Origin** | Meta AI / FAIR |

Three explicit regularization terms: Variance (maintain dimension variance), Invariance (MSE between views), Covariance (decorrelate dimensions). No architectural tricks — symmetric, no momentum, no stop-gradient.

**Strengths:** Interpretable loss terms, symmetric architecture, explicit collapse prevention.
**Weaknesses:** Three hyperparameters to balance (lambda, mu, nu).
**Who uses it:** Self-supervised learning research, when interpretable loss is desired.

##### JEPA

| | |
|---|---|
| **Module** | `Edifice.Contrastive.JEPA` |
| **Paper** | Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (CVPR 2023) |
| **Reference** | [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) |
| **Origin** | Meta AI / Yann LeCun |

Predicts representations of masked regions (not pixel values), learning more abstract features. Context encoder processes visible patches, narrow predictor bridges to target encoder's representation space. EMA target updates.

**Strengths:** Learns abstract representations (not pixel-level), aligns with Yann LeCun's world model vision.
**Weaknesses:** Complex training setup (two encoders + predictor), relatively new paradigm.
**Who uses it:** Meta AI (core research direction for Yann LeCun's AGI vision).

---

### Composition & Enhancement

#### Meta-Learning (10 architectures)

Architectures for routing, adaptation, composition, and model enhancement. These are typically composed with other architectures rather than used standalone.

##### MoE (Mixture of Experts)

| | |
|---|---|
| **Module** | `Edifice.Meta.MoE` |
| **Paper** | Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (ICLR 2017) |
| **Origin** | Google Brain |

Routes each input to a subset of specialized expert networks via learned routing. Top-K, switch (top-1), soft (all experts), and hash routing strategies.

**Strengths:** Massive capacity with sparse activation, expert specialization, established technique.
**Weaknesses:** Load balancing issues, routing instability, requires auxiliary loss.
**Who uses it:** GPT-4 (rumored), Mixtral, Google Switch Transformer, large-scale models.

##### Switch MoE

| | |
|---|---|
| **Module** | `Edifice.Meta.SwitchMoE` |
| **Paper** | Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (JMLR 2022) |
| **Reference** | [arXiv:2101.03961](https://arxiv.org/abs/2101.03961) |
| **Origin** | Google Brain |

Top-1 expert routing — each token goes to exactly one expert. Simplifies MoE routing while maintaining capacity benefits. Peaked softmax distribution for near-one-hot routing.

**Strengths:** Simpler than top-K MoE, best load balance, scales to trillion parameters.
**Weaknesses:** Single expert per token may limit quality, token dropping on imbalance.
**Who uses it:** Google (Switch Transformer), large-scale sparse model research.

##### Soft MoE

| | |
|---|---|
| **Module** | `Edifice.Meta.SoftMoE` |
| **Paper** | Puigcerver et al., "From Sparse to Soft Mixtures of Experts" (ICLR 2024) |
| **Reference** | [arXiv:2308.00951](https://arxiv.org/abs/2308.00951) |
| **Origin** | Google Brain |

Fully differentiable soft routing — computes weighted combination of all expert outputs for every token. Eliminates token dropping, load balancing issues, and routing instability.

**Strengths:** Fully differentiable, no token dropping, stable training, no load balancing loss needed.
**Weaknesses:** All experts computed for every token (no sparsity benefit at inference).
**Who uses it:** Soft routing research, when training stability is paramount.

##### LoRA

| | |
|---|---|
| **Module** | `Edifice.Meta.LoRA` |
| **Paper** | Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022) |
| **Reference** | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) |
| **Origin** | Microsoft Research |

Freezes original weights, injects trainable low-rank decomposition: output = Wx + (α/r)·B(Ax). Reduces trainable parameters by orders of magnitude while maintaining quality.

**Strengths:** Orders of magnitude fewer trainable params, no inference overhead (merge weights), simple.
**Weaknesses:** Limited capacity (rank constraint), may not match full fine-tuning on complex tasks.
**Who uses it:** Virtually all LLM fine-tuning — now the default PEFT method.

##### Adapter

| | |
|---|---|
| **Module** | `Edifice.Meta.Adapter` |
| **Paper** | Houlsby et al., "Parameter-Efficient Transfer Learning for NLP" (ICML 2019) |
| **Reference** | [arXiv:1902.00751](https://arxiv.org/abs/1902.00751) |
| **Origin** | Google Research |

Small bottleneck modules (down-project → activation → up-project + residual) inserted between frozen pretrained layers. Adds minimal trainable parameters.

**Strengths:** Simple, modular, can be stacked/combined, minimal interference with pretrained weights.
**Weaknesses:** Adds inference overhead (extra layers), largely superseded by LoRA.
**Who uses it:** Transfer learning, multi-task adaptation, historical PEFT method.

##### Hypernetwork

| | |
|---|---|
| **Module** | `Edifice.Meta.Hypernetwork` |
| **Paper** | Ha et al., "HyperNetworks" (2016) |
| **Reference** | [arXiv:1609.09106](https://arxiv.org/abs/1609.09106) |
| **Origin** | Google Brain |

Networks that generate other networks' weights. A hypernetwork takes conditioning input and produces weight matrices for a target network. Enables conditional computation and task adaptation.

**Strengths:** Conditional weight generation, task adaptation, compression (small hyper → large target).
**Weaknesses:** Complex training dynamics, hypernetwork capacity limits target quality.
**Who uses it:** Multi-task learning, neural architecture search, dynamic network research.

##### Capsule

| | |
|---|---|
| **Module** | `Edifice.Meta.Capsule` |
| **Paper** | Sabour et al., "Dynamic Routing Between Capsules" (2017) |
| **Reference** | [arXiv:1710.09829](https://arxiv.org/abs/1710.09829) |
| **Origin** | Google Brain / Geoffrey Hinton |

Vector "capsules" encoding both entity existence (vector length) and properties (vector direction). Dynamic routing by agreement replaces max-pooling, preserving spatial hierarchies.

**Strengths:** Preserves spatial hierarchies, rotation/pose equivariance, interpretable.
**Weaknesses:** Computationally expensive (routing iterations), difficult to scale, largely abandoned.
**Who uses it:** Research on equivariant representations, medical imaging (small-scale).

##### Mixture of Depths

| | |
|---|---|
| **Module** | `Edifice.Meta.MixtureOfDepths` |
| **Paper** | Raposo et al., "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models" (2024) |
| **Reference** | [arXiv:2404.02258](https://arxiv.org/abs/2404.02258) |
| **Origin** | Google DeepMind |

Per-token routing where a learned router scores tokens and only top-C% receive full transformer processing; rest skip via residual. Dynamic compute allocation.

**Strengths:** Dynamic compute allocation (easy tokens skip layers), reduces average FLOPs.
**Weaknesses:** Differentiable approximation (soft gating) less efficient than true sparsity.
**Who uses it:** Efficient transformer research, dynamic compute allocation.

##### Mixture of Agents

| | |
|---|---|
| **Module** | `Edifice.Meta.MixtureOfAgents` |
| **Paper** | Wang et al., "Mixture-of-Agents Enhances Large Language Model Capabilities" (2024) |
| **Origin** | Together AI |

N independent proposer transformer stacks process input in parallel, outputs concatenated and fed to a larger aggregator transformer that combines proposals.

**Strengths:** Diverse proposals from multiple models, aggregator learns to combine best aspects.
**Weaknesses:** N× compute for proposers, complex architecture, relatively new.
**Who uses it:** Multi-model combination research, ensemble-style architectures.

##### RLHF Head

| | |
|---|---|
| **Module** | `Edifice.Meta.RLHFHead` |
| **Paper** | Ouyang et al. (RLHF, 2022); Rafailov et al. (DPO, 2023) |
| **Origin** | OpenAI / Stanford |

Composable head modules for alignment: Reward head (sequence → scalar reward) and DPO head (chosen/rejected → preference logit). Designed to be attached to any backbone.

**Strengths:** Composable with any backbone, supports both RLHF reward modeling and DPO.
**Weaknesses:** Head only (requires backbone), reward modeling has known limitations.
**Who uses it:** LLM alignment pipelines, reward model training, DPO fine-tuning.

---

### Specialized

#### Energy-Based (3 architectures)

Architectures modeling energy landscapes and continuous dynamics.

##### EBM

| | |
|---|---|
| **Module** | `Edifice.Energy.EBM` |
| **Paper** | Du & Mordatch, "Implicit Generation and Modeling with Energy-Based Models" (2019); LeCun et al., "A Tutorial on Energy-Based Learning" (2006) |
| **Origin** | MIT / NYU |

Energy function network assigning scalar energy to inputs. Trained via contrastive divergence: push down energy on real data, push up on negatives from Langevin dynamics MCMC.

**Strengths:** Flexible unnormalized density model, can model complex distributions, no mode collapse.
**Weaknesses:** MCMC sampling is slow, training instability, energy landscape hard to control.
**Who uses it:** Energy-based generative modeling research, anomaly detection.

##### Hopfield

| | |
|---|---|
| **Module** | `Edifice.Energy.Hopfield` |
| **Paper** | Ramsauer et al., "Hopfield Networks is All You Need" (2020) |
| **Reference** | [arXiv:2008.02217](https://arxiv.org/abs/2008.02217) |
| **Origin** | Johannes Kepler University |

Modern continuous Hopfield networks with exponential interaction function. Stores exponentially many patterns (vs polynomial in classical). Update rule softmax(β·X·Y^T)·Y is exactly the attention mechanism.

**Strengths:** Exponential storage capacity, single-step convergence, mathematical attention connection.
**Weaknesses:** Primarily theoretical interest, specialized use case.
**Who uses it:** Associative memory research, attention-memory connection studies.

##### Neural ODE

| | |
|---|---|
| **Module** | `Edifice.Energy.NeuralODE` |
| **Paper** | Chen et al., "Neural Ordinary Differential Equations" (NeurIPS 2018) |
| **Reference** | [arXiv:1806.07366](https://arxiv.org/abs/1806.07366) |
| **Origin** | University of Toronto |

Parameterizes continuous hidden state dynamics: dh/dt = f(h(t), t; θ). Constant memory cost via adjoint method, adaptive computation via solver control, continuous-depth networks.

**Strengths:** Continuous-depth, constant memory, adaptive computation, elegant mathematics.
**Weaknesses:** Slow (ODE solving), adjoint method can be unstable, harder to train than discrete layers.
**Who uses it:** Physics-informed ML, continuous dynamics modeling, normalizing flows.

---

#### Probabilistic (3 architectures)

Architectures for principled uncertainty quantification.

##### Bayesian NN

| | |
|---|---|
| **Module** | `Edifice.Probabilistic.Bayesian` |
| **Paper** | Blundell et al., "Weight Uncertainty in Neural Networks" (2015) |
| **Reference** | [arXiv:1505.05424](https://arxiv.org/abs/1505.05424) |
| **Origin** | Google DeepMind |

Each weight is a distribution (mu, rho) sampled via W = mu + softplus(rho)·ε. Trained by maximizing ELBO. Provides uncertainty estimation and learned regularization.

**Strengths:** Principled uncertainty, learned regularization, multiple weight samples reduce overconfidence.
**Weaknesses:** 2x parameters (mu + rho), expensive sampling, difficult to scale.
**Who uses it:** Safety-critical applications (medical, autonomous), uncertainty research.

##### MC Dropout

| | |
|---|---|
| **Module** | `Edifice.Probabilistic.MCDropout` |
| **Paper** | Gal & Ghahramani, "Dropout as a Bayesian Approximation" (2016) |
| **Reference** | [arXiv:1506.02142](https://arxiv.org/abs/1506.02142) |
| **Origin** | University of Cambridge |

Keep dropout ON at inference, run N forward passes, compute mean (prediction) and variance (uncertainty). Practical Bayesian approximation requiring zero training modification.

**Strengths:** Zero training modification needed, practical uncertainty estimation, simple.
**Weaknesses:** N forward passes at inference (slow), approximate (not exact Bayesian).
**Who uses it:** Any model needing uncertainty estimates with minimal effort, medical AI.

##### Evidential NN

| | |
|---|---|
| **Module** | `Edifice.Probabilistic.EvidentialNN` |
| **Paper** | Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty" (NeurIPS 2018) |
| **Reference** | [arXiv:1806.01768](https://arxiv.org/abs/1806.01768) |
| **Origin** | National Defense University (Turkey) |

Places Dirichlet distribution over class probabilities. Network outputs evidence parameters (alpha) for single-pass epistemic and aleatoric uncertainty estimation.

**Strengths:** Single forward pass uncertainty, separates epistemic and aleatoric uncertainty, no ensemble needed.
**Weaknesses:** Dirichlet assumption may not hold, evidence calibration can be tricky.
**Who uses it:** Out-of-distribution detection, safety-critical classification.

---

#### Memory-Augmented (2 architectures)

Architectures with external differentiable memory.

##### NTM (Neural Turing Machine)

| | |
|---|---|
| **Module** | `Edifice.Memory.NTM` |
| **Paper** | Graves et al., "Neural Turing Machines" (2014) |
| **Reference** | [arXiv:1410.5401](https://arxiv.org/abs/1410.5401) |
| **Origin** | Google DeepMind |

Neural network controller with external memory matrix read/written via differentiable attention. 4-stage addressing: content-based → interpolation → circular shift → sharpening.

**Strengths:** Can learn algorithms (copy, sort, recall), external memory decoupled from computation.
**Weaknesses:** Difficult to train, complex addressing mechanism, largely superseded.
**Who uses it:** Algorithm learning research, differentiable memory research (historical significance).

##### Memory Network

| | |
|---|---|
| **Module** | `Edifice.Memory.MemoryNetwork` |
| **Paper** | Sukhbaatar et al., "End-To-End Memory Networks" (2015) |
| **Reference** | [arXiv:1503.08895](https://arxiv.org/abs/1503.08895) |
| **Origin** | Meta AI / FAIR |

Iterative reasoning over memory slots via multi-hop attention. Each hop: attend to memory, read, update query. Different embedding matrices at each hop for different attention aspects.

**Strengths:** Multi-hop reasoning, iterative query refinement, interpretable attention.
**Weaknesses:** Fixed memory slots, largely superseded by transformer attention.
**Who uses it:** Question answering research, multi-hop reasoning (historical significance).

---

#### Neuromorphic (2 architectures)

Biologically-plausible architectures for ultra-low-power deployment on neuromorphic hardware.

##### SNN (Spiking Neural Network)

| | |
|---|---|
| **Module** | `Edifice.Neuromorphic.SNN` |
| **Paper** | Neftci et al., "Surrogate Gradient Learning in SNNs" (2019) |
| **Reference** | [arXiv:1901.09948](https://arxiv.org/abs/1901.09948) |
| **Origin** | UC San Diego |

Leaky Integrate-and-Fire (LIF) neurons processing information as discrete spikes. Surrogate gradients (sigmoid approximation) enable backpropagation through non-differentiable spike function.

**Strengths:** Energy-efficient on neuromorphic hardware (Intel Loihi, IBM TrueNorth), biologically plausible.
**Weaknesses:** Surrogate gradients are approximate, spike-based processing limits precision, specialized hardware needed.
**Who uses it:** Neuromorphic computing research, ultra-low-power applications, brain-inspired AI.

##### ANN2SNN

| | |
|---|---|
| **Module** | `Edifice.Neuromorphic.ANN2SNN` |
| **Paper** | Diehl et al. (IJCNN 2015); Rueckauer et al. (2017) |
| **Origin** | ETH Zurich |

Converts trained ANNs to spiking networks by replacing ReLU with integrate-and-fire neurons that encode activation magnitudes as spike rates. Train with standard backprop, deploy on neuromorphic hardware.

**Strengths:** Train with standard tools, deploy on neuromorphic hardware, leverages existing ANN training.
**Weaknesses:** Rate coding requires many timesteps for accuracy, conversion loss.
**Who uses it:** ANN-to-neuromorphic deployment pipeline, edge computing.

---

#### Feedforward (5 architectures)

Foundational and specialized feedforward architectures.

##### MLP

| | |
|---|---|
| **Module** | `Edifice.Feedforward.MLP` |
| **Origin** | Rosenblatt (1958), Rumelhart et al. (1986) |

Multi-layer perceptron — stacked dense layers with activations. Configurable hidden sizes, activation, dropout, layer norm, and residual connections.

**Strengths:** Universal approximator, simple, fast, building block for everything.
**Weaknesses:** No inductive bias (doesn't exploit structure in data), requires more data.
**Who uses it:** Everywhere — backbone component of virtually every architecture.

##### KAN (Kolmogorov-Arnold Network)

| | |
|---|---|
| **Module** | `Edifice.Feedforward.KAN` |
| **Paper** | Liu et al., "KAN: Kolmogorov-Arnold Networks" (2024) |
| **Reference** | [arXiv:2404.19756](https://arxiv.org/abs/2404.19756) |
| **Origin** | MIT / Ziming Liu |

Learnable activation functions on edges (not fixed activations on nodes). Based on Kolmogorov-Arnold representation theorem. Supports B-spline, sine, Chebyshev, Fourier, and RBF basis functions.

**Strengths:** Learnable activations, interpretable (visualizable), good for symbolic/scientific tasks.
**Weaknesses:** O(n²·g) parameters (g=grid_size), slower than MLP, less general-purpose.
**Who uses it:** Scientific ML, symbolic regression, interpretable AI research.

##### KAT (KAN-Attention Transformer)

| | |
|---|---|
| **Module** | `Edifice.Feedforward.KAT` |
| **Paper** | Combines Liu et al. (KAN, 2024) with Vaswani et al. (Attention, 2017) |
| **Origin** | Edifice composition |

Standard multi-head attention with KAN layers replacing the FFN sublayer. Learnable activation functions provide more expressive feature transformation than fixed-activation FFNs.

**Strengths:** More expressive FFN via learnable activations, attention + KAN synergy.
**Weaknesses:** Higher parameter count (grid_size multiplier), slower than standard transformer.
**Who uses it:** Hybrid architecture research, when standard FFN is the bottleneck.

##### TabNet

| | |
|---|---|
| **Module** | `Edifice.Feedforward.TabNet` |
| **Paper** | Arik & Pfister, "TabNet: Attentive Interpretable Tabular Learning" (AAAI 2021) |
| **Reference** | [arXiv:1908.07442](https://arxiv.org/abs/1908.07442) |
| **Origin** | Google Cloud AI |

Sequential attention for instance-wise feature selection in tabular data. Each decision step selects relevant features via sparse mask (sparsemax), with a relaxation factor controlling feature reuse.

**Strengths:** Inherently interpretable, instance-wise feature selection, designed for tabular data.
**Weaknesses:** Tabular-specific, sequential decision steps, complex training.
**Who uses it:** Tabular ML (alternative to gradient boosting), feature importance analysis.

##### BitNet

| | |
|---|---|
| **Module** | `Edifice.Feedforward.BitNet` |
| **Paper** | Wang et al., "BitNet" (2023); Ma et al., "The Era of 1-bit LLMs" (2024) |
| **Reference** | [arXiv:2310.11453](https://arxiv.org/abs/2310.11453) |
| **Origin** | Microsoft Research |

Quantizes weights to ternary {-1, 0, +1} in forward pass while maintaining full-precision for gradients. BitLinear layers with straight-through estimator. Binary (1-bit) and ternary (1.58-bit) modes.

**Strengths:** Massive memory reduction (1-1.58 bits/weight), faster inference (integer ops), maintains quality.
**Weaknesses:** Quantization-aware training required, straight-through estimator is approximate.
**Who uses it:** Efficient LLM deployment, edge inference, Microsoft Research.

---

#### Liquid (1 architecture)

##### Liquid Neural Network

| | |
|---|---|
| **Module** | `Edifice.Liquid` |
| **Paper** | Hasani et al., "Liquid Time-constant Networks" (AAAI 2021) |
| **Origin** | MIT CSAIL / Liquid AI |

Continuous-time ODE dynamics with learnable time constants: dx/dt = -x/τ + f(x, I, θ)/τ. Multiple ODE solvers: exact, Euler, midpoint, RK4, Dormand-Prince (adaptive).

**Strengths:** Continuous adaptation during inference, robust to distributional drift, physically interpretable.
**Weaknesses:** ODE solving is computationally expensive, more complex than discrete RNNs.
**Who uses it:** Liquid AI ($250M from AMD), autonomous driving, time series with drift.

---

## Architecture Gaps

Missing architectures identified through survey of current research, organized by priority.

### High Priority

| Architecture | Family | Why It Matters |
|---|---|---|
| **Hymba** | SSM | Hybrid Mamba + attention with learned routing — next-gen hybrid architecture |
| **RWKV-6/7 (full)** | Attention | Updated RWKV with improved token shift and time mixing — current v7 is simplified |
| **DeltaNet v2** | Recurrent | Improved delta rule with gated memory — significant accuracy improvements |
| **xLSTM v2 (sLSTM)** | Recurrent | Scalar LSTM with exponential gating — sequential state-tracking specialist |
| **Sparse Attention** | Attention | Block-sparse and sliding window patterns — essential for long-context LLMs |

### Medium Priority

| Architecture | Family | Why It Matters |
|---|---|---|
| **Hawk/Griffin v2** | Attention | Improved gated linear recurrence — Google's RecurrentGemma successor |
| **RetNet v2** | Attention | Improved multi-scale retention with better decay schedules |
| **GLA v2** | Attention | Enhanced gated linear attention with improved forget gates |
| **HGRN v2** | Attention | Multi-resolution hierarchical gated RNN |
| **FlashLinearAttention** | Attention | Hardware-efficient linear attention kernel for production |
| **LoRA+/DoRA** | Meta | Enhanced parameter-efficient fine-tuning with direction-aware updates |
| **DiT v2** | Generative | Improved diffusion transformer conditioning |
| **Byte Latent Transformer** | Transformer | Byte-level processing with latent patching — emerging paradigm |

### Low Priority / Exploratory

| Architecture | Family | Why It Matters |
|---|---|---|
| **Test-Time Compute** | Meta | Dynamic inference-time scaling strategies |
| **Mixture of Tokenizers** | Meta | Multi-granularity tokenization routing |
| **Medusa/EAGLE** | Meta | Speculative decoding heads for faster inference |
| **Distillation Head** | Meta | Knowledge distillation output head |
| **QAT utilities** | Meta | Quantization-aware training beyond BitNet |
| **Native Recurrence** | Recurrent | Hardware-native recurrent implementations for specialized accelerators |
| **State Space Transformer** | SSM/Transformer | Deeper SSM-attention integration (beyond simple interleaving) |

See [ARCHITECTURE_ROADMAP.md](../ARCHITECTURE_ROADMAP.md) for implementation timeline and prioritization.

---

## Cross-Reference

Every architecture in `Edifice.list_architectures/0` appears in this taxonomy. Family counts match `Edifice.list_families/0`:

| Family | Count | Verified |
|--------|-------|----------|
| ssm | 15 | yes |
| attention | 19 | yes |
| recurrent | 10 | yes |
| transformer | 1 | yes |
| vision | 9 | yes |
| convolutional | 6 | yes |
| graph | 8 | yes |
| sets | 2 | yes |
| generative | 11 | yes |
| contrastive | 6 | yes |
| meta | 10 | yes |
| energy | 3 | yes |
| probabilistic | 3 | yes |
| memory | 2 | yes |
| neuromorphic | 2 | yes |
| feedforward | 5 | yes |
| liquid | 1 | yes |
| **Total** | **113** | |
