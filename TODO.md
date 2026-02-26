# Edifice — Architecture TODO

## v0.2.0 (done)

- [x] Transformer (decoder-only) — GPT-style with GQA+RoPE+SwiGLU+RMSNorm
- [x] Mixture of Depths — Dynamic per-token compute allocation
- [x] RLHF/DPO Head — Reward model and preference heads
- [x] KAT — KAN + attention hybrid
- [x] mLSTM — Registry alias for xLSTM variant
- [x] RoPE option — Added to MultiHead and GQA
- [x] TTT variants — :linear and :mlp inner models
- [x] Based — Linear attention with Taylor expansion kernels
- [x] BitNet — Binary/ternary weight quantization
- [x] StripedHyena — Gated conv + Hyena hybrid
- [x] Mega — EMA + single-head gated attention
- [x] Conformer — Conv + Transformer for audio
- [x] FocalNet — Focal modulation for vision
- [x] PoolFormer — Pooling-based MetaFormer
- [x] NeRF — Positional encoding + MLP for radiance fields
- [x] GINv2 — GIN with edge features
- [x] Mixture of Agents — Multi-proposer + aggregator routing
- [x] RingAttention — Chunked attention with ring pattern
- [x] InfiniAttention — Compressive memory + local attention
- [x] CausalMask block — Unified mask creation
- [x] DepthwiseConv block — 1D depthwise separable convolution
- [x] TransformerBlock :custom_ffn — Callback for non-standard FFN

- [x] Mamba-3 — Complex states, trapezoidal discretization, MIMO rank-r
- [x] MLA — Multi-Head Latent Attention (DeepSeek-style KV compression)
- [x] JEPA — Joint Embedding Predictive Architecture (self-supervised)
- [x] Differential Transformer — Dual softmax attention with noise cancellation

## v0.3.0 (done)
- [x] **Hymba** — Hybrid Mamba+attention with learnable meta tokens
- [x] **sLSTM** — Scalar LSTM with exponential gating (xLSTM component)
- [x] **GSS** — Gated State Space (simplified S4 with multiplicative gating)
- [x] **Hawk/RecurrentGemma** — Google's RG-LRU recurrent model
- [x] **DiT v2** — Updated diffusion transformer with improved adaptive norm conditioning
- [x] **Mixture of Experts v2** — Expert choice routing, shared expert slots
- [x] **State Space Duality (SSD)** — Improved Mamba-2 structured masking
- [x] **xLSTM v2** — Updated mLSTM with matrix memory improvements
- [x] **Hyena v2** — Improved implicit long convolution filters
- [x] **RetNet v2** — Retention with improved chunkwise formulation
- [x] **MEGALODON** — Mega-scale sequence model (Meta)
- [x] **KV Cache support** — Inference-time KV caching for autoregressive models
- [ ] **Flash Attention** — IO-aware exact attention (requires EXLA backend work)
- [x] **Quantization toolkit** — GPTQ, AWQ, SqueezeLLM weight quantization
- [x] **LoRA+ / DoRA** — Improved low-rank adaptation variants

## 2026 Wave 1 (done)
- [x] **Gated DeltaNet** — Linear attention with data-dependent gating (Qwen3-Next, Kimi Linear)
- [x] **RWKV-7** — Generalized delta rule, "Goose" architecture
- [x] **TTT-E2E** — End-to-end test-time training
- [x] **MMDiT** — Multimodal Diffusion Transformer (FLUX.1, SD3)
- [x] **SoFlow** — Flow matching + consistency loss
- [x] **KDA** — Kimi Delta Attention (channel-wise decay)
- [x] **MambaVision** — 4-stage hierarchical CNN+Mamba+Attention
- [x] **Multimodal MLP Fusion** — MLP projection, cross-attention, Perceiver resampler
- [x] **RL Integration** — PPOTrainer, GAE, CartPole, GridWorld environments
- [x] **iRoPE** — Interleaved RoPE in decoder_only (Llama 4 pattern)
- [x] **Aux-loss-free MoE** — Bias-based load balancing in MoE v2

## 2026 Wave 2 (done)
- [x] **Gated Attention** — Sigmoid post-attention gate (NeurIPS 2025 best paper)
- [x] **NSA** — Native Sparse Attention (DeepSeek three-path)
- [x] **Scalable-Softmax** — Drop-in softmax replacement
- [x] **Softpick** — Non-saturating sparse attention function
- [x] **VAR** — Visual Autoregressive (next-scale prediction, NeurIPS 2024 best paper)
- [x] **Transfusion** — Unified AR text + diffusion images
- [x] **Linear DiT (SANA)** — Linear attention for diffusion
- [x] **SiT** — Scalable Interpolant Transformer
- [x] **MAR** — Masked Autoregressive generation
- [x] **DINOv2** — Self-distillation vision backbone
- [x] **MetaFormer / CAFormer** — Architecture-first framework
- [x] **EfficientViT** — Linear attention ViT
- [x] **SigLIP** — Sigmoid contrastive learning
- [x] **FNO** — Fourier Neural Operator (scientific ML)
- [x] **EGNN** — E(n)-Equivariant GNN
- [x] **DPO** — Direct Preference Optimization
- [x] **GRPO** — Group Relative Policy Optimization
- [x] **KTO** — Kahneman-Tversky Optimization
- [x] **Engram** — O(1) hash-based associative memory
- [x] **RNoPE-SWA** — No positional encoding + sliding window
- [x] **YaRN** — RoPE context extension
- [x] **Dual Chunk Attention** — Long-context chunked attention
- [x] **TMRoPE** — Time-aligned Multimodal RoPE
- [x] **Medusa** — Multi-head speculative decoding
- [x] **Gaussian Splatting** — 3D Gaussian Splatting (NeRF successor)
- [x] **TRELLIS** — Sparse 3D lattice generation
- [x] **CogVideoX** — 3D causal video generation
- [x] **ACT** — Action Chunking Transformer (robotics)
- [x] **OpenVLA** — Vision-Language-Action model
- [x] **EnCodec** — Neural audio codec
- [x] **VALL-E** — Codec language model for TTS
- [x] **SoundStorm** — Parallel audio token generation
- [x] **GGUF Export** — Model export to GGUF format

## 2026 Wave 3 — New Families & Gap Fills

### Detection / Segmentation (new family)
- [ ] **DETR** — DEtection TRansformer (set-based object detection with bipartite matching). Encoder-decoder transformer + learned object queries + Hungarian loss. Family: `detection`.
- [ ] **RT-DETR** — Real-Time DETR (Baidu). Hybrid CNN+transformer encoder, anchor-free, NMS-free. 53-55% AP at 108 FPS. Practical real-time detection baseline.
- [ ] **SAM 2** — Segment Anything Model 2 (Meta). Promptable segmentation for images + video. Image encoder + prompt encoder + mask decoder + memory attention for video. Major 2024/2025 release.

### Attention
- [ ] **Sigmoid Self-Attention** — Drop-in softmax replacement using properly normalized sigmoid (ICLR 2025). FlashSigmoid yields 17% kernel speedup over FlashAttention2 on H100. Eliminates token competition. Standalone mechanism, distinct from Gated Attention's post-SDPA sigmoid gate.

### RL
- [ ] **Decision Transformer** — Offline RL as conditional sequence generation (Chen et al. 2021). Frames RL as sequence modeling: conditions on desired return, state, action triples. Causal transformer predicts next action given (R, s, a) history. Directly relevant to ExPhil imitation learning pipeline.

### Audio
- [ ] **Whisper** — Encoder-decoder ASR (OpenAI). Log-mel spectrogram frontend + transformer encoder-decoder with multitask training (transcription, translation, timestamps, language ID). Fills the ASR gap — audio family has TTS but no recognition.

### Generative
- [ ] **Mercury/MDLM** — Discrete diffusion LM (Inception Labs, arXiv:2506.17298). Parallel token denoising instead of autoregressive generation. Transformer backbone + discrete noise process + iterative refinement. 10x decoding speedup. Related work: MDLM, SEDD, Plaid. New family: `diffusion_lm` or under `generative`.
- [ ] **Rectified Flow** — Straight-trajectory flow matching variant. ODE paths trained to be straight lines, enabling 10-100x fewer inference steps than vanilla diffusion. Can be a variant/option on existing FlowMatching or standalone module.

### Vision
- [ ] **DINOv3** — Self-supervised vision backbone (Meta AI, Aug 2025). CLIP-like image-text alignment + axial RoPE + Gram anchoring, scaled to 7B params. Major upgrade over DINOv2.

### Meta / Efficiency
- [ ] **EAGLE-3** — Multi-level speculative draft head. Extracts low/mid/high features from target model for multi-step draft prediction. 4-6x decoding speedup. Scaling law for speculative decoding.
- [ ] **ReMoE** — Fully differentiable MoE routing (ICLR 2025). Replaces discrete top-k with continuous relaxation via Gumbel-Softmax. Better gradient flow through routing.
- [ ] **mHC** — Manifold Hyper-Connections (DeepSeek-V4). Multi-rate residual streams.

### Graph
- [ ] **DimeNet** — Directional message passing with angle information between atoms. Important for molecular property prediction.
- [ ] **SE(3)-Transformer** — Equivariant transformer for structural biology.

### Remaining Candidates
- [ ] **Flash Attention** — IO-aware exact attention (requires EXLA backend work)
- [ ] **SPLA** — Sparse + Linear Attention hybrid
- [ ] **InfLLM-V2** — Block-partitioned KV cache selection
- [ ] **F5-TTS** — Non-autoregressive flow-matching TTS
- [ ] **JanusFlow** — AR text + rectified flow images
- [ ] **Show-o** — AR + discrete diffusion
- [ ] **Diffusion Policy** — Diffusion for robot action generation
- [ ] **CausVid** — Causal video DiT distillation
- [ ] **DeepONet** — Branch-trunk operator learning
- [ ] **MAGVIT-v2** — Lookup-free quantization for image/video tokens
- [ ] **MIRAS** — Google's Titans extension framework
- [ ] **MoR** — Mixture of Recursions
- [ ] **MoED** — Mixture of Expert Depths
- [ ] **Agent swarm patterns** — Multi-agent coordination framework
- [ ] **PointNet++** — Hierarchical point cloud processing
- [ ] **Wav2Vec 2.0** — Self-supervised speech backbone
- [ ] **Janus Multimodal** — Decoupled visual encoding for understanding + generation (CVPR 2025)
- [ ] **GPS** — General Powerful Scalable graph transformer

## 🔍 Opus Review Pass — AI-Generated Architecture Implementations (2026-02-26)

All architectures added since Tier 1 (2026-02) were implemented by Claude Code (sonnet).
Reviewed by Opus for correctness, math accuracy, and idiomatic Elixir.

### Clean — no code changes needed (6/8)
- `lib/edifice/attention/nsa.ex` — 3-path sparse attention correct, proper 6-arg Nx.dot batching
- `lib/edifice/generative/transfusion.ex` — mixed AR+diffusion masking correct, dual heads + dual loss
- `lib/edifice/graph/egnn.ex` — equivariant coord update equations correct, proper Nx.dot batching
- `lib/edifice/memory/engram.ex` — LSH hashing via sign-based binary projection correct, EMA sound
- `lib/edifice/attention/yarn.ex` — wavelength-based frequency scaling correct, norm-preserving RoPE
- `lib/edifice/scientific/fno.ex` — spectral convolution correct; O(n^2) DFT matrix (Nx lacks FFT) known limitation

### Fixed (2/8)
- `lib/edifice/meta/moe_v2.ex` — stack_fn fallback was broken for non-standard expert counts (3,5,6,7). Arity-1 generic closure incompatible with Axon.layer positional arg unpacking. Replaced with explicit cases for 2-8 experts.
- `lib/edifice/generative/var.ex` — token embedding used deterministic Nx.iota projection instead of learnable weights. Replaced with Axon.nx (one_hot) + Axon.dense (no bias) for proper learnable embedding table. Note: decoder reshape has a separate pre-existing bug (not addressed here).

---

## CUDA Kernel Fusion for Recurrent Architectures - 2026-02-18 22:45

- **Explore fused RNN kernels for LSTM/GRU/minGRU/minLSTM** - Plan what's needed to make recurrent architectures competitive on GPU inference latency. **Problem:** Axon unrolls each recurrence timestep as separate EXLA kernel launches, causing 70-600ms latency for seq_len=32 vs 14ms for gated_ssm. TensorFlow/PyTorch use cuDNN's fused `cudnnRNNForward` kernel which handles all timesteps in one GPU call. **Files:** `bench/inference_latency.exs`, `lib/edifice/recurrent/lstm.ex`, `lib/edifice/recurrent/gru.ex`, `lib/edifice/recurrent/min_gru.ex`, `lib/edifice/recurrent/min_lstm.ex`. **Solution:** Investigate four approaches: (1) cuDNN fused RNN integration via EXLA/XLA custom calls, (2) custom CUDA kernels for fused LSTM cells callable from Nx, (3) XLA's built-in RNN fusion passes and whether EXLA exposes them, (4) step-by-step inference with explicit state passing (seq_len=1 per frame) which sidesteps unrolling entirely. Reference: slippi-ai achieves real-time LSTM inference via TensorFlow's cuDNN integration. Benchmark data in `tmp/bench_results/`.
