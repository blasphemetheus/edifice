# Edifice — Architecture TODO

## Current Status

196 registered architectures across 26 families, 20 shared blocks, 2170+ tests.

## Completed Milestones

### v0.2.0
Transformer (decoder-only), MixtureOfDepths, RLHF/DPO Head, KAT, mLSTM alias,
RoPE option, TTT variants, Based, BitNet, StripedHyena, Mega, Conformer, FocalNet,
PoolFormer, NeRF, GINv2, MixtureOfAgents, RingAttention, InfiniAttention,
CausalMask block, DepthwiseConv block, TransformerBlock :custom_ffn,
Mamba-3, MLA, JEPA, DiffTransformer.

### v0.3.0
Hymba, sLSTM, GSS, Hawk/RecurrentGemma, DiT v2, MoE v2, SSD, xLSTM v2,
Hyena v2, RetNet v2, MEGALODON, KV Cache, Quantization toolkit (GPTQ/AWQ/SqueezeLLM),
LoRA+/DoRA.

### 2026 Wave 1
Gated DeltaNet, RWKV-7, TTT-E2E, MMDiT, SoFlow, KDA, MambaVision,
Multimodal MLP Fusion, RL Integration (PPO/GAE/CartPole/GridWorld),
iRoPE, Aux-loss-free MoE.

### 2026 Wave 2
Gated Attention, NSA, Scalable-Softmax, Softpick, VAR, Transfusion, Linear DiT (SANA),
SiT, MAR, DINOv2, MetaFormer/CAFormer, EfficientViT, SigLIP, FNO, EGNN,
DPO, GRPO, KTO, Engram, RNoPE-SWA, YaRN, Dual Chunk Attention, TMRoPE, Medusa,
Gaussian Splatting, TRELLIS, CogVideoX, ACT, OpenVLA, EnCodec, VALL-E,
SoundStorm, GGUF Export.

### 2026 Wave 3
Detection family (DETR, RT-DETR, SAM 2), Sigmoid Self-Attention,
Decision Transformer, Whisper, Mercury/MDLM, Rectified Flow, ReMoE.

### Composability Audit (complete)

**TransformerBlock encoder-decoder** (`3a5bb44`):
`layer/3` (3-sublayer decoder), `stack/4`, `:cross_attention_fn` callback.
Adopted by DETR, RT-DETR, ACT, Whisper.

**Shared block adoption** (`7492b56`):
AdaptiveNorm `modulate/3`+`gate/3` (6 modules), CrossAttention `layer/4` (3 modules),
RoPE `apply_rotary_4d/3` (4 modules), SinusoidalPE `timestep_layer/2` (8 modules),
SwiGLU (MDLM), RMSNorm (DiTv2 + TransformerLike bug fix).

**Duplicate extraction** (`fedcf97`):
SDPA.compute (6 copies), SinusoidalPE2D (3 copies), Upsample2x (2 copies),
BBoxHead (2 copies), CausalMask migration (2 modules), TopK sparsify dedup.

**Final SDPA adoption** (`51c2e00`):
VALLE (SDPA + SinusoidalPE.layer), Perceiver (SDPA), Decision Transformer (SDPA + CausalMask).

### Opus Review Pass (2026-02-26)
8 architectures reviewed. 6 clean, 2 fixed (MoE v2 stack_fn, VAR token embedding).

---

## Open — Architecture Candidates

### Near-term
- [ ] **DINOv3** — Self-supervised vision backbone (Meta AI, Aug 2025). CLIP-like alignment + axial RoPE + Gram anchoring, 7B params.
- [ ] **EAGLE-3** — Multi-level speculative draft head. 4-6x decoding speedup.
- [ ] **mHC** — Manifold Hyper-Connections (DeepSeek-V4). Multi-rate residual streams.

### Graph
- [ ] **DimeNet** — Directional message passing with angle information between atoms.
- [ ] **SE(3)-Transformer** — Equivariant transformer for structural biology.

### Backlog
- [ ] Flash Attention — IO-aware exact attention (requires EXLA backend work)
- [ ] SPLA — Sparse + Linear Attention hybrid
- [ ] InfLLM-V2 — Block-partitioned KV cache selection
- [ ] F5-TTS — Non-autoregressive flow-matching TTS
- [ ] JanusFlow — AR text + rectified flow images
- [ ] Show-o — AR + discrete diffusion
- [ ] Diffusion Policy — Diffusion for robot action generation
- [ ] CausVid — Causal video DiT distillation
- [ ] DeepONet — Branch-trunk operator learning
- [ ] MAGVIT-v2 — Lookup-free quantization for image/video tokens
- [ ] MIRAS — Google's Titans extension framework
- [ ] MoR — Mixture of Recursions
- [ ] MoED — Mixture of Expert Depths
- [ ] PointNet++ — Hierarchical point cloud processing
- [ ] Wav2Vec 2.0 — Self-supervised speech backbone
- [ ] Janus Multimodal — Decoupled visual encoding (CVPR 2025)
- [ ] GPS — General Powerful Scalable graph transformer
- [ ] Agent swarm patterns — Multi-agent coordination framework

---

## Open — Infrastructure

- [ ] **CUDA Kernel Fusion** — Fused RNN kernels for LSTM/GRU/minGRU/minLSTM. Axon unrolls each timestep as separate kernel launches (70-600ms for seq_len=32 vs 14ms for gated_ssm). Investigate cuDNN integration, custom CUDA kernels, XLA fusion passes, or seq_len=1 inference. See `bench/inference_latency.exs`.
