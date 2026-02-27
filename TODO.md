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
- [x] **DINOv3** — Self-supervised vision backbone (Meta AI, Aug 2025). Axial 2D RoPE + LayerScale + Sinkhorn-Knopp centering + iBOT + Gram anchoring.
- [x] **EAGLE-3** — Multi-level feature fusion draft head (NeurIPS 2025). Low/mid/high target features, single decoder layer, GQA, SwiGLU, vocabulary mapping.
- [x] **mHC** — Manifold Hyper-Connections (DeepSeek, arXiv:2512.24880). Multi-rate residual streams with Sinkhorn-Knopp doubly stochastic mixing on Birkhoff polytope.

### Graph
- [x] **DimeNet** — Directional message passing (DimeNet++) with radial Bessel basis, Chebyshev angular basis, and Hadamard interaction blocks.
- [x] **SE(3)-Transformer** — 3D roto-translation equivariant attention with fiber features (type-0 scalars + type-1 vectors), invariant attention, and TFN-style direction messages.

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

## Open — Codebase Quality (from 2026-02-27 evaluation)

Full findings in `notebooks/research/codebase_evaluation.md`.

### Testing — Shared Block Test Files (Priority: High) ✓

All 20 shared blocks now have dedicated test files (222 tests in `test/edifice/blocks/`).
TransformerBlock, FFN, CrossAttention, CausalMask, SDPA, RoPE, SinusoidalPE,
ModelBuilder, RMSNorm, SwiGLU, AdaptiveNorm, ALiBi, BBoxHead, DepthwiseConv,
KVCache, PatchEmbed, SinusoidalPE2D, Softpick, SSMax, Upsample2x.

### Testing — Family Coverage Gaps (Priority: Medium)

Coverage has improved significantly since the initial audit. Most families now
have dedicated test files. Remaining gaps are leaf modules or minor variants.

- [x] **Graph family tests** — 8 test files covering all 11 modules (GCN, GAT, GraphSAGE, GIN, GINv2, PNA, SchNet, EGNN, GraphTransformer, MessagePassing, DimeNet).
- [x] **Vision family tests** — 13 test files (ViT, Swin, ConvNeXt, MetaFormer, PoolFormer, FocalNet, EfficientViT, DINOv2, DINOv3, MambaVision, NeRF, GaussianSplat, U-Net).
- [x] **Contrastive family tests** — 5 test files (BYOL, JEPA, TemporalJEPA + contrastive_test + correctness).
- [x] **Detection family tests** — 3/3 coverage (DETR, RT-DETR, SAM2).
- [x] **SSM family tests** — 19 test files covering all modules (Mamba, SSD, S4, S4D, S5, H3, Hyena, GatedSSM, StripedHyena, Mamba3, etc.).
- [x] **Meta family tests** — 22 test files (MoE, MoEv2, MixtureOfDepths, MixtureOfAgents, RLHFHead, Capsules, LoRA, DoRA, DPO, GRPO, KTO, EAGLE-3, mHC, ReMoE, etc.).
- [x] **Attention family tests** — 36 test files covering all attention variants (MultiHead, GQA, Conformer, InfiniAttention, MLA, DiffTransformer, NSA, Sigmoid, etc.).

### Testing — Test Depth Improvements (Priority: Medium)

- [ ] **Add batch=1 edge cases** — Only 27% of test files check batch_size=1. Add `test "handles batch_size=1"` to all architecture test files. Batch=1 is the inference case and frequently exposes broadcasting bugs.
- [ ] **Add Edifice.build/2 integration tests** — Only 14% of test files verify registry access. Each test file should include a `describe "Edifice.build/2"` block confirming the module is reachable via the registry.
- [ ] **Add output_size/1 tests** — Modules exposing `output_size/1` should test it returns the expected value for default and custom options.
- [ ] **ExCoveralls integration** — Add `excoveralls` to CI pipeline. Track line coverage, set minimum threshold (70% initially, increase over time). Add coverage badge to README.

### Documentation — Doctests (Priority: High) ✓

19 doctests across registry, shared blocks, and representative architectures.

- [x] **Doctests for Edifice registry** — `Edifice.build/2`, `list_architectures/0`, `list_families/0`, `module_for/1` (6 doctests in `test/edifice_test.exs`).
- [x] **Doctests for shared blocks** — `TransformerBlock.layer/2`, `FFN.layer/2`, `CrossAttention.layer/3`, `CausalMask.causal/1`, `CausalMask.window/2`, `SinusoidalPE.build_table/1`, `RoPE.precompute_freqs/3`, `SDPA.compute/5` (8 doctests in `test/edifice/blocks/doctest_test.exs`).
- [x] **Doctests for 5 representative architectures** — MLP (with full forward pass), LSTM, Mamba, GAN, ViT (5 doctests in `test/edifice/architecture_doctest_test.exs`).

### Documentation — Guides & Notebooks (Priority: Medium)

- [ ] **Composition guide** — New guide: `guides/composing_architectures.md`. Show how to use TransformerBlock callbacks to mix attention mechanisms, swap FFN variants, combine encoder/decoder from different families. Include before/after code showing custom block vs shared block usage.
- [ ] **Livebook notebooks** — Create 3-5 `.livemd` notebooks: (1) "Build your first model" — walk through build/init/predict cycle, (2) "Architecture comparison" — benchmark 5 architectures on same task, (3) "Custom architecture from blocks" — compose a novel model from shared blocks, (4) "Whisper ASR demo" — end-to-end encoder-decoder usage, (5) "Training a small model" — connect to Axon training loop.
- [ ] **CODE_OF_CONDUCT.md** — Add Contributor Covenant (GitHub OSPO community standard). Signals the project welcomes contributions.

### Module Decomposition (Priority: Low-Medium)

- [ ] **Split multi_head.ex** (1,152 lines) — Extract 5 attention algorithms into focused modules: `attention/standard_attention.ex`, `attention/sliding_window_attention.ex`, `attention/chunked_attention.ex`, `attention/memory_efficient_attention.ex`, `attention/online_softmax_attention.ex`. Keep `multi_head.ex` as the entry point that dispatches by option. Each extracted module gets its own test file.
- [ ] **Vision backbone interface** — Define a shared interface for vision modules (ViT, Swin, DeiT, ConvNeXt, etc.) so they can be used interchangeably as feature extractors. Consider an `Edifice.Vision.Backbone` behaviour with `build_encoder/1` callback.

### CI/CD Improvements (Priority: Medium)

- [ ] **Multi-version test matrix** — Test against Elixir 1.18 + 1.19 + 1.20 on OTP 27 + 28. The `mix.exs` claims `~> 1.18` compatibility; CI should verify it.
- [ ] **Benchmark regression CI** — Run Benchee on 5-10 key architectures in CI. Store baseline timings, fail if >10% regression. Candidate architectures: MLP, LSTM, Mamba, GQA, ViT, DETR (covers major families and input patterns).
- [ ] **Normalize git tag format** — Use consistent `v` prefix on all tags. Current mix: `0.1.1` and `v0.2.0`.

### ML-Specific Quality (Priority: Low)

- [ ] **Pretrained weight loading** — Support loading weights for 2-3 reference architectures (e.g., Whisper tiny, a small MLP). Even minimal weight support differentiates from "just architecture definitions." Investigate SafeTensors format for Elixir.
- [ ] **ONNX integration guide** — Document workflow: Edifice.build → Axon model → axon_onnx export → inference in other runtimes. Even if axon_onnx is a separate package, showing the integration path is valuable.
- [x] **Architecture visualization** — `mix edifice.viz mamba` prints layer structure as table (default), ASCII tree (`--format tree`), or Mermaid diagram (`--format mermaid`). Handles tuple-returning models via `--component`. See `Edifice.Display` module.
- [ ] **Gradient smoke tests** — For each family, verify gradients flow through the model (non-zero grads on all params after a forward+backward pass). Catches dead layers and broken backward paths.
