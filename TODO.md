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

## Open — Codebase Quality (from 2026-02-27 evaluation)

Full findings in `notebooks/research/codebase_evaluation.md`.

### Testing — Shared Block Test Files (Priority: High)

The 20 shared blocks are the foundation for all composability improvements but
only 1/20 has a dedicated test file. Each block needs isolated tests so that
refactoring doesn't silently break consumers.

- [ ] **TransformerBlock tests** — `test/edifice/blocks/transformer_block_test.exs`. Test `layer/2` (2-sublayer), `layer/3` (3-sublayer decoder), `stack/3`, `stack/4`. Verify pre-norm vs post-norm, dropout, custom_ffn callback, cross_attention_fn callback. Edge cases: 1 layer, hidden_dim not divisible by num_heads.
- [ ] **FFN tests** — `test/edifice/blocks/ffn_test.exs`. Test `layer/2` (standard) and `gated_layer/2` (SwiGLU). Verify expansion_factor, inner_size override, dropout, activation options. Shape assertions for each variant.
- [ ] **CrossAttention tests** — `test/edifice/blocks/cross_attention_test.exs`. Test `layer/3` with different Q/KV sequence lengths. Verify output shape matches query sequence length. Multi-head correctness (num_heads option).
- [ ] **CausalMask tests** — `test/edifice/blocks/causal_mask_test.exs`. Test `causal/1` (lower triangular), `window/2` (sliding window). Verify mask shapes, boolean values, edge cases (seq_len=1). Test `to_binary_backend/1`.
- [ ] **SDPA tests** — `test/edifice/blocks/sdpa_test.exs`. Test `compute/4` with and without mask. Verify batched attention shapes, scaling factor, softmax output sums to 1.
- [ ] **RoPE tests** — `test/edifice/blocks/rope_test.exs`. Test `precompute_freqs/2`, `apply_rotary/2`. Verify rotation preserves norms. Test YaRN scaling. Shape assertions for different dim/seq_len combos.
- [ ] **SinusoidalPE tests** — `test/edifice/blocks/sinusoidal_pe_test.exs`. Test `build_table/1` and `layer/2`. Verify output shape, orthogonality of PE vectors, encoding is deterministic (no learned params).
- [ ] **ModelBuilder tests** — `test/edifice/blocks/model_builder_test.exs`. Test `build_sequence_model/1` with different output_mode values (:last_timestep, :all, :mean_pool). Verify input projection, block stacking, final norm.
- [ ] **RMSNorm tests** — `test/edifice/blocks/rms_norm_test.exs`. Test `layer/2`. Verify output has unit RMS. Compare to Axon.layer_norm behavior.
- [ ] **SwiGLU tests** — `test/edifice/blocks/swiglu_test.exs`. Test `layer/2`. Verify gated output shape, activation variants (:silu, :gelu, :relu).
- [ ] **Remaining block tests** — AdaptiveNorm, ALiBi, BBoxHead, DepthwiseConv, KVCache, PatchEmbed, SinusoidalPE2D, Softpick, SSMax, Upsample2x. One test file per block, following the same build→init→predict→shape pattern.

### Testing — Family Coverage Gaps (Priority: Medium)

107 of 223 source files have no dedicated test file (48%). The registry sweep
provides shallow coverage but doesn't catch option handling, edge cases, or
numerical issues. Prioritized by risk (shared blocks > frequently-used families
> leaf modules).

- [ ] **Graph family tests** — 0/10 coverage. Need test files for GCN, GAT, GraphSAGE, GIN, GINv2, PNA, SchNet, EGNN, GraphTransformer, MessagePassing. Graph modules have unique input shapes (adjacency matrices) that the registry sweep may not exercise well.
- [ ] **Vision family tests** — 0/15 coverage. Need test files for ViT, Swin, DeiT, ConvNeXt, MLP-Mixer, MetaFormer, PoolFormer, FocalNet, EfficientViT, DINOv2, MambaVision, NeRF, GaussianSplat, U-Net. Vision modules use 4D inputs that need specific shape testing.
- [ ] **Contrastive family tests** — 0/8 coverage. SimCLR, BYOL, BarlowTwins, VICReg, SigLIP, MAE, JEPA, TemporalJEPA. These return tuple outputs (encoder, projector) that need structural tests.
- [ ] **Detection family tests** — 0/3 coverage. DETR, RT-DETR, SAM2. Multi-input models with container outputs — the registry sweep may not cover the full input interface.
- [ ] **SSM family tests** — 2/21 coverage. 19 untested modules. Core SSM primitives (S4, S4D, S5, Mamba variants, Hyena variants) should have dedicated shape and scan-correctness tests.
- [ ] **Meta family tests** — 1/23 coverage. LoRA, DoRA, MoE, MoEv2, Adapters, etc. Meta-modules wrap other modules — need tests that verify wrapping behavior doesn't alter base model shapes.
- [ ] **Attention family tests** — 5/34 coverage. 29 untested. Priority: GQA, Conformer, InfiniAttention, MLA (most-used attention variants after MultiHead).

### Testing — Test Depth Improvements (Priority: Medium)

- [ ] **Add batch=1 edge cases** — Only 27% of test files check batch_size=1. Add `test "handles batch_size=1"` to all architecture test files. Batch=1 is the inference case and frequently exposes broadcasting bugs.
- [ ] **Add Edifice.build/2 integration tests** — Only 14% of test files verify registry access. Each test file should include a `describe "Edifice.build/2"` block confirming the module is reachable via the registry.
- [ ] **Add output_size/1 tests** — Modules exposing `output_size/1` should test it returns the expected value for default and custom options.
- [ ] **ExCoveralls integration** — Add `excoveralls` to CI pipeline. Track line coverage, set minimum threshold (70% initially, increase over time). Add coverage badge to README.

### Documentation — Doctests (Priority: High)

- [ ] **Doctests for Edifice registry** — Add iex examples to `Edifice.build/2`, `list_architectures/0`, `list_families/0`, `module_for/1`. These render on the main HexDocs page and are the first thing visitors see.
- [ ] **Doctests for shared blocks** — Add iex examples to `TransformerBlock.layer/2`, `FFN.layer/2`, `CrossAttention.layer/3`, `CausalMask.causal/1`, `SinusoidalPE.layer/2`, `RoPE.precompute_freqs/2`, `SDPA.compute/4`. These are the building blocks users will compose from.
- [ ] **Doctests for 3-5 representative architectures** — Pick one module per major family (e.g., MLP, LSTM, Mamba, ViT, GAN) and add iex examples showing build + forward pass. Demonstrates the consistent API pattern.

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
- [ ] **Architecture visualization** — `mix edifice.viz :mamba` that prints layer structure as ASCII tree or generates Mermaid diagram. Helps users understand what `build/1` actually constructs.
- [ ] **Gradient smoke tests** — For each family, verify gradients flow through the model (non-zero grads on all params after a forward+backward pass). Catches dead layers and broken backward paths.
