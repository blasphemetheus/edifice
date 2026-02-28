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

### Interpretability (Priority)

Full research notes in `notebooks/research/interpretability_architectures.md`.

**Tier 1 — High value, straightforward:**
- [x] **Gated SAE** — Gated encoder decouples feature selection from magnitude (DeepMind, NeurIPS 2024). Near-drop-in SAE improvement, ~50% better reconstruction at same sparsity.
- [x] **JumpReLU SAE** — Per-feature learned threshold replaces TopK (DeepMind/Gemma Scope, 2024). Adaptive sparsity without rigid k constraint.
- [x] **BatchTopK SAE** — Batch-global top-k instead of per-sample (Bussmann, ICLR 2025). Variable per-sample sparsity within batch budget.
- [x] **Linear Probe** — Single linear layer for concept detection in frozen activations (Alain & Bengio 2016). Foundational interpretability tool, trivial architecture.
- [x] **Crosscoder** — Joint SAE across multiple model checkpoints/layers with shared dictionary (Anthropic, Dec 2024). Finds features shared across training stages.

**Tier 2 — Moderate complexity:**
- [x] **Concept Bottleneck** — Intermediate interpretable concept layer before task prediction (Koh et al., ICML 2020). Inherently interpretable, enables concept interventions.
- [x] **DAS Probe** — Distributed Alignment Search finds causal linear subspaces for concepts (Geiger et al., ICLR 2024). Stronger than linear probes, needs orthogonal parameterization.
- [x] **LEACE** — Least-squares concept erasure via projection (Belrose et al., ICML 2023). Closed-form, gold-standard concept removal.
- [x] **Matryoshka SAE** — Nested multi-scale SAE with ordered features (Bussmann, 2025). One model, multiple granularity levels.
- [x] **Cross-Layer Transcoder** — Extends Transcoder to all MLP layers simultaneously with shared dictionary (Anthropic, Feb 2025). Enables full circuit-level sparse analysis.

### Backlog
- [ ] Flash Attention — IO-aware exact attention (requires EXLA backend work)
- [ ] SPLA — Sparse + Linear Attention hybrid
- [ ] InfLLM-V2 — Block-partitioned KV cache selection
- [x] F5-TTS — Non-autoregressive flow-matching TTS (DiT backbone + ConvNeXt V2 text encoder + RoPE + conv PE)
- [ ] JanusFlow — AR text + rectified flow images
- [x] Show-o — AR + discrete diffusion (unified transformer with omni-attention mask)
- [x] Diffusion Policy — ConditionalUnet1D with FiLM conditioning, cosine noise schedule
- [ ] CausVid — Causal video DiT distillation
- [x] DeepONet — Branch-trunk operator learning (branch MLP + trunk MLP + dot-product combine)
- [x] MAGVIT-v2 — Lookup-free quantization for image/video tokens
- [ ] MIRAS — Google's Titans extension framework
- [x] MoR — Mixture of Recursions (weight-tied recursive transformer with per-token depth routing)
- [x] MoED — Mixture of Expert Depths (integrated MoDE with no-op expert for depth routing)
- [x] PointNet++ — Hierarchical point cloud processing (FPS + ball query + mini-PointNet SA layers)
- [x] Wav2Vec 2.0 — Self-supervised speech backbone (7-layer CNN encoder + conv PE + Transformer + product quantizer)
- [x] Janus Multimodal — Decoupled visual encoding (ViT encoder + MLP aligner + VQ gen head)
- [x] GPS — General Powerful Scalable graph transformer (GIN MPNN + global attention dual-branch with RWSE PE)
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

- [x] **Batch=1 edge cases** — Covered centrally by `registry_sweep_test.exs` which tests batch=1, 4, 16 for every architecture via `Edifice.build/2`. Catches broadcasting bugs across all 200+ architectures without per-file duplication.
- [x] **Edifice.build/2 integration tests** — Covered centrally by `registry_integrity_test.exs` (build-only) and `registry_sweep_test.exs` (build + forward pass) for every registered architecture. Per-file tests would duplicate centralized coverage.
- [x] **output_size/1 tests** — Covered by `output_size_sweep_test.exs` which discovers and validates all modules exporting `output_size/1`.
- [x] **ExCoveralls integration** — `excoveralls` dep added, `coveralls.json` configured (70% minimum), CI calls `mix coveralls.github`, coverage + CI badges in README.

### Documentation — Doctests (Priority: High) ✓

19 doctests across registry, shared blocks, and representative architectures.

- [x] **Doctests for Edifice registry** — `Edifice.build/2`, `list_architectures/0`, `list_families/0`, `module_for/1` (6 doctests in `test/edifice_test.exs`).
- [x] **Doctests for shared blocks** — `TransformerBlock.layer/2`, `FFN.layer/2`, `CrossAttention.layer/3`, `CausalMask.causal/1`, `CausalMask.window/2`, `SinusoidalPE.build_table/1`, `RoPE.precompute_freqs/3`, `SDPA.compute/5` (8 doctests in `test/edifice/blocks/doctest_test.exs`).
- [x] **Doctests for 5 representative architectures** — MLP (with full forward pass), LSTM, Mamba, GAN, ViT (5 doctests in `test/edifice/architecture_doctest_test.exs`).

### Documentation — Guides & Notebooks (Priority: Medium)

- [x] **Composition guide** — `guides/composing_architectures.md`. Covers TransformerBlock callbacks (attention_fn, cross_attention_fn, custom_ffn), ModelBuilder skeletons (sequence + vision), shared blocks table, and 3 composition recipes (custom attention, hybrid encoder-decoder, SSM+attention interleaving).
- [ ] **Livebook notebooks** — Create 3-5 `.livemd` notebooks: (1) "Build your first model" — walk through build/init/predict cycle, (2) "Architecture comparison" — benchmark 5 architectures on same task, (3) "Custom architecture from blocks" — compose a novel model from shared blocks, (4) "Whisper ASR demo" — end-to-end encoder-decoder usage, (5) "Training a small model" — connect to Axon training loop.
- [ ] **CODE_OF_CONDUCT.md** — Copy Contributor Covenant from contributor-covenant.org (content filters block generation). Manual task.

### Module Decomposition (Priority: Low-Medium)

- [x] **Split multi_head.ex** — Extracted pure tensor attention computations into `Edifice.Attention.Primitives` (~580 lines). Slimmed `multi_head.ex` to ~634 lines (Axon layer/model builders only). Deduplicated `causal_mask`/`window_mask` via `defdelegate` to `Edifice.Blocks.CausalMask`. All public APIs preserved via delegation.
- [ ] **Vision backbone interface** — Define a shared interface for vision modules (ViT, Swin, DeiT, ConvNeXt, etc.) so they can be used interchangeably as feature extractors. Consider an `Edifice.Vision.Backbone` behaviour with `build_encoder/1` callback.

### CI/CD Improvements (Priority: Medium)

- [ ] **Multi-version test matrix** — Test against Elixir 1.18 + 1.19 + 1.20 on OTP 27 + 28. The `mix.exs` claims `~> 1.18` compatibility; CI should verify it.
- [ ] **Benchmark regression CI** — Run Benchee on 5-10 key architectures in CI. Store baseline timings, fail if >10% regression. Candidate architectures: MLP, LSTM, Mamba, GQA, ViT, DETR (covers major families and input patterns).
- [ ] **Normalize git tag format** — Use consistent `v` prefix on all tags. Current mix: `0.1.1` and `v0.2.0`.

### ML-Specific Quality (Priority: Low)

- [ ] **Pretrained weight loading** — Support loading weights for 2-3 reference architectures (e.g., Whisper tiny, a small MLP). Even minimal weight support differentiates from "just architecture definitions." Investigate SafeTensors format for Elixir.
- [ ] **ONNX integration guide** — Document workflow: Edifice.build → Axon model → axon_onnx export → inference in other runtimes. Even if axon_onnx is a separate package, showing the integration path is valuable.
- [x] **Architecture visualization** — `mix edifice.viz mamba` prints layer structure as table (default), ASCII tree (`--format tree`), or Mermaid diagram (`--format mermaid`). Handles tuple-returning models via `--component`. See `Edifice.Display` module.
- [x] **Gradient smoke tests** — 176 passing tests across all 26 families (analytical gradients via `value_and_grad` + parameter sensitivity fallback). Covers sequence models, transformers, vision, detection, audio, robotics, RL, generative, graph, meta/PEFT, contrastive, interpretability, world model, multimodal, scientific, and memory architectures.
