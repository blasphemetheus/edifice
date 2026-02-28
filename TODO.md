# Edifice — Architecture TODO

## Current Status

234 registered architectures across 26 families, 20 shared blocks, 2500+ tests.

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

### 2026 Wave 4 Candidates

**Attention / Sequence:**
- [x] **FoX (Forgetting Transformer)** — Learnable forget gate on softmax attention (ICLR 2025). Per-head sigmoid gate modulates attention weights, enabling bounded memory in standard transformers. Adopted by Microsoft.
- [x] **Log-Linear Attention** — O(log T) space attention bridging linear and softmax (arXiv Jun 2025). Segment-based attention with hierarchical aggregation tree. Optimal memory-quality tradeoff.
- [x] **Native Hybrid Attention (NHA)** — Unified per-layer selection of linear vs full attention with shared KV projection (ICML 2025). Jointly trains attention type selection.

**Generative / Flow:**
- [x] **TarFlow** — Transformer-based normalizing flow (Apple, ICML 2025). Autoregressive flow using masked self-attention on image patches. Competitive with diffusion, exact likelihood.
- [x] **STARFlow** — Stacked TarFlow with multi-scale latent hierarchy. Extension of TarFlow with progressive resolution refinement.

**Graph / Scientific:**
- [x] **KA-GNN** — KAN-augmented GNN (KAN activation functions replace MLPs in GNN message passing). Improves expressivity on molecular property prediction.

**Meta / Reasoning:**
- [x] **Coconut (Continuous Chain of Thought)** — Meta ICLR 2025. Internal reasoning in continuous latent space rather than discrete tokens. Breadth-first reasoning without text generation overhead.
- [x] **Memory Layers** — Meta 2025. Sparse key-value lookup layers (1M+ keys) that replace dense FFN. Product-quantized nearest-neighbor retrieval provides massive memory at constant compute.

**Vision / Multimodal:**
- [x] **V-JEPA 2** — Meta 2025. Video world model with 3D-RoPE, sequence-level ViT encoder, and lightweight predictor with mask tokens.

**Dynamic Inference:**
- [x] **FreeTransformer** — Meta 2025. Decoder with latent variable per layer enabling speculative decoding without separate draft model. Samples latent → deterministic generation.

### 2026 Wave 5 Candidates

**Attention / Sequence:**
- [x] **LASER** — Attention with exp(V) transformation for larger gradient signals (ICML 2025, arXiv:2411.03493). Log-Weighted-Sum-Exp trick for stability. Low complexity.
- [x] **MoBA** — Mixture of Block Attention (Moonshot/Kimi, arXiv:2502.13189). MoE-style gating router selects relevant KV blocks per query. Production-proven. Medium complexity.
- [x] **Multi-Token Attention (MTA)** — Convolution over Q/K/head dims before attention (Meta, arXiv:2504.00927). Conditions attention on multiple tokens simultaneously. Medium complexity.
- [x] **DeltaProduct** — Multi-step DeltaNet via products of Householder transformations (NeurIPS 2025, arXiv:2502.10297). Extends existing DeltaNet/Gated DeltaNet. Medium complexity.
- [x] **Gated Slot Attention (GSA)** — Two-layer GLA linked via softmax with adaptive forgetting (NeurIPS 2024, arXiv:2409.07146). Fixed-size memory slots, linear time. Medium complexity.
- [x] **Residual Linear Attention** — Auxiliary recurrent state correcting base linear attention errors (arXiv:2509.25223). Also Residual DeltaNet variant. Low-medium complexity.
- [x] **TNN (Toeplitz Neural Network)** — Position-based token mixing via learned Toeplitz convolutions with RPE (ICLR 2023, arXiv:2305.04749). O(n log n), excellent length extrapolation. Low complexity.

**SSM / Recurrent:**
- [x] **Longhorn** — SSM from online associative recall closed-form solution (ICLR 2025, arXiv:2407.14207). Drop-in Mamba replacement, no explicit forget gate. Low complexity.
- [x] **Samba** — Hybrid Mamba + SWA + MLP interleaving for unlimited context (ICLR 2025, arXiv:2406.07522). First hybrid beating Transformers on short+long context. Medium complexity.
- [x] **Huginn** — Depth-recurrent transformer with adaptive iteration for latent reasoning (NeurIPS 2025, arXiv:2502.05171). Weight-tied recurrent blocks. Medium complexity.
- [x] **Mixture-of-Mamba (MoM)** — Modality-aware SSM sparsity inside Mamba blocks (ICLR 2025, arXiv:2501.16295). Per-modality SSM routing. Medium complexity.

**Generative / Flow:**
- [x] **LLaDA** — Masked discrete diffusion LLM, vanilla transformer backbone (ICLR 2025, arXiv:2502.09992). First 8B diffusion LM. Extends MDLM family. Medium complexity.
- [x] **CaDDi** — Non-Markovian discrete diffusion with causal trajectory conditioning (NeurIPS 2025, arXiv:2502.09767). Unifies AR and diffusion. Medium complexity.
- [x] **DeepFlow** — Deeply supervised flow with VeRA velocity refinement blocks (ICCV 2025, arXiv:2503.14494). Extends SiT. 8x faster convergence. Medium complexity.
- [x] **Meissonic** — Masked generative transformer for images with VQ + iterative mask-predict (ICLR 2025, arXiv:2410.08261). SDXL-level quality, 1B params. Medium complexity.

**Meta / PEFT:**
- [x] **VeRA** — Shared frozen random matrices + per-layer scaling vectors (ICLR 2025, arXiv:2310.11454). 10x fewer params than LoRA. Low complexity.
- [x] **Kron-LoRA** — Kronecker-product + low-rank LoRA hybrid (arXiv:2508.01961). 4x fewer params than rank-8 LoRA. Low complexity.
- [x] **Mixture of Transformers (MoT)** — Per-modality parameter decoupling with shared global attention (Meta/Stanford, TMLR 2025, arXiv:2411.04996). 55% FLOP savings. Medium complexity.

**Vision:**
- [x] **Vision KAN** — Hierarchical RBFKAN vision backbone without attention (arXiv:2601.21541). Patch-wise KAN + depthwise conv. Extends existing KAN module. Medium complexity.

**Graph / Scientific:**
- [ ] **Temporal Neural Operator (TNO)** — Temporal branch augmenting DeepONet for time-dependent PDEs (Nature Sci Reports 2025, arXiv:2504.20249). Extends existing DeepONet. Medium complexity.

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
- [x] SPLA — Block-sparse + residual linear attention with 2nd-order Taylor selection
- [x] InfLLM-V2 — Dense-sparse switchable attention with multi-level block selection
- [x] F5-TTS — Non-autoregressive flow-matching TTS (DiT backbone + ConvNeXt V2 text encoder + RoPE + conv PE)
- [x] JanusFlow — AR text + rectified flow images (velocity prediction network: ShallowUViT + ConvNeXt V2 + transformer backbone)
- [x] Show-o — AR + discrete diffusion (unified transformer with omni-attention mask)
- [x] Diffusion Policy — ConditionalUnet1D with FiLM conditioning, cosine noise schedule
- [x] CausVid — ~~Causal video DiT distillation~~ Training technique only, not architecture (skip)
- [x] DeepONet — Branch-trunk operator learning (branch MLP + trunk MLP + dot-product combine)
- [x] MAGVIT-v2 — Lookup-free quantization for image/video tokens
- [x] MIRAS — Moneta (p-norm), Yaad (Huber loss), Memora (KL-divergence) memory variants
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
- [ ] **EXLA GPU Custom Call Infrastructure** — Prototype for GPU-native custom calls bypassing NIF copy overhead. Branch `gpu-custom-calls` in `/home/dori/git/melee/nx` (commit `5a7d8cc8`). Adds nvcc compilation to EXLA Makefile, `gpu_add.cu` prototype kernel, `Value.gpu_add/3` stablehlo.custom_call binding, and `GPUCustomCall` module. Validated pattern for FlashAttention. Intent: clean up and PR to `elixir-nx/nx` separately.

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
- [x] **Vision backbone interface** — `Edifice.Vision.Backbone` behaviour with `build_backbone/1`, `feature_size/1`, and `input_shape/1` callbacks. Adopted by 12 modules: ViT, DeiT, Swin, ConvNeXt, MLPMixer, PoolFormer, FocalNet, MetaFormer, EfficientViT, MambaVision, DINOv2, DINOv3. Dispatch helper `Backbone.build_backbone(Module, opts)`.

### CI/CD Improvements (Priority: Medium)

- [ ] **Multi-version test matrix** — Test against Elixir 1.18 + 1.19 + 1.20 on OTP 27 + 28. The `mix.exs` claims `~> 1.18` compatibility; CI should verify it.
- [ ] **Benchmark regression CI** — Run Benchee on 5-10 key architectures in CI. Store baseline timings, fail if >10% regression. Candidate architectures: MLP, LSTM, Mamba, GQA, ViT, DETR (covers major families and input patterns).
- [x] **Normalize git tag format** — All tags now use `v` prefix (`v0.1.1`, `v0.2.0`).

### Pretrained Weight Loading (Priority: Medium)

Load HuggingFace SafeTensors checkpoints into Edifice models. The `safetensors` Elixir package
(v0.1.3, `elixir-nx/safetensors`) handles serialization; the work is key mapping and tensor
transformation between PyTorch and Axon conventions.

**Phase 1 — Core infrastructure (`lib/edifice/pretrained/`):**

- [ ] **Add `{:safetensors, "~> 0.1.3"}` dependency** — Add to `mix.exs` as optional dep. Zero runtime cost for users who don't load weights.
- [ ] **Key mapping behaviour** — `Edifice.Pretrained.KeyMap` behaviour with `@callback map_key(pytorch_key :: String.t()) :: String.t() | :skip`. Each architecture implements this to translate PyTorch param names (e.g. `model.layers.0.self_attn.q_proj.weight`) to Axon layer names. Include `@callback tensor_transforms() :: [{String.t(), (Nx.Tensor.t() -> Nx.Tensor.t())}]` for per-key reshape/transpose rules.
- [ ] **Tensor transformation layer** — `Edifice.Pretrained.Transform` module. Handles: (1) Linear weight transpose (`[out, in]` PyTorch -> `[in, out]` Axon), (2) Conv weight permutation (PyTorch OIHW -> Axon format), (3) Dtype casting (fp16/bf16 -> f32 or preserve), (4) Nested key grouping (flat `"layer.0.weight"` -> `%{"layer" => %{"0" => %{"weight" => tensor}}}`).
- [ ] **Loader API** — `Edifice.Pretrained.load(module, path_or_url, opts)` that: (1) Reads `.safetensors` file via `Safetensors.read!/2` with lazy loading, (2) Applies module's key mapping, (3) Applies tensor transforms, (4) Returns `Axon.ModelState.t()`. Options: `:backend` (default BinaryBackend), `:dtype` (cast all to given type), `:strict` (error on unmapped keys vs warn).

**Phase 2 — Reference architecture key maps (2-3 models):**

- [ ] **ViT key map** — `Edifice.Vision.ViT.KeyMap`. Map from HuggingFace `vit-base-patch16-224` checkpoint. Covers patch embedding, positional embedding, transformer encoder layers, layernorm, classifier head. Test with `google/vit-base-patch16-224` (~86M params, 330MB safetensors).
- [ ] **Whisper key map** — `Edifice.Audio.Whisper.KeyMap`. Map from HuggingFace `whisper-tiny` checkpoint. Covers: mel encoder conv layers, encoder transformer, decoder transformer, cross-attention projections. Test with `openai/whisper-tiny` (~39M params, 151MB safetensors). Exercises encoder-decoder architecture.
- [ ] **ConvNeXt key map** (stretch) — `Edifice.Vision.ConvNeXt.KeyMap`. Map from `facebook/convnext-tiny-224`. Exercises a pure-CNN architecture with depthwise convs, LayerNorm, different param naming patterns from transformers.

**Phase 3 — HuggingFace Hub integration (optional):**

- [ ] **Hub download helper** — `Edifice.Pretrained.Hub.download(repo_id, opts)`. Downloads `.safetensors` from HuggingFace Hub via HTTP. Handles: multi-file sharded checkpoints (`model-00001-of-00003.safetensors`), `model.safetensors.index.json` shard index, local caching in `~/.cache/edifice/`, progress reporting via Logger. Optional dep on `req` for HTTP.
- [ ] **Model card metadata** — Parse `config.json` from HuggingFace repos to auto-detect architecture type and build opts (hidden_size, num_heads, etc.) so users can do `Edifice.Pretrained.from_hub("google/vit-base-patch16-224")` without specifying opts.

**Phase 4 — Validation & docs:**

- [ ] **Round-trip tests** — For each reference model: (1) Build Edifice model, (2) Init random weights, (3) `Safetensors.write!` -> `Pretrained.load` round-trip, (4) Assert all params match. Also test with actual HuggingFace checkpoints (tagged `:external`, skipped in CI).
- [ ] **Numerical validation** — For ViT and Whisper: compare Edifice forward pass output against known PyTorch outputs on reference inputs. Store expected outputs as fixtures. Tolerance: 1e-4 for f32.
- [ ] **Guide** — `guides/loading_pretrained_weights.md`. Walk through: installing dep, downloading checkpoint, loading into model, running inference. Include troubleshooting for common key mismatches.

### ML-Specific Quality (Priority: Low)
- [ ] **ONNX integration guide** — Document workflow: Edifice.build → Axon model → axon_onnx export → inference in other runtimes. Even if axon_onnx is a separate package, showing the integration path is valuable.
- [x] **Architecture visualization** — `mix edifice.viz mamba` prints layer structure as table (default), ASCII tree (`--format tree`), or Mermaid diagram (`--format mermaid`). Handles tuple-returning models via `--component`. See `Edifice.Display` module.
- [x] **Gradient smoke tests** — 176 passing tests across all 26 families (analytical gradients via `value_and_grad` + parameter sensitivity fallback). Covers sequence models, transformers, vision, detection, audio, robotics, RL, generative, graph, meta/PEFT, contrastive, interpretability, world model, multimodal, scientific, and memory architectures.
