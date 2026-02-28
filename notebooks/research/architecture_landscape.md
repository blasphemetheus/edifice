# The ML Architecture Landscape

> Where Edifice stands, what's happening at the frontier, and what to build next.

Edifice currently implements **234 registered architectures** spanning 26 families — from transformers and state space models to graph networks, generative models, multimodal fusion, audio synthesis, robotics, RL integration, scientific ML, interpretability, and neuromorphic computing. This document maps the broader landscape to identify gaps, emerging paradigms, and high-impact build targets.

---

## Table of Contents

1. [The Philosophical Divide](#1-the-philosophical-divide)
2. [Attention & Efficiency Innovations](#2-attention--efficiency-innovations)
3. [Mixture of Experts & Routing](#3-mixture-of-experts--routing)
4. [Multimodal Architectures](#4-multimodal-architectures)
5. [Interpretability & Safety](#5-interpretability--safety)
6. [Deployment & Serving](#6-deployment--serving)
7. [Agentic & Multi-Agent](#7-agentic--multi-agent)
8. [Frontier Model Architectures](#8-frontier-model-architectures)
9. [What Edifice Should Build Next](#9-what-edifice-should-build-next)
10. [2026 Update #2: Frontier Architecture Survey](#2026-update-2-frontier-architecture-survey-feb-2026)

---

## 1. The Philosophical Divide

The ML research community is split along a fundamental question: **what is the right way to build intelligence?** Two camps have crystallized, and understanding them shapes everything downstream.

### Sutton's Bitter Lesson

Rich Sutton's 2019 essay ["The Bitter Lesson"](http://www.incompleteideas.net/IncsightBitter/TheBitterLesson.html) argues that the history of AI teaches one thing: **general methods that leverage computation scale better than clever, human-designed features.** Every time researchers hand-engineer solutions (chess evaluation functions, speech features, vision pipelines), those approaches eventually get crushed by simpler methods + more compute.

The LLM revolution is the ultimate vindication of this thesis. GPT, Claude, and their kin are "mimicry engines" — they predict the next token, and that single objective, applied at massive scale, produces emergent capabilities that no amount of hand-engineering could match.

**But there's a counterargument:** These models don't *understand* anything. They learn statistical co-occurrence patterns over tokens. They can't form new hypotheses, run experiments, or adapt to genuinely novel situations. They hallucinate because they're pattern-matching, not reasoning. The Bitter Lesson says "scale more" — but some researchers argue we need a fundamentally different approach.

The case for **continual reinforcement learning**: Instead of pre-training on static datasets and then fine-tuning, build agents that learn continuously from interaction with an environment. This is how biological intelligence works — trial and error, reward signals, adaptation. DeepMind's work on game-playing agents (AlphaGo, AlphaZero, MuZero) showed that RL can discover strategies no human ever conceived. The question is whether this approach can generalize beyond games.

### LeCun's JEPA Vision

Yann LeCun (Meta's chief AI scientist) has been the most vocal critic of autoregressive language models. His alternative: **Joint Embedding Predictive Architecture (JEPA)**.

The core idea: instead of predicting raw pixels or tokens (which forces the model to model every irrelevant detail), predict in an **abstract representation space**. A JEPA model has:

1. A **context encoder** that maps observations to representations
2. A **predictor** that predicts the representation of missing/future parts
3. A **target encoder** that provides the prediction targets (updated via EMA)

This is **non-generative** — the model never reconstructs raw data. It learns what matters for prediction and ignores noise. LeCun argues this is closer to how humans build mental models of the world.

**V-JEPA** (Video JEPA) showed this works for video: predict masked video patches in representation space, and the model learns strong visual features without ever generating pixels. **VL-JEPA** extends this to vision-language.

### The Tension

| Autoregressive (GPT/Claude) | Abstract Prediction (JEPA) |
|------------------------------|---------------------------|
| Predict next token in sequence | Predict representations of missing parts |
| Must model every detail | Can ignore irrelevant details |
| Scale-first philosophy | Architecture-first philosophy |
| Generative (can produce outputs) | Non-generative (representations only) |
| Proven at massive scale | Promising but less validated |
| Hallucination is inherent | Could be more grounded |

### What This Means for Edifice

Edifice already has:
- `:decoder_only` — the autoregressive camp
- `:jepa` — the JEPA camp (context encoder + predictor)
- `:mae` — Masked Autoencoder (reconstruction-based, between the two camps)

**Gaps to consider:**
- **RL environments** — Edifice has no RL integration. Building even a simple environment loop (observations → model → actions → rewards) would open a whole new paradigm.
- **JEPA v2 / V-JEPA** — The current JEPA module is image-based. A video/temporal version would be a significant addition.
- **World models** — The intersection of JEPA + RL: learn a world model in abstract space, then plan in that space. This is LeCun's full vision.

---

## 2. Attention & Efficiency Innovations

The attention mechanism is the beating heart of transformers, and it's also the bottleneck. Standard attention scales as O(n²) with sequence length, which means a 128K-context model needs 16,000x more compute for attention than a 1K-context model. The field has exploded with approaches to fix this.

### DeepSeek Multi-head Latent Attention (MLA)

**What it is:** Instead of caching full key-value pairs (which dominate memory at long contexts), MLA compresses KV into a low-rank **latent space**. During inference, only the compressed latents are cached, then projected back to full KV on the fly.

**Why it matters:** At 128K context, standard KV cache for a 7B model can eat 16+ GB of memory. MLA cuts this dramatically — DeepSeek-V2 reported 93.3% KV cache reduction with negligible quality loss.

**Edifice status:** ✅ Implemented as `:mla` (`Edifice.Attention.MLA`).

### Lightning Attention (MiniMax)

**What it is:** A hybrid attention mechanism from MiniMax (the company behind MiniMax-01, 456B MoE). It combines:
1. **Linear attention** for intra-block computation (O(n) within blocks)
2. **Softmax attention** for inter-block interactions (standard quality)
3. **I/O-aware tiling** — structures computation to minimize GPU memory transfers, which are the real bottleneck on modern hardware

**Why it matters:** Standard "efficient attention" methods (Performer, linear transformers) sacrifice quality. Lightning Attention maintains softmax-quality where it matters while using linear attention where exact attention isn't critical. MiniMax-01 used it to achieve 1M+ token context.

**Paper:** "Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths" (MiniMax, 2024)

**Edifice status:** ✅ Implemented as `:lightning_attention` (`Edifice.Attention.LightningAttention`).

### iRoPE (Meta, Llama 4)

**What it is:** **Interleaved Rotary Position Embeddings**. Instead of applying RoPE to every attention layer, Meta's Llama 4 applies it to alternating layers. Layers without RoPE use NoPE (no positional encoding) — they attend based purely on content similarity.

**Why it matters:** This decouples position-dependent attention (where RoPE helps) from content-dependent attention (where position can be a distraction). The result: better generalization to sequence lengths far beyond training. Llama 4 was trained on shorter contexts but reportedly supports 10M+ tokens at inference.

**Edifice status:** ✅ `:decoder_only` supports `interleave_rope: true` option (alternating RoPE/NoPE layers).

### YARN + Dual Chunk Attention (Qwen)

**What it is:** Two complementary techniques for extending context at inference time:

- **YARN (Yet Another RoPE Extension):** Modifies RoPE frequencies to handle longer sequences than training. More principled than simple NTK-aware scaling — YARN identifies which frequency bands need modification and applies targeted adjustments.
- **Dual Chunk Attention:** Splits long sequences into chunks, computes intra-chunk attention normally, then uses a separate mechanism for inter-chunk attention. Reduces peak memory while maintaining long-range connections.

**Why it matters:** These let a model trained on 4K context work at 32K+ context without retraining. Qwen2.5 used this combination to achieve 128K context.

**Edifice status:** ✅ YARN implemented as `:yarn` (`Edifice.Attention.YARN`). Dual Chunk Attention implemented as `:dual_chunk_attention` (`Edifice.Attention.DualChunk`).

### What Edifice Has vs. What's Missing

| Technique | Edifice? | Module |
|-----------|----------|--------|
| Multi-Head Attention | ✅ | `:attention` |
| Grouped Query Attention (GQA) | ✅ | `:gqa` (used in `:decoder_only`) |
| Multi-head Latent Attention (MLA) | ✅ | `:mla` |
| Differential Transformer | ✅ | `:diff_transformer` |
| Linear Transformer | ✅ | `:linear_transformer` |
| Performer | ✅ | `:performer` |
| Nystromformer | ✅ | `:nystromformer` |
| FNet (Fourier) | ✅ | `:fnet` |
| Ring Attention | ✅ | `:ring_attention` |
| Infini-Attention | ✅ | `:infini_attention` |
| RoPE | ✅ | Built into DecoderOnly |
| Lightning Attention (hybrid) | ✅ | `:lightning_attention` |
| Gated Attention | ✅ | `:gated_attention` |
| Scalable-Softmax (SSMax) | ✅ | `:ssmax` |
| Softpick | ✅ | `:softpick` |
| iRoPE (interleaved) | ✅ | `:decoder_only` (interleave_rope: true) |
| YARN context extension | ✅ | `:yarn` |
| Dual Chunk Attention | ✅ | `:dual_chunk_attention` |
| NSA (Native Sparse Attention) | ✅ | `:nsa` |
| TMRoPE | ✅ | `:tmrope` |
| RNoPE-SWA | ✅ | `:rnope_swa` |
| FlashAttention (kernel-level) | ❌ | Requires custom CUDA, beyond Nx |

---

## 3. Mixture of Experts & Routing

MoE has become the dominant paradigm for scaling to enormous parameter counts efficiently. The idea: instead of activating all parameters for every input, **route each token to a subset of specialized "expert" networks**. A 671B-parameter MoE model might only activate ~37B parameters per token.

### DeepSeek Auxiliary-Loss-Free Load Balancing

**The problem:** MoE routing tends to collapse — a few experts get all the traffic while others go unused. Traditional fix: add an auxiliary loss that penalizes imbalanced routing. But this auxiliary loss conflicts with the main training objective, degrading quality.

**DeepSeek's solution:** Instead of a loss, they use a **bias term** in the routing mechanism. Experts that receive too few tokens get a small positive bias (making them more likely to be selected), and vice versa. The bias is adjusted dynamically based on a running average of expert utilization. No auxiliary loss needed.

**Why it matters:** Cleaner training, better quality. DeepSeek-V3 used this to train a 671B MoE that matches or exceeds models trained with auxiliary losses.

**Paper:** "DeepSeek-V3 Technical Report" (DeepSeek, December 2024)

**Edifice status:** Partially addressed. `:moe_v2` has expert-choice routing (where experts choose tokens, not vice versa) and shared experts, but doesn't implement the specific bias-based balancing.

### Edifice's MoE Family

| Variant | Module | Key Feature |
|---------|--------|-------------|
| Standard MoE | `:moe` | Top-k routing with auxiliary loss |
| Switch MoE | `:switch_moe` | Top-1 routing (Google Switch Transformer) |
| Soft MoE | `:soft_moe` | Soft token merging (no discrete routing) |
| MoE v2 | `:moe_v2` | Expert-choice routing + shared experts |

**What's next:**
- **Auxiliary-loss-free balancing** — Implement DeepSeek's bias-based approach in `:moe_v2`
- **Mixture of Tokenizers** — Different experts process different tokenizations of the input
- **Speculative decoding** — Draft with a small model, verify with a large model. Not MoE per se, but a related efficiency technique

---

## 4. Multimodal Architectures

The frontier has moved decisively toward **multimodal** models — systems that process text, images, audio, and video in a unified architecture. This is arguably the biggest gap in Edifice today.

### Qwen2.5-VL

**What it is:** Alibaba's vision-language model with several innovations:
- **Dynamic resolution:** Instead of resizing all images to a fixed size, Qwen2.5-VL processes images at their native resolution using a variable number of visual tokens
- **Window attention for vision:** Instead of full attention over all visual tokens (prohibitively expensive at high resolution), uses sliding window attention within the vision encoder
- **Absolute position embeddings for vision, RoPE for text** — Different positional encoding schemes for different modalities, unified in the cross-attention layers

**Why it matters:** Most VLMs force images into a fixed grid (e.g., 224×224). This destroys detail in high-resolution images and wastes compute on low-resolution ones. Dynamic resolution is the natural solution.

### Qwen2.5-Omni "Thinker-Talker"

**What it is:** A truly omni-modal model with a novel dual-track architecture:
- **Thinker:** Processes all input modalities (text, image, audio, video) through a shared transformer, producing a text-based "chain of thought"
- **Talker:** Takes the Thinker's output and generates speech in real-time, streaming audio tokens while the Thinker is still producing text

**Why it matters:** This enables genuinely real-time voice conversation — the model starts speaking before it finishes "thinking." Previous voice models either had to wait for full text generation or sacrificed quality.

### V-JEPA / VL-JEPA

**What it is:** Meta's video understanding models based on LeCun's JEPA framework:
- **V-JEPA:** Masks patches of video frames, predicts their representations (not pixels). Learns strong temporal features without reconstruction.
- **VL-JEPA:** Extends to vision-language — predict masked visual representations conditioned on text.

**Why it matters:** These models learn video understanding without generating pixels, which means they focus on semantics rather than appearance. They outperform reconstruction-based methods (like VideoMAE) on downstream tasks while being more efficient to train.

### TMRoPE (Time-aligned Multimodal RoPE)

**What it is:** A positional encoding scheme from Qwen2.5-Omni that aligns different modalities in a shared temporal space. Each modality (text, image, audio) gets position embeddings that reflect its real-world timing, so the model can reason about temporal relationships across modalities.

**Why it matters:** Without temporal alignment, a multimodal model can't answer questions like "what was happening when the person said X?" TMRoPE provides this alignment without requiring explicit timestamps.

### What Edifice Would Need for Multimodal

Edifice is currently **text/sequence-first**. To support multimodal:

1. **Image encoders** — Patch embedding, ViT backbone (Edifice has `:vit`, but it's standalone, not integrated as an encoder for a language model)
2. **Cross-modal fusion** — Attention layers that combine visual and textual representations
3. **Dynamic resolution handling** — Variable-length visual token sequences
4. **Audio processing** — Mel spectrogram encoding, streaming audio tokens
5. **Modality-specific positional encoding** — TMRoPE or similar

This is a large undertaking, but the building blocks exist: `:vit` for vision, `:decoder_only` for language, `:conformer` for audio. The gap is the **glue** — the cross-modal attention and fusion layers.

---

## 5. Interpretability & Safety

Understanding *why* a model makes its predictions is increasingly important — for scientific understanding, safety, and regulation. The field has made real progress here, along with some sobering discoveries.

### Sparse Autoencoders (SAEs) — Promise and Limitations

**What they are:** Train a sparse autoencoder on a model's activations. Each neuron in the SAE's hidden layer corresponds to an interpretable "feature" — a concept the model has learned (e.g., "present tense verbs", "code syntax", "sarcasm").

**The promise:** Anthropic's 2024 work on Claude 3 Sonnet extracted millions of interpretable features, including concepts like "Golden Gate Bridge" that could be artificially amplified.

**The reality check (2025):** Subsequent research revealed significant limitations:
- **Only ~30% feature overlap** between different SAE training runs on the same model — suggesting SAEs capture arbitrary decompositions, not the model's "true" features
- **Linear probes outperform SAEs** for many classification tasks — a simpler method works better
- **Feature splitting** — a single concept often gets spread across multiple SAE features depending on context

SAEs are still useful for exploration and hypothesis generation, but they're not the mechanistic interpretability silver bullet they were initially hoped to be.

### Transcoders > SAEs

**What they are:** Instead of compressing activations through a bottleneck (autoencoder), transcoders learn to map from one layer's activations directly to the next layer's, with a sparsity constraint. Each transcoder feature represents a *computation* the model performs, not just a *representation* it stores.

**Why they're better:** Transcoders capture the model's actual processing steps — "when feature A is active, the model is deciding to output B" — rather than just "this pattern exists in the activations." This is closer to true mechanistic understanding.

**Paper:** "Transcoders Find Interpretable LLM Feature Circuits" (Anthropic, 2024)

### Anthropic's Circuit Tracing (March 2025)

**What it is:** A new methodology for understanding model behavior end-to-end:
1. Identify a behavior (e.g., "the model answers in French when asked in French")
2. Use **attribution** to trace which components contribute to the output
3. Build an **attribution graph** — a subnetwork of the full model that explains the behavior
4. Verify by **ablating** components and confirming the behavior changes

**Key finding:** Many model behaviors can be explained by surprisingly compact circuits — just a few dozen attention heads and MLP neurons out of billions of parameters.

**Paper:** "On the Biology of a Large Language Model" (Anthropic, March 2025)

### Causal Abstraction Framework

**What it is:** A unified theoretical framework (Geiger et al., JMLR 2025) that subsumes most interpretability techniques:
- **Activation patching** → special case of causal intervention
- **Circuit discovery** → finding a causal model that abstracts the neural network
- **SAE features** → candidate variables in the causal model
- **Steering vectors** → interventions in the causal model

**Why it matters:** The field has been fragmented — dozens of techniques with unclear relationships. Causal Abstraction provides a common language and lets researchers compare methods rigorously.

### What Edifice Could Build

Edifice is an architecture library, not an interpretability toolkit. But some additions would bridge the gap:

- **Activation extraction hooks** — Let users tap into any layer's activations during forward passes. Essential for probing, SAEs, and circuit analysis.
- **Sparse autoencoder module** — A trainable SAE that attaches to any Edifice model's intermediate layers.
- **Transcoder module** — Same idea, but mapping between layers instead of compressing within a layer.
- **Probing layers** — Simple linear classifiers that attach to hidden states. The simplest interpretability tool and often the most effective.

---

## 6. Deployment & Serving

Building an architecture is only half the story. Getting it running efficiently on real hardware is the other half.

### The Ollama Model

[Ollama](https://ollama.ai) has become the de facto standard for running LLMs locally. Its approach:

1. Models are stored in **GGUF format** — a quantized format designed for efficient CPU/GPU inference
2. A simple CLI: `ollama run llama3` downloads and runs the model
3. An HTTP API for integration with applications
4. Automatic hardware detection and optimization

**The GGUF format** (by Georgi Gerganov, creator of llama.cpp) is key: it stores quantized weights in a self-describing binary format with metadata about the model architecture. Supported quantization levels range from Q2 (2-bit, very aggressive) to Q8 (8-bit, near-lossless) and FP16.

### Edifice's Quantization Toolkit

Edifice already has quantization infrastructure:

| Method | Description |
|--------|-------------|
| **RTN** | Round-to-nearest — simplest, applies per-tensor |
| **GPTQ** | Post-training quantization using Hessian information |
| **AWQ** | Activation-aware weight quantization — protects salient channels |

These produce quantized Axon models that run in the Nx/EXLA ecosystem. The missing link is **export** — converting these quantized models to formats that other runtimes can consume.

### Could Edifice Models Be Served via Ollama?

**The gap:**

1. **GGUF export** — Edifice would need to serialize model weights + architecture metadata in GGUF format. This is a significant engineering effort but well-defined.
2. **Architecture support** — GGUF/llama.cpp support a specific set of architectures (LLaMA-family, Mamba, etc.). Edifice's more exotic architectures (Hyena, RetNet, Liquid) would need custom llama.cpp backends.
3. **Tokenizer** — Edifice doesn't include tokenizers. GGUF models bundle their tokenizer. This could be solved by integrating with existing tokenizer libraries (Tokenizers in Rust/Python, or Elixir wrappers).

**Realistic path:** Start with GGUF export for `:decoder_only` models (which map cleanly to LLaMA architecture), then expand.

---

## 7. Agentic & Multi-Agent

The "agent" paradigm — models that take actions, use tools, and operate autonomously — has become the dominant application pattern for LLMs.

### Kimi K2.5 Agent Swarm

**What it is:** Moonshot AI's approach to complex agentic tasks:
- A **coordinator agent** decomposes a task into subtasks
- **Specialist agents** handle each subtask in parallel
- Agents communicate through a shared message bus
- Results are aggregated by the coordinator

**Why it matters:** Single-agent systems hit a ceiling on complex tasks — they lose context, make cascading errors, and can't parallelize. Multi-agent decomposition mirrors how human teams work: divide, conquer, integrate.

### Tool Calling Architectures

Modern LLMs need to call external tools (search, code execution, APIs). The architectures for this:

1. **Function calling** (OpenAI, Anthropic, Ollama) — Model outputs structured JSON specifying which function to call and with what arguments
2. **ReAct** (Reasoning + Acting) — Model alternates between reasoning steps and action steps in a chain
3. **Code-as-action** — Model writes executable code instead of structured function calls

### Edifice's MixtureOfAgents

Edifice has `:mixture_of_agents` — a multi-model ensemble where different "agent" models process the input independently, and their outputs are aggregated. This is conceptually related to agent swarms but operates at the inference level (parallel model calls) rather than the task level (parallel subtask execution).

**What's different about real agent systems:** They operate over multiple turns, maintain state, use tools, and coordinate. Edifice's MixtureOfAgents is a single-turn ensemble. Bridging this gap would require:

- **State management** — Persistent memory across agent turns
- **Tool integration** — Hooks for calling external functions
- **Coordination protocols** — Message passing between agents
- **Planning modules** — Task decomposition and scheduling

This is more of an application-layer concern than an architecture-layer one, but Edifice could provide building blocks (the RLHFHead module is already a step in this direction).

---

## 8. Frontier Model Architectures

Let's look at the specific architectures powering today's strongest models.

### DeepSeek-V3 (December 2024)

**Parameters:** 671B total, 37B active per token (MoE)

**Key innovations:**
- **Multi-head Latent Attention (MLA)** — KV cache compression (see Section 2)
- **Auxiliary-loss-free load balancing** — Dynamic bias-based expert routing (see Section 3)
- **Multi-token prediction (MTP)** — Instead of predicting just the next token, predict the next N tokens simultaneously. During training, this provides richer gradients. During inference, the extra predictions can be used for speculative decoding.
- **FP8 mixed-precision training** — Trained large portions of the model in 8-bit floating point, saving memory and compute

**Training cost:** $5.6M reported — dramatically lower than comparable models, largely due to MoE efficiency and FP8 training.

**What Edifice has:** MLA (✅), MoE v2 (✅), Multi-token prediction (✅), FP8 training (❌ — depends on Nx/EXLA support)

### Kimi K2 (June 2025)

**Parameters:** 1T total (MoE), largest openly-described model

**Key innovations:**
- **MuonClip optimizer** — A variant of the Muon optimizer (momentum + orthogonalization) with gradient clipping. Claims better training stability than Adam at scale.
- **Massive expert count** — Hundreds of experts with fine-grained routing
- **Agent-native training** — Trained from the start with tool-use and multi-turn capabilities, not fine-tuned post-hoc

**What Edifice has:** No optimizer implementations (optimizers are handled by Polaris in the Nx ecosystem). Could add custom optimizer wrappers.

### Qwen3 (2025)

**Key features of the Qwen model line:**
- Aggressive **context extension** via YARN + Dual Chunk Attention
- Strong **multilingual** capabilities (especially CJK languages)
- **Qwen-Agent** framework for tool use
- **Qwen-Audio** and **Qwen-VL** multimodal variants
- Progressive scaling: Qwen 0.5B → 1.8B → 7B → 14B → 72B → 110B

### Common Patterns Across Frontier Models

| Pattern | DeepSeek-V3 | Kimi K2 | Qwen3 | Llama 4 |
|---------|:-----------:|:-------:|:-----:|:-------:|
| MoE | ✅ | ✅ | ✅ | ✅ |
| Long context (128K+) | ✅ | ✅ | ✅ | ✅ |
| GQA or MLA | MLA | GQA | GQA | GQA |
| RoPE variant | Standard | Standard | YARN | iRoPE |
| Multi-token prediction | ✅ | ❌ | ❌ | ❌ |
| SwiGLU FFN | ✅ | ✅ | ✅ | ✅ |
| RMSNorm (pre-norm) | ✅ | ✅ | ✅ | ✅ |
| FP8 / quantized training | ✅ | ✅ | Partial | ✅ |

**Takeaway:** The frontier has converged on a recipe: **MoE + GQA/MLA + RoPE + SwiGLU + RMSNorm + long context**. The differentiation is in the details — routing strategy, positional encoding variant, context extension method.

---

## 9. What Edifice Should Build Next

Based on this landscape survey, here's a prioritized build list. Tiers reflect impact (how much capability it adds to Edifice), feasibility (how hard it is to implement), and strategic importance (whether it fills a gap that matters).

### Tier 1: High Impact

These are the most impactful additions — each one fills a significant gap or enables a new class of experiments.

#### Lightning Attention (Hybrid Linear/Softmax)

| | |
|---|---|
| **What** | Hybrid attention: linear attention within blocks, softmax between blocks. I/O-aware tiling for GPU efficiency. |
| **Paper** | "Lightning Attention-2" (MiniMax, 2024) |
| **Difficulty** | Medium — the attention math is straightforward, the I/O tiling is the hard part (may need custom Nx operations) |
| **Builds on** | `:linear_transformer`, `:performer`, `:attention` |
| **Why Tier 1** | Bridges the quality gap of linear attention while keeping subquadratic scaling. The most practical efficient attention approach. |

#### Transcoders / SAE Module

| | |
|---|---|
| **What** | Trainable sparse autoencoder or transcoder that attaches to any model's intermediate layers. Extracts interpretable features. |
| **Paper** | "Transcoders Find Interpretable LLM Feature Circuits" (Anthropic, 2024) |
| **Difficulty** | Low-Medium — the architecture is simple (encoder-decoder with sparsity), the integration requires activation hooks |
| **Builds on** | `:vae` (encoder-decoder pattern), any model as the analysis target |
| **Why Tier 1** | Interpretability is a killer feature. Being able to say "build model + attach SAE + extract features" in three lines of Elixir would be unique. |

#### iRoPE (Interleaved Rotary Position Embeddings)

| | |
|---|---|
| **What** | Apply RoPE to alternating transformer layers, use NoPE (content-only attention) for the others. |
| **Paper** | Meta Llama 4 technical report |
| **Difficulty** | Low — modify `:decoder_only` to accept an `interleave_rope: true` option |
| **Builds on** | `:decoder_only` (RoPE is already implemented) |
| **Why Tier 1** | Simple change, big impact on long-context generalization. Directly applicable to the SLM notebook. |

#### JEPA v2 / V-JEPA (Temporal JEPA)

| | |
|---|---|
| **What** | Extend JEPA from single images to video/temporal sequences. Mask-and-predict in time. |
| **Paper** | "Revisiting Feature Prediction for Learning Visual Representations from Video" (Meta, 2024) |
| **Difficulty** | Medium — requires temporal masking, 3D patch embeddings, and video-aware data loading |
| **Builds on** | `:jepa`, `:vit` (patch embedding), `:mae` (masking strategy) |
| **Why Tier 1** | Represents the cutting edge of self-supervised learning. Unique in the Elixir ecosystem. |

#### Auxiliary-Loss-Free MoE Routing

| | |
|---|---|
| **What** | Replace auxiliary load-balancing loss with dynamic bias-based routing in `:moe_v2`. |
| **Paper** | DeepSeek-V3 technical report (December 2024) |
| **Difficulty** | Low — add a bias tensor to the router, update it based on expert utilization running average |
| **Builds on** | `:moe_v2` (expert-choice routing, shared experts) |
| **Why Tier 1** | Simple to implement, directly improves MoE quality. Removes a known training pain point. |

### Tier 2: Medium Impact

Valuable additions that fill specific gaps or enable new use cases.

#### Multi-Token Prediction Head

| | |
|---|---|
| **What** | Instead of one output head predicting the next token, N independent heads predict the next N tokens. Richer training signal, enables speculative decoding. |
| **Paper** | "Better & Faster Large Language Models via Multi-token Prediction" (Meta, 2024) |
| **Difficulty** | Low — add N parallel dense layers after the transformer backbone |
| **Builds on** | `:decoder_only`, any sequence model |
| **Why Tier 2** | Simple implementation, proven training benefit. But the speculative decoding integration (where MTP shines at inference) requires more work. |

#### YARN Context Extension

| | |
|---|---|
| **What** | Modify RoPE frequency bands to extrapolate to longer contexts than training. |
| **Paper** | "YaRN: Efficient Context Window Extension" (2023) |
| **Difficulty** | Low — modify the RoPE computation with frequency band scaling factors |
| **Builds on** | RoPE in `:decoder_only` |
| **Why Tier 2** | Useful for inference-time context extension. But requires careful calibration and isn't needed at small Edifice-notebook scale. |

#### Speculative Decoding

| | |
|---|---|
| **What** | Use a small "draft" model to propose N tokens, then verify them in parallel with the large model. Accepted tokens skip the expensive large-model inference. |
| **Paper** | "Fast Inference from Transformers via Speculative Decoding" (Google, 2023) |
| **Difficulty** | Medium — requires coordinating two models, managing the acceptance/rejection logic |
| **Builds on** | Any pair of Edifice models (small + large) |
| **Why Tier 2** | 2-4x inference speedup with zero quality loss. But it's an inference optimization, not an architecture, and requires the deployment story to be further along. |

#### GGUF Export

| | |
|---|---|
| **What** | Serialize Edifice model weights and architecture metadata into GGUF format for use with llama.cpp / Ollama. |
| **Difficulty** | High — requires understanding the GGUF binary format, mapping Edifice architectures to GGUF architecture tags, handling quantization metadata |
| **Builds on** | Edifice's quantization toolkit (RTN/GPTQ/AWQ) |
| **Why Tier 2** | Would make Edifice models deployable in the wider ecosystem. But the architecture diversity of Edifice (Hyena, RetNet, Liquid) means most models wouldn't have llama.cpp backends. |

### Tier 3: Exploratory

Longer-term additions that open new frontiers but require significant investment.

#### RL Environment Integration

| | |
|---|---|
| **What** | Define a standard environment interface (observations, actions, rewards) and build a training loop that runs an Edifice model as a policy. |
| **Difficulty** | High — RL training is fundamentally different from supervised training. Needs PPO/SAC algorithms, environment wrappers, reward shaping. |
| **Builds on** | `:rlhf_head` (already produces value/policy outputs) |
| **Why Tier 3** | Opens the entire RL paradigm. But it's a large scope expansion and the Nx ecosystem's RL support is limited. |

#### Agent Swarm Patterns

| | |
|---|---|
| **What** | Multi-agent coordination — task decomposition, specialist routing, message passing between model instances. |
| **Difficulty** | High — more of an application framework than an architecture module |
| **Builds on** | `:mixture_of_agents`, `:moe_v2` |
| **Why Tier 3** | The agentic paradigm is dominant in applications, but Edifice is an architecture library. This might be better as a separate library that uses Edifice as a dependency. |

#### Multimodal Fusion Layers

| | |
|---|---|
| **What** | Cross-attention layers that combine visual tokens (from `:vit`) with text tokens (from `:decoder_only`). Dynamic resolution handling. |
| **Difficulty** | High — requires designing the fusion interface, handling variable-length visual sequences, training on paired data |
| **Builds on** | `:vit`, `:decoder_only`, `:perceiver` (which already does cross-modal attention) |
| **Why Tier 3** | The biggest gap in Edifice's coverage. But multimodal requires data pipelines and pre-trained vision encoders that Edifice doesn't have yet. |

#### TMRoPE (Time-aligned Multimodal RoPE)

| | |
|---|---|
| **What** | Positional encoding that aligns different modalities (text, image, audio) in a shared temporal space. |
| **Paper** | Qwen2.5-Omni technical report |
| **Difficulty** | Medium — the positional encoding math is tractable, but it requires the multimodal fusion layers to exist first |
| **Builds on** | RoPE in `:decoder_only`, requires multimodal fusion |
| **Why Tier 3** | Prerequisite: multimodal fusion layers. Without those, TMRoPE has nothing to attach to. |

---

### Summary Matrix

| Addition | Tier | Difficulty | Builds On |
|----------|:----:|:----------:|-----------|
| Lightning Attention | 1 | Medium | ✅ Done |
| Transcoders/SAE Module | 1 | Low-Med | ✅ Done |
| iRoPE | 1 | Low | ✅ Done |
| JEPA v2 / V-JEPA | 1 | Medium | ✅ Done |
| Aux-loss-free MoE | 1 | Low | ✅ Done (bias routing in moe_v2) |
| Multi-token prediction | 2 | Low | ✅ Done |
| YARN | 2 | Low | ✅ Done |
| Speculative decoding | 2 | Medium | ✅ Done |
| GGUF export | 2 | High | ✅ Done (lib/edifice/export/gguf.ex) |
| RL environments | 3 | High | ✅ Done (PPO, GAE, CartPole, GridWorld) |
| Agent swarm patterns | 3 | High | ❌ Not done |
| Multimodal fusion | 3 | High | ✅ Done (multimodal_mlp_fusion) |
| TMRoPE | 3 | Medium | ✅ Done |

---

## 10. 2026 Update: Revised Priorities

> Updated 2026-02-23 based on late-2025 / early-2026 research trends.

The landscape has shifted significantly since the original tiers were written. Several Tier 1/2 items have been completed, and new architectures have emerged as production-critical.

### Completed Since Original Tiers

| Original Item | Tier | Status |
|---------------|:----:|--------|
| Lightning Attention | 1 | Done — `:lightning_attention` |
| Transcoders/SAE Module | 1 | Done — `:sparse_autoencoder`, `:transcoder` |
| V-JEPA / Temporal JEPA | 1 | Done — `:temporal_jepa` |
| Multi-token prediction | 2 | Done — `:multi_token_prediction` |
| Speculative decoding | 2 | Done — `:speculative_decoding` |

### Still Open From Original Tiers

| Original Item | Tier | Status |
|---------------|:----:|--------|
| iRoPE | 1 | ✅ Done — `interleave_rope: true` in `:decoder_only` |
| Aux-loss-free MoE | 1 | ✅ Done — bias routing in `:moe_v2` |
| YARN context extension | 2 | ✅ Done — `:yarn` |
| GGUF export | 2 | ✅ Done — `lib/edifice/export/gguf.ex` |
| RL environments | 3 | ✅ Done — PPOTrainer, GAE, CartPole, GridWorld |
| Agent swarm patterns | 3 | **Not done** |
| Multimodal fusion | 3 | ✅ Done — `:multimodal_mlp_fusion` |
| TMRoPE | 3 | ✅ Done — `:tmrope` |

### New Tier 1: High Impact (2026)

These reflect the biggest shifts in production architectures since the original tiers.

#### Gated DeltaNet

| | |
|---|---|
| **What** | Linear attention with data-dependent gating on the delta rule. Replaces 75% of attention layers in production models. |
| **Adopted by** | Qwen3-Next (80B MoE), Kimi Linear (Moonshot AI) |
| **Difficulty** | Medium — extends existing `:delta_net` with gating mechanism |
| **Builds on** | `:delta_net`, `:gla` (conceptual ancestor) |
| **Why Tier 1** | The hottest linear attention mechanism of 2025-2026. Two major labs adopted it for production models. |

#### iRoPE (Interleaved RoPE) — ✅ DONE

Implemented as `interleave_rope: true` option in `:decoder_only`.

#### Aux-loss-free MoE Routing — ✅ DONE

Implemented in `:moe_v2` with bias-based load balancing.

#### RWKV-7 Update (Generalized Delta Rule)

| | |
|---|---|
| **What** | Update `:rwkv` to v7 "Goose" with generalized delta rule and vector-valued gating |
| **Paper** | "Eagle and Finch" / RWKV-7 (March 2025) |
| **Difficulty** | Medium — significant architecture changes to gating and state update |
| **Why Tier 1** | RWKV-7 comprehensively surpasses Transformers on efficiency. Most active open-source non-transformer LLM project. |

#### Configurable Hybrid Builder

| | |
|---|---|
| **What** | Builder that lets users set SSM/attention ratio (e.g., 90% Mamba + 10% attention) |
| **Pattern** | Nemotron-H (92% Mamba-2 + 8% attention), Qwen3-Next (75% DeltaNet + 25% attention) |
| **Difficulty** | Medium — generalize `:jamba`/`:zamba` pattern with configurable layer schedule |
| **Builds on** | `:jamba`, `:zamba`, any SSM + any attention module |
| **Why Tier 1** | The 90/10 hybrid is THE dominant production pattern of 2025-2026. |

### New Tier 2: Medium Impact (2026) — All Done

| Addition | Difficulty | Status |
|----------|-----------|--------|
| **TTT-E2E** | Medium | ✅ Done — `:ttt_e2e` |
| **MMDiT** | Medium | ✅ Done — `:mmdit` |
| **YARN** | Low | ✅ Done — `:yarn` |
| **SoFlow** | Low-Med | ✅ Done — `:soflow` |

### New Tier 3: Exploratory (2026) — All Done

| Addition | Difficulty | Status |
|----------|-----------|--------|
| **Multimodal fusion layers** | High | ✅ Done — `:multimodal_mlp_fusion` (MLP projection, cross-attention, Perceiver resampler) |
| **RL environment integration** | High | ✅ Done — `PPOTrainer`, `GAE`, `CartPole`, `GridWorld` environments |
| **MambaVision** | Medium | ✅ Done — `:mamba_vision` (4-stage hierarchical CNN+Mamba+Attention) |
| **KDA (Kimi Delta Attention)** | Medium | ✅ Done — `:kda` (channel-wise decay, low-rank alpha gate) |

### Notebook / Documentation Gaps

Edifice has 184 registered architectures but only ~11 notebooks. Highest-value additions:

| Gap | Architectures Uncovered | Priority |
|-----|------------------------|----------|
| **Vision** | 9 (ViT, DeiT, Swin, U-Net, ConvNeXt, MLP-Mixer, FocalNet, PoolFormer, NeRF) | High |
| **Attention deep-dive** | 24 attention variants, zero dedicated visualization | High |
| **Contrastive/self-supervised** | 7 (SimCLR, BYOL, BarlowTwins, MAE, VICReg, JEPA, Temporal JEPA) | High |
| **RNN evolution** | 12 recurrent architectures, only basic LSTM/GRU covered | Medium |
| **Diffusion from scratch** | Existing notebook covers VAE, not diffusion training | Medium |
| **MoE routing visualization** | 4 MoE variants, zero visualization of expert routing | Medium |
| **Memory networks** | 2 (NTM, Memory Network), zero coverage | Low |
| **Interpretability** | 2 (SAE, Transcoder), zero coverage | Low |
| **World Models / RL** | 2 (WorldModel, PolicyValue), zero coverage | Low |

### Research Trend Summary (Early 2026)

| Trend | Heat | Signal |
|-------|------|--------|
| Hybrid SSM+Attention (90/10) | Very Hot | Nemotron-H, Jamba 1.5, Qwen3-Next in production |
| Gated DeltaNet / linear attention | Very Hot | Adopted by Qwen3-Next, Kimi Linear |
| Test-time compute / reasoning | Very Hot | o1/o3, DeepSeek-R1, TTT-E2E |
| MoE (DeepSeek-style) | Hot | Standard for all frontier models |
| Rectified Flow + DiT | Hot | Replaced UNet+DDPM for image generation |
| BitNet 1.58-bit | Hot | Microsoft strategic initiative, CPU inference |
| RWKV-7 | Hot | Best open-source non-transformer lineage |
| xLSTM at scale | Hot | 7B model competitive, industrial deployment |
| Native multimodality | Hot | GPT-5, unified AR+Diffusion |
| SAE / interpretability | Uncertain | DeepMind published negative results, pivoting to pragmatic methods |

---

## Appendix: Edifice Architecture Inventory

For reference, here is every architecture currently registered in Edifice (184 entries across 25 families), grouped by family.

### Transformer (4)
- `decoder_only` — GPT-style autoregressive transformer (GQA, RoPE, iRoPE, SwiGLU, RMSNorm)
- `multi_token_prediction` — Predict next N tokens simultaneously
- `byte_latent_transformer` — Encoder + latent transformer + decoder for byte-level processing
- `nemotron_h` — NVIDIA's hybrid Mamba-Transformer architecture

### Feedforward (5)
- `mlp` — Multi-layer perceptron
- `kan` — Kolmogorov-Arnold Networks (learnable activation functions)
- `kat` — KAN-Transformer hybrid
- `tabnet` — Attentive tabular learning
- `bitnet` — 1-bit weight networks

### Convolutional (6)
- `conv1d` — 1D convolution blocks
- `resnet` — Residual networks
- `densenet` — Dense connections
- `tcn` — Temporal convolutional networks
- `mobilenet` — Depthwise separable convolutions
- `efficientnet` — Compound scaling

### Recurrent (15)
- `lstm`, `gru` — Classic recurrent cells
- `xlstm`, `xlstm_v2`, `mlstm`, `slstm` — Extended LSTM family
- `min_gru`, `min_lstm` — Minimal gated units
- `delta_net` — Delta rule-based RNN
- `gated_delta_net` — Gated DeltaNet (linear attention with data-dependent gating)
- `ttt`, `ttt_e2e` — Test-Time Training (standard + end-to-end)
- `titans` — Memory-augmented RNN
- `reservoir` — Echo State Networks
- `native_recurrence` — Native recurrence block

### State Space Models (19)
- `mamba`, `mamba_ssd`, `mamba_cumsum`, `mamba_hillis_steele` — Mamba family (4 scan algorithms)
- `mamba3` — Mamba v3 (complex states, trapezoidal discretization, MIMO)
- `s4`, `s4d`, `s5` — Structured state space family
- `h3` — Hungry Hungry Hippos
- `hyena`, `hyena_v2` — Long convolution models
- `bimamba` — Bidirectional Mamba
- `gated_ssm` — Gated state space model
- `jamba`, `zamba` — Hybrid SSM-Transformer
- `striped_hyena` — Alternating attention + Hyena
- `gss` — Gated State Spaces
- `hymba` — Hybrid Mamba
- `ss_transformer` — State Space Transformer

### Attention Variants (34)
- `attention` — Standard multi-head attention
- `gqa` — Grouped query attention
- `mla` — Multi-head latent attention (DeepSeek)
- `diff_transformer` — Differential attention
- `retnet`, `retnet_v2` — Retentive networks
- `rwkv` — RWKV-7 "Goose" (generalized delta rule)
- `gla`, `gla_v2` — Gated Linear Attention
- `hgrn`, `hgrn_v2` — Hierarchically Gated Recurrent Network
- `griffin`, `hawk` — Google's RNN-attention hybrids
- `based` — Linear attention with Taylor expansion
- `perceiver` — Cross-attention with latent bottleneck
- `fnet` — Fourier transform attention
- `linear_transformer` — Linear attention
- `nystromformer` — Nystrom approximation
- `performer` — Random feature attention
- `mega`, `megalodon` — Exponential moving average attention
- `infini_attention` — Compressive memory attention
- `conformer` — Convolution + attention (speech)
- `ring_attention` — Distributed attention across devices
- `lightning_attention` — Hybrid linear/softmax attention with I/O-aware tiling
- `flash_linear_attention` — Flash Linear Attention
- `kda` — Kimi Delta Attention (channel-wise decay)
- `gated_attention` — Sigmoid post-attention gate (NeurIPS 2025 best paper)
- `nsa` — Native Sparse Attention (DeepSeek three-path)
- `rnope_swa` — No positional encoding + sliding window
- `yarn` — YaRN context extension (RoPE frequency scaling)
- `tmrope` — Time-aligned Multimodal RoPE
- `dual_chunk_attention` — Dual Chunk Attention for long-context
- `ssmax` — Scalable-Softmax (drop-in softmax replacement)
- `softpick` — Non-saturating sparse attention function

### Vision (15)
- `vit` — Vision Transformer
- `deit` — Data-efficient Image Transformer
- `swin` — Shifted window transformer
- `unet` — Encoder-decoder with skip connections
- `convnext` — Modernized ConvNet
- `mlp_mixer` — All-MLP architecture
- `focalnet` — Focal modulation
- `poolformer` — Pooling as token mixing
- `nerf` — Neural Radiance Fields
- `gaussian_splat` — 3D Gaussian Splatting
- `mamba_vision` — MambaVision (4-stage hierarchical CNN+Mamba+Attention)
- `dino_v2` — DINOv2 self-distillation vision backbone
- `metaformer` — MetaFormer (architecture-first framework)
- `caformer` — CAFormer (Conv + Attention stages)
- `efficient_vit` — EfficientViT (linear attention vision)

### Generative (22)
- `vae`, `vq_vae` — Variational autoencoders
- `gan` — Generative adversarial network
- `diffusion`, `ddim` — Diffusion models
- `dit`, `dit_v2` — Diffusion Transformer
- `mmdit` — Multimodal Diffusion Transformer (FLUX.1, SD3)
- `linear_dit` / `sana` — Linear DiT (linear attention for diffusion)
- `sit` — Scalable Interpolant Transformer
- `latent_diffusion` — Latent space diffusion
- `consistency_model` — Single-step generation
- `score_sde` — Score-based SDE
- `flow_matching` — Continuous normalizing flows
- `soflow` — SoFlow (flow matching + consistency)
- `normalizing_flow` — Invertible transformations
- `var` — Visual Autoregressive (next-scale prediction)
- `transfusion` — Unified AR text + diffusion images
- `mar` — Masked Autoregressive generation
- `cogvideox` — CogVideoX video generation
- `trellis` — TRELLIS 3D generation (sparse lattices + rectified flow)

### Graph (9)
- `gcn` — Graph Convolutional Network
- `gat` — Graph Attention Network
- `graph_sage` — Inductive graph learning
- `gin`, `gin_v2` — Graph Isomorphism Network
- `pna` — Principal Neighbourhood Aggregation
- `graph_transformer` — Transformer on graphs
- `schnet` — Continuous-filter convolution (molecular)
- `egnn` — E(n)-Equivariant GNN (rotation/translation invariant)

### Sets & Point Clouds (2)
- `deep_sets` — Permutation-invariant set functions
- `pointnet` — 3D point cloud processing

### Energy-Based (3)
- `ebm` — Energy-Based Model
- `hopfield` — Modern Hopfield networks
- `neural_ode` — Neural Ordinary Differential Equations

### Probabilistic (3)
- `bayesian` — Bayesian neural networks
- `mc_dropout` — Monte Carlo dropout
- `evidential` — Evidential deep learning

### Memory (3)
- `ntm` — Neural Turing Machine
- `memory_network` — End-to-end memory networks
- `engram` — O(1) hash-based associative memory

### Meta / Composition (22)
- `moe`, `switch_moe`, `soft_moe`, `moe_v2` — Mixture of Experts family
- `lora`, `dora` — Low-rank adaptation
- `adapter` — Adapter layers
- `hypernetwork` — Weight generation
- `capsule` — Capsule networks
- `mixture_of_depths` — Dynamic computation allocation
- `mixture_of_agents` — Multi-model ensemble
- `mixture_of_tokenizers` — Different experts process different tokenizations
- `rlhf_head` — Reward/value heads for RLHF
- `dpo` — Direct Preference Optimization
- `grpo` — Group Relative Policy Optimization (DeepSeek-R1)
- `kto` — Kahneman-Tversky Optimization (binary feedback)
- `speculative_decoding` — Draft + verify inference acceleration
- `speculative_head` — Multi-head speculative prediction
- `distillation_head` — Knowledge distillation
- `test_time_compute` — Adaptive test-time compute
- `qat` — Quantization-Aware Training
- `hybrid_builder` — Configurable SSM/Attention ratio builder

### Contrastive / Self-Supervised (8)
- `simclr` — Contrastive learning
- `byol` — Bootstrap Your Own Latent
- `barlow_twins` — Redundancy reduction
- `mae` — Masked Autoencoder
- `vicreg` — Variance-Invariance-Covariance Regularization
- `jepa` — Joint Embedding Predictive Architecture
- `temporal_jepa` — V-JEPA (video/temporal JEPA)
- `siglip` — Sigmoid contrastive learning (CLIP improvement)

### Interpretability (2)
- `sparse_autoencoder` — Trainable SAE for feature extraction
- `transcoder` — Cross-layer transcoder for mechanistic interpretability

### World Model (1)
- `world_model` — Encoder + dynamics + reward head

### Multimodal (1)
- `multimodal_mlp_fusion` — MLP projection, cross-attention, Perceiver resampler

### RL (1 registered + infrastructure)
- `policy_value` — Actor-critic policy-value network
- `PPOTrainer` — Proximal Policy Optimization trainer
- `GAE` — Generalized Advantage Estimation
- `CartPole` — Classic cart-pole balancing environment
- `GridWorld` — Discrete grid navigation environment

### Liquid (1)
- `liquid` — Liquid Neural Networks (continuous-time ODE)

### Scientific (1)
- `fno` — Fourier Neural Operator (PDE solving)

### Neuromorphic (2)
- `snn` — Spiking Neural Network
- `ann2snn` — ANN to SNN conversion

### Inference (1)
- `medusa` — Multi-head speculative decoding

### Robotics (2)
- `act` — Action Chunking Transformer (imitation learning)
- `openvla` — Vision-Language-Action model

### Audio (3)
- `encodec` — Neural audio codec (encoder → RVQ → decoder)
- `valle` — VALL-E codec language model (zero-shot TTS)
- `soundstorm` — Parallel audio token generation

---

## 2026 Update #2: Frontier Architecture Survey (Feb 2026)

> A comprehensive survey of architectures Edifice doesn't yet support, organized into
> implementable tiers. Research conducted across attention/SSM, generative/diffusion,
> vision/multimodal, robotics, scientific ML, and dynamic inference domains.

**Current state**: 234 registered architectures across 26 families.

### How Tiers Are Assigned

| Tier | Criteria | Expected Effort |
|------|----------|-----------------|
| **Tier 1** | Deployed in production frontier models, or NeurIPS/ICML best-paper tier. Direct, concrete impact. | 1-3 days per item |
| **Tier 2** | Strong papers with code, adopted by multiple research groups, fills important Edifice gaps. | 1-2 days per item |
| **Tier 3** | Interesting/emerging, fills niche gaps, or requires significant new infrastructure. | Variable |

---

### New Tier 1: High Impact (2026 Update #2)

These were architectures deployed in production models or with best-paper-level recognition. **All 6 are now implemented.**

#### 1. Gated Attention — ✅ DONE
Implemented as `:gated_attention` (`Edifice.Attention.GatedAttention`).

#### 2. Native Sparse Attention (NSA) — ✅ DONE
Implemented as `:nsa` (`Edifice.Attention.NSA`). Three-path sparse attention (compressed, top-k block selection, sliding window).

#### 3. DiffTransformer V2 — ✅ DONE
`:diff_transformer` updated with simplified formulation.

#### 4. VAR (Visual Autoregressive Modeling) — ✅ DONE
Implemented as `:var` (`Edifice.Generative.VAR`). Next-scale prediction with multi-scale VQ tokenizer.

#### 5. Scalable-Softmax (SSMax) — ✅ DONE
Implemented as `:ssmax` (`Edifice.Blocks.SSMax`). Drop-in softmax replacement.

#### 6. Transfusion — ✅ DONE
Implemented as `:transfusion` (`Edifice.Generative.Transfusion`). Unified AR text + diffusion images.

---

### New Tier 2: Strong Research (2026 Update #2)

#### Attention & Sequence Innovations

| Architecture | What | Status |
|-------------|------|--------|
| **Softpick** | Non-saturating sparse attention function | ✅ Done — `:softpick` |
| **Engram** | O(1) hash-based associative memory | ✅ Done — `:engram` |
| **SPLA** | Sparse + Linear Attention hybrid | ❌ Not done |
| **InfLLM-V2** | Block-partitioned KV cache selection | ❌ Not done |
| **RNoPE-SWA** | No positional encoding + sliding window | ✅ Done — `:rnope_swa` |

#### Generative Models

| Architecture | What | Status |
|-------------|------|--------|
| **SANA (Linear DiT)** | Linear attention for diffusion, 100x speedup | ✅ Done — `:linear_dit` / `:sana` |
| **SiT** | Scalable Interpolant Transformer | ✅ Done — `:sit` |
| **MAR** | Masked Autoregressive generation | ✅ Done — `:mar` |
| **F5-TTS** | Non-autoregressive flow-matching TTS | ❌ Not done |
| **JanusFlow** | AR text + rectified flow images | ❌ Not done |
| **Show-o** | AR + discrete diffusion | ❌ Not done |

#### Vision & Multimodal

| Architecture | What | Status |
|-------------|------|--------|
| **DINOv2** | Self-distillation vision backbone | ✅ Done — `:dino_v2` |
| **SigLIP** | Sigmoid contrastive learning | ✅ Done — `:siglip` |
| **MetaFormer / CAFormer** | Architecture-first framework | ✅ Done — `:metaformer`, `:caformer` |
| **EfficientViT** | Linear attention ViT | ✅ Done — `:efficient_vit` |

#### Scientific & Specialized

| Architecture | What | Status |
|-------------|------|--------|
| **FNO** | Fourier Neural Operator for PDEs | ✅ Done — `:fno` |
| **EGNN** | E(n)-Equivariant GNN | ✅ Done — `:egnn` |
| **Diffusion Policy** | Diffusion for robot actions | ❌ Not done |

#### Training & Optimization

| Architecture | What | Status |
|-------------|------|--------|
| **DPO** | Direct Preference Optimization | ✅ Done — `:dpo` |
| **GRPO** | Group Relative Policy Optimization | ✅ Done — `:grpo` |

---

### New Tier 3: Exploratory (2026 Update #2)

#### Dynamic Inference

| Architecture | What | Status |
|-------------|------|--------|
| **Medusa** | Multi-head speculative decoding, 2-3x speedup | ✅ Done — `:medusa` |
| **MoR (Mixture of Recursions)** | Dynamic depth per token | ❌ Not done |
| **MoED (Mixture of Expert Depths)** | Per-expert depth routing | ❌ Not done |

#### Audio & Speech

| Architecture | What | Status |
|-------------|------|--------|
| **EnCodec** | Neural audio codec (encoder → RVQ → decoder) | ✅ Done — `:encodec` |
| **VALL-E** | Codec language model for TTS | ✅ Done — `:valle` |
| **SoundStorm** | Parallel audio token generation | ✅ Done — `:soundstorm` |

#### Video Generation

| Architecture | What | Status |
|-------------|------|--------|
| **CogVideoX** | 3D causal VAE + expert transformer for video | ✅ Done — `:cogvideox` |
| **CausVid / Causal Forcing** | Causal video DiT distillation | ❌ Not done |

#### 3D & Spatial

| Architecture | What | Status |
|-------------|------|--------|
| **3D Gaussian Splatting** | Differentiable rasterization of 3D Gaussians | ✅ Done — `:gaussian_splat` |
| **TRELLIS** | Sparse 3D lattice + rectified flow generation | ✅ Done — `:trellis` |

#### Robotics & Embodied AI

| Architecture | What | Status |
|-------------|------|--------|
| **ACT** | Action Chunking Transformer for imitation learning | ✅ Done — `:act` |
| **OpenVLA** | Vision-Language-Action model | ✅ Done — `:openvla` |

#### Scientific ML (Extended)

| Architecture | What | Status |
|-------------|------|--------|
| **DeepONet** | Branch-trunk operator learning | ❌ Not done |
| **SE(3)-Transformer** | Equivariant transformer for structural biology | ❌ Not done |

#### Modern Tokenizers

| Architecture | What | Status |
|-------------|------|--------|
| **MAGVIT-v2** | Lookup-free quantization for image/video tokens | ❌ Not done |

#### Miscellaneous High-Interest

| Architecture | What | Status |
|-------------|------|--------|
| **mHC (Manifold Hyper-Connections)** | Riemannian manifold residual stream | ❌ Not done |
| **KTO** | Binary feedback RLHF (Kahneman-Tversky) | ✅ Done — `:kto` |
| **MIRAS (Moneta/Memora/Yaad)** | Google's Titans extension framework | ❌ Not done |

---

### Priority Recommendation — Updated Status

All 16 items from the original priority list are now implemented:

**Quick Wins — ✅ All Done**:
1. ~~Gated Attention~~ → `:gated_attention`
2. ~~Scalable-Softmax~~ → `:ssmax`
3. ~~DiffTransformer V2~~ → `:diff_transformer` (updated)
4. ~~Softpick~~ → `:softpick`
5. ~~SigLIP~~ → `:siglip`
6. ~~RNoPE-SWA~~ → `:rnope_swa`

**Medium Builds — ✅ All Done**:
7. ~~VAR~~ → `:var`
8. ~~SANA / Linear DiT~~ → `:linear_dit` / `:sana`
9. ~~DINOv2~~ → `:dino_v2`
10. ~~FNO~~ → `:fno`
11. ~~DPO / GRPO~~ → `:dpo`, `:grpo`
12. ~~Transfusion~~ → `:transfusion`

**Ambitious — ✅ All Done**:
13. ~~NSA~~ → `:nsa`
14. ~~EGNN~~ → `:egnn`
15. ~~EnCodec~~ → `:encodec`
16. ~~3D Gaussian Splatting~~ → `:gaussian_splat`

### Remaining Unimplemented (from all tiers)

| Architecture | Category | Difficulty |
|-------------|----------|-----------|
| SPLA (Sparse+Linear Attention) | Attention | Medium |
| InfLLM-V2 | Attention | Medium |
| F5-TTS | Audio/TTS | Medium |
| JanusFlow | Generative | Medium-High |
| Show-o | Generative | Medium |
| Diffusion Policy | Robotics | Medium |
| CausVid / Causal Forcing | Video | High |
| DeepONet | Scientific ML | Medium |
| SE(3)-Transformer | Scientific ML | High |
| MAGVIT-v2 | Tokenizers | Medium-High |
| mHC (Manifold Hyper-Connections) | Architecture | High |
| MIRAS (Moneta/Memora/Yaad) | Memory | High |
| MoR (Mixture of Recursions) | Dynamic inference | Medium-High |
| MoED (Mixture of Expert Depths) | Dynamic inference | Medium |
| Agent swarm patterns | Application | High |

---

### Research Trend Summary (Feb 2026)

| Trend | Heat | Signal |
|-------|------|--------|
| Gated Attention (sigmoid post-attention gate) | 🔥🔥🔥 | NeurIPS best paper, Qwen3.5 deployment |
| Unified AR + Diffusion (Transfusion, Show-o, JanusFlow) | 🔥🔥🔥 | Every frontier lab pursuing native multimodal |
| Native Sparse Attention (hardware-aligned) | 🔥🔥🔥 | DeepSeek-V3/V4 core mechanism |
| Linear attention in DiT (SANA) | 🔥🔥 | 100x speedup for image generation |
| Next-scale prediction (VAR) | 🔥🔥 | NeurIPS best paper, new generation paradigm |
| Scalable-Softmax / Softpick | 🔥🔥 | Drop-in improvements to all attention |
| Scientific ML (FNO, EGNN, weather models) | 🔥🔥 | Massive real-world impact, underserved by frameworks |
| Neural audio codecs (EnCodec → LM) | 🔥🔥 | Foundation for speech/music generation |
| VLA for robotics (OpenVLA, Pi0) | 🔥 | Early but high potential |
| RLHF alternatives (DPO, GRPO, KTO) | 🔥🔥 | Standard practice, replaces classical RLHF |
| Speculative decoding (Medusa) | 🔥 | Practical inference optimization |
| 3D Gaussian Splatting | 🔥🔥 | Replacing NeRF across the board |

### Notebook Ideas Unlocked by New Tiers

| Notebook | Architectures Used | Priority |
|----------|-------------------|----------|
| **"Softmax Shootout"** — compare softmax, SSMax, Softpick, ASEntmax on same task | Scalable-Softmax, Softpick, attention | High |
| **"Image Generation Paradigms"** — VAR vs DiT vs consistency vs flow matching | VAR, DiT, consistency_model, flow_matching | High |
| **"Self-Supervised Vision"** — DINOv2 vs MAE vs SimCLR vs JEPA | DINOv2, MAE, SimCLR, JEPA | High |
| **"Scientific ML: Solving PDEs"** — FNO vs DeepONet vs neural ODE | FNO, DeepONet, neural_ode | Medium |
| **"RLHF Without Tears"** — DPO vs GRPO vs KTO on simple preference tasks | DPO, GRPO, KTO, rlhf_head | Medium |
| **"Audio from Scratch"** — EnCodec tokenization + VALL-E generation | EnCodec, VALL-E | Medium |
| **"Unified Multimodal"** — Transfusion: one model for text + images | Transfusion, decoder_only, dit | High |

---

## 2026 Update #3: Wave 4 Architecture Candidates (Feb 28, 2026)

> Research survey of architectures not yet in Edifice's 234-module registry.
> Cross-referenced against all existing implementations to identify genuine gaps.

**Current state**: 234 registered architectures across 26 families. Since Update #2,
added 50 architectures including full interpretability family (10 modules), detection
family (3), expanded audio (6), expanded graph (DimeNet, SE3, GPS), sets family
(DeepSets, PointNet, PointNet++), and many more.

### Top Candidates for Wave 4

#### FoX (Forgetting Transformer) — ICLR 2025

**What**: Standard softmax attention augmented with a learnable per-head forget gate.
Each head has a sigmoid gate that modulates attention weights, effectively giving the
transformer bounded memory. Tokens beyond the "forgetting horizon" receive exponentially
decayed attention, enabling O(1) memory during inference while maintaining training-time
full-attention quality.

**Why it matters**: FoX unifies the transformer/RNN divide. During training, the forget
gate is near 1.0 (standard attention). During inference, it enables streaming with
bounded KV cache. Microsoft has adopted this for production models.

**Key innovation**: The forget gate is multiplicative on the attention logits (before
softmax), not on the attention weights. This means it can be fused with standard
FlashAttention kernels with minimal overhead.

**Paper**: "FoX: Forgetting Transformer" (ICLR 2025)

**Difficulty**: Low-medium. Extends existing `MultiHead` attention with per-head sigmoid
gate. Main work is the decay-based attention modification.

**Builds on**: `:attention`, `:gated_attention` (similar pattern of post-attention gating)

#### Log-Linear Attention — arXiv Jun 2025

**What**: Attention mechanism with O(log T) memory that bridges linear attention (O(1)
memory, poor quality) and softmax attention (O(T) memory, full quality). Uses a
hierarchical segment tree where each level stores aggregated KV pairs at
exponentially increasing granularity.

**Why it matters**: Linear attention sacrifices quality. Full attention sacrifices
memory. Log-Linear achieves a Pareto-optimal tradeoff — provably the best possible
memory-quality curve for any attention mechanism.

**Key innovation**: Segment-based attention with hierarchical aggregation. Recent
tokens get exact attention, older tokens get coarser-grained attention. The tree
structure makes this O(log T) in both time and space.

**Difficulty**: Medium. Requires segment tree data structure and hierarchical
attention aggregation. Novel implementation pattern not seen in existing modules.

**Builds on**: `:lightning_attention` (hybrid attention concept), `:infini_attention`
(compressive memory concept)

#### TarFlow / STARFlow — Apple, ICML 2025

**What**: Normalizing flow built entirely from transformer blocks. Uses masked
self-attention on flattened image patches to define an autoregressive flow.
Unlike diffusion models, provides exact log-likelihood computation and single-pass
generation. STARFlow extends this with stacked multi-scale latent hierarchy.

**Why it matters**: First normalizing flow competitive with diffusion on image
quality (FID ~2 on ImageNet 256x256). Exact likelihood enables principled density
estimation, out-of-distribution detection, and lossless compression.

**Key innovation**: Reuses standard transformer architecture (attention + FFN) as
flow coupling layers. The autoregressive masking pattern defines the triangular
Jacobian needed for tractable likelihood.

**Paper**: "Autoregressive Image Generation without Vector Quantization" (Apple, ICML 2025)

**Difficulty**: Medium. Core transformer infrastructure exists. Main work is the
flow coupling layer pattern and multi-scale hierarchy for STARFlow.

**Builds on**: `:normalizing_flow` (flow framework), `:decoder_only` (masked attention)

#### Native Hybrid Attention (NHA) — ICML 2025

**What**: Unified framework that jointly selects linear vs. full (softmax) attention
on a per-layer basis with shared KV projections. Instead of designing the hybrid
ratio by hand (like Jamba's 87.5% Mamba / 12.5% attention), NHA learns the optimal
allocation during training.

**Why it matters**: The hybrid SSM+attention pattern is dominant (Nemotron-H,
Jamba, Zamba), but the ratio is always hand-tuned. NHA makes this learnable,
potentially finding better ratios than human designers.

**Difficulty**: Medium. Extends existing `:hybrid_builder` with learned per-layer
selection mechanism.

**Builds on**: `:hybrid_builder`, `:lightning_attention` (linear attention),
`:attention` (softmax attention)

#### Coconut (Continuous Chain of Thought) — Meta, ICLR 2025

**What**: Instead of generating chain-of-thought reasoning as discrete text tokens,
Coconut operates in continuous latent space. The model's hidden states from one
"thought step" are fed back as input to the next step, enabling breadth-first
exploration of reasoning paths without the bottleneck of text generation.

**Why it matters**: Text-based CoT is sequential and lossy (the model must compress
its reasoning into discrete tokens). Continuous CoT can maintain richer intermediate
representations and explore multiple reasoning paths simultaneously.

**Key innovation**: Hidden state recycling — the last hidden state of one forward
pass becomes the "thought token" input for the next pass. No text generation overhead.

**Paper**: "Training Large Language Models to Reason in a Continuous Latent Space"
(Meta, ICLR 2025)

**Difficulty**: Medium. The architecture modification is straightforward (hidden
state feedback), but training requires the multi-stage curriculum from the paper.

**Builds on**: `:decoder_only` (base architecture), `:test_time_compute` (related
concept of adaptive inference)

#### Memory Layers — Meta, 2025

**What**: Sparse key-value lookup layers with 1M+ keys that replace dense FFN
layers. Each "memory layer" stores a large dictionary of key-value pairs. During
forward pass, the input is used to retrieve the top-k most relevant memories via
product-quantized approximate nearest neighbor search.

**Why it matters**: Dense FFN layers scale linearly with hidden dimension and are
the majority of parameters in large models. Memory layers provide massive
associative storage at constant compute cost — only top-k memories are activated.

**Key innovation**: Product quantization makes the lookup O(sqrt(N)) instead of
O(N). The memories are trained end-to-end with the rest of the model.

**Difficulty**: Medium-high. Requires product quantization and efficient
nearest-neighbor search. Novel pattern not in existing modules.

**Builds on**: `:engram` (hash-based associative memory concept), `:moe` (sparse
activation concept)

#### V-JEPA 2 — Meta, 2025

**What**: Next-generation video world model extending V-JEPA with attentive pooling
and a multimodal decoder. Improved temporal abstraction through hierarchical
prediction targets at multiple temporal scales.

**Why it matters**: V-JEPA showed that predicting in representation space (not pixel
space) learns strong video features. V-JEPA 2 adds the ability to decode back to
multiple modalities (text, action) while maintaining the non-generative training.

**Difficulty**: Medium. Extends existing `:temporal_jepa` with attentive pooling
and multimodal decoder heads.

**Builds on**: `:temporal_jepa`, `:jepa`, `:perceiver` (cross-attention pooling)

#### KA-GNN — KAN + GNN Hybrid

**What**: Replaces the MLP message functions in GNNs with KAN (Kolmogorov-Arnold
Network) layers. The learnable B-spline activation functions in KAN provide more
expressive edge/node transformations than standard ReLU MLPs.

**Why it matters**: Molecular property prediction and drug discovery rely heavily
on GNNs. KA-GNN shows consistent improvements on molecular benchmarks by making
the message passing more expressive.

**Difficulty**: Low. Swap MLP layers for KAN layers in existing GNN modules.

**Builds on**: `:kan` (KAN architecture), `:gcn`/`:gat`/`:gin` (GNN family)

#### FreeTransformer — Meta, 2025

**What**: Decoder architecture with a per-layer latent variable that enables
speculative decoding without a separate draft model. Each layer samples a latent
from a learned prior; given the latent, generation becomes deterministic. This
means you can speculate by sampling multiple latents and verifying in parallel.

**Why it matters**: Speculative decoding currently requires maintaining a separate
small "draft" model. FreeTransformer eliminates this requirement — the model is
its own draft model via latent sampling.

**Difficulty**: Medium-high. Requires latent variable integration into the
transformer forward pass and modified sampling procedure.

**Builds on**: `:decoder_only`, `:speculative_decoding` (inference pattern)

### Research Trend Summary (Late Feb 2026)

| Trend | Heat | Signal |
|-------|------|--------|
| Forgetting/bounded-memory attention (FoX) | Very Hot | ICLR 2025, Microsoft adoption |
| Learned hybrid attention ratios (NHA) | Hot | ICML 2025, replaces hand-tuned ratios |
| Continuous reasoning (Coconut) | Hot | Meta ICLR 2025, alternative to text CoT |
| Transformer-based normalizing flows (TarFlow) | Hot | Apple ICML 2025, competitive with diffusion |
| Sparse memory layers | Hot | Meta production research, massive parameter efficiency |
| Log-space attention (Log-Linear) | Emerging | Theoretical optimality proof, early adoption |
| KAN in everything (KA-GNN, KAT) | Warm | Consistent improvements across domains |
| Self-speculative decoding (FreeTransformer) | Emerging | Eliminates draft model requirement |
