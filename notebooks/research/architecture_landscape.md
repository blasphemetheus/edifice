# The ML Architecture Landscape

> Where Edifice stands, what's happening at the frontier, and what to build next.

Edifice currently implements **113+ architectures** spanning 15 families — from transformers and state space models to graph networks, generative models, and neuromorphic computing. This document maps the broader landscape to identify gaps, emerging paradigms, and high-impact build targets.

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

**Edifice status:** ❌ Not implemented. Edifice has `:linear_transformer` and `:performer` (both pure linear attention) but not the hybrid approach.

### iRoPE (Meta, Llama 4)

**What it is:** **Interleaved Rotary Position Embeddings**. Instead of applying RoPE to every attention layer, Meta's Llama 4 applies it to alternating layers. Layers without RoPE use NoPE (no positional encoding) — they attend based purely on content similarity.

**Why it matters:** This decouples position-dependent attention (where RoPE helps) from content-dependent attention (where position can be a distraction). The result: better generalization to sequence lengths far beyond training. Llama 4 was trained on shorter contexts but reportedly supports 10M+ tokens at inference.

**Edifice status:** ❌ Not implemented. Edifice's `:decoder_only` applies RoPE uniformly. Adding an `interleave_rope: true` option would be straightforward.

### YARN + Dual Chunk Attention (Qwen)

**What it is:** Two complementary techniques for extending context at inference time:

- **YARN (Yet Another RoPE Extension):** Modifies RoPE frequencies to handle longer sequences than training. More principled than simple NTK-aware scaling — YARN identifies which frequency bands need modification and applies targeted adjustments.
- **Dual Chunk Attention:** Splits long sequences into chunks, computes intra-chunk attention normally, then uses a separate mechanism for inter-chunk attention. Reduces peak memory while maintaining long-range connections.

**Why it matters:** These let a model trained on 4K context work at 32K+ context without retraining. Qwen2.5 used this combination to achieve 128K context.

**Edifice status:** ❌ Not implemented. Edifice has standard RoPE but no context extension methods.

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
| Lightning Attention (hybrid) | ❌ | — |
| iRoPE (interleaved) | ❌ | — |
| YARN context extension | ❌ | — |
| Dual Chunk Attention | ❌ | — |
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

**What Edifice has:** MLA (✅), MoE v2 (✅ partial), Multi-token prediction (❌), FP8 training (❌ — depends on Nx/EXLA support)

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
| Lightning Attention | 1 | Medium | linear_transformer, performer |
| Transcoders/SAE Module | 1 | Low-Med | vae, any model |
| iRoPE | 1 | Low | decoder_only |
| JEPA v2 / V-JEPA | 1 | Medium | jepa, vit, mae |
| Aux-loss-free MoE | 1 | Low | moe_v2 |
| Multi-token prediction | 2 | Low | decoder_only |
| YARN | 2 | Low | decoder_only |
| Speculative decoding | 2 | Medium | any model pair |
| GGUF export | 2 | High | quantization toolkit |
| RL environments | 3 | High | rlhf_head |
| Agent swarm patterns | 3 | High | mixture_of_agents |
| Multimodal fusion | 3 | High | vit, decoder_only |
| TMRoPE | 3 | Medium | multimodal fusion |

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
| iRoPE | 1 | **Not done** — still high-value, used by Llama 4 |
| Aux-loss-free MoE | 1 | **Not done** — DeepSeek-V3's bias routing |
| YARN context extension | 2 | **Not done** |
| GGUF export | 2 | **Not done** — high effort |
| RL environments | 3 | **Not done** |
| Agent swarm patterns | 3 | **Not done** |
| Multimodal fusion | 3 | **Not done** |
| TMRoPE | 3 | **Not done** |

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

#### iRoPE (Interleaved RoPE) — carried from original Tier 1

| | |
|---|---|
| **What** | Apply RoPE to alternating transformer layers, NoPE (content-only) for others |
| **Difficulty** | Low — add `interleave_rope: true` option to `:decoder_only` |
| **Why Tier 1** | Trivial change, proven by Llama 4 for long-context generalization |

#### Aux-loss-free MoE Routing — carried from original Tier 1

| | |
|---|---|
| **What** | Replace auxiliary load-balancing loss with dynamic bias tensor in `:moe_v2` |
| **Difficulty** | Low — add bias tensor, update based on utilization running average |
| **Why Tier 1** | Standard practice for frontier MoE models |

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

### New Tier 2: Medium Impact (2026)

| Addition | Difficulty | Why |
|----------|-----------|-----|
| **TTT-E2E** | Medium | End-to-end test-time training — mutates 25% of MLP layers at inference for long context. Extends existing `:ttt`. |
| **MMDiT** | Medium | Multimodal Diffusion Transformer — joint text-image attention blocks. Standard for FLUX.1, SD3, Sora. |
| **YARN** | Low | Carried from original Tier 2. Still the standard context extension method. |
| **SoFlow** | Low-Med | Combined flow matching + consistency loss for one-step generation. Extends `:flow_matching` + `:consistency_model`. |

### New Tier 3: Exploratory (2026)

| Addition | Difficulty | Status |
|----------|-----------|--------|
| **Multimodal fusion layers** | High | Done — `:multimodal_mlp_fusion` (MLP projection, cross-attention, Perceiver resampler) |
| **RL environment integration** | High | Done — `PPOTrainer`, `GAE`, `CartPole`, `GridWorld` environments |
| **MambaVision** | Medium | Done — `:mamba_vision` (4-stage hierarchical CNN+Mamba+Attention) |
| **KDA (Kimi Delta Attention)** | Medium | Done — `:kda` (channel-wise decay, low-rank alpha gate) |

### Notebook / Documentation Gaps

Edifice has 113+ architectures but only ~11 notebooks. Highest-value additions:

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

For reference, here is every architecture currently in Edifice, grouped by family.

### Transformer
- `decoder_only` — GPT-style autoregressive transformer (GQA, RoPE, SwiGLU, RMSNorm)

### Feedforward
- `mlp` — Multi-layer perceptron
- `kan` — Kolmogorov-Arnold Networks (learnable activation functions)
- `kat` — KAN-Transformer hybrid
- `tabnet` — Attentive tabular learning
- `bitnet` — 1-bit weight networks

### Convolutional
- `conv1d` — 1D convolution blocks
- `resnet` — Residual networks
- `densenet` — Dense connections
- `tcn` — Temporal convolutional networks
- `mobilenet` — Depthwise separable convolutions
- `efficientnet` — Compound scaling

### Recurrent
- `lstm`, `gru` — Classic recurrent cells
- `xlstm`, `xlstm_v2`, `mlstm`, `slstm` — Extended LSTM family
- `min_gru`, `min_lstm` — Minimal gated units
- `delta_net` — Delta rule-based RNN
- `ttt` — Test-Time Training
- `titans` — Memory-augmented RNN
- `reservoir` — Echo State Networks

### State Space Models
- `mamba`, `mamba_ssd`, `mamba_cumsum`, `mamba_hillis_steele` — Mamba family (4 implementations)
- `mamba3` — Mamba v3
- `s4`, `s4d`, `s5` — Structured state space family
- `h3` — Hungry Hungry Hippos
- `hyena`, `hyena_v2` — Long convolution models
- `bimamba` — Bidirectional Mamba
- `gated_ssm` — Gated state space model
- `jamba`, `zamba` — Hybrid SSM-Transformer
- `striped_hyena` — Alternating attention + Hyena
- `gss` — Gated State Spaces
- `hymba` — Hybrid Mamba

### Attention Variants
- `attention` — Standard multi-head attention
- `gqa` — Grouped query attention
- `mla` — Multi-head latent attention (DeepSeek)
- `diff_transformer` — Differential attention
- `retnet`, `retnet_v2` — Retentive networks
- `rwkv` — Receptance Weighted Key Value
- `gla` — Gated Linear Attention
- `hgrn` — Hierarchically Gated Recurrent Network
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

### Vision
- `vit` — Vision Transformer
- `deit` — Data-efficient Image Transformer
- `swin` — Shifted window transformer
- `unet` — Encoder-decoder with skip connections
- `convnext` — Modernized ConvNet
- `mlp_mixer` — All-MLP architecture
- `focalnet` — Focal modulation
- `poolformer` — Pooling as token mixing
- `nerf` — Neural Radiance Fields

### Generative
- `vae`, `vq_vae` — Variational autoencoders
- `gan` — Generative adversarial network
- `diffusion`, `ddim` — Diffusion models
- `dit`, `dit_v2` — Diffusion Transformer
- `latent_diffusion` — Latent space diffusion
- `consistency_model` — Single-step generation
- `score_sde` — Score-based SDE
- `flow_matching` — Continuous normalizing flows
- `normalizing_flow` — Invertible transformations

### Graph
- `gcn` — Graph Convolutional Network
- `gat` — Graph Attention Network
- `graph_sage` — Inductive graph learning
- `gin`, `gin_v2` — Graph Isomorphism Network
- `pna` — Principal Neighbourhood Aggregation
- `graph_transformer` — Transformer on graphs
- `schnet` — Continuous-filter convolution (molecular)

### Sets & Point Clouds
- `deep_sets` — Permutation-invariant set functions
- `pointnet` — 3D point cloud processing

### Energy-Based
- `ebm` — Energy-Based Model
- `hopfield` — Modern Hopfield networks
- `neural_ode` — Neural Ordinary Differential Equations

### Probabilistic
- `bayesian` — Bayesian neural networks
- `mc_dropout` — Monte Carlo dropout
- `evidential` — Evidential deep learning

### Memory
- `ntm` — Neural Turing Machine
- `memory_network` — End-to-end memory networks

### Meta / Composition
- `moe`, `switch_moe`, `soft_moe`, `moe_v2` — Mixture of Experts family
- `lora`, `dora` — Low-rank adaptation
- `adapter` — Adapter layers
- `hypernetwork` — Weight generation
- `capsule` — Capsule networks
- `mixture_of_depths` — Dynamic computation allocation
- `mixture_of_agents` — Multi-model ensemble
- `rlhf_head` — Reward/value heads for RLHF

### Contrastive / Self-Supervised
- `simclr` — Contrastive learning
- `byol` — Bootstrap Your Own Latent
- `barlow_twins` — Redundancy reduction
- `mae` — Masked Autoencoder
- `vicreg` — Variance-Invariance-Covariance Regularization
- `jepa` — Joint Embedding Predictive Architecture

### Liquid
- `liquid` — Liquid Neural Networks (continuous-time ODE)

### Neuromorphic
- `snn` — Spiking Neural Network
- `ann2snn` — ANN to SNN conversion
