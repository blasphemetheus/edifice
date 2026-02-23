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

## v0.3.0 Candidates
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

## 🔍 Opus Review Pass — AI-Generated Architecture Implementations

All architectures added since Tier 1 (2026-02) were implemented by Claude Code (sonnet).
**Bradley to review with Opus for correctness, math accuracy, and idiomatic Elixir.**

Priority review targets (most complex / most novel):
- `lib/edifice/attention/nsa.ex` — NSA three-path sparse attention (complex)
- `lib/edifice/generative/transfusion.ex` — mixed AR+diffusion mask logic
- `lib/edifice/generative/var.ex` — next-scale VQ tokenizer
- `lib/edifice/scientific/fno.ex` — spectral convolution via FFT
- `lib/edifice/graph/egnn.ex` — equivariant coord update equations
- `lib/edifice/memory/engram.ex` — LSH hash routing
- `lib/edifice/attention/yarn.ex` — RoPE frequency band interpolation
- `lib/edifice/meta/moe_v2.ex` — bias-based load balancing

Suggested process:
1. Open each file + the reference paper
2. Ask Opus: "Verify this Elixir implementation matches the paper's equations. Flag any math errors, missing features, or non-idiomatic patterns."
3. Commit verified files with `verified: true` in @moduledoc metadata

---

## CUDA Kernel Fusion for Recurrent Architectures - 2026-02-18 22:45

- **Explore fused RNN kernels for LSTM/GRU/minGRU/minLSTM** - Plan what's needed to make recurrent architectures competitive on GPU inference latency. **Problem:** Axon unrolls each recurrence timestep as separate EXLA kernel launches, causing 70-600ms latency for seq_len=32 vs 14ms for gated_ssm. TensorFlow/PyTorch use cuDNN's fused `cudnnRNNForward` kernel which handles all timesteps in one GPU call. **Files:** `bench/inference_latency.exs`, `lib/edifice/recurrent/lstm.ex`, `lib/edifice/recurrent/gru.ex`, `lib/edifice/recurrent/min_gru.ex`, `lib/edifice/recurrent/min_lstm.ex`. **Solution:** Investigate four approaches: (1) cuDNN fused RNN integration via EXLA/XLA custom calls, (2) custom CUDA kernels for fused LSTM cells callable from Nx, (3) XLA's built-in RNN fusion passes and whether EXLA exposes them, (4) step-by-step inference with explicit state passing (seq_len=1 per frame) which sidesteps unrolling entirely. Reference: slippi-ai achieves real-time LSTM inference via TensorFlow's cuDNN integration. Benchmark data in `tmp/bench_results/`.
