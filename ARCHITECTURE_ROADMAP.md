# Edifice Architecture Roadmap

113 architectures across 17 families implemented. Remaining targets from the research survey,
organized by priority tier. Check marks indicate completed implementations.

## Tier 1 — High Impact (next release candidates)

<<<<<<< HEAD
- [ ] **Mamba-4 / Hymba** — Hybrid Mamba + attention with learned routing (SSM.Hymba)
- [ ] **RWKV-6/7 (full)** — Updated RWKV with improved token shift and time mixing (Attention.RWKVv6)
- [ ] **DeltaNet v2** — Improved delta rule with gated memory (Recurrent.DeltaNetv2)
- [ ] **xLSTM v2 (sLSTM)** — Scalar LSTM variant with exponential gating (Recurrent.SLSTM)
- [ ] **Sparse Attention** — Block-sparse and sliding window patterns (Attention.SparseAttention)
- [ ] **Hawk/RecurrentGemma** — Google's RG-LRU recurrent model (Recurrent.Hawk)

## Tier 2 — Moderate Impact (architectural diversity)

- [ ] **RetNet v2** — Multi-scale retention with improved decay schedules (Attention.RetNetv2)
- [ ] **GLA v2** — Gated linear attention with improved forget gates (Attention.GLAv2)
- [ ] **HGRN v2** — Hierarchical gated recurrent with multi-resolution (Attention.HGRNv2)
- [ ] **FlashLinearAttention** — Hardware-efficient linear attention kernel (Attention.FlashLinear)
- [ ] **LoRA+/DoRA** — Enhanced parameter-efficient fine-tuning (Meta.DoRA)
- [ ] **DiT v2** — Improved diffusion transformer conditioning (Generative.DiTv2)
- [ ] **Byte Latent Transformer** — Byte-level processing with latent patching (Transformer.BLT)

- [ ] **MEGALODON** — Mega-scale sequence model with chunked attention (Attention.Megalodon)
- [ ] **Mixture of Experts v2** — Expert choice routing, shared expert slots (Meta.MoEv2)

## Tier 3 — Research / Exploratory

- [ ] **Test-Time Compute** — Dynamic inference-time scaling strategies (Meta.TestTimeCompute)
- [ ] **Mixture of Tokenizers** — Multi-granularity tokenization routing (Meta.MoT)
- [ ] **Native Recurrence** — Hardware-native recurrent implementations (Recurrent.NativeRecurrence)
- [ ] **Medusa / EAGLE** — Speculative decoding heads (Meta.SpeculativeHead)
- [ ] **Distillation Head** — Knowledge distillation output head (Meta.DistillationHead)
- [ ] **Quantization-Aware Training** — QAT utilities beyond BitNet (Meta.QAT)
- [ ] **State Space Transformer** — Deeper SSM-attention integration beyond interleaving (SSM.SSTransformer)

## Completed (v0.2.0)

- [x] DecoderOnly — GPT-style transformer (Transformer.DecoderOnly)
- [x] MixtureOfDepths — Dynamic compute per token (Meta.MixtureOfDepths)
- [x] RLHFHead — Reward model head (Meta.RLHFHead)
- [x] KAT — Kolmogorov-Arnold Transformer (Feedforward.KAT)
- [x] mLSTM — Matrix-memory xLSTM variant (alias via Recurrent.XLSTM)
- [x] Based — Taylor expansion linear attention (Attention.Based)
- [x] BitNet — Ternary/binary weight networks (Feedforward.BitNet)
- [x] StripedHyena — Gated conv + Hyena hybrid (SSM.StripedHyena)
- [x] Mega — Moving average gated attention (Attention.Mega)
- [x] Conformer — Conv + Transformer (Attention.Conformer)
- [x] FocalNet — Focal modulation (Vision.FocalNet)
- [x] PoolFormer — Pooling token mixer (Vision.PoolFormer)
- [x] NeRF — Neural radiance field (Vision.NeRF)
- [x] GINv2 — Enhanced GIN with edge features (Graph.GINv2)
- [x] MixtureOfAgents — Multi-model routing (Meta.MixtureOfAgents)
- [x] RingAttention — Distributed sequence attention (Attention.RingAttention)
- [x] InfiniAttention — Compressive memory attention (Attention.InfiniAttention)
- [x] Mamba-3 — Complex states, trapezoidal discretization, MIMO (SSM.Mamba3)
- [x] MLA — Multi-Head Latent Attention (Attention.MLA)
- [x] JEPA — Joint Embedding Predictive Architecture (Contrastive.JEPA)
- [x] Titans — Surprise-gated neural long-term memory (Recurrent.Titans)
- [x] DiffTransformer — Noise-cancelling differential attention (Attention.DiffTransformer)

## Notes

- Priority is based on relevance to ExPhil's Melee AI backbone selection
- Tier 1 focuses on sequence modeling architectures competitive at 60fps inference
- Each implementation follows the standard `Module.build(opts) :: Axon.t()` API
- Current: 113 architectures across 17 families (v0.2.0)
- Target: ~130 architectures by v0.3.0 (17 roadmap items remaining)
- See [guides/architecture_taxonomy.md](guides/architecture_taxonomy.md) for comprehensive taxonomy with paper references, strengths/weaknesses, and adoption context
