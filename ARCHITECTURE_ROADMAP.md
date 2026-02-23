# Edifice Architecture Roadmap

147 architectures across 20 families implemented. All three priority tiers are complete.

## Tier 1 — High Impact ✓

- [x] **Hymba** — Hybrid Mamba + attention with learned routing (SSM.Hymba)
- [x] **Hawk** — Google's RG-LRU recurrent model (Attention.Hawk)
- [x] **xLSTM v2 (sLSTM)** — Scalar LSTM variant with exponential gating (Recurrent.SLSTM)
- [x] **DeltaNet** — Delta rule linear attention (Recurrent.DeltaNet)
- [x] **Gated DeltaNet** — Gated delta rule with improved memory (Recurrent.GatedDeltaNet)
- [x] **Lightning Attention** — Hardware-efficient linear attention (Attention.LightningAttention)
- [x] **Flash Linear Attention** — IO-aware linear attention kernel (Attention.FlashLinearAttention)

## Tier 2 — Moderate Impact ✓

- [x] **RetNet v2** — Multi-scale retention with improved decay schedules (Attention.RetNetV2)
- [x] **GLA v2** — Gated linear attention with improved forget gates (Attention.GLAv2)
- [x] **HGRN v2** — Hierarchical gated recurrent with multi-resolution (Attention.HGRNv2)
- [x] **DoRA** — Weight-decomposed low-rank adaptation (Meta.DoRA)
- [x] **DiT v2** — Improved diffusion transformer conditioning (Generative.DiTv2)
- [x] **Byte Latent Transformer** — Byte-level processing with latent patching (Transformer.ByteLatentTransformer)
- [x] **MEGALODON** — Mega-scale sequence model with chunked attention (Attention.Megalodon)
- [x] **Mixture of Experts v2** — Expert choice routing, shared expert slots (Meta.MoEv2)

## Tier 3 — Research / Exploratory ✓

- [x] **Test-Time Compute** — Backbone + scorer for best-of-N inference scaling (Meta.TestTimeCompute)
- [x] **Mixture of Tokenizers** — Multi-granularity tokenization routing (Meta.MixtureOfTokenizers)
- [x] **Native Recurrence** — ELU-GRU, Real-GRU, diagonal linear variants (Recurrent.NativeRecurrence)
- [x] **Speculative Head (Medusa/EAGLE)** — Multi-head parallel draft decoding (Meta.SpeculativeHead)
- [x] **Distillation Head** — Knowledge distillation projection + KL/MSE losses (Meta.DistillationHead)
- [x] **Quantization-Aware Training** — Binary/ternary/int4/int8 quantized transformer (Meta.QAT)
- [x] **State Space Transformer** — Parallel SSM + attention with learned gating (SSM.SSTransformer)

## Additional Implementations (beyond roadmap)

- [x] TTT End-to-End — Test-time training with full backprop (Recurrent.TTTe2e)
- [x] MMDiT — Multi-modal diffusion transformer (Generative.MMDiT)
- [x] SoFlow — Second-order flow matching (Generative.SoFlow)
- [x] Speculative Decoding — Draft + verifier coordination (Meta.SpeculativeDecoding)

## Completed (v0.2.0 baseline — 113 architectures)

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

## Architecture Summary

| Family | Count | Architectures |
|--------|-------|---------------|
| Attention | 26 | MultiHead, GQA, MLA, DiffTransformer, Perceiver, FNet, LinearTransformer, Nystromformer, Performer, RetNet, RetNetV2, RWKV, GLA, GLAv2, HGRN, HGRNv2, Griffin, Hawk, Based, InfiniAttention, Conformer, Mega, Megalodon, RingAttention, LightningAttention, FlashLinearAttention |
| SSM | 19 | Mamba, MambaSSD, MambaCumsum, MambaHillisSteele, S4, S4D, S5, H3, Hyena, HyenaV2, BiMamba, GatedSSM, Jamba, Zamba, StripedHyena, Mamba3, GSS, Hymba, SSTransformer |
| Meta | 18 | MoE, SwitchMoE, SoftMoE, MoEv2, LoRA, DoRA, Adapter, Hypernetwork, Capsule, MixtureOfDepths, MixtureOfAgents, RLHFHead, SpeculativeDecoding, TestTimeCompute, MixtureOfTokenizers, SpeculativeHead, DistillationHead, QAT |
| Recurrent | 15 | LSTM, GRU, xLSTM, mLSTM, sLSTM, xLSTMv2, MinGRU, MinLSTM, DeltaNet, GatedDeltaNet, TTT, TTTe2e, Titans, Reservoir, NativeRecurrence |
| Generative | 14 | Diffusion, DDIM, DiT, DiTv2, LatentDiffusion, ConsistencyModel, ScoreSDE, FlowMatching, VAE, VQVAE, GAN, NormalizingFlow, MMDiT, SoFlow |
| Vision | 9 | ViT, DeiT, Swin, UNet, ConvNeXt, MLPMixer, FocalNet, PoolFormer, NeRF |
| Graph | 8 | GCN, GAT, GraphSAGE, GIN, GINv2, PNA, GraphTransformer, SchNet |
| Contrastive | 7 | SimCLR, BYOL, BarlowTwins, MAE, VICReg, JEPA, TemporalJEPA |
| Convolutional | 6 | Conv1D, ResNet, DenseNet, TCN, MobileNet, EfficientNet |
| Feedforward | 5 | MLP, KAN, KAT, TabNet, BitNet |
| Transformer | 3 | DecoderOnly, MultiTokenPrediction, ByteLatentTransformer |
| Energy | 3 | EBM, Hopfield, NeuralODE |
| Probabilistic | 3 | Bayesian, MCDropout, Evidential |
| Sets | 2 | DeepSets, PointNet |
| Memory | 2 | NTM, MemoryNetwork |
| Interpretability | 2 | SparseAutoencoder, Transcoder |
| Neuromorphic | 2 | SNN, ANN2SNN |
| Liquid | 1 | Liquid |
| RL | 1 | PolicyValue |
| World Model | 1 | WorldModel |

## Notes

- Priority was based on relevance to ExPhil's Melee AI backbone selection
- Tier 1 focused on sequence modeling architectures competitive at 60fps inference
- Each implementation follows the standard `Module.build(opts) :: Axon.t()` API
- Current: 147 architectures across 20 families
- Roadmap complete — all 22 planned items plus 4 bonus implementations
- See [guides/architecture_taxonomy.md](guides/architecture_taxonomy.md) for comprehensive taxonomy with paper references, strengths/weaknesses, and adoption context
