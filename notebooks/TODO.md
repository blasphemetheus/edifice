# Livebook Notebooks — Master Plan

## Current Coverage

12 notebooks covering ~30 of 184 registered architectures.

### Done

- [x] `architecture_zoo.livemd` — Build/inspect tour of all 17 families (no training)
- [x] `architecture_comparison.livemd` — MLP, TabNet, Bayesian, MC Dropout, EBM, Hopfield, SNN, ANN2SNN on 5-class classification with decision boundaries
- [x] `sequence_modeling.livemd` — Mamba, GatedSSM, GRU on sine wave prediction
- [x] `training_mlp.livemd` — First training loop: MLP + TabNet on 3-class clusters
- [x] `graph_classification.livemd` — GCN, GAT, GIN on synthetic graph classification
- [x] `generative_models.livemd` — VAE on 2D crescent moon point cloud
- [x] `small_language_model.livemd` — Decoder-Only + Mamba character-level LM
- [x] `liquid_neural_networks.livemd` — Liquid vs LSTM vs GRU, ODE solver comparison
- [x] `lm_architecture_shootout.livemd` — 8 architectures from 5 families on char-level LM
- [x] `softmax_shootout.livemd` — Softmax vs SSMax vs Softpick normalization comparison
- [x] `demo_for_dad.livemd` — Guided tour for newcomers (MLP, TabNet, MC Dropout, Mamba, GRU, GCN, VAE)
- [x] `index.livemd` — Notebook index (needs update)

---

## Tier 1 — High Impact, Fill Major Gaps

These cover the biggest uncovered families and teach broadly useful ML concepts.
Priority order: build these first.

### 1. `vision_classification.livemd`

**Architectures:** `:vit`, `:convnext`, `:swin`, `:mlp_mixer`, `:resnet`
**Task:** MNIST or CIFAR-10 classification via `scidata`
**What it teaches:**
- Patch embedding (how ViT turns images into sequences)
- CNN vs ViT vs pure-MLP (MLPMixer) comparison
- First notebook with real image data from a standard dataset
- Hierarchical features (Swin's shifted windows, ConvNeXt's modernized convolutions)

**Key sections:** Data loading + visualization, model building, training loop,
accuracy/loss charts, per-class confusion analysis, sample predictions with
confidence scores

### 2. `diffusion_from_scratch.livemd`

**Architectures:** `:diffusion`, `:ddim`, `:flow_matching`, `:consistency_model`
**Task:** 2D point cloud generation (swiss roll or concentric moons)
**What it teaches:**
- Forward noising process visualized step-by-step (t=0 to t=T)
- Reverse denoising — how the model learns to undo noise
- DDPM vs DDIM — deterministic sampling for 10-50x faster generation
- Flow matching as an ODE alternative to the SDE diffusion framework
- Consistency models — single-step generation via consistency training
- The generative family's flagship notebook

**Key sections:** Noise schedule visualization, forward process animation grid,
train denoiser, sample at different timestep counts, quality vs speed chart

### 3. `mixture_of_experts.livemd`

**Architectures:** `:moe`, `:moe_v2`, `:switch_moe`, `:soft_moe`, `:mixture_of_depths`
**Task:** Sequence classification or character-level LM
**What it teaches:**
- Expert routing — how tokens get assigned to experts
- Load balancing — why naive routing leads to expert collapse
- Aux-loss-free routing (MoEv2 bias correction) vs auxiliary loss
- Soft MoE — fully differentiable, no discrete routing decisions
- Mixture of Depths — skip layers for easy tokens, go deep for hard ones
- Why MoE gives you more capacity without proportional compute cost

**Key sections:** Expert utilization heatmaps, routing distribution charts,
capacity factor visualization, comparison of routing strategies

### 4. `lora_and_adapters.livemd`

**Architectures:** `:lora`, `:dora`, `:adapter`, `:qat`, `:bitnet`
**Task:** Fine-tune a frozen base model on a new task
**What it teaches:**
- Parameter-efficient fine-tuning — why you don't need to retrain everything
- LoRA's rank decomposition (W = W_frozen + BA where B,A are tiny)
- DoRA's weight decomposition improvement
- Adapter bottleneck modules for transfer learning
- QAT and BitNet for efficient deployment
- Trainable vs frozen param count comparison

**Key sections:** Frozen param count table, rank vs quality sweep,
LoRA vs DoRA vs full fine-tune comparison chart

### 5. `contrastive_learning.livemd`

**Architectures:** `:simclr`, `:byol`, `:barlow_twins`, `:vicreg`, `:jepa`
**Task:** Synthetic 2D augmentation pairs or simple feature learning
**What it teaches:**
- Evolution: contrastive (need negatives) -> non-contrastive (no negatives) -> predictive (JEPA)
- NT-Xent loss (SimCLR) — why you need large batches
- BYOL — bootstrap without negatives via momentum encoder
- Barlow Twins — redundancy reduction through cross-correlation
- VICReg — variance/invariance/covariance as explicit objectives
- JEPA — predict representations, not pixels

**Key sections:** Learned representation scatter plots (color-coded by class),
representation quality comparison, timeline diagram of the progression

### 6. `rlhf_without_tears.livemd`

**Architectures:** `:rlhf_head`, `:dpo`, `:grpo`, `:kto`
**Task:** Synthetic preference pairs (sentence quality ranking)
**What it teaches:**
- Reward modeling — how to learn human preferences
- DPO — optimize preferences directly without a reward model
- GRPO — group relative policy optimization (DeepSeek-R1's approach)
- KTO — binary good/bad feedback instead of pairwise comparisons
- Why RLHF matters for aligning language models

**Key sections:** Preference pair visualization, reward score distributions,
training curves for each method, win-rate comparison

### 7. `world_model_rl.livemd`

**Architectures:** `:world_model`, `:policy_value`, `:neural_ode`
**Task:** Simple grid-world or CartPole-like environment
**What it teaches:**
- Encode observations into latent space
- Learn latent dynamics (MLP vs NeuralODE vs GRU variants)
- Predict rewards from latent state
- Plan in imagination (latent rollouts without real environment)
- Policy-value networks for actor-critic RL

**Key sections:** Real vs imagined trajectory comparison, latent space
visualization, reward prediction accuracy, policy improvement over episodes

### 8. `audio_pipeline.livemd`

**Architectures:** `:encodec`, `:valle`, `:soundstorm`
**Task:** Synthetic audio (sine waves, chirps, simple waveforms)
**What it teaches:**
- Neural audio codec — how EnCodec compresses waveforms to discrete tokens
- Residual Vector Quantization — multi-codebook token representation
- VALL-E — codec language model for voice synthesis from a short prompt
- SoundStorm — parallel masked generation for fast audio synthesis

**Key sections:** Waveform visualization, codebook utilization chart,
reconstruction quality comparison, token sequence visualization

---

## Tier 2 — Deepen Existing Coverage

These go deeper on families that have some coverage but deserve dedicated exploration.

### 9. `image_generation.livemd`

**Architectures:** `:dit`, `:var`, `:latent_diffusion`, `:linear_dit`, `:mmdit`
**Task:** 2D point clouds or tiny image patches
**What it teaches:**
- DiT — diffusion meets transformers (AdaLN-Zero conditioning)
- VAR — next-scale prediction instead of next-token (coarse to fine)
- Latent diffusion — diffuse in VAE latent space (Stable Diffusion-style)
- LinearDiT/SANA — linear attention for ~100x diffusion speedup
- MMDiT — parallel text+image streams (FLUX.1/SD3 architecture)

### 10. `memory_and_reasoning.livemd`

**Architectures:** `:ntm`, `:memory_network`, `:engram`, `:hopfield`
**Task:** Sequence copy, sort, or associative recall
**What it teaches:**
- Differentiable memory read/write (NTM)
- Multi-hop reasoning (Memory Networks)
- O(1) hash-based associative memory (Engram)
- Modern Hopfield networks with exponential storage capacity

### 11. `scientific_ml.livemd`

**Architectures:** `:fno`, `:neural_ode`
**Task:** Heat equation or 1D wave equation PDE
**What it teaches:**
- Fourier Neural Operator — learn mappings between function spaces
- Operator learning vs pointwise learning
- How FNO handles resolution invariance
- NeuralODE for continuous dynamics

### 12. `hybrid_architectures.livemd`

**Architectures:** `:jamba`, `:zamba`, `:hymba`, `:griffin`, `:nemotron_h`, `:hybrid_builder`
**Task:** Character-level LM (reuse LM shootout corpus)
**What it teaches:**
- Why hybrid SSM+Attention beats pure approaches
- Configurable layer ratios via HybridBuilder
- Jamba's interleaved Mamba+Attention pattern
- Zamba's shared attention layer trick
- Hymba's learnable meta tokens

### 13. `interpretability_pipeline.livemd`

**Architectures:** `:sparse_autoencoder`, `:transcoder`
**Task:** Activations from a small trained transformer
**What it teaches:**
- End-to-end mechanistic interpretability workflow
- Train SAE on model activations
- Identify interpretable dictionary features
- Ablate features to confirm causal role
- Transcoder for cross-layer representation analysis

### 14. `graph_deep_dive.livemd`

**Architectures:** `:graph_sage`, `:pna`, `:graph_transformer`, `:schnet`, `:egnn`
**Task:** Molecular property prediction or social network classification
**What it teaches:**
- GraphSAGE for inductive learning (unseen nodes at test time)
- PNA's multiple aggregators (mean, max, sum, std)
- Full transformer attention over graphs
- SchNet/EGNN for 3D molecular geometry (equivariant networks)

---

## Tier 3 — Specialized / Research-Focused

These target specific audiences or go deep on niche topics.

### 15. `self_supervised_vision.livemd`

**Architectures:** `:mae`, `:dino_v2`, `:siglip`, `:temporal_jepa`
**Task:** Synthetic image patches or simple feature learning
**Teaches:** Masked autoencoding, self-distillation (DINOv2), vision-language
alignment (SigLIP), temporal prediction (V-JEPA)

### 16. `neuromorphic_computing.livemd`

**Architectures:** `:snn`, `:ann2snn`
**Task:** Temporal spike pattern classification
**Teaches:** Spiking neural networks, surrogate gradients, ANN-to-SNN conversion,
biological plausibility, event-driven computation

### 17. `sets_and_point_clouds.livemd`

**Architectures:** `:deep_sets`, `:pointnet`
**Task:** 3D point cloud classification or set cardinality prediction
**Teaches:** Permutation invariance, why order shouldn't matter, T-Net spatial
alignment, DeepSets' sum-decomposition theorem

### 18. `robotics_imitation.livemd`

**Architectures:** `:act`, `:openvla`
**Task:** Synthetic trajectory data (2D robot arm reaching)
**Teaches:** Action chunking for robot control, VLA models, imitation learning
from demonstrations, temporal action consistency

### 19. `multimodal_fusion.livemd`

**Architectures:** `:multimodal_mlp_fusion`, `:transfusion`, `:perceiver`
**Task:** Synthetic multi-modal data (text features + image features)
**Teaches:** MLP projection fusion, cross-attention fusion, Perceiver's
input-agnostic design, Transfusion's unified text+image approach

### 20. `speculative_decoding.livemd`

**Architectures:** `:speculative_decoding`, `:speculative_head`, `:medusa`,
`:test_time_compute`, `:multi_token_prediction`
**Task:** Character-level LM inference optimization
**Teaches:** Draft-verify acceleration, Medusa multi-head parallel drafting,
test-time compute scaling, multi-token prediction. Pure inference — no training.

---

## Tier 4 — LM Concept Deep-Dives

These don't introduce new architectures but teach important concepts using
existing LM infrastructure.

### 21. `tokenization_matters.livemd`

Compare char-level vs word-level vs BPE on the same corpus. Show how
tokenization affects context window, vocabulary size, and generation quality.

### 22. `sampling_strategies.livemd`

Greedy, top-k, top-p (nucleus), and temperature sampling side by side.
Visualize probability distributions at each step of generation.

### 23. `attention_visualization.livemd`

Plot attention heatmaps for different layers/heads on example sentences.
Show what attention heads actually learn to focus on.

### 24. `scaling_laws.livemd`

Train the same architecture at 3-4 different sizes, plot loss vs parameters
vs data size. Demonstrate when more data beats more parameters.

---

## Coverage Summary

| Status | Atoms | Notebooks |
|--------|-------|-----------|
| Already covered | ~30 | 12 |
| Tier 1-2 would add | ~60 | +14 |
| Tier 3-4 would add | ~25 | +10 |
| Per-architecture notebooks | ~89 | +23 |
| **Total** | **184 of 184** | **59** |

With all tiers + per-architecture notebooks, every registered atom has a
dedicated notebook home.

---

## Build Priority

If building sequentially, recommended order:

1. `vision_classification` — biggest gap, real data, broadly interesting
2. `diffusion_from_scratch` — hottest topic in ML, visually stunning
3. `mixture_of_experts` — very timely (every frontier model uses MoE)
4. `lora_and_adapters` — every practitioner needs this
5. `contrastive_learning` — beautiful conceptual progression
6. `rlhf_without_tears` — extremely practical for LLM practitioners
7. `hybrid_architectures` — leverages existing LM shootout infrastructure
8. `image_generation` — extends diffusion notebook to transformers
9. `memory_and_reasoning` — unique and underexplored
10. `interpretability_pipeline` — mechanistic interp is growing fast

Remaining 14 can be interleaved based on interest.

---

## Notebook Conventions (all notebooks follow these)

- Dual setup cells (standalone Mix.install / attached EXLA)
- "What you'll learn" bullets in Introduction
- "What to look for" sections before visualizations
- Heavy code comments explaining WHY, not just what
- IO.puts progress messages for long-running cells
- VegaLite visualizations (use `Vl.concat/2` with `:horizontal` for multi-chart)
- Default 10-15 epochs with comments suggesting more
- Experiment suggestions at the end

---

## Per-Architecture Notebooks

Every architecture in the registry deserves at least a mention in a notebook.
The thematic notebooks above (Tiers 1-4) cover many, but ~89 atoms still lack
dedicated exploration. Below are specialized notebooks grouped by theme, each
covering the remaining uncovered atoms.

### 25. `byte_latent_transformer.livemd`

**Atom:** `:byte_latent_transformer`
**What it covers:** Byte-level processing via an encoder-latent-decoder triple.
Raw bytes go in, get compressed to latent patches, processed by a transformer,
then decoded back. No tokenizer needed.
**Task:** Character-level text — feed raw UTF-8 bytes, compare to char-level
decoder-only transformer on the same corpus.
**Key demo:** Visualize the encoder's patch boundaries, show how the model
learns to group bytes into meaningful units, compare perplexity with vs
without the byte-level bottleneck.

### 26. `kan_and_kat.livemd`

**Atoms:** `:kan`, `:kat`
**What it covers:** Kolmogorov-Arnold Networks put learnable activations on
edges instead of nodes. KAT puts KAN inside a transformer's FFN sublayer.
**Task:** Function approximation — fit a complex 1D or 2D function (e.g.
`sin(x) * cos(3x)`), compare KAN vs MLP on interpolation AND extrapolation.
Then KAT vs standard transformer on sequence classification.
**Key demo:** Plot the learned activation functions on each edge. Show that
KAN can extrapolate beyond training range where MLP fails. Visualize edge
sparsity.

### 27. `convolutional_zoo.livemd`

**Atoms:** `:conv1d`, `:densenet`, `:tcn`, `:mobilenet`, `:efficientnet`
**What it covers:** The full convolutional family beyond ResNet.
**Task:** Image classification (MNIST or synthetic) for DenseNet/MobileNet/
EfficientNet. Sequence classification for Conv1D/TCN.
**Key demo:** Parameter count vs accuracy scatter plot. Show DenseNet's
dense connectivity pattern diagram. TCN's dilated receptive field growth.
MobileNet's depthwise separable factorization (param savings). EfficientNet's
compound scaling (depth x width x resolution).

### 28. `modern_recurrent.livemd`

**Atoms:** `:mlstm`, `:slstm`, `:xlstm_v2`, `:min_lstm`, `:native_recurrence`
**What it covers:** The modern recurrent renaissance — minimal and extended
LSTM variants that are parallel-scannable or have matrix memory.
**Task:** Char-level LM (reuse LM shootout corpus) with all 5 + GRU/LSTM
baselines from existing notebooks.
**Key demo:** Loss curves for all variants, highlight that MinLSTM matches
LSTM quality with far fewer parameters. Show mLSTM's matrix memory capacity
vs sLSTM's scalar memory. xlstm_v2 scaling improvements.

### 29. `delta_rule_attention.livemd`

**Atoms:** `:delta_net`, `:gated_delta_net`
**What it covers:** Linear attention with delta-rule weight updates — the
model maintains an associative memory that gets updated with each token.
**Task:** Associative recall — present key-value pairs, then query a key
and expect the correct value. This directly tests the delta-rule memory.
**Key demo:** Compare DeltaNet vs standard attention on recall accuracy
as the number of stored pairs grows. Show that GatedDeltaNet's data-dependent
gating improves selective forgetting.

### 30. `test_time_training.livemd`

**Atoms:** `:ttt`, `:ttt_e2e`
**What it covers:** TTT layers perform self-supervised weight updates at
inference time — the model literally trains itself on the input sequence
as it processes it.
**Task:** Long-context sequence with distributional shift — the beginning
of the sequence follows one pattern, the end follows another. TTT should
adapt in real-time.
**Key demo:** Compare TTT vs frozen transformer on a sequence where the
"rules" change midway. Visualize how TTT's internal weights evolve during
inference. Show TTT-E2E's end-to-end gradient flow.

### 31. `titans_long_memory.livemd`

**Atom:** `:titans`
**What it covers:** Neural long-term memory with surprise-gated updates —
the model stores information when it encounters something unexpected.
**Task:** Long sequence with rare important events (e.g. a signal token
that appears once every 100 positions).
**Key demo:** Visualize the surprise gate activations — when does the model
decide something is worth remembering? Compare retrieval accuracy vs
sequence length against standard attention and LSTM.

### 32. `reservoir_computing.livemd`

**Atom:** `:reservoir`
**What it covers:** Echo State Networks — a fixed random recurrent network
where only the readout layer is trained. Extremely fast training.
**Task:** Chaotic time series prediction (Lorenz attractor or Mackey-Glass).
**Key demo:** Show that the reservoir captures dynamics without backprop
through time. Compare training time vs quality against LSTM/GRU. Visualize
the echo state property — how information decays in the reservoir.

### 33. `ssm_family_tree.livemd`

**Atoms:** `:s4d`, `:s5`, `:h3`, `:gss`, `:ss_transformer`
**What it covers:** The S4 family beyond vanilla S4 — diagonal simplification
(S4D), MIMO extension (S5), language-focused H3, gated variant GSS, and the
SSM+attention hybrid (SSTransformer).
**Task:** Char-level LM shootout restricted to SSM variants.
**Key demo:** S4 vs S4D (does diagonal lose quality?). S5's single-layer MIMO
vs S4's multiple SISO. H3's multiplicative gating. GSS's additive simplicity.
Loss curves + perplexity table for all 5.

### 34. `hyena_long_convolution.livemd`

**Atoms:** `:hyena`, `:hyena_v2`, `:striped_hyena`
**What it covers:** Implicit long convolutions as an attention alternative —
O(L log L) via FFT instead of O(L^2).
**Task:** Long-range sequence task (e.g. ListOps-style or long-context LM).
**Key demo:** Visualize the learned implicit convolution filters. Show
frequency response of the filters. Compare Hyena vs Hyena v2 filter quality.
Striped Hyena's interleaved architecture diagram.

### 35. `mamba_variants.livemd`

**Atoms:** `:mamba_ssd`, `:mamba3`, `:mamba_cumsum`, `:mamba_hillis_steele`, `:bimamba`
**What it covers:** The full Mamba family — SSD (tensor-core efficient),
Mamba-3 (complex states), scan algorithm variants, and bidirectional Mamba.
**Task:** Causal LM for Mamba/SSD/Mamba3, bidirectional classification for BiMamba.
**Key demo:** Scan algorithm comparison (cumsum vs Hillis-Steele — work vs
depth tradeoff). Mamba-3's complex state dynamics visualization. BiMamba on
a task where future context helps (e.g. sentiment classification).

### 36. `linear_attention_landscape.livemd`

**Atoms:** `:linear_transformer`, `:performer`, `:nystromformer`, `:based`,
`:flash_linear_attention`, `:lightning_attention`
**What it covers:** All the ways to approximate or replace quadratic softmax
attention with linear-time alternatives.
**Task:** Sequence modeling or char-level LM — same task, 6 different
linear attention methods.
**Key demo:** Quality vs speed scatter plot. Show how each method approximates
the full attention matrix (feature maps, Nystrom landmarks, Taylor expansion).
Lightning Attention's intra/inter block split diagram.

### 37. `retention_and_gated_rnn.livemd`

**Atoms:** `:retnet`, `:retnet_v2`, `:gla`, `:gla_v2`, `:hgrn`, `:hgrn_v2`
**What it covers:** The linear-recurrence family — models that train in
parallel like transformers but infer in O(1) like RNNs.
**Task:** Char-level LM with parallel training, then demonstrate O(1)
recurrent inference mode.
**Key demo:** Multi-scale retention visualization (RetNet). GLA's data-dependent
decay heatmap. HGRN's hierarchical gating across layers. Show the parallel
vs recurrent mode equivalence.

### 38. `attention_efficiency.livemd`

**Atoms:** `:gqa`, `:mla`, `:ring_attention`, `:infini_attention`,
`:dual_chunk_attention`
**What it covers:** Efficiency improvements to standard attention — fewer
KV heads, compressed KV cache, distributed computation, infinite context.
**Task:** Build models at increasing sequence lengths, measure memory usage
and throughput.
**Key demo:** GQA's KV head sharing (4 heads share 1 KV = 4x KV-cache
savings). MLA's low-rank compression ratio. InfiniAttention's compressive
memory growing with context. Ring Attention's chunk distribution diagram.

### 39. `specialized_attention.livemd`

**Atoms:** `:diff_transformer`, `:fnet`, `:conformer`, `:mega`, `:megalodon`,
`:gated_attention`, `:hawk`, `:kda`
**What it covers:** Attention mechanisms designed for specific use cases.
**Task:** Varies per architecture — audio features for Conformer, noise
cancellation demo for DiffTransformer, FNet's FFT replacement.
**Key demo:** DiffTransformer's dual-softmax noise cancellation visualized.
FNet's pure-FFT attention (no learned parameters!). Conformer's
conv-attention sandwich for audio. Mega/Megalodon's exponential moving
average visualization. Hawk as recurrence-only Griffin.

### 40. `positional_encoding.livemd`

**Atoms:** `:yarn`, `:nsa`, `:rnope_swa`, `:tmrope`
**What it covers:** Advanced positional encoding and context-window strategies.
**Task:** Train a small LM, then test on sequences longer than training length.
**Key demo:** YaRN's frequency scaling for context extension. RNoPE-SWA's
no-positional-encoding approach. NSA's three-path sparse attention pattern.
TMRoPE's multimodal position alignment.

### 41. `vision_architectures.livemd`

**Atoms:** `:deit`, `:unet`, `:focalnet`, `:poolformer`, `:mamba_vision`,
`:metaformer`, `:caformer`, `:efficient_vit`
**What it covers:** The full vision family beyond ViT/ConvNeXt/Swin.
**Task:** Image classification for most; segmentation for UNet.
**Key demo:** DeiT's distillation token mechanism. PoolFormer proving that
the MetaFormer structure matters more than the mixer. FocalNet's multi-scale
focal modulation. MambaVision's hybrid CNN+Mamba+Attention. CAFormer's
conv-to-attention stage transition. EfficientViT's linear attention.

### 42. `3d_neural_rendering.livemd`

**Atoms:** `:nerf`, `:gaussian_splat`
**What it covers:** Neural 3D scene representation and rendering.
**Task:** Synthetic 3D scene (spheres/cubes) — train NeRF and Gaussian
Splatting to render novel views.
**Key demo:** Novel view synthesis from trained models. NeRF's coordinate
MLP (position -> color + density). Gaussian Splatting's explicit 3D
Gaussians vs NeRF's implicit field. Rendering speed comparison.

### 43. `generative_deep_dive.livemd`

**Atoms:** `:vq_vae`, `:gan`, `:normalizing_flow`, `:score_sde`, `:dit_v2`,
`:soflow`, `:sit`
**What it covers:** Generative models beyond basic diffusion — discrete
tokens (VQ-VAE), adversarial (GAN), invertible (flows), score-based (SDE).
**Task:** 2D point cloud generation for all, compare sample quality and
diversity.
**Key demo:** VQ-VAE's codebook utilization heatmap. GAN's generator vs
discriminator loss dynamics (the training dance). Normalizing flow's
invertible transformations visualized. Score SDE's VP vs VE comparison.
SiT's interpolant generalization of DiT.

### 44. `video_and_3d_generation.livemd`

**Atoms:** `:cogvideox`, `:trellis`, `:mar`, `:sana`
**What it covers:** Generation beyond 2D images — video, 3D assets, and
masked autoregressive approaches.
**Task:** Synthetic sequential frames for CogVideoX, 3D lattice for TRELLIS,
2D for MAR/SANA.
**Key demo:** CogVideoX's 3D causal VAE temporal compression. TRELLIS's
sparse lattice structure. MAR's bridge between autoregressive and masked
prediction. SANA as an efficiency-focused LinearDiT variant.

### 45. `graph_edges_and_molecules.livemd`

**Atom:** `:gin_v2`
**What it covers:** GINv2 adds edge features to the maximally-expressive GIN.
**Task:** Molecular property prediction where bond types (edge features) matter.
**Key demo:** Compare GIN vs GINv2 on a molecular task — show that edge
features improve accuracy on bond-dependent properties. Visualize molecular
graphs with colored edges.

### 46. `evidential_uncertainty.livemd`

**Atom:** `:evidential`
**What it covers:** Evidential deep learning — place a Dirichlet prior over
class probabilities for principled uncertainty without MC sampling.
**Task:** Classification with out-of-distribution detection.
**Key demo:** Compare Evidential vs MC Dropout uncertainty estimates.
Visualize the Dirichlet concentration parameters. Show that evidential
uncertainty is high for OOD inputs without needing multiple forward passes.

### 47. `meta_learning_advanced.livemd`

**Atoms:** `:hypernetwork`, `:capsule`, `:mixture_of_agents`,
`:mixture_of_tokenizers`, `:distillation_head`
**What it covers:** Advanced meta-learning and composition patterns.
**Task:** Hypernetwork generating weights for a tiny target network. Capsule
dynamic routing on simple shapes. MoA ensemble. Distillation from a larger
to smaller model.
**Key demo:** Hypernetwork's weight generation visualized (one network
controls another). Capsule routing coefficients showing part-whole
relationships. MoA's proposer-aggregator pipeline. Knowledge distillation
loss curves (student approaching teacher quality).

---

## Original Ideas Archive

These ideas were in the original TODO.md and are preserved here. Many are
now covered by the tiered plan above, but keeping them for reference and
any unique details.

### Architecture Walkthroughs (original list)

- [ ] World Model — Encode observations, learn latent dynamics (MLP vs NeuralODE vs GRU), predict rewards. Train on a simple grid-world or CartPole-like environment, visualize latent trajectories and imagined rollouts
- [ ] RL PolicyValue + Environment — Build a policy-value network, implement the Environment behaviour, run PPO-style training on a toy environment (e.g. bandit or cliff-walking). Show policy improvement over episodes
- [ ] Lightning Attention — Compare standard softmax attention vs Lightning Attention on a sequence task, benchmark the speed/memory tradeoff, visualize intra-block vs inter-block attention contributions
- [ ] Sparse Autoencoder — Train an SAE on activations from a pre-trained model, visualize the learned dictionary features, show top-k sparsity in action, demonstrate how L1 coefficient affects feature quality
- [ ] Transcoder — Train a cross-layer transcoder, show how representations transform between layers, compare to a regular SAE
- [ ] Temporal JEPA (V-JEPA) — Mask timesteps from a sequence, predict masked representations, show EMA target divergence, compare to pixel-level reconstruction (MAE-style)
- [ ] iRoPE — Compare standard RoPE vs interleaved RoPE (odd/even layers) on a language modeling task, show how NoPE layers learn different attention patterns
- [ ] Aux-loss-free MoE — Visualize expert utilization with and without the load-balance bias, show how the bias corrects routing imbalance without auxiliary loss
- [ ] Image Generation Paradigms — VAR vs DiT vs consistency model vs flow matching on 2D data
- [ ] Self-Supervised Vision — DINOv2 vs MAE vs SimCLR vs JEPA feature quality comparison
- [ ] RLHF Without Tears — DPO vs GRPO vs KTO on simple preference tasks
- [ ] Audio from Scratch — EnCodec tokenization + VALL-E generation pipeline
- [ ] Unified Multimodal — Transfusion: one model for text + images
- [ ] Scientific ML — FNO for solving simple PDEs vs neural ODE

### Other Module Candidates (original list)

- [ ] Mechanistic interpretability pipeline — End-to-end: train a small transformer, extract activations, train SAE, identify interpretable features, ablate them to confirm causal role
- [ ] Model-based RL loop — Full Dreamer-style pipeline: train world model on environment interactions, plan in latent space, compare model-based vs model-free sample efficiency
- [ ] Contrastive learning evolution — SimCLR -> BYOL -> BarlowTwins -> VICReg -> JEPA -> Temporal JEPA, the progression from contrastive to non-contrastive to predictive
- [ ] Hybrid attention architectures — Combine Lightning Attention with Mamba blocks (like Jamba/Hymba), show how hybrid models get the best of both worlds

### Language Model Deep-Dives (original list)

- [ ] Tokenization matters — Compare char-level vs word-level vs BPE on the same corpus, show how tokenization affects context window and generation quality
- [ ] Temperature and sampling — Explore greedy, top-k, top-p (nucleus), and temperature sampling side by side, visualize probability distributions
- [ ] Attention visualization — Show what attention heads actually look at, plot attention heatmaps for different layers/heads on example sentences
- [ ] Scaling laws — Train the same architecture at 3-4 different sizes, plot loss vs parameters vs data size, demonstrate when more data beats more parameters
- [ ] Context window and memory — Demonstrate how sequence length affects what the model can learn, show failure modes when context is too short
- [ ] Perplexity and evaluation — Explain perplexity as a metric, compare models by perplexity vs accuracy, show how to evaluate generation quality
