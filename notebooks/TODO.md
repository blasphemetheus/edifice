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
| **Total** | **~115 of 184** | **36** |

The remaining ~69 atoms are mostly variants (v2s, aliases like `:sana`),
specialized attention patterns (`:yarn`, `:nsa`, `:dual_chunk_attention`),
or niche architectures that appear in the zoo tour but don't need their
own dedicated notebooks.

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
