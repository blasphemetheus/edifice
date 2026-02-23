# Livebook Notebooks TODO

## Done

- [x] Architecture zoo tour — `architecture_zoo.livemd`
- [x] Sequence modeling — `sequence_modeling.livemd`
- [x] Architecture comparison — `architecture_comparison.livemd`

## Planned

- [ ] Vision — Train a ViT or ConvNeXt on MNIST/CIFAR-10 via scidata
- [x] Graph classification — `graph_classification.livemd`
- [x] Generative models — `generative_models.livemd`
- [ ] Diffusion from scratch — Build a tiny diffusion model on 2D data, show denoising step by step
- [ ] LoRA/Adapter fine-tuning — Show meta-learning adapters wrapping a frozen backbone
- [x] Small language model — `small_language_model.livemd`

### New Architecture Walkthroughs (Tier 1 Build)

- [ ] World Model — Encode observations, learn latent dynamics (MLP vs NeuralODE vs GRU), predict rewards. Train on a simple grid-world or CartPole-like environment, visualize latent trajectories and imagined rollouts
- [ ] RL PolicyValue + Environment — Build a policy-value network, implement the Environment behaviour, run PPO-style training on a toy environment (e.g. bandit or cliff-walking). Show policy improvement over episodes
- [ ] Lightning Attention — Compare standard softmax attention vs Lightning Attention on a sequence task, benchmark the speed/memory tradeoff, visualize intra-block vs inter-block attention contributions
- [ ] Sparse Autoencoder — Train an SAE on activations from a pre-trained model, visualize the learned dictionary features, show top-k sparsity in action, demonstrate how L1 coefficient affects feature quality
- [ ] Transcoder — Train a cross-layer transcoder, show how representations transform between layers, compare to a regular SAE
- [ ] Temporal JEPA (V-JEPA) — Mask timesteps from a sequence, predict masked representations, show EMA target divergence, compare to pixel-level reconstruction (MAE-style)
- [ ] iRoPE — Compare standard RoPE vs interleaved RoPE (odd/even layers) on a language modeling task, show how NoPE layers learn different attention patterns
- [ ] Aux-loss-free MoE — Visualize expert utilization with and without the load-balance bias, show how the bias corrects routing imbalance without auxiliary loss

### Other New Module Candidates

- [ ] Mechanistic interpretability pipeline — End-to-end: train a small transformer, extract activations, train SAE, identify interpretable features, ablate them to confirm causal role
- [ ] Model-based RL loop — Full Dreamer-style pipeline: train world model on environment interactions, plan in latent space, compare model-based vs model-free sample efficiency
- [ ] Contrastive learning evolution — SimCLR → BYOL → BarlowTwins → VICReg → JEPA → Temporal JEPA, the progression from contrastive to non-contrastive to predictive
- [ ] Hybrid attention architectures — Combine Lightning Attention with Mamba blocks (like Jamba/Hymba), show how hybrid models get the best of both worlds

### Language Model Deep-Dives (one key concept per notebook)

- [ ] Tokenization matters — Compare char-level vs word-level vs BPE on the same corpus, show how tokenization affects context window and generation quality
- [ ] Temperature and sampling — Explore greedy, top-k, top-p (nucleus), and temperature sampling side by side, visualize probability distributions
- [ ] Attention visualization — Show what attention heads actually look at, plot attention heatmaps for different layers/heads on example sentences
- [ ] Scaling laws — Train the same architecture at 3-4 different sizes, plot loss vs parameters vs data size, demonstrate when more data beats more parameters
- [ ] Context window and memory — Demonstrate how sequence length affects what the model can learn, show failure modes when context is too short
- [ ] Perplexity and evaluation — Explain perplexity as a metric, compare models by perplexity vs accuracy, show how to evaluate generation quality
