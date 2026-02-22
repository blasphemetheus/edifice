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

### Language Model Deep-Dives (one key concept per notebook)

- [ ] Tokenization matters — Compare char-level vs word-level vs BPE on the same corpus, show how tokenization affects context window and generation quality
- [ ] Temperature and sampling — Explore greedy, top-k, top-p (nucleus), and temperature sampling side by side, visualize probability distributions
- [ ] Attention visualization — Show what attention heads actually look at, plot attention heatmaps for different layers/heads on example sentences
- [ ] Scaling laws — Train the same architecture at 3-4 different sizes, plot loss vs parameters vs data size, demonstrate when more data beats more parameters
- [ ] Context window and memory — Demonstrate how sequence length affects what the model can learn, show failure modes when context is too short
- [ ] Perplexity and evaluation — Explain perplexity as a metric, compare models by perplexity vs accuracy, show how to evaluate generation quality
