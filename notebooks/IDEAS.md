# Livebook Notebook Ideas

A brainstorm of demonstrations that could be built as `.livemd` notebooks for Edifice.
Organized by problem space / theme. Items marked with `*` are already done or planned in TODO.md.

---

## 1. Language & Text

- **Small character-level language model** `*planned` — Train a decoder-only transformer on a small text corpus (Shakespeare, Elixir code, etc.), generate text with temperature sampling
- **Architecture shootout for text** — Compare transformer vs Mamba vs RWKV vs GRU on next-character prediction (perplexity, generation quality, speed)
- **Byte-level tokenizer + model** — Build a tiny BPE or character-level tokenizer, then train a model end-to-end, showing the full pipeline
- **Text classification** — Sentiment or topic classification using attention-based architectures on synthetic labeled sequences
- **Sequence-to-sequence toy translation** — Encoder-decoder on a synthetic "language" (e.g., reverse a string, simple cipher), demonstrating seq2seq without real NLP data
- **Autoregressive generation deep dive** — Greedy vs top-k vs nucleus sampling, temperature effects, visualized token probabilities
- **Repetition and memory in LMs** — Show how transformer vs Mamba vs RWKV handle long-range repetition tasks (copy, reverse, lookup)

## 2. Vision

- **Image classification** `*planned` — Train ViT or ConvNeXt on MNIST/CIFAR-10 via scidata
- **CNN evolution** — Train Conv → ResNet → DenseNet → MobileNet → EfficientNet on same dataset, compare accuracy/param count
- **ViT vs CNN showdown** — ViT vs ConvNeXt vs ResNet on MNIST, showing that architecture choice matters even on simple image tasks
- **Patch embedding visualization** — Show how ViT chops an image into patches, visualize attention maps across patches
- **U-Net segmentation** — Synthetic image segmentation task (circles/squares on background), train U-Net, visualize masks
- **MLP-Mixer on images** — Demonstrate the "no attention, no convolution" approach on a small image dataset
- **Swin Transformer windows** — Visualize the shifted window mechanism on a small image
- **FocalNet vs PoolFormer** — Compare token-mixing strategies that aren't attention
- **NeRF basics (2D)** — Simplified 2D neural radiance field: learn a continuous image representation from sparse samples

## 3. Sequence & Time Series

- **Multi-architecture sequence benchmark** — Extend existing sequence notebook to all sequence-capable families: SSM (S4, Mamba, Hyena), RNN (LSTM, GRU, xLSTM, MinGRU), Attention (RetNet, RWKV), Liquid
- **Multi-step forecasting** — Predict multiple future values instead of one, compare autoregressive rollout vs direct multi-output
- **Anomaly detection in time series** — Train on normal patterns, detect anomalies via reconstruction error or prediction confidence
- **Financial/synthetic market data** — Generate synthetic stock-like data, predict trends, visualize predictions vs actuals with confidence bands
- **Audio/signal processing** — Synthetic audio signals (superposed sine waves), classify or denoise using Conformer or Mega
- **Chaotic systems** — Train on Lorenz attractor or logistic map, show where models succeed and fail at predicting chaos
- **Variable-length sequences** — Demonstrate padding/masking strategies, show how different architectures handle varying sequence lengths

## 4. Generative Models

- **Diffusion from scratch** `*planned` — Build tiny diffusion model on 2D data, show denoising step by step
- **GAN on 2D distributions** — Train a GAN on 2D point clouds (rings, spirals), visualize generator vs discriminator dynamics, mode collapse
- **VQ-VAE discrete codes** — Show how VQ-VAE learns a discrete codebook, visualize code usage and reconstruction
- **Flow matching** — Demonstrate flow matching on 2D data, show the learned vector field and transport paths
- **Normalizing flows** — Visualize invertible transformations warping a Gaussian into a complex distribution
- **Score-based / SDE** — Show the score function, Langevin dynamics sampling, noise schedule effects
- **Consistency model** — Single-step generation, compare quality vs diffusion multi-step
- **Conditional generation** — Class-conditional VAE or diffusion: generate specific classes on demand
- **Latent space interpolation** — Walk through VAE/VQ-VAE latent space, show smooth transitions between data points
- **DiT (Diffusion Transformer)** — Compare U-Net-based vs transformer-based diffusion on same 2D task
- **Latent diffusion pipeline** — Encode → diffuse in latent space → decode, the full stable-diffusion-style pipeline on toy data

## 5. Graph Neural Networks

- **Node classification** — Zachary's karate club or synthetic community detection, per-node labels
- **Molecular property prediction** — Synthetic molecules (atoms as nodes, bonds as edges), predict properties with SchNet
- **GNN message passing visualization** — Step through GCN/GAT/GIN layers, visualize how node features propagate and aggregate
- **Graph generation** — Train a model to generate valid graph structures
- **Heterogeneous graphs** — Different node/edge types, demonstrate typed message passing
- **Graph-level vs node-level tasks** — Same GNN backbone, different pooling and heads, comparing the two paradigms
- **Scalability: PNA vs GCN vs GAT** — Compare message-passing strategies on graphs of increasing size
- **GraphTransformer** — Show how attention operates over graph structure vs sequential position

## 6. Self-Supervised & Contrastive Learning

- **SimCLR on synthetic data** — Learn representations from augmented views without labels, then linear probe
- **BYOL vs BarlowTwins vs VICReg** — Compare contrastive objectives on same data, visualize learned embedding spaces
- **MAE (Masked Autoencoder)** — Mask portions of input, reconstruct, show what the model learns to represent
- **JEPA demonstration** — Joint-embedding predictive architecture, show prediction in representation space vs pixel space
- **Representation quality evaluation** — Train contrastive model, freeze backbone, evaluate on downstream tasks with linear probes
- **Augmentation sensitivity** — How do different augmentation strategies affect contrastive learning quality?

## 7. Meta-Learning & Efficiency

- **LoRA/Adapter fine-tuning** `*planned` — Meta-learning adapters wrapping a frozen backbone
- **Mixture of Experts routing** — Visualize which experts activate for which inputs, show load balancing
- **Switch MoE vs Soft MoE** — Compare hard routing vs soft routing, visualize expert utilization
- **Mixture of Depths** — Show dynamic per-token compute allocation, which tokens get more/less processing
- **Hypernetwork** — A network that generates weights for another network, demonstrate on a simple task
- **Capsule network routing** — Visualize the iterative routing-by-agreement mechanism
- **LoRA rank ablation** — Same task with different LoRA ranks, show the accuracy-efficiency tradeoff
- **Adapter composition** — Stack multiple adapters for multi-task learning, show task-specific behavior
- **Mixture of Agents** — Multiple model instances collaborating, demonstrate ensemble-like behavior

## 8. Uncertainty & Probabilistic

- **Bayesian uncertainty quantification** — Train Bayesian NN, show predictive uncertainty grows away from training data
- **MC Dropout as cheap Bayesian** — Compare MC Dropout uncertainty to Bayesian NN uncertainty on same task
- **Evidential deep learning** — Learn uncertainty directly as output, compare to sampling-based methods
- **Out-of-distribution detection** — Train on one distribution, show uncertainty spikes on OOD inputs
- **Calibration curves** — Are model confidences well-calibrated? Plot reliability diagrams
- **Prediction intervals** — Regression with uncertainty: show predicted bands, coverage analysis
- **Uncertainty-aware decision making** — Use uncertainty to abstain from low-confidence predictions, improve accuracy on remaining

## 9. Energy-Based & Dynamic Systems

- **Energy landscape visualization** — Train EBM, plot energy surface, show how low-energy regions correspond to data
- **Hopfield associative memory** — Store patterns, demonstrate pattern completion from partial/noisy input
- **Modern Hopfield vs classical** — Compare storage capacity and retrieval quality
- **Neural ODE trajectories** — Visualize learned continuous dynamics, compare to discrete residual networks
- **Phase portraits** — Neural ODE learned vector fields overlaid on data trajectories
- **Liquid neural network dynamics** — Continuous-time ODE neurons, show how dynamics adapt to input
- **Attractor landscapes** — Train energy models with multiple attractors, visualize basins of attraction

## 10. Neuromorphic & Spiking

- **Spiking neural network basics** — Leaky integrate-and-fire neurons, visualize membrane potentials and spike trains
- **ANN-to-SNN conversion** — Train a standard ANN, convert to SNN, compare accuracy and show temporal coding
- **Temporal coding** — Demonstrate information encoding in spike timing vs rate coding
- **Event-driven processing** — Show how SNNs naturally handle event-based (sparse temporal) data
- **Energy efficiency argument** — Compare "compute" (multiply-accumulate ops) between SNN and ANN on same task

## 11. Memory & Associative

- **Neural Turing Machine** — Train NTM on copy task, visualize read/write head attention over memory
- **Memory network for Q&A** — Simple factoid storage and retrieval, show attention over memory slots
- **Content-addressable memory** — Demonstrate similarity-based memory lookup and retrieval
- **Algorithmic tasks** — Copy, reverse, sort — show which memory architectures learn algorithmic patterns

## 12. Attention Mechanism Deep Dives

- **Attention visualization gallery** — MultiHead, GQA, Linear, Retention side by side on same input
- **Positional encoding showdown** — RoPE vs ALiBi vs Sinusoidal vs Learned: how they encode position, extrapolation behavior
- **Linear attention approximations** — Show how linear attention trades quality for speed on a sequence task
- **RetNet retention patterns** — Visualize the exponential decay retention mechanism
- **GQA efficiency** — Group query attention: same quality, fewer KV heads, show the memory savings
- **DiffTransformer** — Differential attention mechanism visualization
- **Long-context attention** — Ring Attention and InfiniAttention for sequences beyond single-context length

## 13. Building Blocks & Fundamentals

- **Normalization techniques** — LayerNorm vs RMSNorm vs BatchNorm: when each helps, training stability comparison
- **Activation functions** — SwiGLU vs GELU vs ReLU: gradient flow, training dynamics
- **Residual connections** — With vs without skip connections: gradient flow visualization, training depth limits
- **Loss landscape exploration** — Visualize 2D slices of the loss surface for different architectures
- **Optimizer comparison** — SGD vs Adam vs AdamW: convergence speed, generalization
- **Learning rate schedules** — Constant vs cosine vs warmup: training curves comparison
- **Batch size effects** — Small vs large batch: noise, convergence, generalization tradeoffs
- **Overfitting clinic** — Deliberately overfit, then show regularization techniques (dropout, weight decay, early stopping, data augmentation)
- **Gradient flow analysis** — Monitor gradient magnitudes through layers during training, show vanishing/exploding gradients

## 14. Architecture Evolution Stories

- **RNN evolution** — SimpleRNN → LSTM → GRU → xLSTM → MinGRU → DeltaNet → TTT, trained on same task, showing progressive improvements
- **CNN evolution** — Conv → ResNet → DenseNet → MobileNet → EfficientNet, param count vs accuracy trajectory
- **From attention to state space** — MHA → Linear Attention → RetNet → S4 → Mamba, the journey from quadratic to linear
- **VAE → VQ-VAE → Diffusion** — Evolution of generative approaches, quality vs training stability
- **Transformer to efficient transformer** — Standard → GQA → MoE → MoD → BitNet, making transformers cheaper
- **The Mamba family** — S4 → S4D → S5 → H3 → Hyena → Mamba → Mamba-2 → Mamba-3, SSM evolution

## 15. Interpretability

- **Sparse Autoencoder feature discovery** — Train a small transformer, extract activations, train an SAE on them. Visualize the sparse dictionary: which features fire for which inputs? Show top-k vs L1 sparsity modes
- **Transcoder cross-layer analysis** — Train a transcoder between two layers of a model, show how it captures the transformation between representations. Compare reconstruction quality to a same-layer SAE
- **Feature ablation study** — After training an SAE, zero out individual learned features and observe the effect on model output. Demonstrate causal interpretability
- **Dictionary size ablation** — Same model, SAEs with dict_size 64 vs 256 vs 1024 vs 4096. Show how overcomplete dictionaries find finer-grained features
- **Superposition and polysemanticity** — Demonstrate how neurons in small models represent multiple concepts, and how SAEs disentangle them into monosemantic features

## 16. World Models & Model-Based RL

- **World model basics** — Train encoder + dynamics + reward head on a simple environment (CartPole or grid-world). Visualize latent space, show predicted vs actual trajectories
- **Dynamics model comparison** — MLP vs Neural ODE vs GRU dynamics on the same environment. Compare rollout stability, multi-step prediction error, and training speed
- **Imagination and planning** — Train a world model, then "dream" trajectories in latent space. Show how imagined experience can train a policy without real environment interaction
- **World model reconstruction** — Enable the decoder, train with reconstruction loss. Visualize decoded imagined observations vs real observations
- **Latent space structure** — Visualize the learned latent space with t-SNE/PCA. Show that similar observations cluster together, and dynamics follow smooth paths

## 17. Reinforcement Learning

- **PolicyValue from scratch** — Build a discrete policy-value network, implement a simple environment (multi-armed bandit), train with REINFORCE. Visualize policy convergence
- **PPO on a toy environment** — Full PPO loop: rollout collection, GAE advantage estimation, clipped surrogate update. Show reward curves and policy improvement over episodes
- **Discrete vs continuous actions** — Same task framed as discrete (grid navigation) and continuous (reaching). Show how action_type changes the policy head and loss
- **Environment behaviour showcase** — Implement 3 environments (bandit, cliff-walking, CartPole-like) using the Environment behaviour, show the common interface
- **Value function visualization** — Plot the learned value function over the state space, show how it guides policy improvement
- **Model-based vs model-free** — Train PolicyValue model-free vs train WorldModel + plan. Compare sample efficiency and final performance

## 18. Lightning Attention & Efficient Transformers

- **Block attention visualization** — Show how Lightning Attention splits sequences into blocks, visualize intra-block softmax patterns vs inter-block linear attention contributions
- **Complexity benchmark** — Standard attention vs Lightning Attention at increasing sequence lengths. Plot wall-clock time and memory usage, show the linear scaling advantage
- **Lightning vs Linear vs Performer** — Three approaches to efficient attention on the same sequence task. Compare quality-speed tradeoffs
- **Long-context performance** — Tasks that require long-range dependencies (copy, retrieval). Show where Lightning Attention's inter-block mechanism helps vs pure local attention

## 19. Applied / Cross-Cutting

- **Tabular data with TabNet** — Structured data classification, visualize feature attention masks
- **Ensemble methods** — Combine predictions from multiple architectures, show ensemble > individual
- **Transfer learning workflow** — Pre-train on one task, fine-tune on another, show the benefit
- **Model compression comparison** — Full model vs BitNet vs pruned: accuracy-size tradeoff
- **Inference latency profiling** — Benchmark wall-clock time across architectures, CPU vs GPU
- **Architecture search** — Use Edifice registry to programmatically sweep architectures, find best for a task
- **Multi-task learning** — Single backbone, multiple heads, show how shared representations help
- **Curriculum learning** — Start with easy examples, gradually increase difficulty, compare to random ordering
- **Data efficiency** — How much data does each architecture need? Learning curves with varying dataset sizes
- **Reproducibility** — Same architecture, same data, different random seeds: variance in outcomes

## 20. Fun & Creative

- **Cellular automata learner** — Train a model to predict Game of Life or Rule 110 next states
- **Maze solver** — Graph neural network that finds paths through mazes
- **Music as sequences** — Generate simple melodies (MIDI note sequences) with sequence models
- **Fractal generator** — Train a generative model on fractal point clouds
- **Elixir code generation** — Character-level LM trained on Elixir source code
- **Drawing generator** — VAE trained on simple stroke sequences (synthetic handwriting)
- **Game state prediction** — Predict next state of a simple game (tic-tac-toe, connect-4) given board state as graph or grid

---

## Summary by Architecture Family Coverage

| Family | Existing Notebooks | Ideas Above |
|--------|-------------------|-------------|
| Feedforward | training_mlp, architecture_comparison | TabNet, building blocks |
| Transformer | small_language_model | Small LM, text tasks, evolution, iRoPE |
| SSM | sequence_modeling | Multi-arch benchmark, Mamba family |
| Attention | — | Visualization gallery, positional encoding, Lightning Attention |
| Recurrent | sequence_modeling | RNN evolution, multi-arch benchmark |
| Vision | — | MNIST/CIFAR, U-Net, patch viz |
| Convolutional | — | CNN evolution, image classification |
| Graph | graph_classification | Node classification, molecules, viz |
| Sets | — | (could add point cloud classification) |
| Generative | generative_models | Diffusion, GAN, flows, VQ-VAE |
| Contrastive | — | SimCLR, BYOL, MAE, JEPA, Temporal JEPA |
| Meta | — | MoE routing, LoRA, adapters, aux-loss-free MoE |
| Energy | architecture_comparison | Energy landscapes, Hopfield memory |
| Probabilistic | architecture_comparison | Uncertainty, OOD detection |
| Memory | — | NTM copy task, memory viz |
| Neuromorphic | architecture_comparison | SNN basics, ANN→SNN |
| Liquid | — | Continuous dynamics visualization |
| Interpretability | — | SAE features, transcoder, ablation, superposition |
| World Model | — | Dynamics comparison, imagination, latent structure |
| RL | — | PolicyValue, PPO, model-based vs model-free |
