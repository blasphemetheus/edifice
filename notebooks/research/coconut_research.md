# Coconut (Continuous Chain of Thought) — Research Notes

## Paper
- **Title**: Training Large Language Models to Reason in a Continuous Latent Space
- **Authors**: Hao et al. (Meta AI)
- **Venue**: ICLR 2025
- **arXiv**: https://arxiv.org/abs/2412.06769

## Key Idea

Standard Chain-of-Thought (CoT) reasons via discrete text tokens. Coconut
replaces the text-based reasoning with continuous latent representations:
the model's hidden states at special `<latent>` positions are fed back as
input embeddings for the next forward pass, bypassing the vocabulary
bottleneck entirely.

## Mechanism

1. Replace `<latent>` token positions in the input with the last hidden state
   from the previous forward pass
2. Run the transformer forward on the augmented sequence
3. Repeat for N "thought" steps

The continuous thought vectors can encode multiple reasoning paths
simultaneously (unlike discrete tokens which commit to one path). This
enables implicit breadth-first search through the reasoning space.

## Training

The paper uses a curriculum: start with standard language modeling, then
gradually replace CoT tokens with latent thoughts. This requires a
multi-stage training schedule that is beyond Edifice's scope (architecture
only).

## Edifice Implementation

### Design Choices

- **Fixed thought slots**: Axon builds static computation graphs, so we
  pre-allocate `num_thoughts` positions. Each thought step shifts the
  sequence left by 1 and appends the new thought vector at the end.

- **Weight sharing**: Thought steps share transformer weights (name:
  `"shared_thought_block"`), matching the paper's setup and the MoR pattern
  in Edifice. First and last blocks have unique parameters.

- **Thought extraction**: Last-position hidden state is extracted via
  `Axon.nx` slice and written back via `Axon.layer` concatenation.

### Architecture Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| embed_dim | (required) | Input embedding dimension |
| hidden_size | 256 | Transformer hidden dim |
| num_heads | 4 | Attention heads |
| num_thoughts | 3 | Continuous thought steps |
| num_layers | 2 | Layers per thought step |
| dropout | 0.1 | Dropout rate |
| window_size | 60 | Input sequence length |

### Output

`[batch, hidden_size]` — the final thought vector at the last sequence
position after all thought steps and final transformer processing.
