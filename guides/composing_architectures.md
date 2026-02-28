# Composing Architectures
> How to mix attention mechanisms, swap feed-forward networks, build encoder-decoders, and reuse shared blocks -- turning Edifice's 200+ architectures into ingredients rather than finished dishes.

## What This Guide Covers

Most Edifice architectures are ready-to-use: call `build/1`, get a model. But the library's real
power emerges when you start *composing* -- plugging one architecture's attention into another's
transformer stack, replacing a standard FFN with SwiGLU, or wiring an encoder from one family
into a decoder from another.

This guide covers the three composition mechanisms that make this possible:

1. **TransformerBlock callbacks** -- swap attention and FFN via function arguments
2. **ModelBuilder skeletons** -- standardized input/stack/output pipelines that accept any block
3. **Shared blocks** -- deduplicated primitives (SDPA, RoPE, CrossAttention, etc.) used across families

**Prerequisites:** You should be comfortable with [Reading Edifice](reading_edifice.md) and
[Building Blocks](building_blocks.md). Understanding Axon computation graphs is essential --
composition works by passing Axon graph-building functions as callbacks.

## The Callback Pattern

The core idea: separate *structure* (normalization, residuals, dropout) from *computation*
(how attention works, what the FFN does). Structure is handled by `TransformerBlock`. Computation
is provided by you.

### TransformerBlock.layer/2 -- The 2-Sublayer Block

Every attention-based architecture in Edifice follows this skeleton:

```
Input
  |
  +---> Norm -> attention_fn(x) -> Dropout ---+
  |                                           |
  +<------------- Residual <------------------+
  |
  +---> Norm -> FFN(x) -> Dropout ------------+
  |                                           |
  +<------------- Residual <------------------+
  |
Output
```

The `attention_fn` callback is where you inject your attention mechanism. It receives a
normalized input and a name prefix, and returns an Axon node:

```elixir
# Standard multi-head attention
TransformerBlock.layer(input,
  attention_fn: fn x, name ->
    MultiHead.self_attention(x, hidden_size, num_heads, name)
  end,
  hidden_size: 256
)

# Differential attention (same structure, different attention math)
TransformerBlock.layer(input,
  attention_fn: fn x, name ->
    DiffAttention.layer(x, hidden_size: 256, num_heads: 8, name: name)
  end,
  hidden_size: 256
)

# Sigmoid attention (softmax replaced with sigmoid + bias)
TransformerBlock.layer(input,
  attention_fn: fn x, name ->
    SigmoidAttention.layer(x, hidden_size: 256, num_heads: 8, name: name)
  end,
  hidden_size: 256
)
```

The surrounding norm/residual/dropout structure is identical in all three cases. Only the
attention math changes. This is how Edifice achieves 15+ attention variants without duplicating
the transformer skeleton.

### Swapping the FFN

The `:custom_ffn` callback replaces the standard feed-forward network:

```elixir
# Standard FFN (dense -> activation -> dense)
TransformerBlock.layer(input,
  attention_fn: my_attention,
  hidden_size: 256,
  ffn_type: :standard,
  ffn_expansion: 4
)

# SwiGLU FFN (gated linear unit with SiLU activation)
TransformerBlock.layer(input,
  attention_fn: my_attention,
  hidden_size: 256,
  ffn_type: :gated
)

# Fully custom FFN (e.g., Whisper uses explicit inner_size)
TransformerBlock.layer(input,
  attention_fn: my_attention,
  hidden_size: 256,
  custom_ffn: fn x, name ->
    FFN.layer(x, hidden_size: 256, inner_size: 1024, dropout: 0.1, name: name)
  end
)

# KAT-style: replace FFN with a Kolmogorov-Arnold network
TransformerBlock.layer(input,
  attention_fn: my_attention,
  hidden_size: 256,
  custom_ffn: fn x, name ->
    KAT.feedforward(x, hidden_size: 256, name: name)
  end
)
```

When `:custom_ffn` is provided, the `:ffn_type` and `:ffn_expansion` options are ignored.
The callback receives the post-norm input and a name, same pattern as `attention_fn`.

### TransformerBlock.layer/3 -- The 3-Sublayer Block

Encoder-decoder architectures need a third sublayer: cross-attention between the decoder's
representation and the encoder's output. `layer/3` adds this between self-attention and FFN:

```
Input   Memory (encoder output)
  |       |
  +---> Norm -> self_attention_fn(x) -> Residual
  |       |
  +---> Norm -> cross_attention_fn(x, memory) -> Residual
  |
  +---> Norm -> FFN(x) -> Residual
  |
Output
```

The `cross_attention_fn` callback receives three arguments: the normalized query, the memory
tensor, and a name prefix:

```elixir
# Whisper decoder: self-attn + cross-attn + FFN
TransformerBlock.stack(decoder_input, encoder_output, num_layers,
  attention_fn: fn x, name ->
    causal_self_attention(x, hidden_dim, num_heads, name)
  end,
  cross_attention_fn: fn q, memory, name ->
    CrossAttention.layer(q, memory,
      hidden_size: hidden_dim,
      num_heads: num_heads,
      name: name
    )
  end,
  hidden_size: hidden_dim
)
```

This 3-sublayer pattern is used by Whisper, DETR, RT-DETR, ACT, and any architecture that
attends to a separate encoder output.

### Stacking

Both variants have a `stack/3` and `stack/4` function that repeat the block N times with
auto-generated names:

```elixir
# Stack 6 self-attention blocks
TransformerBlock.stack(input, 6,
  attention_fn: my_attention,
  hidden_size: 256,
  name: "encoder"
)
# Creates: encoder_block_1, encoder_block_2, ..., encoder_block_6

# Stack 6 encoder-decoder blocks
TransformerBlock.stack(input, memory, 6,
  attention_fn: my_self_attention,
  cross_attention_fn: my_cross_attention,
  hidden_size: 256,
  name: "decoder"
)
```

## ModelBuilder -- Full Model Skeletons

While `TransformerBlock` handles individual blocks, `ModelBuilder` provides complete model
pipelines: input creation, optional projection, block stacking, final normalization, and
output extraction.

### Sequence Models

```elixir
ModelBuilder.build_sequence_model(
  embed_dim: 287,         # Input dimension (e.g., from game state embedding)
  hidden_size: 256,       # Internal dimension (auto-projects if different)
  num_layers: 4,
  block_builder: fn input, opts ->
    TransformerBlock.layer(input,
      attention_fn: my_attention,
      hidden_size: 256,
      name: "block_#{opts[:layer_idx]}"
    )
  end,
  output_mode: :last_timestep  # or :all, :mean_pool
)
```

This creates:
1. An input node named `"state_sequence"` with shape `[batch, seq_len, embed_dim]`
2. A dense projection from `embed_dim` to `hidden_size` (if they differ)
3. N stacked blocks via your `block_builder` callback
4. Final layer normalization
5. Output extraction (last timestep, all timesteps, or mean pooling)

The `block_builder` callback receives the current Axon node and the full options map (with
`:layer_idx` added). This is how architectures like GatedAttention, SSMax, and Softpick
build complete models -- they define an attention function, wrap it in a TransformerBlock
callback, and hand it to ModelBuilder.

### Vision Models

```elixir
ModelBuilder.build_vision_model(
  image_size: 224,
  patch_size: 16,
  in_channels: 3,
  hidden_size: 768,
  num_layers: 12,
  block_builder: fn input, opts ->
    TransformerBlock.layer(input,
      attention_fn: my_vision_attention,
      hidden_size: 768,
      name: "vit_block_#{opts[:layer_idx]}"
    )
  end,
  num_classes: 1000  # optional classifier head
)
```

This adds patch embedding (via `PatchEmbed.layer`) before the block stack and mean pooling
after it. The same `block_builder` pattern applies.

## Shared Blocks

Edifice extracts commonly duplicated computations into standalone blocks. These are the
building primitives that appear across families:

| Block | What it does | Used by |
|-------|-------------|---------|
| `SDPA.compute` | Multi-head scaled dot-product attention | DETR, RT-DETR, SAM 2, ACT, Whisper, Perceiver, VALL-E, Janus, Decision Transformer |
| `CrossAttention.layer` | Q/K/V projection + SDPA + output projection | Whisper decoder, DETR decoder, ACT decoder |
| `RoPE.apply_rotary_4d` | Rotary position embeddings | MultiHead, DiffTransformer, MLA, DecoderOnly |
| `SinusoidalPE.layer` | Fixed sinusoidal position encoding | Whisper, DETR, SAM 2, SoundStorm, Conformer, DiT, MDLM, SoFlow |
| `SinusoidalPE.timestep_layer` | Scalar timestep embedding for diffusion | DiT, DiTv2, MMDiT, LinearDiT, SiT, RectifiedFlow, SoFlow, MDLM |
| `AdaptiveNorm.modulate` | Condition-dependent scale/shift | DiT, DiTv2, MMDiT, LinearDiT, SiT, MDLM |
| `AdaptiveNorm.gate` | Condition-dependent gating (AdaLN-Zero) | DiT, DiTv2, MMDiT, LinearDiT, SiT, MDLM |
| `FFN.layer` / `FFN.gated_layer` | Standard and SwiGLU feed-forward | TransformerBlock, Whisper, DETR, and any `:custom_ffn` user |
| `RMSNorm.layer` | Root mean square normalization | TransformerBlock (`:norm` option), DiTv2, TransformerLike |
| `SwiGLU.layer` | Gated linear unit with SiLU | MDLM, TransformerBlock (`:ffn_type` option) |
| `PatchEmbed.layer` | Image to patch sequence | ModelBuilder vision, DINOv2, MetaFormer, EfficientViT |
| `CausalMask.causal` | Lower-triangular attention mask | DecoderOnly, Whisper decoder, MDLM |
| `BBoxHead.layer` | Bounding box prediction head | DETR, RT-DETR |

### Why Shared Blocks Matter

Before the composability audit, SDPA was implemented 6 times across the codebase -- each
with slight variations in reshape order, scaling, and softmax stability. Now there's one
`SDPA.compute` that handles reshaping, scaling, optional masking, and numerically stable
softmax (FP32 internal via `FusedOps.fused_softmax`). When a bug is fixed or an optimization
added, all 9+ consumers benefit.

The same story applies to `SinusoidalPE.timestep_layer` (was duplicated in 8 diffusion models)
and `AdaptiveNorm.modulate/gate` (was duplicated in 6 conditional generation models).

## Composition Recipes

### Recipe 1: Custom Attention in a Standard Transformer

Suppose you've implemented a new attention mechanism and want a complete model:

```elixir
defmodule MyAttention do
  def build(opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_layers = Keyword.get(opts, :num_layers, 4)

    ModelBuilder.build_sequence_model(
      embed_dim: hidden_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      block_builder: fn input, block_opts ->
        TransformerBlock.layer(input,
          attention_fn: fn x, name ->
            my_fancy_attention(x, hidden_size, name)
          end,
          hidden_size: hidden_size,
          ffn_type: :gated,
          name: "my_attn_block_#{block_opts[:layer_idx]}"
        )
      end,
      output_mode: :all
    )
  end

  defp my_fancy_attention(x, hidden_size, name) do
    # Your attention implementation here
    # x is [batch, seq, hidden_size], already normalized
    q = Axon.dense(x, hidden_size, name: "#{name}_q")
    k = Axon.dense(x, hidden_size, name: "#{name}_k")
    v = Axon.dense(x, hidden_size, name: "#{name}_v")

    # Use shared SDPA for the actual attention math
    Axon.layer(
      fn q_val, k_val, v_val, _opts ->
        SDPA.compute(q_val, k_val, v_val, 8, div(hidden_size, 8))
      end,
      [q, k, v],
      name: "#{name}_sdpa"
    )
  end
end
```

This gets you normalization, residuals, dropout, SwiGLU FFN, input projection, final norm,
and output extraction -- all from the shared infrastructure.

### Recipe 2: Encoder-Decoder from Different Families

Build an encoder with one attention type and a decoder with another:

```elixir
defmodule HybridEncoderDecoder do
  def build(opts) do
    hidden = opts[:hidden_size]
    heads = opts[:num_heads]

    # Encoder: use sigmoid attention (no softmax)
    encoder_input = Axon.input("encoder_input", shape: {nil, nil, hidden})
    encoded = TransformerBlock.stack(encoder_input, 6,
      attention_fn: fn x, name ->
        SigmoidAttention.layer(x, hidden_size: hidden, num_heads: heads, name: name)
      end,
      hidden_size: hidden,
      name: "encoder"
    )

    # Decoder: use standard attention + cross-attention to encoder
    decoder_input = Axon.input("decoder_input", shape: {nil, nil, hidden})
    decoded = TransformerBlock.stack(decoder_input, encoded, 6,
      attention_fn: fn x, name ->
        causal_self_attention(x, hidden, heads, name)
      end,
      cross_attention_fn: fn q, memory, name ->
        CrossAttention.layer(q, memory,
          hidden_size: hidden, num_heads: heads, name: name
        )
      end,
      hidden_size: hidden,
      name: "decoder"
    )

    Axon.container(%{encoder: encoded, decoder: decoded})
  end
end
```

### Recipe 3: Mixing SSM and Attention (Hymba-style)

Some architectures interleave different block types across layers. The `block_builder`
callback receives `:layer_idx`, so you can alternate:

```elixir
ModelBuilder.build_sequence_model(
  embed_dim: 256,
  hidden_size: 256,
  num_layers: 8,
  block_builder: fn input, opts ->
    idx = opts[:layer_idx]

    if rem(idx, 2) == 1 do
      # Odd layers: Mamba SSM block
      Mamba.block(input, hidden_size: 256, name: "mamba_#{idx}")
    else
      # Even layers: standard attention
      TransformerBlock.layer(input,
        attention_fn: fn x, name ->
          MultiHead.self_attention(x, 256, 8, name)
        end,
        hidden_size: 256,
        name: "attn_block_#{idx}"
      )
    end
  end,
  output_mode: :last_timestep
)
```

## Configuration Options Reference

### TransformerBlock

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:attention_fn` | function | required | `(input, name) -> Axon.t()` |
| `:cross_attention_fn` | function | required for `layer/3` | `(query, memory, name) -> Axon.t()` |
| `:hidden_size` | integer | required | Hidden dimension |
| `:ffn_type` | `:standard` or `:gated` | `:standard` | Feed-forward variant |
| `:ffn_expansion` | integer | 4 | FFN expansion factor |
| `:custom_ffn` | function | nil | `(input, name) -> Axon.t()` replaces FFN |
| `:norm` | `:layer_norm` or `:rms_norm` | `:layer_norm` | Normalization type |
| `:norm_position` | `:pre` or `:post` | `:pre` | Pre-norm or post-norm |
| `:dropout` | float | 0.0 | Dropout rate |
| `:name` | string | `"transformer_block"` | Name prefix |

### ModelBuilder.build_sequence_model

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:embed_dim` | integer | required | Input embedding dimension |
| `:hidden_size` | integer | embed_dim | Internal hidden dimension |
| `:num_layers` | integer | required | Number of blocks to stack |
| `:block_builder` | function | required | `(input, opts) -> Axon.t()` |
| `:seq_len` | integer | 60 | Sequence length for JIT |
| `:output_mode` | atom | `:last_timestep` | `:last_timestep`, `:all`, `:mean_pool` |
| `:final_norm` | boolean | true | Apply final layer norm |
| `:dropout` | float | 0.0 | Inter-block dropout rate |

### ModelBuilder.build_vision_model

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:image_size` | integer | 224 | Square image size |
| `:patch_size` | integer | 16 | Patch size |
| `:in_channels` | integer | 3 | Input channels |
| `:hidden_size` | integer | required | Hidden dimension |
| `:num_layers` | integer | required | Number of blocks |
| `:block_builder` | function | required | `(input, opts) -> Axon.t()` |
| `:num_classes` | integer | nil | Classifier head (omit for features only) |
| `:final_norm` | boolean | true | Apply final layer norm |

## Design Principles

**Callbacks over inheritance.** Elixir doesn't have class inheritance, but even in languages
that do, the callback approach is more flexible. A callback can close over arbitrary state,
compose multiple operations, or dispatch to entirely different implementations based on
runtime configuration.

**Shared blocks over copy-paste.** When you find yourself writing `softmax(QK^T / sqrt(d))V`
for the third time, extract it. Edifice's `SDPA.compute` started as 6 independent
implementations and is now one function with consistent numerics.

**Convention over configuration.** Input names (`"state_sequence"`, `"image"`), output modes
(`:last_timestep`, `:all`), and callback signatures (`(input, name) -> Axon.t()`) are
consistent across the library. Learn the pattern once, apply it everywhere.
