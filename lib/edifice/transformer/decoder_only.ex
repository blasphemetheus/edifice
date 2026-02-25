defmodule Edifice.Transformer.DecoderOnly do
  @moduledoc """
  GPT-style decoder-only transformer with GQA + RoPE + SwiGLU + RMSNorm.

  Combines modern LLM techniques into a single decoder-only transformer:
  - Grouped Query Attention (GQA) for efficient KV cache
  - Rotary Position Embeddings (RoPE) for position encoding
  - SwiGLU gated feed-forward network
  - RMSNorm for faster normalization

  ## Attention Variants

  The `:attention_type` option allows switching between attention mechanisms:
  - `:gqa` (default) — Grouped Query Attention with RoPE
  - `:lightning` — Lightning Attention (hybrid linear/softmax block attention)
  - `:dual_chunk` — Dual Chunk Attention (intra-chunk + inter-chunk for long contexts)

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  Input projection to hidden_size
        |
  +------------------------------------+
  |   Decoder Block (x num_layers)     |
  |                                    |
  |   RMSNorm -> Attention             |
  |     (GQA / Lightning / DualChunk)  |
  |     + RoPE on Q and K (GQA only)   |
  |   -> Residual                      |
  |   RMSNorm -> SwiGLU FFN            |
  |   -> Residual                      |
  +------------------------------------+
        |
  Final LayerNorm
        |
  Last timestep -> [batch, hidden_size]
  ```

  ## Usage

      # Default GQA attention
      model = DecoderOnly.build(
        embed_dim: 287,
        hidden_size: 256,
        num_heads: 8,
        num_kv_heads: 2,
        num_layers: 6
      )

      # Lightning Attention for subquadratic complexity
      model = DecoderOnly.build(
        embed_dim: 287,
        hidden_size: 256,
        num_heads: 8,
        num_layers: 6,
        attention_type: :lightning,
        block_size: 64
      )

      # Dual Chunk Attention for long contexts
      model = DecoderOnly.build(
        embed_dim: 287,
        hidden_size: 256,
        num_heads: 8,
        num_layers: 6,
        attention_type: :dual_chunk,
        chunk_size: 64
      )

  ## References
  - GPT-2/3 decoder-only architecture (Radford et al., 2019; Brown et al., 2020)
  - LLaMA architecture combining GQA + RoPE + SwiGLU + RMSNorm (Touvron et al., 2023)
  - Lightning Attention-2 (Qin et al., 2024)
  - DeepSeek/Qwen2.5 Dual Chunk Attention (2024)
  """

  alias Edifice.Attention.{DualChunk, GQA}
  alias Edifice.Attention.LightningAttention
  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}

  @default_hidden_size 256
  @default_num_heads 8
  @default_num_kv_heads 2
  @default_num_layers 4
  @default_dropout 0.1

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_kv_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:attention_type, :gqa | :lightning | :dual_chunk}
          | {:block_size, pos_integer()}
          | {:chunk_size, pos_integer()}
          | {:use_rope, boolean()}
          | {:interleave_rope, boolean()}
          | {:yarn, boolean()}
          | {:yarn_scale, number()}
          | {:yarn_original_max_position, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  @doc """
  Build a GPT-style decoder-only transformer model.

  ## Options

    - `:embed_dim` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of query heads (default: 8)
    - `:num_kv_heads` - Number of key/value heads for GQA (default: 2)
    - `:num_layers` - Number of decoder blocks (default: 4)
    - `:attention_type` - Attention mechanism to use (default: :gqa)
      - `:gqa` — Grouped Query Attention with optional RoPE
      - `:lightning` — Lightning Attention (hybrid linear/softmax block attention)
      - `:dual_chunk` — Dual Chunk Attention (intra + inter-chunk for long contexts)
    - `:block_size` - Block size for Lightning Attention (default: 64)
    - `:chunk_size` - Chunk size for Dual Chunk Attention (default: 64)
    - `:use_rope` - Apply Rotary Position Embeddings, GQA only (default: true)
    - `:interleave_rope` - When true, even-indexed layers (0,2,4...) use RoPE and
      odd-indexed layers (1,3,5...) use NoPE (content-only attention). Overrides
      `:use_rope` on a per-layer basis. This is the iRoPE pattern used by Llama 4.
      (default: false)
    - `:yarn` - Enable YaRN context extension for longer sequences (default: false)
    - `:yarn_scale` - YaRN scaling factor, e.g., 8 extends 2048 to 16384 (default: 8)
    - `:yarn_original_max_position` - Original trained context length (default: 2048)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length for JIT optimization (default: 60)

  ## Returns

    An Axon model that outputs `[batch, hidden_size]` from the last position.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_kv_heads = Keyword.get(opts, :num_kv_heads, @default_num_kv_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    attention_type = Keyword.get(opts, :attention_type, :gqa)
    block_size = Keyword.get(opts, :block_size, 64)
    chunk_size = Keyword.get(opts, :chunk_size, 64)
    use_rope = Keyword.get(opts, :use_rope, true)
    interleave_rope = Keyword.get(opts, :interleave_rope, false)
    use_yarn = Keyword.get(opts, :yarn, false)
    yarn_scale = Keyword.get(opts, :yarn_scale, 8)
    yarn_original_max_position = Keyword.get(opts, :yarn_original_max_position, 2048)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    head_dim = div(hidden_size, num_heads)

    ModelBuilder.build_sequence_model(
      Keyword.merge(opts,
        hidden_size: hidden_size,
        num_layers: num_layers,
        block_builder: fn input, block_opts ->
          layer_idx = block_opts[:layer_idx]
          name = "decoder_block_#{layer_idx}"

          # iRoPE: even layers (0,2,4...) get RoPE, odd layers get NoPE (content-only)
          apply_rope =
            if interleave_rope do
              rem(layer_idx, 2) == 0
            else
              use_rope
            end

          attn_fn = fn x, attn_name ->
            build_attention(x, attention_type,
              hidden_size: hidden_size,
              num_heads: num_heads,
              head_dim: head_dim,
              num_kv_heads: num_kv_heads,
              block_size: block_size,
              chunk_size: chunk_size,
              rope: apply_rope,
              yarn: use_yarn,
              yarn_scale: yarn_scale,
              yarn_original_max_position: yarn_original_max_position,
              name: attn_name
            )
          end

          TransformerBlock.layer(input,
            attention_fn: attn_fn,
            hidden_size: hidden_size,
            norm: :rms_norm,
            ffn_type: :gated,
            dropout: dropout,
            name: name
          )
        end
      )
    )
  end

  # Build attention based on type
  defp build_attention(input, :gqa, opts) do
    GQA.build_gqa_attention(input,
      hidden_size: opts[:hidden_size],
      num_heads: opts[:num_heads],
      num_kv_heads: opts[:num_kv_heads],
      rope: opts[:rope],
      yarn: opts[:yarn],
      yarn_scale: opts[:yarn_scale],
      yarn_original_max_position: opts[:yarn_original_max_position],
      name: opts[:name]
    )
  end

  defp build_attention(input, :lightning, opts) do
    LightningAttention.build_lightning_attention(input,
      hidden_size: opts[:hidden_size],
      num_heads: opts[:num_heads],
      head_dim: opts[:head_dim],
      block_size: opts[:block_size],
      name: opts[:name]
    )
  end

  defp build_attention(input, :dual_chunk, opts) do
    DualChunk.build_dual_chunk_attention(input,
      hidden_size: opts[:hidden_size],
      num_heads: opts[:num_heads],
      head_dim: opts[:head_dim],
      chunk_size: opts[:chunk_size],
      name: opts[:name]
    )
  end

  @doc """
  Get the output size of a decoder-only model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end
end
