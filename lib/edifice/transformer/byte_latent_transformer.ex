defmodule Edifice.Transformer.ByteLatentTransformer do
  @moduledoc """
  Byte Latent Transformer (BLT) — byte-level processing via encode-process-decode.

  BLT processes raw byte sequences by encoding bytes into latent patches,
  processing them with a powerful latent transformer, then decoding back
  to byte-level predictions. This avoids the need for a fixed tokenizer.

  ## Architecture

  Three-component pipeline:

  ```
  Byte IDs [batch, byte_len]
        |
  +----- Encoder -------------------------------------------+
  |  Embedding(256, byte_dim) + transformer blocks          |
  |  Strided mean pool (patch_size stride) + project        |
  |  → [batch, byte_len/patch_size, latent_dim]             |
  +----------------------------------------------------------+
        |
  +----- Latent Transformer --------------------------------+
  |  GQA + RoPE + SwiGLU (DecoderOnly-style)               |
  |  output_mode: :all                                      |
  |  → [batch, byte_len/patch_size, latent_dim]             |
  +----------------------------------------------------------+
        |
  +----- Decoder -------------------------------------------+
  |  Project + upsample (repeat) + transformer blocks       |
  |  Dense(vocab_size)                                      |
  |  → [batch, byte_len, vocab_size]                        |
  +----------------------------------------------------------+
  ```

  ## Returns

  A 3-tuple `{encoder, latent_transformer, decoder}` where each is an
  independent Axon model.

  ## Usage

      {encoder, latent, decoder} = ByteLatentTransformer.build(
        vocab_size: 256,
        patch_size: 4,
        latent_dim: 256,
        byte_dim: 64,
        max_byte_len: 256
      )

  ## References

  - "Byte Latent Transformer: Patches Scale Better Than Tokens"
    (Meta, 2024) — https://arxiv.org/abs/2412.09871
  """

  alias Edifice.Blocks.TransformerBlock
  alias Edifice.Attention.GQA

  @default_vocab_size 256
  @default_patch_size 4
  @default_latent_dim 256
  @default_byte_dim 64
  @default_num_encoder_layers 2
  @default_num_latent_layers 4
  @default_num_decoder_layers 2
  @default_num_heads 4
  @default_num_kv_heads 2
  @default_max_byte_len 256
  @default_dropout 0.1

  @doc """
  Build a Byte Latent Transformer.

  ## Options

    - `:vocab_size` - Byte vocabulary size (default: 256)
    - `:patch_size` - Number of bytes per latent patch (default: 4)
    - `:latent_dim` - Latent transformer hidden dimension (default: 256)
    - `:byte_dim` - Byte-level encoder/decoder hidden dimension (default: 64)
    - `:num_encoder_layers` - Encoder transformer layers (default: 2)
    - `:num_latent_layers` - Latent transformer layers (default: 4)
    - `:num_decoder_layers` - Decoder transformer layers (default: 2)
    - `:num_heads` - Attention heads for latent transformer (default: 4)
    - `:num_kv_heads` - KV heads for GQA (default: 2)
    - `:max_byte_len` - Maximum byte sequence length (default: 256).
      Must be divisible by `patch_size`.
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns

    `{encoder, latent_transformer, decoder}` — a 3-tuple of Axon models.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:vocab_size, pos_integer()}
          | {:patch_size, pos_integer()}
          | {:latent_dim, pos_integer()}
          | {:byte_dim, pos_integer()}
          | {:num_encoder_layers, pos_integer()}
          | {:num_latent_layers, pos_integer()}
          | {:num_decoder_layers, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:max_byte_len, pos_integer()}
          | {:dropout, float()}

  @spec build([build_opt()]) :: {Axon.t(), Axon.t(), Axon.t()}
  def build(opts \\ []) do
    vocab_size = Keyword.get(opts, :vocab_size, @default_vocab_size)
    patch_size = Keyword.get(opts, :patch_size, @default_patch_size)
    latent_dim = Keyword.get(opts, :latent_dim, @default_latent_dim)
    byte_dim = Keyword.get(opts, :byte_dim, @default_byte_dim)
    num_encoder_layers = Keyword.get(opts, :num_encoder_layers, @default_num_encoder_layers)
    num_latent_layers = Keyword.get(opts, :num_latent_layers, @default_num_latent_layers)
    num_decoder_layers = Keyword.get(opts, :num_decoder_layers, @default_num_decoder_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_kv_heads = Keyword.get(opts, :num_kv_heads, @default_num_kv_heads)
    max_byte_len = Keyword.get(opts, :max_byte_len, @default_max_byte_len)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    num_patches = div(max_byte_len, patch_size)

    encoder =
      build_encoder(
        vocab_size: vocab_size,
        byte_dim: byte_dim,
        latent_dim: latent_dim,
        patch_size: patch_size,
        num_layers: num_encoder_layers,
        max_byte_len: max_byte_len,
        dropout: dropout
      )

    latent_transformer =
      build_latent_transformer(
        latent_dim: latent_dim,
        num_layers: num_latent_layers,
        num_heads: num_heads,
        num_kv_heads: num_kv_heads,
        num_patches: num_patches,
        dropout: dropout
      )

    decoder =
      build_decoder(
        vocab_size: vocab_size,
        byte_dim: byte_dim,
        latent_dim: latent_dim,
        patch_size: patch_size,
        num_layers: num_decoder_layers,
        num_patches: num_patches,
        max_byte_len: max_byte_len,
        dropout: dropout
      )

    {encoder, latent_transformer, decoder}
  end

  # Encoder: byte embeddings -> transformer blocks -> strided mean pool -> project to latent_dim
  defp build_encoder(opts) do
    vocab_size = Keyword.fetch!(opts, :vocab_size)
    byte_dim = Keyword.fetch!(opts, :byte_dim)
    latent_dim = Keyword.fetch!(opts, :latent_dim)
    patch_size = Keyword.fetch!(opts, :patch_size)
    num_layers = Keyword.fetch!(opts, :num_layers)
    max_byte_len = Keyword.fetch!(opts, :max_byte_len)
    dropout = Keyword.fetch!(opts, :dropout)

    # Input: byte IDs [batch, byte_len]
    input = Axon.input("byte_ids", shape: {nil, max_byte_len})

    # Byte embedding: [batch, byte_len] -> [batch, byte_len, byte_dim]
    x = Axon.embedding(input, vocab_size, byte_dim, name: "encoder_embedding")

    # Small transformer blocks at byte level
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        name = "encoder_block_#{layer_idx}"

        attn_fn = fn x_in, attn_name ->
          build_small_attention(x_in,
            hidden_size: byte_dim,
            num_heads: max(1, div(byte_dim, 16)),
            name: attn_name
          )
        end

        TransformerBlock.layer(acc,
          attention_fn: attn_fn,
          hidden_size: byte_dim,
          norm: :layer_norm,
          ffn_type: :standard,
          dropout: dropout,
          name: name
        )
      end)

    # Strided mean pool: [batch, byte_len, byte_dim] -> [batch, num_patches, byte_dim]
    x =
      Axon.layer(
        &strided_mean_pool/2,
        [x],
        name: "encoder_pool",
        patch_size: patch_size,
        op_name: :strided_mean_pool
      )

    # Project to latent dim
    Axon.dense(x, latent_dim, name: "encoder_project")
  end

  # Latent transformer: GQA + RoPE + SwiGLU, output all timesteps
  defp build_latent_transformer(opts) do
    latent_dim = Keyword.fetch!(opts, :latent_dim)
    num_layers = Keyword.fetch!(opts, :num_layers)
    num_heads = Keyword.fetch!(opts, :num_heads)
    num_kv_heads = Keyword.fetch!(opts, :num_kv_heads)
    num_patches = Keyword.fetch!(opts, :num_patches)
    dropout = Keyword.fetch!(opts, :dropout)

    # Input: latent patches [batch, num_patches, latent_dim]
    input = Axon.input("latent_patches", shape: {nil, num_patches, latent_dim})

    x =
      Enum.reduce(1..num_layers, input, fn layer_idx, acc ->
        name = "latent_block_#{layer_idx}"

        attn_fn = fn x_in, attn_name ->
          GQA.build_gqa_attention(x_in,
            hidden_size: latent_dim,
            num_heads: num_heads,
            num_kv_heads: num_kv_heads,
            rope: true,
            name: attn_name
          )
        end

        TransformerBlock.layer(acc,
          attention_fn: attn_fn,
          hidden_size: latent_dim,
          norm: :rms_norm,
          ffn_type: :gated,
          dropout: dropout,
          name: name
        )
      end)

    # Final norm, keep all timesteps
    Axon.layer_norm(x, name: "latent_final_norm")
  end

  # Decoder: project + upsample (repeat) + transformer blocks + dense(vocab_size)
  defp build_decoder(opts) do
    vocab_size = Keyword.fetch!(opts, :vocab_size)
    byte_dim = Keyword.fetch!(opts, :byte_dim)
    latent_dim = Keyword.fetch!(opts, :latent_dim)
    patch_size = Keyword.fetch!(opts, :patch_size)
    num_layers = Keyword.fetch!(opts, :num_layers)
    num_patches = Keyword.fetch!(opts, :num_patches)
    max_byte_len = Keyword.fetch!(opts, :max_byte_len)
    dropout = Keyword.fetch!(opts, :dropout)

    # Input: latent representations [batch, num_patches, latent_dim]
    input = Axon.input("latent_output", shape: {nil, num_patches, latent_dim})

    # Project to byte_dim
    x = Axon.dense(input, byte_dim, name: "decoder_project")

    # Upsample: repeat each patch patch_size times
    # [batch, num_patches, byte_dim] -> [batch, num_patches * patch_size, byte_dim]
    x =
      Axon.layer(
        &upsample_repeat/2,
        [x],
        name: "decoder_upsample",
        patch_size: patch_size,
        max_byte_len: max_byte_len,
        op_name: :upsample_repeat
      )

    # Small transformer blocks at byte level
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        name = "decoder_block_#{layer_idx}"

        attn_fn = fn x_in, attn_name ->
          build_small_attention(x_in,
            hidden_size: byte_dim,
            num_heads: max(1, div(byte_dim, 16)),
            name: attn_name
          )
        end

        TransformerBlock.layer(acc,
          attention_fn: attn_fn,
          hidden_size: byte_dim,
          norm: :layer_norm,
          ffn_type: :standard,
          dropout: dropout,
          name: name
        )
      end)

    # Final projection to vocab_size
    Axon.dense(x, vocab_size, name: "decoder_output")
  end

  # Simple multi-head attention for encoder/decoder (lightweight)
  defp build_small_attention(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.fetch!(opts, :num_heads)
    name = Keyword.get(opts, :name, "small_attn")
    head_dim = div(hidden_size, num_heads)

    qkv = Axon.dense(input, hidden_size * 3, name: "#{name}_qkv")

    output =
      Axon.layer(
        &small_attention_impl/2,
        [qkv],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :small_attention
      )

    Axon.dense(output, hidden_size, name: "#{name}_out")
  end

  defp small_attention_impl(qkv, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    hidden_size = num_heads * head_dim

    batch = Nx.axis_size(qkv, 0)
    seq_len = Nx.axis_size(qkv, 1)

    q = Nx.slice_along_axis(qkv, 0, hidden_size, axis: 2)
    k = Nx.slice_along_axis(qkv, hidden_size, hidden_size, axis: 2)
    v = Nx.slice_along_axis(qkv, hidden_size * 2, hidden_size, axis: 2)

    q = Nx.reshape(q, {batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = Nx.reshape(k, {batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = Nx.reshape(v, {batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    scale = :math.sqrt(head_dim)
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.divide(scale)

    # Causal mask
    row_idx = Nx.iota({seq_len, seq_len}, axis: 0)
    col_idx = Nx.iota({seq_len, seq_len}, axis: 1)
    mask = Nx.greater_equal(row_idx, col_idx)
    mask = Nx.broadcast(mask, Nx.shape(scores), axes: [2, 3])

    scores = Nx.select(mask, scores, Nx.broadcast(-1.0e9, Nx.shape(scores)))

    max_s = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_s = Nx.exp(Nx.subtract(scores, max_s))
    weights = Nx.divide(exp_s, Nx.add(Nx.sum(exp_s, axes: [-1], keep_axes: true), 1.0e-8))

    output = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, hidden_size})
  end

  # Strided mean pool: average groups of patch_size consecutive positions
  defp strided_mean_pool(x, opts) do
    patch_size = opts[:patch_size]
    batch = Nx.axis_size(x, 0)
    seq_len = Nx.axis_size(x, 1)
    dim = Nx.axis_size(x, 2)
    num_patches = div(seq_len, patch_size)

    x
    |> Nx.reshape({batch, num_patches, patch_size, dim})
    |> Nx.mean(axes: [2])
  end

  # Upsample by repeating each position patch_size times
  defp upsample_repeat(x, opts) do
    patch_size = opts[:patch_size]
    batch = Nx.axis_size(x, 0)
    num_patches = Nx.axis_size(x, 1)
    dim = Nx.axis_size(x, 2)

    x
    |> Nx.new_axis(2)
    |> Nx.broadcast({batch, num_patches, patch_size, dim})
    |> Nx.reshape({batch, num_patches * patch_size, dim})
  end

  @doc "Get the output size of the latent transformer."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :latent_dim, @default_latent_dim)
  end
end
