defmodule Edifice.Generative.VAR do
  @moduledoc """
  VAR: Visual Autoregressive Modeling via Next-Scale Prediction.

  Implements the VAR architecture from "Visual Autoregressive Modeling:
  Scalable Image Generation via Next-Scale Prediction" (Tian et al., NeurIPS 2024
  Best Paper). Instead of generating images token-by-token (like traditional AR),
  VAR generates images scale-by-scale: 1×1 → 2×2 → 4×4 → ... → N×N.

  ## Key Innovation: Next-Scale Prediction

  Traditional autoregressive image generation flattens images to 1D sequences
  and predicts pixels/tokens one at a time. This is slow and ignores spatial
  structure. VAR instead:

  1. Encodes images at multiple resolutions via a multi-scale VQ tokenizer
  2. Autoregressively predicts each scale given all coarser scales
  3. Each scale prediction is parallel (all tokens at that scale at once)

  ```
  Scale 1 (1×1):   [tok]           → Predict via GPT
  Scale 2 (2×2):   [tok tok]       → Predict via GPT given Scale 1
                   [tok tok]
  Scale 3 (4×4):   [tok tok tok tok]   → Predict via GPT given Scales 1-2
                   [tok tok tok tok]
                   [tok tok tok tok]
                   [tok tok tok tok]
  ...
  Scale K (N×N):   Full resolution
  ```

  ## Architecture

  ```
  Image [batch, H, W, C]
        |
        v
  +---------------------------+
  | Multi-Scale VQ Tokenizer  |  (encode at scales 1, 2, 4, 8, 16...)
  +---------------------------+
        |
        v
  [Scale tokens at each resolution]
        |
        v
  +---------------------------+
  | GPT-2 Backbone            |  (autoregressive over scale sequence)
  | (decoder_only pattern)    |
  +---------------------------+
        |
        v
  | Predict next scale tokens |
        |
        v
  +---------------------------+
  | Multi-Scale VQ Decoder    |  (decode all scales to image)
  +---------------------------+
        |
        v
  Output Image [batch, H, W, C]
  ```

  ## Usage

      # Build tokenizer for encoding/decoding
      tokenizer = VAR.build_tokenizer(
        image_size: 256,
        scales: [1, 2, 4, 8, 16],
        codebook_size: 1024,
        embed_dim: 256
      )

      # Build the full VAR model (GPT backbone for next-scale prediction)
      model = VAR.build(
        hidden_size: 512,
        num_layers: 12,
        num_heads: 8,
        scales: [1, 2, 4, 8, 16],
        codebook_size: 1024
      )

  ## Reference

  - Paper: "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"
  - Authors: Tian, Yu, et al.
  - arXiv: https://arxiv.org/abs/2404.02905
  - Award: NeurIPS 2024 Best Paper
  """

  @default_hidden_size 512
  @default_num_layers 12
  @default_num_heads 8
  @default_mlp_ratio 4.0
  @default_scales [1, 2, 4, 8, 16]
  @default_codebook_size 1024
  @default_embed_dim 256

  @doc """
  Build a multi-scale VQ tokenizer for VAR.

  The tokenizer encodes images at multiple resolutions, each with its own
  codebook. This enables the coarse-to-fine generation strategy.

  ## Options

    - `:image_size` - Target image size (default: 256)
    - `:scales` - List of scale factors [1, 2, 4, ...] (default: [1, 2, 4, 8, 16])
    - `:codebook_size` - Number of codes per scale (default: 1024)
    - `:embed_dim` - Embedding dimension (default: 256)
    - `:in_channels` - Input image channels (default: 3)

  ## Returns

    A tuple `{encoder, decoder}` where:
    - `encoder` maps images to multi-scale token indices
    - `decoder` maps multi-scale tokens back to images
  """
  @typedoc "Options for `build_tokenizer/1`."
  @type tokenizer_opt ::
          {:codebook_size, pos_integer()}
          | {:embed_dim, pos_integer()}
          | {:image_size, pos_integer()}
          | {:in_channels, pos_integer()}
          | {:scales, [pos_integer()]}

  @spec build_tokenizer([tokenizer_opt()]) :: {Axon.t(), Axon.t()}
  def build_tokenizer(opts \\ []) do
    image_size = Keyword.get(opts, :image_size, 256)
    scales = Keyword.get(opts, :scales, @default_scales)
    codebook_size = Keyword.get(opts, :codebook_size, @default_codebook_size)
    embed_dim = Keyword.get(opts, :embed_dim, @default_embed_dim)
    in_channels = Keyword.get(opts, :in_channels, 3)

    encoder = build_multiscale_encoder(image_size, scales, codebook_size, embed_dim, in_channels)
    decoder = build_multiscale_decoder(image_size, scales, codebook_size, embed_dim, in_channels)

    {encoder, decoder}
  end

  @doc """
  Build the VAR model (GPT-2 style backbone for next-scale prediction).

  ## Options

    - `:hidden_size` - Transformer hidden dimension (default: 512)
    - `:num_layers` - Number of transformer layers (default: 12)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:mlp_ratio` - MLP expansion ratio (default: 4.0)
    - `:scales` - List of scale factors (default: [1, 2, 4, 8, 16])
    - `:codebook_size` - Vocabulary size per scale (default: 1024)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns

    An Axon model that takes scale token embeddings and predicts next-scale logits.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:codebook_size, pos_integer()}
          | {:dropout, float()}
          | {:hidden_size, pos_integer()}
          | {:mlp_ratio, float()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:scales, [pos_integer()]}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    scales = Keyword.get(opts, :scales, @default_scales)
    codebook_size = Keyword.get(opts, :codebook_size, @default_codebook_size)
    dropout = Keyword.get(opts, :dropout, 0.1)

    # Total sequence length across all scales
    total_tokens = Enum.reduce(scales, 0, fn s, acc -> acc + s * s end)

    # Input: embedded tokens from all scales concatenated
    input = Axon.input("scale_embeddings", shape: {nil, total_tokens, hidden_size})

    # Add learnable position embeddings
    x =
      Axon.layer(
        &add_position_embeddings/2,
        [input],
        name: "position_embed",
        seq_len: total_tokens,
        hidden_size: hidden_size,
        op_name: :pos_embed
      )

    # GPT-2 style transformer blocks (causal attention)
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_transformer_block(acc,
          hidden_size: hidden_size,
          num_heads: num_heads,
          mlp_ratio: mlp_ratio,
          dropout: dropout,
          name: "block_#{layer_idx}"
        )
      end)

    # Final layer norm
    x = Axon.layer_norm(x, name: "final_norm")

    # Output heads: one per scale (predicting next scale's tokens)
    # For scale i, predict logits for scale i+1
    build_scale_prediction_heads(x, scales, codebook_size, hidden_size)
  end

  @doc """
  Perform next-scale prediction given current scale tokens.

  This function is used during inference to autoregressively generate
  each scale conditioned on all previous scales.

  ## Parameters

    - `model` - The VAR model
    - `params` - Model parameters
    - `current_tokens` - Token indices for scales 1..k
    - `scale_idx` - Which scale to predict (0-indexed)

  ## Returns

    Logits for the next scale's tokens.
  """
  @spec next_scale_prediction(Axon.t(), map(), Nx.Tensor.t(), non_neg_integer()) :: Nx.Tensor.t()
  def next_scale_prediction(model, params, current_embeddings, scale_idx) do
    # Run the model and extract logits for the target scale
    outputs = Axon.predict(model, params, %{"scale_embeddings" => current_embeddings})

    # The output is a container with predictions per scale
    Map.get(outputs, "scale_#{scale_idx + 1}_logits")
  end

  # ============================================================================
  # Multi-Scale Encoder
  # ============================================================================

  defp build_multiscale_encoder(image_size, scales, codebook_size, embed_dim, in_channels) do
    input = Axon.input("image", shape: {nil, image_size, image_size, in_channels})

    # For each scale, downsample and encode
    scale_outputs =
      Enum.with_index(scales, fn scale, idx ->
        target_size = div(image_size, Enum.max(scales)) * scale
        patch_size = div(image_size, target_size)

        encoded =
          input
          |> build_patchify(patch_size, embed_dim, "encoder_scale_#{idx}")
          |> Axon.dense(embed_dim, name: "encoder_scale_#{idx}_proj")
          |> Axon.activation(:gelu, name: "encoder_scale_#{idx}_act")
          |> Axon.dense(codebook_size, name: "encoder_scale_#{idx}_to_codes")

        # Quantize to indices (argmax)
        quantized =
          Axon.nx(encoded, fn logits -> Nx.argmax(logits, axis: -1) end,
            name: "encoder_scale_#{idx}_quantize"
          )

        {"scale_#{idx}", quantized}
      end)

    Axon.container(Map.new(scale_outputs))
  end

  # ============================================================================
  # Multi-Scale Decoder
  # ============================================================================

  defp build_multiscale_decoder(_image_size, scales, codebook_size, embed_dim, in_channels) do
    # Input: token indices per scale
    scale_inputs =
      Enum.with_index(scales, fn scale, idx ->
        num_tokens = scale * scale
        {"scale_#{idx}", Axon.input("scale_#{idx}_tokens", shape: {nil, num_tokens})}
      end)
      |> Map.new()

    # Embed and upsample each scale, then combine
    max_scale = Enum.max(scales)
    target_tokens = max_scale * max_scale

    combined =
      Enum.with_index(scales)
      |> Enum.map(fn {scale, idx} ->
        input = Map.get(scale_inputs, "scale_#{idx}")

        # Embed tokens
        embedded = build_token_embedding(input, codebook_size, embed_dim, "decoder_scale_#{idx}")

        # Upsample to match highest resolution
        upsample_factor = div(max_scale, scale)

        if upsample_factor > 1 do
          Axon.layer(
            &upsample_tokens/2,
            [embedded],
            name: "decoder_scale_#{idx}_upsample",
            factor: upsample_factor,
            target_tokens: target_tokens,
            embed_dim: embed_dim,
            op_name: :upsample
          )
        else
          embedded
        end
      end)
      |> Enum.reduce(fn a, b -> Axon.add(a, b) end)

    # Decode to image
    combined
    |> Axon.dense(embed_dim * 4, name: "decoder_expand")
    |> Axon.activation(:gelu, name: "decoder_act")
    |> Axon.dense(max_scale * max_scale * in_channels, name: "decoder_to_pixels")
    |> Axon.nx(
      fn x ->
        batch = Nx.axis_size(x, 0)
        Nx.reshape(x, {batch, max_scale, max_scale, in_channels})
      end,
      name: "decoder_reshape"
    )
  end

  # ============================================================================
  # Transformer Block (GPT-2 style with causal attention)
  # ============================================================================

  defp build_transformer_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size)
    num_heads = Keyword.get(opts, :num_heads)
    mlp_ratio = Keyword.get(opts, :mlp_ratio)
    dropout = Keyword.get(opts, :dropout)
    name = Keyword.get(opts, :name)
    mlp_dim = round(hidden_size * mlp_ratio)

    # Pre-norm attention
    x_norm = Axon.layer_norm(input, name: "#{name}_attn_norm")

    attn_out = build_causal_attention(x_norm, hidden_size, num_heads, "#{name}_attn")
    attn_out = maybe_dropout(attn_out, dropout, "#{name}_attn_dropout")
    x = Axon.add(input, attn_out, name: "#{name}_attn_residual")

    # Pre-norm MLP
    x_norm2 = Axon.layer_norm(x, name: "#{name}_mlp_norm")

    mlp_out =
      x_norm2
      |> Axon.dense(mlp_dim, name: "#{name}_mlp_up")
      |> Axon.activation(:gelu, name: "#{name}_mlp_act")
      |> Axon.dense(hidden_size, name: "#{name}_mlp_down")
      |> maybe_dropout(dropout, "#{name}_mlp_dropout")

    Axon.add(x, mlp_out, name: "#{name}_mlp_residual")
  end

  # Causal (autoregressive) self-attention
  defp build_causal_attention(input, hidden_size, num_heads, name) do
    head_dim = div(hidden_size, num_heads)

    q = Axon.dense(input, hidden_size, name: "#{name}_q")
    k = Axon.dense(input, hidden_size, name: "#{name}_k")
    v = Axon.dense(input, hidden_size, name: "#{name}_v")

    attn_out =
      Axon.layer(
        &causal_attention_impl/4,
        [q, k, v],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :causal_attention
      )

    Axon.dense(attn_out, hidden_size, name: "#{name}_out_proj")
  end

  defp causal_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape to multi-head: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    q = q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Scaled dot-product attention
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    # Causal mask: positions can only attend to earlier positions
    mask = causal_mask(seq_len, Nx.type(scores))
    scores = Nx.add(scores, mask)

    # Softmax
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))

    attn_weights =
      Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-8))

    # Apply attention to values
    output = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  defp causal_mask(seq_len, type) do
    # Lower triangular mask: 0 for allowed, -inf for masked
    rows = Nx.iota({seq_len, 1})
    cols = Nx.iota({1, seq_len})
    mask = Nx.greater(cols, rows)
    Nx.select(mask, Nx.tensor(-1.0e9, type: type), Nx.tensor(0.0, type: type))
  end

  # ============================================================================
  # Scale Prediction Heads
  # ============================================================================

  defp build_scale_prediction_heads(x, scales, codebook_size, hidden_size) do
    # For each scale (except the first), create a prediction head
    # The head predicts tokens for scale i+1 given context up to scale i
    scale_offsets = compute_scale_offsets(scales)

    head_outputs =
      scales
      |> Enum.with_index()
      |> Enum.drop(-1)
      |> Enum.map(fn {_scale, idx} ->
        # Extract the relevant context (all tokens up to and including this scale)
        end_offset = Enum.at(scale_offsets, idx + 1)
        next_scale = Enum.at(scales, idx + 1)
        num_next_tokens = next_scale * next_scale

        # Use the last token of current scale to predict next scale
        context_token =
          Axon.nx(
            x,
            fn t ->
              Nx.slice_along_axis(t, end_offset - 1, 1, axis: 1)
              |> Nx.squeeze(axes: [1])
            end,
            name: "scale_#{idx}_context"
          )

        # Predict all tokens for next scale
        logits =
          context_token
          |> Axon.dense(hidden_size, name: "scale_#{idx + 1}_head_proj")
          |> Axon.activation(:gelu, name: "scale_#{idx + 1}_head_act")
          |> Axon.dense(num_next_tokens * codebook_size, name: "scale_#{idx + 1}_head_logits")
          |> Axon.nx(
            fn t ->
              batch = Nx.axis_size(t, 0)
              Nx.reshape(t, {batch, num_next_tokens, codebook_size})
            end,
            name: "scale_#{idx + 1}_reshape"
          )

        {"scale_#{idx + 1}_logits", logits}
      end)

    Axon.container(Map.new(head_outputs))
  end

  defp compute_scale_offsets(scales) do
    scales
    |> Enum.scan(0, fn s, acc -> acc + s * s end)
    |> List.insert_at(0, 0)
  end

  # ============================================================================
  # Helper Functions
  # ============================================================================

  defp add_position_embeddings(x, opts) do
    seq_len = opts[:seq_len]
    hidden_size = opts[:hidden_size]

    # Learnable position embeddings (simplified: sinusoidal)
    positions = Nx.iota({seq_len}, type: :f32)

    freqs =
      Nx.exp(
        Nx.multiply(
          Nx.negate(Nx.log(Nx.tensor(10_000.0))),
          Nx.divide(Nx.iota({div(hidden_size, 2)}, type: :f32), max(div(hidden_size, 2) - 1, 1))
        )
      )

    angles = Nx.outer(positions, freqs)
    pos_embed = Nx.concatenate([Nx.sin(angles), Nx.cos(angles)], axis: 1)
    pos_embed = Nx.reshape(pos_embed, {1, seq_len, hidden_size})

    Nx.add(x, pos_embed)
  end

  defp build_patchify(input, patch_size, embed_dim, name) do
    # Use strided convolution to extract non-overlapping patches
    Axon.conv(input, embed_dim,
      kernel_size: {patch_size, patch_size},
      strides: [patch_size, patch_size],
      padding: :valid,
      name: "#{name}_patch_conv"
    )
    |> Axon.nx(
      fn x ->
        # Flatten spatial dims: [batch, h, w, c] -> [batch, h*w, c]
        {batch, h, w, c} = Nx.shape(x)
        Nx.reshape(x, {batch, h * w, c})
      end,
      name: "#{name}_flatten"
    )
  end

  defp build_token_embedding(input, codebook_size, embed_dim, name) do
    # One-hot encode and project
    Axon.layer(
      &token_embed_impl/2,
      [input],
      name: "#{name}_embed",
      codebook_size: codebook_size,
      embed_dim: embed_dim,
      op_name: :token_embed
    )
  end

  defp token_embed_impl(indices, opts) do
    codebook_size = opts[:codebook_size]
    embed_dim = opts[:embed_dim]

    # One-hot encoding
    one_hot =
      Nx.equal(
        Nx.new_axis(Nx.as_type(indices, :s64), -1),
        Nx.iota({1, 1, codebook_size})
      )
      |> Nx.as_type(:f32)

    # Simple linear projection (in practice would use learnable embedding table)
    # Here we use a deterministic projection for the structure
    proj = Nx.iota({codebook_size, embed_dim}, type: :f32)
    proj = Nx.divide(proj, Nx.tensor(codebook_size * embed_dim, type: :f32))

    Nx.dot(one_hot, proj)
  end

  defp upsample_tokens(x, opts) do
    factor = opts[:factor]
    target_tokens = opts[:target_tokens]
    embed_dim = opts[:embed_dim]

    batch = Nx.axis_size(x, 0)
    current_tokens = Nx.axis_size(x, 1)
    current_side = round(:math.sqrt(current_tokens))
    target_side = round(:math.sqrt(target_tokens))

    # Reshape to spatial, upsample, flatten back
    x
    |> Nx.reshape({batch, current_side, current_side, embed_dim})
    |> Nx.new_axis(2)
    |> Nx.new_axis(4)
    |> Nx.tile([1, 1, factor, 1, factor, 1])
    |> Nx.reshape({batch, target_side, target_side, embed_dim})
    |> Nx.reshape({batch, target_tokens, embed_dim})
  end

  defp maybe_dropout(x, rate, _name) when rate <= 0, do: x
  defp maybe_dropout(x, rate, name), do: Axon.dropout(x, rate: rate, name: name)

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the total number of tokens across all scales.
  """
  @spec total_tokens([pos_integer()]) :: pos_integer()
  def total_tokens(scales \\ @default_scales) do
    Enum.reduce(scales, 0, fn s, acc -> acc + s * s end)
  end

  @doc """
  Get recommended defaults for VAR.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: @default_hidden_size,
      num_layers: @default_num_layers,
      num_heads: @default_num_heads,
      mlp_ratio: @default_mlp_ratio,
      scales: @default_scales,
      codebook_size: @default_codebook_size
    ]
  end
end
