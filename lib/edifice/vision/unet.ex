defmodule Edifice.Vision.UNet do
  @moduledoc """
  U-Net encoder-decoder architecture with skip connections.

  Originally designed for biomedical image segmentation, U-Net uses a symmetric
  encoder-decoder structure with skip connections that concatenate encoder features
  at each level with decoder features, preserving fine-grained spatial information.

  This implementation operates on flattened spatial representations using dense
  layers, making it compatible with Axon's tensor-based pipeline. For image
  inputs, the spatial dimensions are flattened and restored during processing.

  ## Architecture

  ```
  Image [batch, channels, height, width]
        |
  +-----v--------------------+
  | Flatten                   |  [batch, C * H * W]
  +---------------------------+
        |
  +-----v--------------------+       Skip Connections
  | Encoder Level 1           |  ----------+
  |   Dense + Act + Dense     |            |
  |   Downsample (Dense /2)   |            |
  +---------------------------+            |
        |                                  |
  +-----v--------------------+             |
  | Encoder Level 2           |  -----+    |
  |   Dense + Act + Dense     |       |    |
  |   Downsample (Dense /2)   |       |    |
  +---------------------------+       |    |
        |                             |    |
        ... (depth levels)            |    |
        |                             |    |
  +-----v--------------------+       |    |
  | Bottleneck                |       |    |
  |   Dense + Act + Dense     |       |    |
  +---------------------------+       |    |
        |                             |    |
  +-----v--------------------+       |    |
  | Decoder Level 2           |       |    |
  |   Upsample (Dense x2)    |       |    |
  |   Concat skip <-----------+------+    |
  |   Dense + Act + Dense     |            |
  +---------------------------+            |
        |                                  |
  +-----v--------------------+             |
  | Decoder Level 1           |            |
  |   Upsample (Dense x2)    |            |
  |   Concat skip <-----------+------------+
  |   Dense + Act + Dense     |
  +---------------------------+
        |
  +-----v--------------------+
  | Output projection         |  Dense -> out_channels * H * W
  +---------------------------+
        |
  +-----v--------------------+
  | Reshape                   |  [batch, out_channels, H, W]
  +---------------------------+
  ```

  ## Usage

      # Basic U-Net for segmentation
      model = UNet.build(
        in_channels: 3,
        out_channels: 1,
        image_size: 256,
        base_features: 64,
        depth: 4
      )

      # Shallow U-Net for small images
      model = UNet.build(
        in_channels: 1,
        out_channels: 10,
        image_size: 28,
        base_features: 32,
        depth: 3
      )

  ## References

  - "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    (Ronneberger et al., MICCAI 2015)
  """

  require Axon

  @default_in_channels 3
  @default_out_channels 1
  @default_image_size 256
  @default_base_features 64
  @default_depth 4
  @default_dropout 0.0

  # ============================================================================
  # Model Builder
  # ============================================================================

  @doc """
  Build a U-Net model.

  ## Options

    - `:in_channels` - Number of input channels (default: 3)
    - `:out_channels` - Number of output channels (default: 1)
    - `:image_size` - Input image size, square (default: 256)
    - `:base_features` - Feature count at first encoder level (default: 64)
    - `:depth` - Number of encoder/decoder levels (default: 4)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:use_attention` - Add attention at bottleneck (default: false)

  ## Returns

    An Axon model outputting `[batch, out_channels, image_size, image_size]`.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    in_channels = Keyword.get(opts, :in_channels, @default_in_channels)
    out_channels = Keyword.get(opts, :out_channels, @default_out_channels)
    image_size = Keyword.get(opts, :image_size, @default_image_size)
    base_features = Keyword.get(opts, :base_features, @default_base_features)
    depth = Keyword.get(opts, :depth, @default_depth)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    use_attention = Keyword.get(opts, :use_attention, false)

    flat_size = in_channels * image_size * image_size

    # Input: [batch, channels, height, width]
    input = Axon.input("image", shape: {nil, in_channels, image_size, image_size})

    # Flatten to [batch, C * H * W]
    x =
      Axon.nx(
        input,
        fn tensor ->
          batch = Nx.axis_size(tensor, 0)
          Nx.reshape(tensor, {batch, :auto})
        end,
        name: "flatten"
      )

    # Encoder path: collect skip connections
    {x, skips, feature_sizes} = encoder_path(x, flat_size, base_features, depth, dropout)

    # Bottleneck
    bottleneck_features = base_features * round(:math.pow(2, depth))
    x = conv_block(x, bottleneck_features, dropout, name: "bottleneck")

    x =
      if use_attention do
        bottleneck_attention(x, bottleneck_features, name: "bottleneck_attn")
      else
        x
      end

    # Decoder path: use skip connections in reverse
    x = decoder_path(x, skips, feature_sizes, base_features, depth, dropout)

    # Output projection: project to out_channels * H * W
    output_flat_size = out_channels * image_size * image_size

    x = Axon.dense(x, output_flat_size, name: "output_proj")

    # Reshape to [batch, out_channels, H, W]
    Axon.nx(
      x,
      fn tensor ->
        batch = Nx.axis_size(tensor, 0)
        Nx.reshape(tensor, {batch, out_channels, image_size, image_size})
      end,
      name: "output_reshape"
    )
  end

  # ============================================================================
  # Encoder
  # ============================================================================

  defp encoder_path(input, _flat_size, base_features, depth, dropout) do
    Enum.reduce(0..(depth - 1), {input, [], []}, fn level, {x, skips, sizes} ->
      features = base_features * round(:math.pow(2, level))

      # Conv block at this level
      x = conv_block(x, features, dropout, name: "enc_#{level}")

      # Save skip connection
      skips = skips ++ [x]
      sizes = sizes ++ [features]

      # Downsample: reduce spatial representation by 2x (halve feature count)
      down_features = features
      x = Axon.dense(x, div(down_features, 2), name: "enc_#{level}_down")
      x = Axon.activation(x, :relu, name: "enc_#{level}_down_act")

      {x, skips, sizes}
    end)
  end

  # ============================================================================
  # Decoder
  # ============================================================================

  defp decoder_path(x, skips, feature_sizes, _base_features, depth, dropout) do
    # Process decoder levels in reverse order
    reversed_skips = Enum.reverse(skips)
    reversed_sizes = Enum.reverse(feature_sizes)

    Enum.reduce(
      Enum.zip([reversed_skips, reversed_sizes, 0..(depth - 1)]),
      x,
      fn {skip, skip_features, level_idx}, acc ->
        dec_level = depth - 1 - level_idx

        # Upsample: project to match skip connection size
        acc = Axon.dense(acc, skip_features, name: "dec_#{dec_level}_up")
        acc = Axon.activation(acc, :relu, name: "dec_#{dec_level}_up_act")

        # Concatenate with skip connection
        acc = Axon.concatenate(acc, skip, name: "dec_#{dec_level}_cat")

        # Conv block (processes concatenated features: 2x skip_features input)
        conv_block(acc, skip_features, dropout, name: "dec_#{dec_level}")
      end
    )
  end

  # ============================================================================
  # Building Blocks
  # ============================================================================

  defp conv_block(input, features, dropout, opts) do
    name = Keyword.get(opts, :name, "conv_block")

    x =
      input
      |> Axon.dense(features, name: "#{name}_dense1")
      |> Axon.layer_norm(name: "#{name}_norm1")
      |> Axon.activation(:relu, name: "#{name}_act1")
      |> Axon.dense(features, name: "#{name}_dense2")
      |> Axon.layer_norm(name: "#{name}_norm2")
      |> Axon.activation(:relu, name: "#{name}_act2")

    if dropout > 0.0 do
      Axon.dropout(x, rate: dropout, name: "#{name}_drop")
    else
      x
    end
  end

  defp bottleneck_attention(input, dim, opts) do
    name = Keyword.get(opts, :name, "attn")

    # Split bottleneck features into a pseudo-sequence for self-attention
    # [batch, dim] -> [batch, num_tokens, token_dim] -> attend -> [batch, dim]
    num_tokens = 4
    token_dim = div(dim, num_tokens)

    x =
      Axon.nx(
        input,
        fn tensor ->
          batch = Nx.axis_size(tensor, 0)
          Nx.reshape(tensor, {batch, num_tokens, token_dim})
        end,
        name: "#{name}_to_seq"
      )

    # QKV projection and attention
    qkv = Axon.dense(x, token_dim * 3, name: "#{name}_qkv")

    attended =
      Axon.nx(
        qkv,
        fn qkv_tensor ->
          q = Nx.slice_along_axis(qkv_tensor, 0, token_dim, axis: 2)
          k = Nx.slice_along_axis(qkv_tensor, token_dim, token_dim, axis: 2)
          v = Nx.slice_along_axis(qkv_tensor, token_dim * 2, token_dim, axis: 2)

          d_k = Nx.axis_size(k, -1)
          scale = Nx.sqrt(d_k) |> Nx.as_type(Nx.type(q))

          scores = Nx.dot(q, [2], [0], k, [2], [0])
          scores = Nx.divide(scores, scale)

          max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
          exp_scores = Nx.exp(Nx.subtract(scores, max_scores))
          weights = Nx.divide(exp_scores, Nx.sum(exp_scores, axes: [-1], keep_axes: true))

          Nx.dot(weights, [2], [0], v, [1], [0])
        end,
        name: "#{name}_compute"
      )

    attended = Axon.dense(attended, token_dim, name: "#{name}_proj")

    # Residual connection on the sequence
    x = Axon.add(x, attended, name: "#{name}_residual")

    # Flatten back to [batch, dim]
    Axon.nx(
      x,
      fn tensor ->
        batch = Nx.axis_size(tensor, 0)
        Nx.reshape(tensor, {batch, :auto})
      end,
      name: "#{name}_flatten"
    )
  end

  # ============================================================================
  # Output Size
  # ============================================================================

  @doc """
  Get the output size of a UNet model.

  Returns `out_channels * image_size * image_size` (flattened spatial output).
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    out_channels = Keyword.get(opts, :out_channels, @default_out_channels)
    image_size = Keyword.get(opts, :image_size, @default_image_size)
    out_channels * image_size * image_size
  end
end
