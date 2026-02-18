defmodule Edifice.Vision.UNet do
  @moduledoc """
  U-Net encoder-decoder architecture with skip connections.

  Originally designed for biomedical image segmentation, U-Net uses a symmetric
  encoder-decoder structure with skip connections that concatenate encoder features
  at each level with decoder features, preserving fine-grained spatial information.

  This implementation uses real 2D convolutions, max-pooling for downsampling,
  and transposed convolutions for upsampling — faithful to the original paper.

  ## Architecture

  ```
  Image [batch, channels, height, width]
        |
  +-----v--------------------+
  | Transpose to NHWC         |  [batch, H, W, C]
  +---------------------------+
        |
  +-----v--------------------+       Skip Connections
  | Encoder Level 1           |  ----------+
  |   Conv 3x3 + BN + ReLU   |            |
  |   Conv 3x3 + BN + ReLU   |            |
  |   MaxPool 2x2             |            |
  +---------------------------+            |
        |                                  |
  +-----v--------------------+             |
  | Encoder Level 2           |  -----+    |
  |   Conv 3x3 + BN + ReLU   |       |    |
  |   Conv 3x3 + BN + ReLU   |       |    |
  |   MaxPool 2x2             |       |    |
  +---------------------------+       |    |
        |                             |    |
        ... (depth levels)            |    |
        |                             |    |
  +-----v--------------------+       |    |
  | Bottleneck                |       |    |
  |   Conv 3x3 + BN + ReLU   |       |    |
  |   Conv 3x3 + BN + ReLU   |       |    |
  +---------------------------+       |    |
        |                             |    |
  +-----v--------------------+       |    |
  | Decoder Level 2           |       |    |
  |   ConvTranspose 2x2 (up) |       |    |
  |   Concat skip <-----------+------+    |
  |   Conv 3x3 + BN + ReLU   |            |
  |   Conv 3x3 + BN + ReLU   |            |
  +---------------------------+            |
        |                                  |
  +-----v--------------------+             |
  | Decoder Level 1           |            |
  |   ConvTranspose 2x2 (up) |            |
  |   Concat skip <-----------+------------+
  |   Conv 3x3 + BN + ReLU   |
  |   Conv 3x3 + BN + ReLU   |
  +---------------------------+
        |
  +-----v--------------------+
  | Output Conv 1x1           |  [batch, H, W, out_channels]
  +---------------------------+
        |
  +-----v--------------------+
  | Transpose to NCHW         |  [batch, out_channels, H, W]
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
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:base_features, pos_integer()}
          | {:depth, pos_integer()}
          | {:dropout, float()}
          | {:image_size, pos_integer()}
          | {:in_channels, pos_integer()}
          | {:out_channels, pos_integer()}
          | {:use_attention, boolean()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    in_channels = Keyword.get(opts, :in_channels, @default_in_channels)
    out_channels = Keyword.get(opts, :out_channels, @default_out_channels)
    image_size = Keyword.get(opts, :image_size, @default_image_size)
    base_features = Keyword.get(opts, :base_features, @default_base_features)
    depth = Keyword.get(opts, :depth, @default_depth)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    use_attention = Keyword.get(opts, :use_attention, false)

    # Input: NCHW [batch, channels, height, width]
    input = Axon.input("image", shape: {nil, in_channels, image_size, image_size})

    # Transpose to NHWC for Axon convolutions (channels_last default)
    x =
      Axon.nx(
        input,
        fn tensor -> Nx.transpose(tensor, axes: [0, 2, 3, 1]) end,
        name: "nchw_to_nhwc"
      )

    # Encoder path: conv blocks + max pool, collect skip connections
    {x, skips} = encoder_path(x, base_features, depth, dropout)

    # Bottleneck
    bottleneck_features = base_features * round(:math.pow(2, depth))
    x = conv_block(x, bottleneck_features, dropout, name: "bottleneck")

    x =
      if use_attention do
        bottleneck_attention(x, bottleneck_features, name: "bottleneck_attn")
      else
        x
      end

    # Decoder path: upsample + concat skip + conv block
    x = decoder_path(x, skips, base_features, depth, dropout)

    # Output: 1x1 convolution to map to out_channels
    x = Axon.conv(x, out_channels, kernel_size: {1, 1}, name: "output_conv")

    # Transpose back to NCHW [batch, out_channels, H, W]
    Axon.nx(
      x,
      fn tensor -> Nx.transpose(tensor, axes: [0, 3, 1, 2]) end,
      name: "nhwc_to_nchw"
    )
  end

  # ============================================================================
  # Encoder
  # ============================================================================

  defp encoder_path(input, base_features, depth, dropout) do
    Enum.reduce(0..(depth - 1), {input, []}, fn level, {x, skips} ->
      features = base_features * round(:math.pow(2, level))

      # Double conv block at this level
      x = conv_block(x, features, dropout, name: "enc_#{level}")

      # Save skip connection (before pooling)
      skips = skips ++ [x]

      # Downsample: max pool 2x2 with stride 2
      x = Axon.max_pool(x, kernel_size: {2, 2}, strides: [2, 2], name: "enc_#{level}_pool")

      {x, skips}
    end)
  end

  # ============================================================================
  # Decoder
  # ============================================================================

  defp decoder_path(x, skips, base_features, depth, dropout) do
    reversed_skips = Enum.reverse(skips)

    Enum.reduce(Enum.with_index(reversed_skips), x, fn {skip, idx}, acc ->
      dec_level = depth - 1 - idx
      features = base_features * round(:math.pow(2, dec_level))

      # Upsample: transposed convolution (doubles spatial dimensions)
      acc =
        Axon.conv_transpose(acc, features,
          kernel_size: {2, 2},
          strides: [2, 2],
          name: "dec_#{dec_level}_up"
        )

      # Concatenate with skip connection along channel axis (last axis in NHWC)
      acc = Axon.concatenate(acc, skip, name: "dec_#{dec_level}_cat")

      # Double conv block (processes concatenated features)
      conv_block(acc, features, dropout, name: "dec_#{dec_level}")
    end)
  end

  # ============================================================================
  # Building Blocks
  # ============================================================================

  # Standard UNet conv block: two 3x3 convolutions with batch norm and ReLU
  defp conv_block(input, features, dropout, opts) do
    name = Keyword.get(opts, :name, "conv_block")

    x =
      input
      |> Axon.conv(features, kernel_size: {3, 3}, padding: :same, name: "#{name}_conv1")
      |> Axon.batch_norm(name: "#{name}_bn1")
      |> Axon.activation(:relu, name: "#{name}_act1")
      |> Axon.conv(features, kernel_size: {3, 3}, padding: :same, name: "#{name}_conv2")
      |> Axon.batch_norm(name: "#{name}_bn2")
      |> Axon.activation(:relu, name: "#{name}_act2")

    if dropout > 0.0 do
      Axon.dropout(x, rate: dropout, name: "#{name}_drop")
    else
      x
    end
  end

  # SE-Net style channel attention for bottleneck
  defp bottleneck_attention(input, dim, opts) do
    name = Keyword.get(opts, :name, "attn")

    # Global average pool: [batch, H, W, C] -> [batch, 1, 1, C]
    gap =
      Axon.nx(
        input,
        fn tensor -> Nx.mean(tensor, axes: [1, 2], keep_axes: true) end,
        name: "#{name}_gap"
      )

    # Channel attention gate via 1x1 convolutions (squeeze-and-excitation)
    gate =
      gap
      |> Axon.conv(max(div(dim, 4), 1), kernel_size: {1, 1}, name: "#{name}_fc1")
      |> Axon.activation(:relu, name: "#{name}_fc1_act")
      |> Axon.conv(dim, kernel_size: {1, 1}, name: "#{name}_fc2")
      |> Axon.activation(:sigmoid, name: "#{name}_gate")

    # Scale input by attention gate (broadcasts over spatial dims)
    Axon.layer(
      fn feat, g, _opts -> Nx.multiply(feat, g) end,
      [input, gate],
      name: "#{name}_scale",
      op_name: :channel_attention
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
