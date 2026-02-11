defmodule Edifice.Convolutional.Conv do
  @moduledoc """
  Conv1D and Conv2D building blocks for convolutional neural networks.

  Provides composable convolutional blocks and full model builders for both
  1D (sequence) and 2D (image) processing. Each block follows the pattern:
  convolution -> batch normalization -> activation -> dropout.

  ## Architecture (Conv1D)

  ```
  Input [batch, seq_len, channels]
        |
        v
  +---------------------------+
  | Conv1D -> BN -> Act -> Drop|  Block 1
  +---------------------------+
        |
        v
  +---------------------------+
  | Conv1D -> BN -> Act -> Drop|  Block 2
  +---------------------------+
        |        (optional pooling)
        v
  Output [batch, seq_len', channels']
  ```

  ## Usage

      # Build a 1D convolutional model for sequence processing
      model = Conv.build_conv1d(
        input_size: 64,
        channels: [128, 256, 512],
        kernel_sizes: [3, 3, 3],
        activation: :relu,
        dropout: 0.1,
        pooling: :max
      )

      # Build a 2D convolutional model for image processing
      model = Conv.build_conv2d(
        input_shape: {nil, 32, 32, 3},
        channels: [32, 64, 128],
        kernel_sizes: [3, 3, 3],
        activation: :relu,
        dropout: 0.1,
        pooling: :max
      )

      # Use individual blocks as building blocks
      input = Axon.input("input", shape: {nil, 100, 64})
      block = Conv.conv1d_block(input, 128, kernel_size: 3, activation: :gelu)
  """

  require Axon

  @default_activation :relu
  @default_dropout 0.1
  @default_kernel_size 3

  # ============================================================================
  # 1D Convolution
  # ============================================================================

  @doc """
  Build a 1D convolutional model for sequence processing.

  ## Options

    - `:input_size` - Number of input channels/features (required)
    - `:channels` - List of output channel counts per layer (default: [64, 128, 256])
    - `:kernel_sizes` - List of kernel sizes per layer, or single integer for all layers (default: 3)
    - `:activation` - Activation function atom (default: :relu)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:pooling` - Pooling type: `:max`, `:avg`, or `nil` for no pooling (default: nil)
    - `:seq_len` - Expected sequence length, or nil for dynamic (default: nil)

  ## Returns

    An Axon model taking `[batch, seq_len, input_size]` and outputting
    `[batch, seq_len', last_channel_count]`.
  """
  @spec build_conv1d(keyword()) :: Axon.t()
  def build_conv1d(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    channels = Keyword.get(opts, :channels, [64, 128, 256])
    kernel_sizes = Keyword.get(opts, :kernel_sizes, @default_kernel_size)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    pooling = Keyword.get(opts, :pooling, nil)
    seq_len = Keyword.get(opts, :seq_len, nil)

    kernel_sizes = normalize_kernel_sizes(kernel_sizes, length(channels))

    input = Axon.input("input", shape: {nil, seq_len, input_size})

    channels
    |> Enum.zip(kernel_sizes)
    |> Enum.with_index()
    |> Enum.reduce(input, fn {{ch, ks}, idx}, acc ->
      block =
        conv1d_block(acc, ch,
          kernel_size: ks,
          activation: activation,
          dropout: dropout,
          name: "conv1d_block_#{idx}"
        )

      maybe_pool_1d(block, pooling, idx)
    end)
  end

  @doc """
  Build a single conv1d -> batch_norm -> activation -> dropout block.

  ## Parameters

    - `input` - Input Axon node with shape `[batch, seq_len, channels]`
    - `out_channels` - Number of output channels

  ## Options

    - `:kernel_size` - Convolution kernel size (default: 3)
    - `:activation` - Activation function atom (default: :relu)
    - `:dropout` - Dropout rate, 0.0 to skip (default: 0.1)
    - `:name` - Layer name prefix (default: "conv1d_block")
    - `:padding` - Padding mode: `:same` or `:valid` (default: :same)
    - `:strides` - Convolution stride (default: 1)

  ## Returns

    An Axon node with shape `[batch, seq_len', out_channels]`.
  """
  @spec conv1d_block(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def conv1d_block(input, out_channels, opts \\ []) do
    kernel_size = Keyword.get(opts, :kernel_size, @default_kernel_size)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "conv1d_block")
    padding = Keyword.get(opts, :padding, :same)
    strides = Keyword.get(opts, :strides, 1)

    x =
      Axon.conv(input, out_channels,
        kernel_size: {kernel_size},
        padding: padding,
        strides: [strides],
        name: "#{name}_conv"
      )

    x = Axon.batch_norm(x, name: "#{name}_bn")
    x = Axon.activation(x, activation, name: "#{name}_act")

    if dropout > 0.0 do
      Axon.dropout(x, rate: dropout, name: "#{name}_drop")
    else
      x
    end
  end

  # ============================================================================
  # 2D Convolution
  # ============================================================================

  @doc """
  Build a 2D convolutional model for image processing.

  ## Options

    - `:input_shape` - Input shape as `{nil, height, width, channels}` (required)
    - `:channels` - List of output channel counts per layer (default: [32, 64, 128])
    - `:kernel_sizes` - List of kernel sizes per layer, or single integer for all layers (default: 3)
    - `:activation` - Activation function atom (default: :relu)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:pooling` - Pooling type: `:max`, `:avg`, or `nil` for no pooling (default: nil)

  ## Returns

    An Axon model taking `[batch, height, width, channels]` (NHWC) and outputting
    `[batch, height', width', last_channel_count]`.
  """
  @spec build_conv2d(keyword()) :: Axon.t()
  def build_conv2d(opts \\ []) do
    input_shape = Keyword.fetch!(opts, :input_shape)
    channels = Keyword.get(opts, :channels, [32, 64, 128])
    kernel_sizes = Keyword.get(opts, :kernel_sizes, @default_kernel_size)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    pooling = Keyword.get(opts, :pooling, nil)

    kernel_sizes = normalize_kernel_sizes(kernel_sizes, length(channels))

    input = Axon.input("input", shape: input_shape)

    channels
    |> Enum.zip(kernel_sizes)
    |> Enum.with_index()
    |> Enum.reduce(input, fn {{ch, ks}, idx}, acc ->
      block =
        conv2d_block(acc, ch,
          kernel_size: ks,
          activation: activation,
          dropout: dropout,
          name: "conv2d_block_#{idx}"
        )

      maybe_pool_2d(block, pooling, idx)
    end)
  end

  @doc """
  Build a single conv2d -> batch_norm -> activation -> dropout block.

  ## Parameters

    - `input` - Input Axon node with shape `[batch, height, width, channels]`
    - `out_channels` - Number of output channels

  ## Options

    - `:kernel_size` - Convolution kernel size (default: 3)
    - `:activation` - Activation function atom (default: :relu)
    - `:dropout` - Dropout rate, 0.0 to skip (default: 0.1)
    - `:name` - Layer name prefix (default: "conv2d_block")
    - `:padding` - Padding mode: `:same` or `:valid` (default: :same)
    - `:strides` - Convolution stride (default: 1)

  ## Returns

    An Axon node with shape `[batch, height', width', out_channels]`.
  """
  @spec conv2d_block(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def conv2d_block(input, out_channels, opts \\ []) do
    kernel_size = Keyword.get(opts, :kernel_size, @default_kernel_size)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "conv2d_block")
    padding = Keyword.get(opts, :padding, :same)
    strides = Keyword.get(opts, :strides, 1)

    x =
      Axon.conv(input, out_channels,
        kernel_size: {kernel_size, kernel_size},
        padding: padding,
        strides: [strides, strides],
        name: "#{name}_conv"
      )

    x = Axon.batch_norm(x, name: "#{name}_bn")
    x = Axon.activation(x, activation, name: "#{name}_act")

    if dropout > 0.0 do
      Axon.dropout(x, rate: dropout, name: "#{name}_drop")
    else
      x
    end
  end

  # ============================================================================
  # Output Size
  # ============================================================================

  @doc """
  Get the output channel count for a convolutional model with the given options.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    channels = Keyword.get(opts, :channels, [64, 128, 256])
    List.last(channels)
  end

  # ============================================================================
  # Private Helpers
  # ============================================================================

  defp normalize_kernel_sizes(ks, num_layers) when is_integer(ks) do
    List.duplicate(ks, num_layers)
  end

  defp normalize_kernel_sizes(ks, num_layers) when is_list(ks) do
    if length(ks) != num_layers do
      raise ArgumentError,
            "kernel_sizes list length (#{length(ks)}) must match channels list length (#{num_layers})"
    end

    ks
  end

  defp maybe_pool_1d(x, nil, _idx), do: x

  defp maybe_pool_1d(x, :max, idx) do
    Axon.max_pool(x, kernel_size: {2}, strides: [2], name: "max_pool_#{idx}")
  end

  defp maybe_pool_1d(x, :avg, idx) do
    Axon.avg_pool(x, kernel_size: {2}, strides: [2], name: "avg_pool_#{idx}")
  end

  defp maybe_pool_2d(x, nil, _idx), do: x

  defp maybe_pool_2d(x, :max, idx) do
    Axon.max_pool(x, kernel_size: {2, 2}, strides: [2, 2], name: "max_pool_#{idx}")
  end

  defp maybe_pool_2d(x, :avg, idx) do
    Axon.avg_pool(x, kernel_size: {2, 2}, strides: [2, 2], name: "avg_pool_#{idx}")
  end
end
