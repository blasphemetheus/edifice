defmodule Edifice.Convolutional.DenseNet do
  @moduledoc """
  DenseNet (Densely Connected Convolutional Network) implementation.

  DenseNet connects each layer to every other layer in a feed-forward fashion.
  Within a dense block, each layer receives the feature maps of all preceding
  layers as input (via concatenation), encouraging feature reuse and reducing
  parameter count compared to traditional CNNs.

  ## Architecture

  ```
  Input [batch, H, W, C]
        |
  +-----v----------+
  | Stem            |  7x7 conv stride 2, BN, ReLU, 3x3 max pool
  +--+--------------+
     |
  +--v--------------+
  | Dense Block 1    |  Each layer concatenates all previous feature maps
  +--+--------------+
     |
  +--v--------------+
  | Transition 1     |  1x1 conv (compress) + 2x2 avg pool (downsample)
  +--+--------------+
     |
  +--v--------------+
  | Dense Block 2    |
  +--+--------------+
     |
     ... (repeat)
     |
  +--v--------------+
  | Final BN + ReLU  |
  +--+--------------+
     |
  +--v--------------+
  | Global AvgPool   |
  +--+--------------+
     |
  +--v--------------+
  | Dense            |  num_classes outputs
  +-----------------+
  ```

  ## Configurations

  | Model         | block_config       | growth_rate | Params |
  |---------------|-------------------|-------------|--------|
  | DenseNet-121  | [6, 12, 24, 16]  | 32          | ~8M    |
  | DenseNet-169  | [6, 12, 32, 32]  | 32          | ~14M   |
  | DenseNet-201  | [6, 12, 48, 32]  | 32          | ~20M   |
  | DenseNet-264  | [6, 12, 64, 48]  | 32          | ~34M   |

  ## Usage

      # DenseNet-121 for CIFAR-10
      model = DenseNet.build(
        input_shape: {nil, 32, 32, 3},
        num_classes: 10,
        growth_rate: 32,
        block_config: [6, 12, 24, 16]
      )

      # Compact DenseNet for small datasets
      model = DenseNet.build(
        input_shape: {nil, 32, 32, 3},
        num_classes: 10,
        growth_rate: 12,
        block_config: [4, 8, 12, 8],
        compression: 0.5
      )
  """

  require Axon

  @default_growth_rate 32
  @default_block_config [6, 12, 24, 16]
  @default_num_classes 10
  @default_compression 0.5

  # ============================================================================
  # Model Builder
  # ============================================================================

  @doc """
  Build a DenseNet model.

  ## Options

    - `:input_shape` - Input shape as `{nil, height, width, channels}` (required)
    - `:num_classes` - Number of output classes (default: 10)
    - `:growth_rate` - Number of new feature maps per dense layer (default: 32)
    - `:block_config` - List of layer counts per dense block (default: [6, 12, 24, 16])
    - `:compression` - Compression factor for transitions, 0.0-1.0 (default: 0.5)
    - `:initial_channels` - Channels after stem conv (default: growth_rate * 2)
    - `:dropout` - Dropout rate in dense layers (default: 0.0)
    - `:bn_size` - Bottleneck width multiplier for BN-ReLU-1x1 layers (default: 4)

  ## Returns

    An Axon model outputting `[batch, num_classes]`.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    input_shape = Keyword.fetch!(opts, :input_shape)
    num_classes = Keyword.get(opts, :num_classes, @default_num_classes)
    growth_rate = Keyword.get(opts, :growth_rate, @default_growth_rate)
    block_config = Keyword.get(opts, :block_config, @default_block_config)
    compression = Keyword.get(opts, :compression, @default_compression)
    initial_channels = Keyword.get(opts, :initial_channels, growth_rate * 2)
    dropout = Keyword.get(opts, :dropout, 0.0)
    bn_size = Keyword.get(opts, :bn_size, 4)

    input = Axon.input("input", shape: input_shape)

    # Stem: 7x7 conv stride 2 -> BN -> ReLU -> 3x3 max pool
    x =
      Axon.conv(input, initial_channels,
        kernel_size: {7, 7},
        strides: [2, 2],
        padding: :same,
        use_bias: false,
        name: "stem_conv"
      )

    x = Axon.batch_norm(x, name: "stem_bn")
    x = Axon.activation(x, :relu, name: "stem_relu")

    x =
      Axon.max_pool(x,
        kernel_size: {3, 3},
        strides: [2, 2],
        padding: :same,
        name: "stem_pool"
      )

    # Track current number of channels for transitions
    num_blocks = length(block_config)

    {x, _num_channels} =
      block_config
      |> Enum.with_index()
      |> Enum.reduce({x, initial_channels}, fn {num_layers, block_idx}, {acc, num_channels} ->
        # Dense block
        {block_out, new_channels} =
          dense_block(acc, num_layers,
            growth_rate: growth_rate,
            num_channels: num_channels,
            dropout: dropout,
            bn_size: bn_size,
            name: "dense_block_#{block_idx}"
          )

        # Transition layer (except after last block)
        if block_idx < num_blocks - 1 do
          compressed_channels = floor(new_channels * compression)

          t =
            transition_layer(block_out, compressed_channels, name: "transition_#{block_idx}")

          {t, compressed_channels}
        else
          {block_out, new_channels}
        end
      end)

    # Final batch norm + activation
    x = Axon.batch_norm(x, name: "final_bn")
    x = Axon.activation(x, :relu, name: "final_relu")

    # Global average pooling -> classifier
    x = Axon.global_avg_pool(x, name: "global_avg_pool")
    Axon.dense(x, num_classes, name: "classifier")
  end

  # ============================================================================
  # Dense Block
  # ============================================================================

  @doc """
  Build a dense block where each layer receives all previous feature maps.

  Within a dense block, layer `i` receives the concatenation of feature maps
  from layers `0, 1, ..., i-1` as input. Each layer produces `growth_rate`
  new feature maps.

  ## Parameters

    - `input` - Input Axon node `[batch, H, W, C]`
    - `num_layers` - Number of dense layers in this block

  ## Options

    - `:growth_rate` - New feature maps per layer (default: 32)
    - `:num_channels` - Current number of input channels (required for tracking)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:bn_size` - Bottleneck width multiplier (default: 4)
    - `:name` - Layer name prefix (default: "dense_block")

  ## Returns

    Tuple of `{output_node, total_channels}` where `total_channels` is
    `num_channels + num_layers * growth_rate`.
  """
  @spec dense_block(Axon.t(), pos_integer(), keyword()) :: {Axon.t(), pos_integer()}
  def dense_block(input, num_layers, opts \\ []) do
    growth_rate = Keyword.get(opts, :growth_rate, @default_growth_rate)
    num_channels = Keyword.fetch!(opts, :num_channels)
    dropout = Keyword.get(opts, :dropout, 0.0)
    bn_size = Keyword.get(opts, :bn_size, 4)
    name = Keyword.get(opts, :name, "dense_block")

    {output, final_channels} =
      Enum.reduce(0..(num_layers - 1), {input, num_channels}, fn layer_idx,
                                                                 {acc, _current_channels} ->
        # BN -> ReLU -> 1x1 conv (bottleneck) -> BN -> ReLU -> 3x3 conv
        new_features =
          dense_layer(acc, growth_rate,
            bn_size: bn_size,
            dropout: dropout,
            name: "#{name}_layer_#{layer_idx}"
          )

        # Concatenate with all previous feature maps along channel axis
        concatenated = Axon.concatenate(acc, new_features, name: "#{name}_cat_#{layer_idx}")

        # Channel count grows by growth_rate per layer
        new_total = num_channels + (layer_idx + 1) * growth_rate
        {concatenated, new_total}
      end)

    {output, final_channels}
  end

  # ============================================================================
  # Transition Layer
  # ============================================================================

  @doc """
  Build a transition layer between dense blocks.

  Transitions reduce spatial dimensions (2x downsampling via average pooling)
  and optionally compress the number of feature maps with a 1x1 convolution.

  ## Parameters

    - `input` - Input Axon node `[batch, H, W, C]`
    - `out_channels` - Number of output channels after compression

  ## Options

    - `:name` - Layer name prefix (default: "transition")

  ## Returns

    An Axon node with shape `[batch, H/2, W/2, out_channels]`.
  """
  @spec transition_layer(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def transition_layer(input, out_channels, opts \\ []) do
    name = Keyword.get(opts, :name, "transition")

    input
    |> Axon.batch_norm(name: "#{name}_bn")
    |> Axon.activation(:relu, name: "#{name}_relu")
    |> Axon.conv(out_channels,
      kernel_size: {1, 1},
      strides: [1, 1],
      padding: :valid,
      use_bias: false,
      name: "#{name}_conv"
    )
    |> Axon.avg_pool(kernel_size: {2, 2}, strides: [2, 2], name: "#{name}_pool")
  end

  # ============================================================================
  # Output Size
  # ============================================================================

  @doc """
  Get the output size (num_classes) for a DenseNet model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :num_classes, @default_num_classes)
  end

  # ============================================================================
  # Private Helpers
  # ============================================================================

  # A single dense layer: BN -> ReLU -> 1x1 bottleneck -> BN -> ReLU -> 3x3 conv
  defp dense_layer(input, growth_rate, opts) do
    bn_size = Keyword.get(opts, :bn_size, 4)
    dropout = Keyword.get(opts, :dropout, 0.0)
    name = Keyword.get(opts, :name, "dense_layer")

    bottleneck_channels = bn_size * growth_rate

    # Bottleneck: BN -> ReLU -> 1x1 conv
    x = Axon.batch_norm(input, name: "#{name}_bn1")
    x = Axon.activation(x, :relu, name: "#{name}_relu1")

    x =
      Axon.conv(x, bottleneck_channels,
        kernel_size: {1, 1},
        strides: [1, 1],
        padding: :valid,
        use_bias: false,
        name: "#{name}_conv1"
      )

    # Main: BN -> ReLU -> 3x3 conv
    x = Axon.batch_norm(x, name: "#{name}_bn2")
    x = Axon.activation(x, :relu, name: "#{name}_relu2")

    x =
      Axon.conv(x, growth_rate,
        kernel_size: {3, 3},
        strides: [1, 1],
        padding: :same,
        use_bias: false,
        name: "#{name}_conv2"
      )

    if dropout > 0.0 do
      Axon.dropout(x, rate: dropout, name: "#{name}_drop")
    else
      x
    end
  end
end
