defmodule Edifice.Convolutional.ResNet do
  @moduledoc """
  Residual Network (ResNet) implementation.

  Deep residual networks use skip connections to enable training of very deep
  networks by mitigating the vanishing gradient problem. Each residual block
  adds its input to its output, allowing gradients to flow directly through
  identity shortcuts.

  ## Architecture

  ```
  Input [batch, H, W, C]
        |
  +-----v-------+
  | Stem         |  7x7 conv stride 2, BN, ReLU, 3x3 max pool
  +--------------+
        |
  +-----v-------+
  | Stage 1      |  N residual blocks at initial_channels
  +--------------+
        |
  +-----v-------+
  | Stage 2      |  N residual blocks at initial_channels * 2 (stride 2)
  +--------------+
        |
  +-----v-------+
  | Stage 3      |  N residual blocks at initial_channels * 4 (stride 2)
  +--------------+
        |
  +-----v-------+
  | Stage 4      |  N residual blocks at initial_channels * 8 (stride 2)
  +--------------+
        |
  +-----v-------+
  | Global AvgPool|
  +--------------+
        |
  +-----v-------+
  | Dense        |  num_classes outputs
  +--------------+
  ```

  ## Configurations

  | Model      | block_sizes     | Block Type  | Params |
  |------------|----------------|-------------|--------|
  | ResNet-18  | [2, 2, 2, 2]  | residual    | ~11M   |
  | ResNet-34  | [3, 4, 6, 3]  | residual    | ~21M   |
  | ResNet-50  | [3, 4, 6, 3]  | bottleneck  | ~25M   |
  | ResNet-101 | [3, 4, 23, 3] | bottleneck  | ~44M   |
  | ResNet-152 | [3, 8, 36, 3] | bottleneck  | ~60M   |

  ## Usage

      # ResNet-18 for CIFAR-10
      model = ResNet.build(
        input_shape: {nil, 32, 32, 3},
        num_classes: 10,
        block_sizes: [2, 2, 2, 2],
        initial_channels: 64
      )

      # ResNet-50 with bottleneck blocks
      model = ResNet.build(
        input_shape: {nil, 224, 224, 3},
        num_classes: 1000,
        block_sizes: [3, 4, 6, 3],
        block_type: :bottleneck,
        initial_channels: 64
      )
  """

  require Axon

  @default_initial_channels 64
  @default_block_sizes [2, 2, 2, 2]
  @default_num_classes 10

  # ============================================================================
  # Model Builder
  # ============================================================================

  @doc """
  Build a ResNet model.

  ## Options

    - `:input_shape` - Input shape as `{nil, height, width, channels}` (required)
    - `:num_classes` - Number of output classes (default: 10)
    - `:block_sizes` - List of block counts per stage (default: [2, 2, 2, 2] for ResNet-18)
    - `:block_type` - `:residual` or `:bottleneck` (default: :residual)
    - `:initial_channels` - Channels after stem conv (default: 64)
    - `:dropout` - Dropout rate before final dense layer (default: 0.0)

  ## Returns

    An Axon model outputting `[batch, num_classes]`.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    input_shape = Keyword.fetch!(opts, :input_shape)
    num_classes = Keyword.get(opts, :num_classes, @default_num_classes)
    block_sizes = Keyword.get(opts, :block_sizes, @default_block_sizes)
    block_type = Keyword.get(opts, :block_type, :residual)
    initial_channels = Keyword.get(opts, :initial_channels, @default_initial_channels)
    dropout = Keyword.get(opts, :dropout, 0.0)

    input = Axon.input("input", shape: input_shape)

    # Stem: 7x7 conv stride 2 -> BN -> ReLU -> 3x3 max pool
    x =
      Axon.conv(input, initial_channels,
        kernel_size: {7, 7},
        strides: [2, 2],
        padding: :same,
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

    # Build stages with increasing channel counts
    block_fn =
      case block_type do
        :residual -> &residual_block/3
        :bottleneck -> &bottleneck_block/3
      end

    # Track current channel count through stages for identity shortcuts
    expansion = if block_type == :bottleneck, do: 4, else: 1

    {x, _} =
      block_sizes
      |> Enum.with_index()
      |> Enum.reduce({x, initial_channels}, fn {num_blocks, stage_idx}, {acc, in_channels} ->
        channels = initial_channels * round(:math.pow(2, stage_idx))
        out_channels = channels * expansion
        # First stage has no downsampling, subsequent stages stride 2
        first_stride = if stage_idx == 0, do: 1, else: 2

        result =
          build_stage(
            acc,
            num_blocks,
            channels,
            first_stride,
            stage_idx,
            block_fn,
            in_channels,
            out_channels
          )

        {result, out_channels}
      end)

    # Global average pooling -> dense
    x = Axon.global_avg_pool(x, name: "global_avg_pool")

    x =
      if dropout > 0.0 do
        Axon.dropout(x, rate: dropout, name: "final_dropout")
      else
        x
      end

    Axon.dense(x, num_classes, name: "classifier")
  end

  # ============================================================================
  # Residual Block
  # ============================================================================

  @doc """
  Build a single residual block.

  Structure: conv 3x3 -> BN -> ReLU -> conv 3x3 -> BN + skip -> ReLU

  When input and output channels differ, a 1x1 projection is applied to the
  skip connection to match dimensions.

  ## Parameters

    - `input` - Input Axon node `[batch, H, W, C]`
    - `channels` - Number of output channels
    - `opts` - Options:
      - `:strides` - Convolution stride for downsampling (default: 1)
      - `:name` - Layer name prefix (default: "res_block")

  ## Returns

    An Axon node with shape `[batch, H', W', channels]`.
  """
  @spec residual_block(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def residual_block(input, channels, opts \\ []) do
    strides = Keyword.get(opts, :strides, 1)
    name = Keyword.get(opts, :name, "res_block")

    # Main path
    x =
      Axon.conv(input, channels,
        kernel_size: {3, 3},
        strides: [strides, strides],
        padding: :same,
        name: "#{name}_conv1"
      )

    x = Axon.batch_norm(x, name: "#{name}_bn1")
    x = Axon.activation(x, :relu, name: "#{name}_relu1")

    x =
      Axon.conv(x, channels,
        kernel_size: {3, 3},
        strides: [1, 1],
        padding: :same,
        name: "#{name}_conv2"
      )

    x = Axon.batch_norm(x, name: "#{name}_bn2")

    # Skip connection (identity when dims match, 1x1 projection otherwise)
    in_channels = Keyword.get(opts, :in_channels, channels)
    skip = maybe_project_skip(input, channels, strides, name, in_channels)

    # Add residual and activate
    Axon.add(x, skip, name: "#{name}_add")
    |> Axon.activation(:relu, name: "#{name}_relu_out")
  end

  # ============================================================================
  # Bottleneck Block
  # ============================================================================

  @doc """
  Build a bottleneck residual block.

  Structure: 1x1 conv (reduce) -> BN -> ReLU -> 3x3 conv -> BN -> ReLU -> 1x1 conv (expand) -> BN + skip -> ReLU

  Bottleneck blocks use a 4x expansion factor: the final 1x1 conv outputs
  `channels * 4` features. This is more parameter-efficient for deep networks.

  ## Parameters

    - `input` - Input Axon node `[batch, H, W, C]`
    - `channels` - Number of bottleneck channels (output will be `channels * 4`)
    - `opts` - Options:
      - `:strides` - Convolution stride for downsampling (default: 1)
      - `:expansion` - Expansion factor for output channels (default: 4)
      - `:name` - Layer name prefix (default: "bottleneck")

  ## Returns

    An Axon node with shape `[batch, H', W', channels * expansion]`.
  """
  @spec bottleneck_block(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def bottleneck_block(input, channels, opts \\ []) do
    strides = Keyword.get(opts, :strides, 1)
    expansion = Keyword.get(opts, :expansion, 4)
    name = Keyword.get(opts, :name, "bottleneck")

    out_channels = channels * expansion

    # 1x1 reduce
    x =
      Axon.conv(input, channels,
        kernel_size: {1, 1},
        strides: [1, 1],
        padding: :valid,
        name: "#{name}_conv1"
      )

    x = Axon.batch_norm(x, name: "#{name}_bn1")
    x = Axon.activation(x, :relu, name: "#{name}_relu1")

    # 3x3 conv (potentially with stride for downsampling)
    x =
      Axon.conv(x, channels,
        kernel_size: {3, 3},
        strides: [strides, strides],
        padding: :same,
        name: "#{name}_conv2"
      )

    x = Axon.batch_norm(x, name: "#{name}_bn2")
    x = Axon.activation(x, :relu, name: "#{name}_relu2")

    # 1x1 expand
    x =
      Axon.conv(x, out_channels,
        kernel_size: {1, 1},
        strides: [1, 1],
        padding: :valid,
        name: "#{name}_conv3"
      )

    x = Axon.batch_norm(x, name: "#{name}_bn3")

    # Skip connection (identity when dims match, 1x1 projection otherwise)
    in_channels = Keyword.get(opts, :in_channels, out_channels)
    skip = maybe_project_skip(input, out_channels, strides, name, in_channels)

    Axon.add(x, skip, name: "#{name}_add")
    |> Axon.activation(:relu, name: "#{name}_relu_out")
  end

  # ============================================================================
  # Output Size
  # ============================================================================

  @doc """
  Get the output size (num_classes) for a ResNet model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :num_classes, @default_num_classes)
  end

  # ============================================================================
  # Private Helpers
  # ============================================================================

  defp build_stage(
         input,
         num_blocks,
         channels,
         first_stride,
         stage_idx,
         block_fn,
         in_channels,
         out_channels
       ) do
    {result, _} =
      Enum.reduce(0..(num_blocks - 1), {input, in_channels}, fn block_idx, {acc, current_in} ->
        strides = if block_idx == 0, do: first_stride, else: 1

        result =
          block_fn.(acc, channels,
            strides: strides,
            name: "stage#{stage_idx}_block#{block_idx}",
            in_channels: current_in
          )

        {result, out_channels}
      end)

    result
  end

  defp maybe_project_skip(input, out_channels, strides, name, in_channels) do
    if strides == 1 and in_channels == out_channels do
      # Identity shortcut: dimensions match, no projection needed.
      # This is the key insight from He et al. — gradients flow directly
      # through the identity path for better training.
      input
    else
      # Project via 1x1 conv when spatial dims or channels change
      input
      |> Axon.conv(out_channels,
        kernel_size: {1, 1},
        strides: [strides, strides],
        padding: :valid,
        name: "#{name}_skip_proj"
      )
      |> Axon.batch_norm(name: "#{name}_skip_bn")
    end
  end
end
