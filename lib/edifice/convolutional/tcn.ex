defmodule Edifice.Convolutional.TCN do
  @moduledoc """
  Temporal Convolutional Network (TCN) for sequence modeling.

  TCNs use dilated causal convolutions to achieve large receptive fields
  efficiently. Each layer doubles the dilation rate (1, 2, 4, 8, ...),
  allowing the receptive field to grow exponentially with depth while
  maintaining linear parameter growth.

  Key properties:
  - **Causal**: output at time t depends only on inputs at times <= t
  - **Flexible receptive field**: receptive field = num_layers * kernel_size * max_dilation
  - **Parallelizable**: unlike RNNs, all timesteps can be computed in parallel
  - **Residual connections**: each temporal block has a skip connection

  ## Architecture

  ```
  Input [batch, seq_len, features]
        |
  +-----v-----------+
  | Temporal Block 1 |  dilation = 1
  |  dilated conv    |
  |  -> BN -> act    |
  |  -> dilated conv |
  |  -> BN -> act    |
  |  + residual skip |
  +-----------------+
        |
  +-----v-----------+
  | Temporal Block 2 |  dilation = 2
  |  (same pattern)  |
  +-----------------+
        |
  +-----v-----------+
  | Temporal Block 3 |  dilation = 4
  |  (same pattern)  |
  +-----------------+
        |
        v
  Output [batch, seq_len, channels]
  ```

  ## Receptive Field

  For `n` layers with kernel size `k`:

      receptive_field = 1 + 2 * (k - 1) * (2^n - 1)

  Examples:
  - 4 layers, k=3: receptive field = 31
  - 6 layers, k=3: receptive field = 127
  - 8 layers, k=3: receptive field = 511
  - 4 layers, k=7: receptive field = 181

  ## Usage

      # TCN for sequence classification
      model = TCN.build(
        input_size: 64,
        channels: [128, 128, 128, 128],
        kernel_size: 3,
        dropout: 0.1
      )

      # Calculate required layers for a target receptive field
      layers = TCN.layers_for_receptive_field(256, kernel_size: 3)
  """

  require Axon

  @default_channels [64, 64, 64, 64]
  @default_kernel_size 3
  @default_dropout 0.1

  # ============================================================================
  # Model Builder
  # ============================================================================

  @doc """
  Build a Temporal Convolutional Network.

  ## Options

    - `:input_size` - Number of input features per timestep (required)
    - `:channels` - List of channel counts per temporal block (default: [64, 64, 64, 64])
    - `:kernel_size` - Convolution kernel size, must be odd for symmetric padding (default: 3)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:seq_len` - Expected sequence length, or nil for dynamic (default: nil)
    - `:activation` - Activation function (default: :relu)

  ## Returns

    An Axon model taking `[batch, seq_len, input_size]` and outputting
    `[batch, seq_len, last_channel_count]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:activation, atom()}
          | {:channels, pos_integer()}
          | {:dropout, float()}
          | {:input_size, pos_integer()}
          | {:kernel_size, pos_integer()}
          | {:seq_len, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    channels = Keyword.get(opts, :channels, @default_channels)
    kernel_size = Keyword.get(opts, :kernel_size, @default_kernel_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    seq_len = Keyword.get(opts, :seq_len, nil)
    activation = Keyword.get(opts, :activation, :relu)

    input = Axon.input("input", shape: {nil, seq_len, input_size})

    channels
    |> Enum.with_index()
    |> Enum.reduce(input, fn {ch, idx}, acc ->
      dilation = round(:math.pow(2, idx))

      temporal_block(acc, ch,
        kernel_size: kernel_size,
        dilation: dilation,
        dropout: dropout,
        activation: activation,
        name: "tcn_block_#{idx}"
      )
    end)
  end

  # ============================================================================
  # Temporal Block
  # ============================================================================

  @doc """
  Build a single TCN temporal block with dilated causal convolution and residual.

  Each temporal block contains two dilated causal convolution layers with
  batch normalization, activation, and dropout, plus a residual skip connection.

  ## Parameters

    - `input` - Input Axon node `[batch, seq_len, channels]`
    - `out_channels` - Number of output channels

  ## Options

    - `:kernel_size` - Convolution kernel size (default: 3)
    - `:dilation` - Dilation rate for this block (default: 1)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:activation` - Activation function (default: :relu)
    - `:name` - Layer name prefix (default: "tcn_block")

  ## Returns

    An Axon node with shape `[batch, seq_len, out_channels]`.
  """
  @spec temporal_block(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def temporal_block(input, out_channels, opts \\ []) do
    kernel_size = Keyword.get(opts, :kernel_size, @default_kernel_size)
    dilation = Keyword.get(opts, :dilation, 1)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    activation = Keyword.get(opts, :activation, :relu)
    name = Keyword.get(opts, :name, "tcn_block")

    # Causal padding: pad only on the left side so output at time t
    # depends only on inputs at times <= t.
    # Total padding needed = (kernel_size - 1) * dilation
    causal_pad = (kernel_size - 1) * dilation

    # First dilated causal conv
    x = causal_conv1d(input, out_channels, kernel_size, dilation, causal_pad, "#{name}_conv1")
    x = Axon.batch_norm(x, name: "#{name}_bn1")
    x = Axon.activation(x, activation, name: "#{name}_act1")

    x =
      if dropout > 0.0 do
        Axon.dropout(x, rate: dropout, name: "#{name}_drop1")
      else
        x
      end

    # Second dilated causal conv
    x = causal_conv1d(x, out_channels, kernel_size, dilation, causal_pad, "#{name}_conv2")
    x = Axon.batch_norm(x, name: "#{name}_bn2")
    x = Axon.activation(x, activation, name: "#{name}_act2")

    x =
      if dropout > 0.0 do
        Axon.dropout(x, rate: dropout, name: "#{name}_drop2")
      else
        x
      end

    # Residual connection: project input channels if they differ from output
    skip = project_residual(input, out_channels, "#{name}_skip")

    Axon.add(x, skip, name: "#{name}_residual")
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output channel count for a TCN model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    channels = Keyword.get(opts, :channels, @default_channels)
    List.last(channels)
  end

  @doc """
  Calculate the receptive field for a TCN configuration.

  ## Options

    - `:num_layers` - Number of temporal blocks (default: 4)
    - `:kernel_size` - Convolution kernel size (default: 3)

  ## Returns

    The receptive field size in timesteps.
  """
  @spec receptive_field(keyword()) :: pos_integer()
  def receptive_field(opts \\ []) do
    num_layers = Keyword.get(opts, :num_layers, 4)
    kernel_size = Keyword.get(opts, :kernel_size, @default_kernel_size)

    # Each block has 2 conv layers with the same dilation.
    # Block i has dilation 2^i.
    # Per conv layer, the additional receptive field is (kernel_size - 1) * dilation.
    # With 2 conv layers per block:
    #   total = 1 + sum_{i=0}^{n-1} 2 * (kernel_size - 1) * 2^i
    #         = 1 + 2 * (kernel_size - 1) * (2^n - 1)
    1 + 2 * (kernel_size - 1) * (round(:math.pow(2, num_layers)) - 1)
  end

  @doc """
  Calculate the minimum number of layers needed for a target receptive field.

  ## Parameters

    - `target` - Desired receptive field in timesteps

  ## Options

    - `:kernel_size` - Convolution kernel size (default: 3)

  ## Returns

    Number of layers required.
  """
  @spec layers_for_receptive_field(pos_integer(), keyword()) :: pos_integer()
  def layers_for_receptive_field(target, opts \\ []) do
    kernel_size = Keyword.get(opts, :kernel_size, @default_kernel_size)

    # Solve: target <= 1 + 2 * (k - 1) * (2^n - 1)
    # 2^n >= (target - 1) / (2 * (k - 1)) + 1
    k_factor = 2 * (kernel_size - 1)

    if k_factor == 0 do
      raise ArgumentError, "kernel_size must be > 1"
    end

    min_power = (target - 1) / k_factor + 1
    ceil(:math.log2(max(min_power, 1)))
  end

  # ============================================================================
  # Private Helpers
  # ============================================================================

  # Build a causal (left-padded) dilated 1D convolution.
  # Uses Axon.nx for causal padding, then a standard conv.
  defp causal_conv1d(input, out_channels, kernel_size, dilation, causal_pad, name) do
    # Pad on the left for causal behavior
    padded =
      Axon.nx(
        input,
        fn x ->
          # x: [batch, seq_len, channels]
          batch_size = Nx.axis_size(x, 0)
          channels = Nx.axis_size(x, 2)
          pad_shape = {batch_size, causal_pad, channels}
          Nx.concatenate([Nx.broadcast(0.0, pad_shape), x], axis: 1)
        end,
        name: "#{name}_pad"
      )

    # Apply dilated conv with :valid padding (we already padded causally)
    Axon.conv(padded, out_channels,
      kernel_size: {kernel_size},
      strides: [1],
      padding: :valid,
      kernel_dilation: [dilation],
      name: name
    )
  end

  # Project skip connection to match output channels if needed.
  # Uses a 1x1 conv for channel projection.
  defp project_residual(input, out_channels, name) do
    Axon.conv(input, out_channels,
      kernel_size: {1},
      strides: [1],
      padding: :same,
      name: name
    )
  end
end
